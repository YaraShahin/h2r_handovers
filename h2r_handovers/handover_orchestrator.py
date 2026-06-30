"""Handover orchestrator — state machine that coordinates the full H2R handover.

Lifecycle:
    IDLE  →  receive selected_grasp
          →  transform it from camera frame into robot planning frame
          →  open gripper  →  move arm to grasp pose  →  close gripper (grasp)
          →  wait for human to release (HOLDING)
          →  retreat to HOME  →  go to DROPOFF  →  release  →  return HOME
          →  IDLE (ready for next handover)

All arm and gripper commands use the action-client helpers in ``panda_client``.
"""

from __future__ import annotations

import enum

import rclpy
import copy
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

import tf2_ros
import tf2_geometry_msgs 
from h2r_handovers.panda_client import ArmClient, GripperClient


class State(enum.Enum):
    IDLE = 'IDLE'
    OPENING = 'OPENING'
    APPROACHING = 'APPROACHING'
    GRASPING = 'GRASPING'
    HOLDING = 'HOLDING'
    RETREATING = 'RETREATING'
    DROPPING = 'DROPPING'
    RELEASING = 'RELEASING'
    HOMING = 'HOMING'


class HandoverOrchestrator(Node):

    def __init__(self):
        super().__init__('handover_orchestrator')

        # Parameters
        self.declare_parameter('selected_grasp_topic', 'selected_grasp')

        # Arm / MoveIt
        self.declare_parameter('planning_group', 'panda_arm')
        self.declare_parameter('planning_frame', 'panda_link0')
        self.declare_parameter('ee_link', 'panda_hand')
        self.declare_parameter('max_velocity_scaling', 0.2)
        self.declare_parameter('max_acceleration_scaling', 0.1)
        self.declare_parameter('planning_time', 5.0)

        # Gripper
        self.declare_parameter('gripper_open_width', 0.08)
        self.declare_parameter('gripper_open_speed', 0.1)
        self.declare_parameter('grasp_width', 0.0)
        self.declare_parameter('grasp_speed', 0.03)
        self.declare_parameter('grasp_force', 40.0)
        self.declare_parameter('grasp_epsilon_inner', 0.1)
        self.declare_parameter('grasp_epsilon_outer', 0.1)

        # Predefined joint configurations
        # HOME — safe resting pose, arm tucked and out of the way
        self.declare_parameter('home_joints', [
            -0.3017, -0.9131, -1.1905, -1.7694, -0.8113, 1.4376, 0.8624])
        # DROPOFF — where the robot places the received object
        self.declare_parameter('dropoff_joints', [
            0.0761, 0.0087, 0.0108, -2.7316, -0.0448, 2.7889, 0.8629])

        # Grasp pose offset — applied along the robot's Z axis (upward)
        # to approach slightly above the raw grasp point
        self.declare_parameter('approach_offset_z', 0.08)

        # Minimum Z height (in planning frame) — safety floor to prevent
        # the robot from going below the table surface.
        self.declare_parameter('min_grasp_z', 0.05)

        # Release detection mode:
        #   'timeout'  — wait a fixed duration then proceed (default, works now)
        #   'force'    — monitor F/T sensor until weight drops (TODO)
        self.declare_parameter('release_detection_mode', 'timeout')
        self.declare_parameter('release_timeout_sec', 3.0)

        self._release_timeout = self.get_parameter('release_timeout_sec').value

        # Read parameters
        self._planning_frame = self.get_parameter('planning_frame').value
        self._home_joints = list(self.get_parameter('home_joints').value)
        self._dropoff_joints = list(self.get_parameter('dropoff_joints').value)
        self._approach_offset_z = self.get_parameter('approach_offset_z').value
        self._min_grasp_z = self.get_parameter('min_grasp_z').value

        self._gripper_open_width = self.get_parameter('gripper_open_width').value
        self._gripper_open_speed = self.get_parameter('gripper_open_speed').value
        self._grasp_width = self.get_parameter('grasp_width').value
        self._grasp_speed = self.get_parameter('grasp_speed').value
        self._grasp_force = self.get_parameter('grasp_force').value
        self._grasp_eps_inner = self.get_parameter('grasp_epsilon_inner').value
        self._grasp_eps_outer = self.get_parameter('grasp_epsilon_outer').value

        self._release_mode = self.get_parameter('release_detection_mode').value
        self._release_timeout = self.get_parameter('release_timeout_sec').value

        # TF2 setup
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Action clients
        self._arm = ArmClient(
            self,
            planning_group=self.get_parameter('planning_group').value,
            planning_frame=self._planning_frame,
            ee_link=self.get_parameter('ee_link').value,
            max_velocity_scaling=self.get_parameter('max_velocity_scaling').value,
            max_acceleration_scaling=self.get_parameter('max_acceleration_scaling').value,
            planning_time=self.get_parameter('planning_time').value,
        )
        self._gripper = GripperClient(self)

        # State machine initialization
        self._state = State.IDLE
        self._target_grasp: PoseStamped | None = None

        # Use a reentrant callback group so the subscription callback can
        # fire while action clients are blocking in _run_handover
        cb_group = ReentrantCallbackGroup()
        self.create_subscription(
            PoseStamped,
            self.get_parameter('selected_grasp_topic').value,
            self._on_selected_grasp,
            10,
            callback_group=cb_group,
        )

        self.get_logger().info(
            f'Orchestrator ready.  State: {self._state.value}.  '
            f'Waiting for grasps on '
            f"'{self.get_parameter('selected_grasp_topic').value}'…")

        # Schedule the initial homing sequence to run shortly after startup
        self._startup_timer = self.create_timer(1.0, self._on_startup_timer, callback_group=cb_group)

    def _on_startup_timer(self) -> None:
        if self._startup_timer is not None:
            self._startup_timer.cancel()
            self._startup_timer = None
        
        self.get_logger().info('Performing startup homing...')
        self._set_state(State.HOMING)
        if self._arm.move_to_joints(self._home_joints):
            self.get_logger().info('Startup homing complete. Ready for handovers.')
        else:
            self.get_logger().warn('Startup homing failed.')
        self._set_state(State.IDLE)

    # Subscription callbacks

    def _on_selected_grasp(self, msg: PoseStamped) -> None:
        if self._state != State.IDLE:
            self.get_logger().info(
                f'Ignoring grasp — currently in {self._state.value}',
                throttle_duration_sec=2.0)
            return

        self.get_logger().info('Received selected grasp — starting handover sequence')
        self._target_grasp = msg
        self._run_handover()

    # Main sequence

    def _run_handover(self) -> None:
        """Execute the full handover state machine sequentially.

        Each step blocks until its action completes.  On any failure the
        sequence aborts and returns to IDLE so the robot doesn't continue
        moving after an error.
        """
        grasp_in_robot = self._transform_grasp()
        if grasp_in_robot is None:
            self._set_state(State.IDLE)
            return

        # Log the full grasp pose for diagnostics
        p = grasp_in_robot.pose.position
        o = grasp_in_robot.pose.orientation
        approach_dir = self._get_approach_direction(o)
        self.get_logger().info(
            f'GRASP DIAGNOSTICS in {self._planning_frame}:\n'
            f'  position:    ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})\n'
            f'  quaternion:  ({o.x:.4f}, {o.y:.4f}, {o.z:.4f}, {o.w:.4f})\n'
            f'  approach_dir (Z-axis): ({approach_dir[0]:.3f}, {approach_dir[1]:.3f}, {approach_dir[2]:.3f})')

        # Safety clamp: prevent the robot from going below the table
        if grasp_in_robot.pose.position.z < self._min_grasp_z:
            self.get_logger().warn(
                f'Grasp Z={grasp_in_robot.pose.position.z:.3f} is below '
                f'min_grasp_z={self._min_grasp_z:.3f} — clamping to floor.')
            grasp_in_robot.pose.position.z = self._min_grasp_z

        # 1. Open gripper
        if not self._transition(State.OPENING,
                                lambda: self._gripper.open(
                                    self._gripper_open_width,
                                    self._gripper_open_speed)):
            return self._handle_abort()

        # 2. Move arm to the actual grasp pose
        self.get_logger().info('Moving to grasp pose...')
        if not self._transition(State.APPROACHING,
                                lambda: self._arm.move_to_pose(grasp_in_robot)):
            return self._handle_abort()

        # 3. Close gripper on the object
        if not self._transition(State.GRASPING,
                                lambda: self._gripper.grasp(
                                    self._grasp_width,
                                    self._grasp_speed,
                                    self._grasp_force,
                                    self._grasp_eps_inner,
                                    self._grasp_eps_outer)):
            return self._handle_abort()

        # 4. Wait for human to release the object
        if not self._transition(State.HOLDING, self._wait_for_release):
            return self._handle_abort()

        # 5. Retreat to home
        if not self._transition(State.RETREATING,
                                lambda: self._arm.move_to_joints(self._home_joints)):
            return self._handle_abort()

        # 6. Go to dropoff position
        if not self._transition(State.DROPPING,
                                lambda: self._arm.move_to_joints(self._dropoff_joints)):
            return self._handle_abort()

        # 7. Release the object
        if not self._transition(State.RELEASING,
                                lambda: self._gripper.open(
                                    self._gripper_open_width,
                                    self._gripper_open_speed)):
            return self._handle_abort()

        # 8. Return home
        if not self._transition(State.HOMING,
                                lambda: self._arm.move_to_joints(self._home_joints)):
            return self._handle_abort()

        self._set_state(State.IDLE)
        self.get_logger().info('Handover complete — ready for next')

    # Helpers

    def _transition(self, state: State, action) -> bool:
        """Move to *state*, execute *action*; return True on success, False on failure."""
        self._set_state(state)
        success = action()
        if not success:
            self.get_logger().error(f'{state.value} failed.')
        return success

    def _handle_abort(self) -> None:
        """Called when any state transition fails or is cancelled."""
        self.get_logger().warn('Handover sequence aborted.')
        
        self.get_logger().info('Returning to HOME after abort...')
        self._set_state(State.HOMING)
        self._arm.move_to_joints(self._home_joints)
        
        self._set_state(State.IDLE)

    def _set_state(self, state: State) -> None:
        self._state = state
        self.get_logger().info(f'State → {state.value}')

    def _transform_grasp(self) -> PoseStamped | None:
        """Transform the grasp from the camera frame into the robot planning frame."""
        try:
            transformed = self._tf_buffer.transform(
                self._target_grasp, self._planning_frame, timeout=rclpy.duration.Duration(seconds=2))
            
            # Prevent the robot arm from wrapping around 360 degrees or twisting awkwardly:
            # We check if the Franka Hand's X-axis (which points 'up' relative to the fingers)
            # is pointing downwards in the robot's base frame (Z < 0).
            # If so, we rotate the grasp 180 degrees around the approach axis (Z-axis).
            from scipy.spatial.transform import Rotation
            q = [
                transformed.pose.orientation.x,
                transformed.pose.orientation.y,
                transformed.pose.orientation.z,
                transformed.pose.orientation.w
            ]
            rot = Rotation.from_quat(q)
            # The X-axis is the first column of the rotation matrix
            x_axis = rot.as_matrix()[:, 0]
            
            if x_axis[2] < 0:
                self.get_logger().info('Flipping grasp 180 degrees around Z to prevent wrist twist.')
                # Rotate 180 degrees around local Z-axis
                rot_180_z = Rotation.from_euler('z', 180, degrees=True)
                rot_flipped = rot * rot_180_z
                q_flipped = rot_flipped.as_quat()
                transformed.pose.orientation.x = q_flipped[0]
                transformed.pose.orientation.y = q_flipped[1]
                transformed.pose.orientation.z = q_flipped[2]
                transformed.pose.orientation.w = q_flipped[3]

            self.get_logger().info(
                f'Transformed grasp: '
                f'({transformed.pose.position.x:.3f}, '
                f'{transformed.pose.position.y:.3f}, '
                f'{transformed.pose.position.z:.3f}) '
                f'in {self._planning_frame}')
            return transformed
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f'TF lookup failed: {e}')
            self._set_state(State.IDLE)
            return None

    @staticmethod
    def _get_approach_direction(orientation):
        """Extract the Z-axis (approach direction) from a quaternion.

        Returns a tuple (x, y, z) representing the third column of the
        rotation matrix encoded by the quaternion.  Used for diagnostics.
        """
        qx, qy, qz, qw = orientation.x, orientation.y, orientation.z, orientation.w
        return (
            2.0 * (qx * qz + qw * qy),
            2.0 * (qy * qz - qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy),
        )

    def _wait_for_release(self) -> bool:
        """Wait until the human releases the object.

        Current modes:
        - 'timeout': simply wait a fixed duration (works out of the box).

        Future mode:
        - 'force': subscribe to the Panda's F/T sensor (external torque
          estimates from ``/franka_robot_state_broadcaster/robot_state``)
          and wait until the sensed load drops below a threshold,
          indicating the human has let go.  This avoids the robot yanking
          the object away before the human is ready.
        """
        if self._release_mode == 'force':
            # TODO: implement F/T-based release detection
            #   1. Subscribe to robot state topic for external torques
            #   2. Wait until external force magnitude drops below threshold
            #   3. Return True when released, False on timeout
            self.get_logger().warn(
                "release_detection_mode='force' is not yet implemented — "
                "falling back to timeout")

        self.get_logger().info(
            f'HOLDING — waiting {self._release_timeout}s for human to release…')
        self.get_clock().sleep_for(Duration(seconds=self._release_timeout))
        return True


def main(args=None):
    rclpy.init(args=args)
    node = HandoverOrchestrator()
    # MultiThreadedExecutor so subscription callback can fire while
    # _run_handover blocks on action results
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
