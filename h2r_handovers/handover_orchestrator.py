"""Handover orchestrator state machine. Coordinates the full H2R handover pipeline."""

from __future__ import annotations

import enum
import sys
import threading
import time

import rclpy
import copy
from geometry_msgs.msg import Pose, PoseStamped
from scipy.spatial.transform import Rotation
from moveit_msgs.msg import (AllowedCollisionEntry, CollisionObject,
                             PlanningScene, PlanningSceneComponents)
from moveit_msgs.srv import GetPlanningScene
from shape_msgs.msg import SolidPrimitive
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from std_msgs.msg import Empty, String

import tf2_ros
import tf2_geometry_msgs
from h2r_handovers.hand_stabilization_node import STATUS_STABLE
from h2r_handovers.panda_client import ArmClient, GripperClient

import numpy as np
from scipy.spatial.transform import Rotation


def _flush_stdin() -> None:
    """Drop buffered keypresses so a stray Enter can't confirm a motion."""
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass


class State(enum.Enum):
    STARTUP = 'STARTUP'
    IDLE = 'IDLE'
    CAPTURING = 'CAPTURING'
    PLANNING = 'PLANNING'
    OPENING = 'OPENING'
    APPROACHING = 'APPROACHING'
    REACHING = 'REACHING'
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

        # Capture
        self.declare_parameter('capture_trigger_topic', 'capture_trigger')
        self.declare_parameter('hand_status_topic', 'hand_stability_status')
        self.declare_parameter('system_state_topic', 'system_state')
        self.declare_parameter('trigger_mode', 'auto')
        self.declare_parameter('hand_wait_timeout_sec', 30.0)
        self.declare_parameter('capture_timeout_sec', 15.0)

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

        self.declare_parameter('home_joints', [
            -0.3017, -0.9131, -1.1905, -1.7694, -0.8113, 1.4376, 0.8624])
        self.declare_parameter('dropoff_joints', [
            0.0761, 0.0087, 0.0108, -2.7316, -0.0448, 2.7889, 0.8629])

        # Offset to align fingertips (not hand base) with grasp point.
        self.declare_parameter('hand_tcp_offset', 0.1034)

        # Distance backed off along approach axis for final straight approach.
        self.declare_parameter('pregrasp_distance', 0.10)

        # Wait for user Enter to confirm planned motions (requires separate terminal).
        self.declare_parameter('confirm_before_execute', True)

        # Safety floor to prevent collision with table.
        self.declare_parameter('min_grasp_z', 0.05)

        # TCP frame offset (xyz) applied after anti-twist flip.
        self.declare_parameter('grasp_offset_tcp', [-0.09, 0.0, -0.09])

        # Project finger-closing axis to horizontal for straight-down approach (preserves yaw).
        self.declare_parameter('force_top_down_grasp', True)

        self.declare_parameter('table_enable', True)
        self.declare_parameter('table_position', [0.4, 0.0, -0.085])
        self.declare_parameter('table_size', [1.2, 1.2, 0.05])
        self.declare_parameter('table_ignore_collision_links', ['panda_link1'])

        # Release detection mode: 'timeout' or 'force' (TODO: F/T based).
        self.declare_parameter('release_detection_mode', 'timeout')
        self.declare_parameter('release_timeout_sec', 3.0)

        self._release_timeout = self.get_parameter('release_timeout_sec').value

        # Read parameters
        self._planning_frame = self.get_parameter('planning_frame').value
        self._home_joints = list(self.get_parameter('home_joints').value)
        self._dropoff_joints = list(self.get_parameter('dropoff_joints').value)
        self._hand_tcp_offset = self.get_parameter('hand_tcp_offset').value
        self._pregrasp_distance = self.get_parameter('pregrasp_distance').value
        self._confirm_before_execute = self.get_parameter('confirm_before_execute').value
        self._min_grasp_z = self.get_parameter('min_grasp_z').value
        self._grasp_offset_tcp = list(self.get_parameter('grasp_offset_tcp').value)
        self._force_top_down = self.get_parameter('force_top_down_grasp').value

        self._gripper_open_width = self.get_parameter('gripper_open_width').value
        self._gripper_open_speed = self.get_parameter('gripper_open_speed').value
        self._grasp_width = self.get_parameter('grasp_width').value
        self._grasp_speed = self.get_parameter('grasp_speed').value
        self._grasp_force = self.get_parameter('grasp_force').value
        self._grasp_eps_inner = self.get_parameter('grasp_epsilon_inner').value
        self._grasp_eps_outer = self.get_parameter('grasp_epsilon_outer').value

        self._release_mode = self.get_parameter('release_detection_mode').value
        self._release_timeout = self.get_parameter('release_timeout_sec').value

        self._trigger_mode = self.get_parameter('trigger_mode').value
        self._hand_wait_timeout = self.get_parameter('hand_wait_timeout_sec').value
        self._capture_timeout = self.get_parameter('capture_timeout_sec').value

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
        self._state = State.STARTUP
        self._sequence_confirmed = False
        self._target_grasp: PoseStamped | None = None
        self._grasp_was_flipped: bool = False
        self._trigger_time: float | None = None
        self._hand_status: str | None = None
        self._capture_lock = threading.Lock()

        # Latched for late subscribers
        state_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self._state_pub = self.create_publisher(
            String, self.get_parameter('system_state_topic').value, state_qos)
        self._scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)
        self._capture_trigger_pub = self.create_publisher(
            Empty, self.get_parameter('capture_trigger_topic').value, 1)

        # Reentrant callback group allows subscriptions to fire during blocking actions.
        cb_group = ReentrantCallbackGroup()
        self._get_scene_client = self.create_client(
            GetPlanningScene, '/get_planning_scene', callback_group=cb_group)
        self.create_subscription(
            PoseStamped,
            self.get_parameter('selected_grasp_topic').value,
            self._on_selected_grasp,
            10,
            callback_group=cb_group,
        )
        self.create_subscription(
            String,
            self.get_parameter('hand_status_topic').value,
            self._on_hand_status,
            10,
            callback_group=cb_group,
        )

        self.get_logger().info(
            f'Orchestrator ready.  State: {self._state.value}.  '
            f'Waiting for grasps on '
            f"'{self.get_parameter('selected_grasp_topic').value}'…")

        # Schedule the initial homing sequence to run shortly after startup
        self._startup_timer = self.create_timer(1.0, self._on_startup_timer, callback_group=cb_group)

        self._input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self._input_thread.start()

    def _on_startup_timer(self) -> None:
        if self._startup_timer is not None:
            self._startup_timer.cancel()
            self._startup_timer = None
        
        self._add_table_to_scene()

        self.get_logger().info('Performing startup homing...')
        self._set_state(State.HOMING)
        if self._arm.move_to_joints(self._home_joints, confirm=self._confirm('startup homing')):
            self.get_logger().info('Startup homing complete. Ready for handovers.')
        else:
            self.get_logger().warn('Startup homing failed.')
        self._set_state(State.IDLE)

    # Operator console (runs in its own thread)

    def _input_loop(self) -> None:
        while rclpy.ok():
            if self._state != State.IDLE:
                time.sleep(0.2)
                continue
            _flush_stdin()
            try:
                msg = '\nIDLE — press Enter to manual override and capture a grasp… ' if self._trigger_mode == 'auto' else '\nIDLE — press Enter to capture a grasp… '
                input(msg)
            except EOFError:
                return
            if self._state == State.IDLE:
                self.get_logger().info('Manual trigger activated via terminal.')
                threading.Thread(target=self._run_capture, daemon=True).start()

    def _run_capture(self) -> None:
        self._trigger_time = time.monotonic()
        self._set_state(State.CAPTURING)
        self._capture_trigger_pub.publish(Empty())

        deadline = time.monotonic() + self._capture_timeout
        while time.monotonic() < deadline:
            if self._state != State.CAPTURING:
                return
            time.sleep(0.2)

        with self._capture_lock:
            if self._state == State.CAPTURING:
                self.get_logger().warn(
                    f'No grasp within {self._capture_timeout:.0f}s — back to IDLE '
                    '(check grasp_debug_image / GraspNet logs).')
                self._set_state(State.IDLE)

    # Subscription callbacks

    def _on_hand_status(self, msg: String) -> None:
        prev_status = self._hand_status
        if msg.data != prev_status:
            self.get_logger().info(f'Hand status: {msg.data}')
        self._hand_status = msg.data
        
        # Auto trigger logic
        if self._trigger_mode == 'auto' and self._state == State.IDLE:
            if prev_status != STATUS_STABLE and self._hand_status == STATUS_STABLE:
                self.get_logger().info('Hand is stable. Auto-triggering capture.')
                threading.Thread(target=self._run_capture, daemon=True).start()

    def _on_selected_grasp(self, msg: PoseStamped) -> None:
        with self._capture_lock:
            if self._state != State.CAPTURING:
                self.get_logger().info(
                    f'Ignoring grasp — currently in {self._state.value}',
                    throttle_duration_sec=2.0)
                return
            self._set_state(State.PLANNING)

        self.get_logger().info('Received selected grasp — starting handover sequence')
        self._target_grasp = msg
        self._run_handover()

    # Main sequence

    def _run_handover(self) -> None:
        self._sequence_confirmed = False
        grasp_in_robot = self._transform_grasp()
        if grasp_in_robot is None:
            self._set_state(State.IDLE)
            return

        if self._force_top_down:
            self._force_top_down_orientation(grasp_in_robot.pose)
        self._apply_grasp_offset(grasp_in_robot.pose)

        # Log the full grasp pose for diagnostics
        p = grasp_in_robot.pose.position
        o = grasp_in_robot.pose.orientation
        approach_dir = self._get_approach_direction(o)
        self.get_logger().info(
            f'GRASP DIAGNOSTICS in {self._planning_frame}:\n'
            f'  position:    ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})\n'
            f'  quaternion:  ({o.x:.4f}, {o.y:.4f}, {o.z:.4f}, {o.w:.4f})\n'
            f'  approach_dir (Z-axis): ({approach_dir[0]:.3f}, {approach_dir[1]:.3f}, {approach_dir[2]:.3f})')


        # Shift pose back so fingertips align with grasp point.
        hand_grasp = copy.deepcopy(grasp_in_robot)
        hand_grasp.pose.position.x -= approach_dir[0] * self._hand_tcp_offset
        hand_grasp.pose.position.y -= approach_dir[1] * self._hand_tcp_offset
        hand_grasp.pose.position.z -= approach_dir[2] * self._hand_tcp_offset

        # Back off for straight-line final approach.
        pre_grasp = copy.deepcopy(hand_grasp)
        pre_grasp.pose.position.x -= approach_dir[0] * self._pregrasp_distance
        pre_grasp.pose.position.y -= approach_dir[1] * self._pregrasp_distance
        pre_grasp.pose.position.z -= approach_dir[2] * self._pregrasp_distance
        pre_grasp.pose.position.z = max(pre_grasp.pose.position.z, self._min_grasp_z)

        hp = hand_grasp.pose.position
        pp = pre_grasp.pose.position
        self.get_logger().info(
            f'Hand-base targets in {self._planning_frame} '
            f'(hand_tcp_offset={self._hand_tcp_offset:.4f} m, '
            f'pregrasp_distance={self._pregrasp_distance:.3f} m):\n'
            f'  pre-grasp: ({pp.x:.3f}, {pp.y:.3f}, {pp.z:.3f})\n'
            f'  grasp:     ({hp.x:.3f}, {hp.y:.3f}, {hp.z:.3f})')

        # 1. Open gripper
        if not self._transition(State.OPENING,
                                lambda: self._gripper.open(
                                    self._gripper_open_width,
                                    self._gripper_open_speed)):
            return self._handle_abort()

        # 2. Move arm to the pre-grasp pose
        self.get_logger().info('Moving to pre-grasp pose...')
        if not self._transition(State.APPROACHING,
                                lambda: self._arm.move_to_pose(
                                    pre_grasp, confirm=self._confirm('pre-grasp'))):
            return self._handle_abort()

        # 3. Final approach along the grasp axis onto the object
        self.get_logger().info('Final approach to grasp pose...')
        if not self._transition(State.REACHING,
                                lambda: self._final_approach(hand_grasp)):
            return self._handle_abort()

        # 4. Close gripper on the object
        if not self._transition(State.GRASPING,
                                lambda: self._gripper.grasp(
                                    self._grasp_width,
                                    self._grasp_speed,
                                    self._grasp_force,
                                    self._grasp_eps_inner,
                                    self._grasp_eps_outer)):
            return self._handle_abort()

        trigger_to_grasp = (time.monotonic() - self._trigger_time
                            if self._trigger_time is not None else None)
        if trigger_to_grasp is not None:
            self.get_logger().info(
                f'Trigger→grasp time: {trigger_to_grasp:.2f} s '
                '(Enter press to gripper closed).')

        # 5. Wait for human to release the object
        if not self._transition(State.HOLDING, self._wait_for_release):
            return self._handle_abort()

        # 6. Retreat to home
        if not self._transition(State.RETREATING,
                                lambda: self._arm.move_to_joints(
                                    self._home_joints, confirm=self._confirm('retreat home'))):
            return self._handle_abort()

        # 7. Go to dropoff position
        if not self._transition(State.DROPPING,
                                lambda: self._arm.move_to_joints(
                                    self._dropoff_joints, confirm=self._confirm('dropoff'))):
            return self._handle_abort()

        # 8. Release the object
        if not self._transition(State.RELEASING,
                                lambda: self._gripper.open(
                                    self._gripper_open_width,
                                    self._gripper_open_speed)):
            return self._handle_abort()

        # 9. Return home
        if not self._transition(State.HOMING,
                                lambda: self._arm.move_to_joints(
                                    self._home_joints, confirm=self._confirm('return home'))):
            return self._handle_abort()

        self._set_state(State.IDLE)
        timing = (f' (trigger→grasp {trigger_to_grasp:.2f} s)'
                  if trigger_to_grasp is not None else '')
        self.get_logger().info(f'Handover complete{timing} — ready for next')

    # Helpers

    def _add_table_to_scene(self) -> None:
        if not self.get_parameter('table_enable').value:
            return
        position = list(self.get_parameter('table_position').value)
        size = list(self.get_parameter('table_size').value)

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = size

        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = position
        pose.orientation.w = 1.0

        obj = CollisionObject()
        obj.header.frame_id = self._planning_frame
        obj.id = 'table'
        obj.primitives.append(box)
        obj.primitive_poses.append(pose)
        obj.operation = CollisionObject.ADD

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(obj)
        self._scene_pub.publish(scene)
        self.get_logger().info(
            f'Added table collision box to planning scene: centre {position}, size {size}.')

        links = list(self.get_parameter('table_ignore_collision_links').value)
        if links:
            self._allow_table_collisions(links)

    def _allow_table_collisions(self, links: list[str]) -> None:
        """Injects the table object into the runtime allowed collision matrix (ACM) for the given links, bypassing URDF limitations."""
        if not self._get_scene_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(
                'get_planning_scene service unavailable — collisions between '
                f'the table and {links} stay enabled.')
            return
        request = GetPlanningScene.Request()
        request.components.components = \
            PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        future = self._get_scene_client.call_async(request)
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout=5.0):
            self.get_logger().warn(
                'get_planning_scene did not respond — collisions between '
                f'the table and {links} stay enabled.')
            return
        acm = future.result().scene.allowed_collision_matrix

        for name in ['table'] + links:
            if name not in acm.entry_names:
                for row in acm.entry_values:
                    row.enabled.append(False)
                acm.entry_names.append(name)
                acm.entry_values.append(
                    AllowedCollisionEntry(enabled=[False] * len(acm.entry_names)))
        table_idx = acm.entry_names.index('table')
        for link in links:
            link_idx = acm.entry_names.index(link)
            acm.entry_values[table_idx].enabled[link_idx] = True
            acm.entry_values[link_idx].enabled[table_idx] = True

        scene = PlanningScene()
        scene.is_diff = True
        scene.allowed_collision_matrix = acm
        self._scene_pub.publish(scene)
        self.get_logger().info(
            f'Allowed collisions between the table and {links}.')

    def _confirm(self, label: str):
        """Blocks on stdin to allow RViz trajectory inspection before execution. Only prompts on the first segment."""
        if not self._confirm_before_execute:
            return None

        def _ask() -> bool:
            if self._sequence_confirmed:
                return True
            _flush_stdin()
            answer = input(
                f"[{label}] planned — check the trajectory in RViz. "
                "Press Enter to run the full sequence, or 'n' + Enter to abort: ")
            if answer.strip().lower() in ('n', 'no'):
                return False
            self._sequence_confirmed = True
            return True

        return _ask

    def _final_approach(self, grasp_pose: PoseStamped) -> bool:
        """Prefer straight-line (LIN) motion for the final approach, falling back to joint-space (PTP)."""
        if self._arm.move_to_pose(grasp_pose, planner_id='LIN',
                                  confirm=self._confirm('final approach (LIN)')):
            return True
        self.get_logger().warn('LIN final approach failed — falling back to PTP.')
        return self._arm.move_to_pose(grasp_pose, planner_id='PTP',
                                      confirm=self._confirm('final approach (PTP)'))

    def _transition(self, state: State, action) -> bool:
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
        self._arm.move_to_joints(self._home_joints, confirm=self._confirm('abort homing'))
        
        self._set_state(State.IDLE)

    def _set_state(self, state: State) -> None:
        self._state = state
        self.get_logger().info(f'State → {state.value}')
        msg = String()
        msg.data = state.value
        self._state_pub.publish(msg)

    def _transform_grasp(self) -> PoseStamped | None:
        """Transform the grasp from the camera frame into the robot planning frame."""
        try:
            transformed = self._tf_buffer.transform(
                self._target_grasp, self._planning_frame, timeout=rclpy.duration.Duration(seconds=2))
            
            # Flip grasp 180° around approach axis if hand's "up" vector points down to prevent wrist twist.
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
            
            self._grasp_was_flipped = x_axis[2] < 0
            if self._grasp_was_flipped:
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

    def _force_top_down_orientation(self, pose: Pose) -> None:
        """Projects the finger-closing axis onto the horizontal plane to force a vertical approach while maintaining yaw alignment."""
        q = pose.orientation
        closing = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()[:, 1]
        closing[2] = 0.0
        norm = np.linalg.norm(closing)
        y_axis = closing / norm if norm > 1e-6 else np.array([0.0, 1.0, 0.0])
        z_axis = np.array([0.0, 0.0, -1.0])
        x_axis = np.cross(y_axis, z_axis)
        q_new = Rotation.from_matrix(
            np.column_stack([x_axis, y_axis, z_axis])).as_quat()
        (pose.orientation.x, pose.orientation.y,
         pose.orientation.z, pose.orientation.w) = q_new.tolist()
        self.get_logger().info('Forced top-down grasp orientation (yaw kept).')

    def _apply_grasp_offset(self, pose: Pose) -> None:
        """Applies a translational offset in the TCP frame for fine-tuning grasp depth/centering."""
        offset = list(self._grasp_offset_tcp)
        if not any(offset):
            return
        if self._grasp_was_flipped:
            offset[0] = -offset[0]
            offset[1] = -offset[1]
        from scipy.spatial.transform import Rotation
        q = pose.orientation
        shift = Rotation.from_quat([q.x, q.y, q.z, q.w]).apply(offset)
        pose.position.x += shift[0]
        pose.position.y += shift[1]
        pose.position.z += shift[2]
        self.get_logger().info(
            f'Applied TCP-frame grasp offset {offset}'
            f'{" (flip-compensated)" if self._grasp_was_flipped else ""} → shift in '
            f'{self._planning_frame}: ({shift[0]:.3f}, {shift[1]:.3f}, {shift[2]:.3f})')

    @staticmethod
    def _get_approach_direction(orientation):
        rot = Rotation.from_quat(
            [orientation.x, orientation.y, orientation.z, orientation.w])
        return rot.as_matrix()[:, 2]

    def _wait_for_release(self) -> bool:
        """Waits for the user to release the object via timeout or F/T sensor load drop."""
        if self._release_mode == 'force':
            # TODO: implement F/T-based release detection
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
    # MultiThreadedExecutor allows subscription callbacks during blocking action results.
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
