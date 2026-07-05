"""Helper action clients for the Franka Panda arm and gripper.

Each class is constructed with a reference to the parent ``rclpy.Node`` so it
can create action clients, log messages, and spin futures without owning its
own node.  This keeps the orchestrator focused on state-machine logic.

Robot interfaces used (all standard Bosch franka_ros2 / MoveIt 2):

* ``moveit_msgs/action/MoveGroup``     →  ``/move_action``
* ``franka_msgs/action/Move``          →  ``/panda_gripper/move``
* ``franka_msgs/action/Grasp``         →  ``/panda_gripper/grasp``
"""

from __future__ import annotations

import math
from typing import Callable

from rclpy.node import Node

from franka_msgs.action import Grasp as GraspAction
from franka_msgs.action import Move as MoveAction
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    MoveItErrorCodes,
    OrientationConstraint,
    PlanningOptions,
    PositionConstraint,
)
from rclpy.action import ActionClient
from rclpy.duration import Duration
from shape_msgs.msg import SolidPrimitive
from threading import Event

def _wait_future(future):
    event = Event()
    future.add_done_callback(lambda f: event.set())
    event.wait()
    return future.result()




# MoveIt 2 MoveGroup action client wrapper
class ArmClient:
    """Thin wrapper around the ``/move_action`` MoveGroup action server."""

    # Default joint names for the 7-DOF Panda arm
    JOINT_NAMES = [
        'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
        'panda_joint5', 'panda_joint6', 'panda_joint7',
    ]

    def __init__(
        self,
        node: Node,
        *,
        action_name: str = '/move_action',
        planning_group: str = 'panda_arm',
        planning_frame: str = 'panda_link0',
        ee_link: str = 'panda_hand',
        max_velocity_scaling: float = 0.2,
        max_acceleration_scaling: float = 0.1,
        planning_time: float = 5.0,
        num_attempts: int = 5,
        position_tolerance: float = 0.005,
        orientation_tolerance: float = 0.02,
        joint_tolerance: float = 0.01,
    ):
        self._node = node
        self._group = planning_group
        self._frame = planning_frame
        self._ee_link = ee_link
        self._max_vel = max_velocity_scaling
        self._max_acc = max_acceleration_scaling
        self._planning_time = planning_time
        self._num_attempts = num_attempts
        self._pos_tol = position_tolerance
        self._orient_tol = orientation_tolerance
        self._joint_tol = joint_tolerance

        self._client = ActionClient(node, MoveGroup, action_name)
        node.get_logger().info(f'ArmClient: waiting for {action_name}…')
        self._client.wait_for_server()
        self._exec_client = ActionClient(node, ExecuteTrajectory, '/execute_trajectory')
        node.get_logger().info('ArmClient: waiting for /execute_trajectory…')
        self._exec_client.wait_for_server()
        node.get_logger().info(f'ArmClient: connected to {action_name}')

    def move_to_pose(self, pose_stamped: PoseStamped, planner_id: str = 'PTP',
                     confirm: Callable[[], bool] | None = None) -> bool:
        """Plan and execute to a Cartesian pose goal.  Blocks until done.

        If *confirm* is given, it is called after planning succeeds (the planned
        trajectory is visible in RViz at that point); execution only proceeds
        if it returns True.
        """
        request = self._base_request(planner_id=planner_id)

        # Position constraint — a tiny sphere around the target
        pc = PositionConstraint()
        pc.header = pose_stamped.header
        pc.link_name = self._ee_link
        pc.weight = 1.0
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [self._pos_tol]
        region = BoundingVolume()
        region.primitives.append(sphere)
        region.primitive_poses.append(pose_stamped.pose)
        pc.constraint_region = region

        # Orientation constraint
        oc = OrientationConstraint()
        oc.header = pose_stamped.header
        oc.link_name = self._ee_link
        oc.orientation = pose_stamped.pose.orientation
        oc.absolute_x_axis_tolerance = self._orient_tol
        oc.absolute_y_axis_tolerance = self._orient_tol
        oc.absolute_z_axis_tolerance = self._orient_tol
        oc.weight = 1.0

        constraint = Constraints()
        constraint.position_constraints.append(pc)
        constraint.orientation_constraints.append(oc)
        request.goal_constraints.append(constraint)

        return self._plan_then_execute(request, 'move_to_pose', confirm)

    def move_to_joints(self, joint_values: list[float], planner_id: str = 'PTP',
                       confirm: Callable[[], bool] | None = None) -> bool:
        """Plan and execute to a joint-space goal.  Blocks until done."""
        if len(joint_values) != len(self.JOINT_NAMES):
            self._node.get_logger().error(
                f'Expected {len(self.JOINT_NAMES)} joint values, got {len(joint_values)}')
            return False

        request = self._base_request(planner_id=planner_id)

        constraint = Constraints()
        for name, value in zip(self.JOINT_NAMES, joint_values):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = value
            jc.tolerance_above = self._joint_tol
            jc.tolerance_below = self._joint_tol
            jc.weight = 1.0
            constraint.joint_constraints.append(jc)
        request.goal_constraints.append(constraint)

        return self._plan_then_execute(request, 'move_to_joints', confirm)



    def _base_request(self, planner_id: str = 'PTP') -> MotionPlanRequest:
        req = MotionPlanRequest()
        req.planner_id = planner_id
        req.group_name = self._group
        req.num_planning_attempts = self._num_attempts
        req.allowed_planning_time = self._planning_time
        req.max_velocity_scaling_factor = self._max_vel
        req.max_acceleration_scaling_factor = self._max_acc
        return req

    def _plan_then_execute(self, request: MotionPlanRequest, label: str,
                           confirm: Callable[[], bool] | None = None) -> bool:
        """Plan the motion, optionally wait for confirmation, then execute it.

        Planning and execution are two separate MoveIt actions so the planned
        trajectory can be inspected in RViz (move_group publishes it on
        /display_planned_path) before the robot moves.
        """
        goal = MoveGroup.Goal()
        goal.request = request
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = True

        self._node.get_logger().info(f'ArmClient: planning {label}…')
        goal_handle = _wait_future(self._client.send_goal_async(goal))
        if not goal_handle.accepted:
            self._node.get_logger().error(f'ArmClient: {label} planning goal REJECTED')
            return False

        plan_result = _wait_future(goal_handle.get_result_async()).result
        if plan_result.error_code.val != MoveItErrorCodes.SUCCESS:
            self._node.get_logger().error(
                f'ArmClient: {label} planning failed (error code {plan_result.error_code.val})')
            return False

        if confirm is not None and not confirm():
            self._node.get_logger().warn(f'ArmClient: {label} execution declined by user.')
            return False

        exec_goal = ExecuteTrajectory.Goal()
        exec_goal.trajectory = plan_result.planned_trajectory

        self._node.get_logger().info(f'ArmClient: executing {label}…')
        exec_handle = _wait_future(self._exec_client.send_goal_async(exec_goal))
        if not exec_handle.accepted:
            self._node.get_logger().error(f'ArmClient: {label} execution goal REJECTED')
            return False

        exec_result = _wait_future(exec_handle.get_result_async()).result
        if exec_result.error_code.val == MoveItErrorCodes.SUCCESS:
            self._node.get_logger().info(f'ArmClient: {label} succeeded')
            return True
        self._node.get_logger().error(
            f'ArmClient: {label} execution failed (error code {exec_result.error_code.val})')
        return False


# Franka gripper action servers wrapper
class GripperClient:
    """Thin wrapper around ``/panda_gripper/move`` and ``/panda_gripper/grasp``."""

    def __init__(
        self,
        node: Node,
        *,
        move_action: str = '/panda_gripper/move',
        grasp_action: str = '/panda_gripper/grasp',
    ):
        self._node = node

        self._move_client = ActionClient(node, MoveAction, move_action)
        self._grasp_client = ActionClient(node, GraspAction, grasp_action)

        node.get_logger().info(f'GripperClient: waiting for {move_action}…')
        self._move_client.wait_for_server()
        node.get_logger().info(f'GripperClient: waiting for {grasp_action}…')
        self._grasp_client.wait_for_server()
        node.get_logger().info('GripperClient: connected')

    def open(self, width: float = 0.08, speed: float = 0.1) -> bool:
        """Open the gripper to *width* metres at *speed* m/s."""
        goal = MoveAction.Goal()
        goal.width = width
        goal.speed = speed
        return self._send(self._move_client, goal, 'open')

    def grasp(
        self,
        width: float,
        speed: float = 0.03,
        force: float = 40.0,
        epsilon_inner: float = 0.005,
        epsilon_outer: float = 0.005,
    ) -> bool:
        """Close the gripper on an object with the given force."""
        goal = GraspAction.Goal()
        goal.width = width
        goal.speed = speed
        goal.force = force
        goal.epsilon.inner = epsilon_inner
        goal.epsilon.outer = epsilon_outer
        return self._send(self._grasp_client, goal, 'grasp')

    def _send(self, client: ActionClient, goal, label: str) -> bool:
        self._node.get_logger().info(f'GripperClient: sending {label} goal…')
        future = client.send_goal_async(goal)
        goal_handle = _wait_future(future)

        if not goal_handle.accepted:
            self._node.get_logger().error(f'GripperClient: {label} goal REJECTED')
            return False

        result_future = goal_handle.get_result_async()
        result = _wait_future(result_future)

        if result.result.success:
            self._node.get_logger().info(f'GripperClient: {label} succeeded')
            return True
        else:
            self._node.get_logger().warn(
                f'GripperClient: {label} reported failure '
                f'(error: {getattr(result.result, "error", "unknown")})')
            return False


