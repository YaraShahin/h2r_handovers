"""Grasp selection node.

Subscribes to the scored grasp candidates from GraspNet and applies a
configurable selection policy to pick the single best grasp, published as
a PoseStamped for the orchestrator to act on.

The selected pose is republished unchanged (position + orientation) in the
camera frame; the orchestrator transforms it into the planning frame via TF.
The only frame handling here is an optional frame_id override used to route
the pose through the calibration-correction frame published in
handover.launch.xml, instead of baking that correction into the pose values.
"""

from functools import partial

import rclpy
from geometry_msgs.msg import PointStamped, PoseArray, PoseStamped
from rclpy.node import Node
import tf2_ros

from h2r_handovers.policies import POLICIES


class GraspSelectionNode(Node):

    def __init__(self):
        super().__init__('grasp_selection_node')

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self.declare_parameter('input_topic', 'grasp_candidates')
        self.declare_parameter('output_topic', 'selected_grasp')
        self.declare_parameter('policy', 'highest_score')
        # 'top_down' policy only: maximum tilt (degrees) of the approach axis
        # from straight-down in panda_link0, and whether candidates outside
        # that cone are rejected outright (strict) or the most downward-
        # pointing one is used as a fallback (non-strict, old behaviour).
        self.declare_parameter('top_down_max_tilt_deg', 45.0)
        self.declare_parameter('top_down_strict', True)
        # 'ergonomic' policy only: 3-D hand centroid published by the GraspNet
        # driver (same camera frame and header stamp as the candidates).
        self.declare_parameter('hand_center_topic', 'hand_center')
        # If non-empty, replaces the frame_id on the published pose. Used to
        # stamp poses with the calibration-correction frame from
        # handover.launch.xml; set to '' once the camera is recalibrated.
        self.declare_parameter('output_frame', 'camera_color_optical_corrected')

        policy_name = self.get_parameter('policy').value
        if policy_name not in POLICIES:
            self.get_logger().error(
                f"Unknown policy '{policy_name}'. "
                f"Available: {list(POLICIES.keys())}.  Falling back to 'highest_score'.")
            policy_name = 'highest_score'
        self._policy = POLICIES[policy_name]
        self._policy_name = policy_name
        if policy_name == 'top_down':
            max_tilt = self.get_parameter('top_down_max_tilt_deg').value
            strict = self.get_parameter('top_down_strict').value
            self._policy = partial(self._policy, max_tilt_deg=max_tilt, strict=strict)
            self.get_logger().info(
                f'top_down policy: grasps tilted more than {max_tilt:.0f} deg '
                f'from vertical are '
                + ('rejected (strict).' if strict
                   else 'allowed as fallback (non-strict).'))

        self._output_frame = self.get_parameter('output_frame').value
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        self._pub = self.create_publisher(PoseStamped, output_topic, 1)
        self.create_subscription(PoseArray, input_topic, self._on_candidates, 10)
        self._last_hand_center: PointStamped | None = None
        self.create_subscription(
            PointStamped, self.get_parameter('hand_center_topic').value,
            self._on_hand_center, 10)

        self.get_logger().info(
            f"Grasp selection: policy='{policy_name}', "
            f"subscribing to '{input_topic}', publishing to '{output_topic}'")

    def _on_hand_center(self, msg: PointStamped) -> None:
        self._last_hand_center = msg

    def _fresh_hand_center(self, stamp):
        """Hand centroid matching this inference's stamp, or None.

        The driver publishes the centroid right before the candidates with
        the same header stamp, so a mismatch means the hand was not visible
        in this inference (or the cached point is from an older capture)."""
        hc = self._last_hand_center
        if hc is None or hc.header.stamp != stamp:
            return None
        return (hc.point.x, hc.point.y, hc.point.z)

    def _on_candidates(self, msg: PoseArray) -> None:
        if not msg.poses:
            self.get_logger().warn('Received empty PoseArray, skipping.', throttle_duration_sec=5.0)
            return

        frame_id = self._output_frame or msg.header.frame_id

        kwargs = {'logger': self.get_logger()}
        if self._policy_name == 'ergonomic':
            kwargs['hand_center'] = self._fresh_hand_center(msg.header.stamp)

        idx = self._policy(msg.poses, frame_id, self._tf_buffer, **kwargs)
        # ergonomic returns (index, debug string) — log the string from this
        # node so it lands in its console output and /rosout.
        if isinstance(idx, tuple):
            idx, debug = idx
            if debug:
                self.get_logger().info(debug)
        if idx is None:
            self.get_logger().warn(
                f"Policy '{self._policy_name}' rejected all {len(msg.poses)} "
                'candidates — nothing published for this capture.')
            return
        idx = max(0, min(idx, len(msg.poses) - 1))  # clamp

        result = PoseStamped()
        result.header.stamp = msg.header.stamp
        result.header.frame_id = frame_id
        result.pose = msg.poses[idx]

        self._pub.publish(result)

        p = result.pose.position
        self.get_logger().info(
            f"Selected grasp {idx + 1}/{len(msg.poses)} via '{self._policy_name}': "
            f"({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) in '{frame_id}'")


def main(args=None):
    rclpy.init(args=args)
    node = GraspSelectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
