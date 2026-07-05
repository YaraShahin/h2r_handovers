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

import rclpy
from geometry_msgs.msg import PoseArray, PoseStamped
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

        self._output_frame = self.get_parameter('output_frame').value
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        self._pub = self.create_publisher(PoseStamped, output_topic, 1)
        self.create_subscription(PoseArray, input_topic, self._on_candidates, 10)

        self.get_logger().info(
            f"Grasp selection: policy='{policy_name}', "
            f"subscribing to '{input_topic}', publishing to '{output_topic}'")

    def _on_candidates(self, msg: PoseArray) -> None:
        if not msg.poses:
            self.get_logger().warn('Received empty PoseArray, skipping.', throttle_duration_sec=5.0)
            return

        frame_id = self._output_frame or msg.header.frame_id

        idx = self._policy(msg.poses, frame_id, self._tf_buffer)
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
