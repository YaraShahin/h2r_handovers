"""Grasp selection node.

Subscribes to the scored grasp candidates from GraspNet and applies a
configurable selection policy to pick the single best grasp, published as
a PoseStamped for the orchestrator to act on.
"""

import rclpy
from geometry_msgs.msg import PoseArray, PoseStamped
from rclpy.node import Node

from h2r_handovers.policies import POLICIES


class GraspSelectionNode(Node):

    def __init__(self):
        super().__init__('grasp_selection_node')

        self.declare_parameter('input_topic', 'grasp_candidates')
        self.declare_parameter('output_topic', 'selected_grasp')
        self.declare_parameter('policy', 'highest_score')

        policy_name = self.get_parameter('policy').value
        if policy_name not in POLICIES:
            self.get_logger().error(
                f"Unknown policy '{policy_name}'. "
                f"Available: {list(POLICIES.keys())}.  Falling back to 'highest_score'.")
            policy_name = 'highest_score'
        self._policy = POLICIES[policy_name]
        self._policy_name = policy_name

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

        idx = self._policy(msg.poses, msg.header.frame_id)
        idx = max(0, min(idx, len(msg.poses) - 1))  # clamp

        result = PoseStamped()
        result.header = msg.header
        result.pose = msg.poses[idx]
        self._pub.publish(result)

        self.get_logger().info(
            f"Selected grasp {idx}/{len(msg.poses)} via '{self._policy_name}'",
            throttle_duration_sec=2.0)


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
