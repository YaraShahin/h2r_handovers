"""Grasp selection node.

Subscribes to the scored grasp candidates from GraspNet and applies a
configurable selection policy to pick the single best grasp, published as
a PoseStamped for the orchestrator to act on.
"""

import rclpy
from geometry_msgs.msg import PoseArray, PoseStamped
from rclpy.node import Node
import tf2_ros
import numpy as np
from scipy.spatial.transform import Rotation as R

from h2r_handovers.policies import POLICIES


class GraspSelectionNode(Node):

    def __init__(self):
        super().__init__('grasp_selection_node')

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self.declare_parameter('input_topic', 'grasp_candidates')
        self.declare_parameter('output_topic', 'selected_grasp')
        self.declare_parameter('policy', 'top_k_aligned')

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

        self.get_logger().warn(f'hyeyyyy {msg.header.frame_id}')
        idx = self._policy(msg.poses, msg.header.frame_id, self._tf_buffer)
        idx = max(0, min(idx, len(msg.poses) - 1))  # clamp

        try:
            # Get transform from camera to panda_link0
            transform = self._tf_buffer.lookup_transform(
                'panda_link0',
                msg.header.frame_id,
                rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().error(f"Failed to lookup transform: {e}")
            return

        import numpy as np
        from scipy.spatial.transform import Rotation as R
        
        t = transform.transform.translation
        r = transform.transform.rotation
        rot_cam = R.from_quat([r.x, r.y, r.z, r.w])
        
        # Invert X and Y to perfectly compensate for the opposite corner mirroring
        pos_cam = np.array([-msg.poses[idx].position.x, -msg.poses[idx].position.y, msg.poses[idx].position.z])
        pos_world = rot_cam.apply(pos_cam) + np.array([t.x, t.y, t.z])
        
        # Apply the exact offsets from the old ROS1 pipeline
        pos_world[1] += 0.09
        pos_world[2] += 0.09

        result = PoseStamped()
        result.header.stamp = msg.header.stamp
        result.header.frame_id = 'panda_link0'
        
        result.pose.position.x = float(pos_world[0])
        result.pose.position.y = float(pos_world[1])
        result.pose.position.z = float(pos_world[2])
        
        # Hardcode orientation to point perfectly straight down
        result.pose.orientation.x = 1.0
        result.pose.orientation.y = 0.0
        result.pose.orientation.z = 0.0
        result.pose.orientation.w = 0.0

        self.get_logger().info('Published straight-down grasp in panda_link0')
        self._pub.publish(result)

        self.get_logger().info(
            f"Selected grasp {idx}/{len(msg.poses)} via '{self._policy_name}' with value {msg.poses[idx]}",
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
