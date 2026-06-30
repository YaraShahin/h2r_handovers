from collections import deque

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

NOTHING_LABEL = 0
HAND_LABEL = 1
OBJECT_LABEL = 2

STATUS_NO_HAND = 'no_hand'
STATUS_UNSTABLE = 'hand_unstable'
STATUS_STABLE = 'hand_stable'


class HandStabilizationNode(Node):
    """Tracks the hand centroid in a segmentation mask and reports stabilization."""

    def __init__(self):
        super().__init__('hand_stabilization_node')

        self.declare_parameter('mask_topic', 'segmentation_mask')
        self.declare_parameter('status_topic', 'hand_stability_status')
        self.declare_parameter('hand_label', HAND_LABEL)
        self.declare_parameter('min_hand_pixels', 50)
        self.declare_parameter('position_tolerance_px', 10.0)
        self.declare_parameter('min_samples', 10)

        self._hand_label = self.get_parameter('hand_label').value
        self._min_hand_pixels = self.get_parameter('min_hand_pixels').value
        self._position_tolerance_px = self.get_parameter('position_tolerance_px').value
        self._min_samples = self.get_parameter('min_samples').value

        self._bridge = CvBridge()
        self._history = deque(maxlen=self._min_samples)  # entries: (cx, cy)
        self._last_status = None

        mask_topic = self.get_parameter('mask_topic').value
        status_topic = self.get_parameter('status_topic').value

        self._status_pub = self.create_publisher(String, status_topic, 10)
        self.create_subscription(Image, mask_topic, self._mask_callback, 10)

        self.get_logger().info(
            f"Tracking hand label {self._hand_label} on '{mask_topic}', "
            f"publishing status on '{status_topic}' "
            f"(tolerance={self._position_tolerance_px}px over {self._min_samples} samples)"
        )

    def _mask_callback(self, msg: Image) -> None:
        mask = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

        ys, xs = np.nonzero(mask == self._hand_label)

        if xs.size < self._min_hand_pixels:
            # Hand lost: stale history would otherwise be stitched to a
            # reappearance at a different location, so drop it.
            self._history.clear()
            self._publish_status(STATUS_NO_HAND)
            return

        centroid = (float(xs.mean()), float(ys.mean()))
        self._history.append(centroid)

        self._publish_status(self._evaluate_stability())

    def _evaluate_stability(self) -> str:
        if len(self._history) < self._min_samples:
            return STATUS_UNSTABLE

        positions = np.array(self._history)
        mean_pos = positions.mean(axis=0)
        max_dist = float(np.linalg.norm(positions - mean_pos, axis=1).max())

        return STATUS_STABLE if max_dist <= self._position_tolerance_px else STATUS_UNSTABLE

    def _publish_status(self, status: str) -> None:
        # Edge-triggered: only emit a message when the status actually changes,
        # so consumers see exactly one "hand_stable" / "hand_unstable" / "no_hand"
        # event per transition instead of a continuous stream.
        if status == self._last_status:
            return
        self._last_status = status

        msg = String()
        msg.data = status
        self._status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = HandStabilizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
