import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading
import time

STATUS_UNSTABLE = 'hand_unstable'
STATUS_STABLE = 'hand_stable'

class TriggerNode(Node):
    """A keyboard trigger node that replaces hand stabilization by sending 'hand_stable' manually."""

    def __init__(self):
        super().__init__('trigger_node')
        
        self.declare_parameter('status_topic', 'hand_stability_status')
        status_topic = self.get_parameter('status_topic').value
        
        self._status_pub = self.create_publisher(String, status_topic, 10)
        
        self.get_logger().info(f"Keyboard Trigger Node started. Publishing to '{status_topic}'")
        
        self._input_thread = threading.Thread(target=self._wait_for_enter, daemon=True)
        self._input_thread.start()

    def _wait_for_enter(self):
        while rclpy.ok():
            try:
                input("Press Enter to trigger grasp inference...\n")
                
                # Publish unstable first to ensure an edge transition in driver.py
                msg = String()
                msg.data = STATUS_UNSTABLE
                self._status_pub.publish(msg)
                
                time.sleep(0.1)
                
                msg.data = STATUS_STABLE
                self._status_pub.publish(msg)
                
            except EOFError:
                break

def main(args=None):
    rclpy.init(args=args)
    node = TriggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
