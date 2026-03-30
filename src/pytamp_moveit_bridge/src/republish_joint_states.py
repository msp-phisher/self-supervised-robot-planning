#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class JointStateRepublisher(Node):
    def __init__(self):
        super().__init__('joint_state_republisher')
        
        self.sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.callback,
            10
        )
        
        self.pub = self.create_publisher(
            JointState,
            '/joint_states_fixed',
            10
        )
        self.get_logger().info("Joint State Republisher Started: /joint_states -> /joint_states_fixed")

    def callback(self, msg):
        # Stamp with current ROS time
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = JointStateRepublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
