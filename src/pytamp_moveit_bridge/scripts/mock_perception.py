#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from yolov8_msgs.msg import Yolov8Inference, InferenceResult
from cv_bridge import CvBridge
import numpy as np
import time

class MockPerception(Node):
    def __init__(self):
        super().__init__('mock_perception')
        
        self.yolo_pub = self.create_publisher(Yolov8Inference, '/Yolov8_Inference', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', 10)
        
        self.bridge = CvBridge()
        self.timer = self.create_timer(1.0, self.timer_callback) # 1Hz
        
        self.get_logger().info("Mock Perception Publisher Started")

    def timer_callback(self):
        now = self.get_clock().now().to_msg()
        frame_id = "camera_link" # Match the TF frame you expect
        
        # 1. Camera Info (Required for Pinhole Model)
        info_msg = CameraInfo()
        info_msg.header.stamp = now
        info_msg.header.frame_id = frame_id
        info_msg.height = 480
        info_msg.width = 640
        # Simple intrinsic matrix (approx 640x480)
        fx, fy = 500.0, 500.0
        cx, cy = 320.0, 240.0
        info_msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info_msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.info_pub.publish(info_msg)

        # 2. Depth Image (Flat plane at 0.9 meter -> Z=0.1m if CamZ=1.0)
        depth_img = np.ones((480, 640), dtype=np.float32) * 0.9 # 0.9 meters away
        depth_msg = self.bridge.cv2_to_imgmsg(depth_img, encoding="32FC1")
        depth_msg.header.stamp = now
        depth_msg.header.frame_id = frame_id
        self.depth_pub.publish(depth_msg)
        
        # 3. YOLO OBB Detection
        yolo_msg = Yolov8Inference()
        yolo_msg.header.stamp = now
        yolo_msg.header.frame_id = frame_id
        
        # Create one mock object (RedBox)
        result = InferenceResult()
        result.class_name = "mock_red_box"
        
        # 4 points for OBB (pixels) centered at 320, 240
        # 100x50 box, rotated slightly
        # Points: TopLeft, TopRight, BottomRight, BottomLeft
        # Let's just make a square for simplicity
        # x, y sequence
        result.coordinates = [
            270.0, 190.0, # TL
            370.0, 190.0, # TR
            370.0, 290.0, # BR
            270.0, 290.0  # BL
        ] # flattened
        
        yolo_msg.yolov8_inference.append(result)
        self.yolo_pub.publish(yolo_msg)
        
        self.get_logger().info("Published Mock Data: 1 Object at 1.5m depth")

def main(args=None):
    rclpy.init(args=args)
    node = MockPerception()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
