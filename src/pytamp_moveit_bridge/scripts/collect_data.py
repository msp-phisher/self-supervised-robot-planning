#!/usr/bin/env python3

import os
import time
import math
import random
import subprocess
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')
        self.bridge = CvBridge()
        
        # Ensure the dataset directory exists
        self.dataset_dir = os.path.join(os.getcwd(), 'dataset', 'images')
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        self.image_count = 0
        self.target_images = 1000
        self.scene_updated = False
        self.got_first_image = False
        
        # Use permissive QoS to match any publisher
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            qos)
        
        self.timer = self.create_timer(0.5, self.randomize_scene)
        self.diag_timer = self.create_timer(5.0, self.check_topics)
        self.get_logger().info('Data collection node started. Waiting for images...')
        self.get_logger().info(f'Images will be saved to: {self.dataset_dir}')

    def check_topics(self):
        """Print diagnostic info if we haven't received any images yet."""
        if not self.got_first_image:
            topic_names = [t[0] for t in self.get_topic_names_and_types()]
            cam_topics = [t for t in topic_names if 'camera' in t or 'image' in t]
            self.get_logger().warn(f'Still waiting for images. Camera-related topics: {cam_topics}')
            self.get_logger().warn('Make sure Gazebo + bridge are running. Try: ros2 topic hz /camera/image_raw')

    def randomize_scene(self):
        if self.image_count >= self.target_images:
            self.get_logger().info('Finished collecting 1000 images! Shutting down...')
            rclpy.shutdown()
            return

        if self.scene_updated:
            # Still waiting for the camera callback to save the last scene
            return

        # Randomize boxes
        boxes = ['red_box', 'green_box', 'blue_box']
        
        for box in boxes:
            x = random.uniform(0.3, 0.7)
            y = random.uniform(-0.35, 0.35)
            z = 1.015
            
            # Random quaternion (simulating random object angles, which acts as different camera angles)
            yaw = random.uniform(-math.pi, math.pi)
            pitch = random.uniform(-0.5, 0.5)
            roll = random.uniform(-0.5, 0.5)
            
            cy = math.cos(yaw * 0.5)
            sy = math.sin(yaw * 0.5)
            cp = math.cos(pitch * 0.5)
            sp = math.sin(pitch * 0.5)
            cr = math.cos(roll * 0.5)
            sr = math.sin(roll * 0.5)

            qw = cr * cp * cy + sr * sp * sy
            qx = sr * cp * cy - cr * sp * sy
            qy = cr * sp * cy + sr * cp * sy
            qz = cr * cp * sy - sr * sp * cy

            req = f'name: "{box}", position: {{x: {x}, y: {y}, z: {z}}}, orientation: {{x: {qx}, y: {qy}, z: {qz}, w: {qw}}}'
            cmd = [
                'gz', 'service', '-s', '/world/arm_and_table/set_pose',
                '--reqtype', 'gz.msgs.Pose',
                '--reptype', 'gz.msgs.Boolean',
                '--timeout', '500',
                '--req', req
            ]
            
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                self.get_logger().error(f"Failed to call gz service: {e}")
        
        # After randomizing all boxes, prompt camera callback to save
        self.scene_updated = True

    def image_callback(self, msg):
        if not self.scene_updated or self.image_count >= self.target_images:
            return
        
        if not self.got_first_image:
            self.got_first_image = True
            self.get_logger().info('Receiving images from camera!')
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            filename = os.path.join(self.dataset_dir, f'image_{self.image_count:04d}.png')
            cv2.imwrite(filename, cv_image)
            self.image_count += 1
            
            if self.image_count % 50 == 0:
                self.get_logger().info(f'Saved {self.image_count} / {self.target_images} images.')
                
            self.scene_updated = False
                
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = DataCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()

if __name__ == '__main__':
    main()
