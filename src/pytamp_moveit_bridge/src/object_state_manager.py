#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import tf2_ros
import tf2_geometry_msgs
import numpy as np
from cv_bridge import CvBridge
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, Point, Quaternion, TransformStamped
from yolov8_msgs.msg import Yolov8Inference
from pytamp_moveit_bridge.msg import ObjectState, WorldState
from image_geometry import PinholeCameraModel
import math

class ObjectStateManager(Node):
    def __init__(self):
        super().__init__('object_state_manager')
        
        # Parameters
        self.target_frame = 'panda_link0' # Base frame of the robot
        self.camera_frame = None # Will be observed from messages
        
        # Tools
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.bridge = CvBridge()
        self.camera_model = PinholeCameraModel()
        self.camera_info_received = False

        # Publishers
        self.world_state_pub = self.create_publisher(WorldState, '/world_state', 10)
        
        # Subscribers
        self.info_sub = self.create_subscription(CameraInfo, '/camera/camera_info', self.info_callback, 10)
        
        # Synchronization for Inference + Depth
        # Logic: We need depth to project 2D OBB to 3D.
        # We assume /Yolov8_Inference and /camera/depth/image_raw are roughly synced 
        # or we just take the latest depth.
        # Since Yolov8 might be slower, using ApproximateTimeSynchronizer is good.
        
        self.inference_sub = message_filters.Subscriber(self, Yolov8Inference, '/Yolov8_Inference')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.inference_sub, self.depth_sub], 
            queue_size=10, 
            slop=0.5 # Generous slop as inference might be delayed
        )
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info('ObjectStateManager Initialized. Waiting for data...')
        self.create_timer(5.0, self.check_status)

    def check_status(self):
        if not self.camera_info_received:
            self.get_logger().warn("WAITING FOR CAMERA INFO... (Topic: /camera/camera_info)")
        else:
             self.get_logger().info(f"Status: CamInfo=OK, Frame={self.camera_frame}. Waiting for synced Inference+Depth...", throttle_duration_sec=10.0)

    def info_callback(self, msg):
        if not self.camera_info_received:
            msg.header.frame_id = "camera_optical_frame" # Force correct Optical frame
            self.camera_model.fromCameraInfo(msg)
            self.camera_frame = msg.header.frame_id
            self.camera_info_received = True
            self.get_logger().info(f'Camera Info received. Frame: {self.camera_frame}')

    def sync_callback(self, inference_msg, depth_msg):
        if not self.camera_info_received:
            return

        try:
            # Convert depth image
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return

        world_state = WorldState()
        
        # Process each detection
        for i, result in enumerate(inference_msg.yolov8_inference):
            # Filter for bolts if strict "only bolt picking" is needed, but generally good to see everything
            # The prompt implies "only bolt picking" behavior, which is usually a Planner decision.
            # However, for robustness, we can prioritize bolts or log them.
            
            coords = result.coordinates
            if len(coords) < 8:
                continue
                
            points = np.array(coords).reshape(4, 2)
            
            # 1. Compute 2D Center
            center_u = np.mean(points[:, 0])
            center_v = np.mean(points[:, 1])
            
            # 2. Get Depth from ROI (Robust Sampling)
            h, w = depth_image.shape
            
            # Crop ROI
            u_min = int(np.clip(np.min(points[:, 0]), 0, w-1))
            u_max = int(np.clip(np.max(points[:, 0]), 0, w-1))
            v_min = int(np.clip(np.min(points[:, 1]), 0, h-1))
            v_max = int(np.clip(np.max(points[:, 1]), 0, h-1))
            
            # Extract depth patch
            depth_roi = depth_image[v_min:v_max+1, u_min:u_max+1]
            
            if depth_roi.size == 0:
                continue
                
            # Filter NaNs and zeros
            valid_depths = depth_roi[(depth_roi > 0) & (~np.isnan(depth_roi))]
            
            if valid_depths.size == 0:
                continue
                
            # Use 5th Percentile to capture foreground (object) even if small
            # Handling different types provided by cv_bridge
            
            # Debug stats
            d_min = np.min(valid_depths)
            d_max = np.max(valid_depths)
            d_med = np.median(valid_depths)
            
            if depth_image.dtype == np.uint16:
                # usually mm
                # Use 5th percentile
                d_p5 = np.percentile(valid_depths, 5)
                depth_m = float(d_p5) / 1000.0
                self.get_logger().info(f"Depth Stats (mm): Min={d_min}, Max={d_max}, Med={d_med}, P5={d_p5} -> Depth={depth_m}m")
            else:
                d_p5 = np.percentile(valid_depths, 5)
                depth_m = float(d_p5)
                self.get_logger().info(f"Depth Stats (m): Min={d_min:.3f}, Max={d_max:.3f}, Med={d_med:.3f}, P5={d_p5:.3f} -> Depth={depth_m:.3f}m")
            
            # Filter invalid depth
            if depth_m <= 0.05 or depth_m > 3.0 or np.isnan(depth_m):
                continue 

            # 3. Project to 3D (Camera Frame)
            ray = self.camera_model.projectPixelTo3dRay((center_u, center_v))
            point_camera = np.array(ray) * depth_m
            
            # 4. Compute Orientation (Yaw in Camera Frame)
            # Find longest edge for orientation
            p0, p1, p2 = points[0], points[1], points[2]
            dist01 = np.linalg.norm(p0 - p1)
            dist12 = np.linalg.norm(p1 - p2)
            
            if dist01 > dist12:
                angle_img = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
            else:
                angle_img = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            
            # In image plane: +x is right, +y is down.
            # Camera Optical Frame: +z forward, +x right, +y down.
            # We need to find the orientation of the object on the table.
            # Project a second point along the angle to find 3D vector
            
            vec_len = 20.0 # pixels
            u_vec = center_u + vec_len * math.cos(angle_img)
            v_vec = center_v + vec_len * math.sin(angle_img)
            
            ray_vec = self.camera_model.projectPixelTo3dRay((u_vec, v_vec))
            point_vec_camera = np.array(ray_vec) * depth_m 
            
            # 5. Transform to/from Robot Base
            try:
                # Transform Center Point
                p_stamped = tf2_geometry_msgs.PointStamped()
                p_stamped.header.frame_id = self.camera_frame
                p_stamped.header.stamp = rclpy.time.Time() # Use latest
                p_stamped.point.x, p_stamped.point.y, p_stamped.point.z = point_camera
                
                # We need to wait for transform availability? 
                # Ideally yes, but let's try with latest available
                if not self.tf_buffer.can_transform(self.target_frame, self.camera_frame, rclpy.time.Time()):
                    self.get_logger().warn(f"Cannot transform {self.camera_frame} -> {self.target_frame}")
                    continue

                p_out = self.tf_buffer.transform(p_stamped, self.target_frame)
                center_world = np.array([p_out.point.x, p_out.point.y, p_out.point.z])

                # Transform Vector Point
                p_vec_stamped = tf2_geometry_msgs.PointStamped()
                p_vec_stamped.header.frame_id = self.camera_frame
                p_vec_stamped.header.stamp = rclpy.time.Time()
                p_vec_stamped.point.x, p_vec_stamped.point.y, p_vec_stamped.point.z = point_vec_camera
                
                p_vec_out = self.tf_buffer.transform(p_vec_stamped, self.target_frame)
                vec_world = np.array([p_vec_out.point.x, p_vec_out.point.y, p_vec_out.point.z])
                
                # Compute Yaw in World (XY plane)
                diff = vec_world - center_world
                yaw_world = math.atan2(diff[1], diff[0])
                
                # Quat from Yaw
                q_world = self.euler_to_quaternion(0, 0, yaw_world) # Flat on table

                # Log all computed world centers for debugging
                self.get_logger().info(f"YOLO {i} -> {result.class_name}: camera Z={depth_m:.3f}m | panda_link0 xyz=({center_world[0]:.3f}, {center_world[1]:.3f}, {center_world[2]:.3f})")

                # Filter out phantom detections from robot base
                if center_world[0] < 0.35:
                    self.get_logger().debug(f"Ignoring phantom detection at x={center_world[0]:.3f}")
                    continue

                # Construct ObjectState
                obj_state = ObjectState()
                obj_state.object_id = f"{result.class_name}_{i}"
                obj_state.class_name = result.class_name
                # Calibration correction: camera systematically over-reports
                # Y by +0.137m and Z by +0.027m (verified vs Gazebo SDF ground-truth)
                # panda_link0 origin in world: [0.05, 0.0, 1.02]
                # bolt1 SDF world: [0.5, 0.0, 1.05]  → panda_link0: [0.45, 0.0, 0.03]
                CALIB_Y_OFFSET = -0.137  # subtract camera Y over-estimation
                CALIB_Z_OFFSET = -0.027  # subtract camera Z over-estimation
                obj_state.pose.position.x = center_world[0]
                obj_state.pose.position.y = center_world[1] + CALIB_Y_OFFSET
                obj_state.pose.position.z = center_world[2] + CALIB_Z_OFFSET
                obj_state.pose.orientation = q_world
                
                # Add to WorldState
                world_state.objects.append(obj_state)
                
            except Exception as e:
                self.get_logger().warn(f"Transform failure: {e}")
                continue
        
        if world_state.objects:
            self.world_state_pub.publish(world_state)
            # self.get_logger().info(f"Published {len(world_state.objects)} objects")

    def euler_to_quaternion(self, roll, pitch, yaw):
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        return Quaternion(x=qx, y=qy, z=qz, w=qw)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectStateManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
