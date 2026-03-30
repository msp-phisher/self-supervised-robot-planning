#!/usr/bin/env python3
import os
import cv2
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from pytamp_moveit_bridge.msg import WorldState, ObjectState

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
IMAGE_SIZE = 224

class SSLPerceptionNode(Node):
    def __init__(self):
        super().__init__('ssl_perception_node')
        self.bridge = CvBridge()
        
        # Load SimCLR Backbone
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.backbone = models.resnet18()
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        backbone_path = os.path.join(MODEL_DIR, 'simclr_backbone.pt')
        self.backbone.load_state_dict(torch.load(backbone_path, map_location=self.device))
        self.backbone.to(self.device).eval()
        
        # Load K-Means Model
        kmeans_path = os.path.join(MODEL_DIR, 'kmeans_model.pkl')
        with open(kmeans_path, 'rb') as f:
            self.kmeans = pickle.load(f)
            
        # Load Cluster Mapping
        map_path = os.path.join(MODEL_DIR, 'cluster_mapping.json')
        with open(map_path, 'r') as f:
            data = json.load(f)
            self.mapping = {int(k): v for k, v in data.items()}
            
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # ROS 2 Sub/Pub
        self.sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.world_pub = self.create_publisher(WorldState, '/world_state', 10)
        self.get_logger().info(f"SSL Perception Node started. Grounding: {self.mapping}")

    def extract_object_crops(self, cv_img):
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        # Relaxed saturation for simulation lighting
        mask = cv2.inRange(hsv, (0, 50, 50), (180, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        crops = []
        bboxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Cubes are ~440 area in current view
            if 300 < area < 1000:
                x, y, w, h = cv2.boundingRect(cnt)
                crop = cv_img[y:y+h, x:x+w]
                crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                bboxes.append((x, y, w, h))
        return crops, bboxes

    def image_callback(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        crops, bboxes = self.extract_object_crops(cv_img)
        
        if not crops: return
        
        # Batch inference
        with torch.no_grad():
            crop_tensors = torch.stack([self.transform(c) for c in crops]).to(self.device)
            feats = self.backbone(crop_tensors).squeeze(-1).squeeze(-1).cpu().numpy()
            
            # Hybrid Color Features (x1000 weight)
            color_feats = []
            for c in crops:
                mean_rgb = np.mean(c, axis=(0, 1)) / 255.0
                color_feats.append(mean_rgb * 1000.0)
            color_feats = np.stack(color_feats)
            
            hybrid = np.hstack([feats, color_feats])
            
        preds = self.kmeans.predict(hybrid)
        
        # Projection Params (Camera at 1.38m, Table at 1.05m)
        cam_x_world, cam_y_world, dist = 0.325, -0.02, 0.33
        fov_h, w_img, h_img = 1.047198, 640, 480
        pixel_size = (2 * dist * np.tan(fov_h/2)) / w_img
        
        world_msg = WorldState()
        
        for cid, bbox in zip(preds, bboxes):
            obj_name = self.mapping.get(cid, "unknown")
            if obj_name == "unknown": continue
            
            # 2D Centroid -> 3D World
            u, v = bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2
            du, dv = u - w_img/2, v - h_img/2
            
            # True Pinhole Projection
            # camera is at X=0.325, Y=-0.02, Z=1.38 relative to panda_link0
            # Cube center Z = 0.02, so depth = 1.36m
            # focal length (fx, fy) ~ 554 for 60-deg FOV at 640x480
            # dv (down in image) → +X in world, du (right in image) → -Y in world
            dz = 1.36
            fx = fy = 554.0

            x_world = float(cam_x_world - (dv * dz / fy))  # above center = further forward
            y_world = float(cam_y_world - (du * dz / fx))  # right in image = more negative Y
            z_world = 0.02  # Cube center in panda_link0 frame
            
            # Populate ObjectState
            obj = ObjectState()
            obj.object_id = f"{obj_name}_{int(u)}_{int(v)}"
            obj.class_name = obj_name
            obj.pose.position.x = x_world
            obj.pose.position.y = y_world
            obj.pose.position.z = z_world
            obj.pose.orientation.w = 1.0
            world_msg.objects.append(obj)
            
            self.get_logger().info(f"GROUNDED: {obj_name} at X={x_world:.3f}, Y={y_world:.3f}", once=True)

        self.world_pub.publish(world_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SSLPerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
