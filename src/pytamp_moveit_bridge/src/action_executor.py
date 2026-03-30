#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import threading
import time
import math
import numpy as np
import subprocess

from geometry_msgs.msg import PoseStamped, Point, Quaternion
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, BoundingVolume, PlanningOptions, MotionPlanRequest, RobotState
from shape_msgs.msg import SolidPrimitive
from control_msgs.action import GripperCommand
from std_msgs.msg import String

from pytamp_moveit_bridge.msg import WorldState, TaskAction

class ActionExecutor(Node):

    def __init__(self):
        super().__init__('action_executor_node')
        print("DEBUG: ActionExecutor (Action Client Version) started.", flush=True)
        
        self.lock = threading.Lock()
        self.world_state = None
        
        # Use ReentrantCallbackGroup so gripper goals can be processed
        # concurrently while execute_pick is sleeping
        self.cb_group = ReentrantCallbackGroup()
        
        # Action Clients
        self.move_client = ActionClient(self, MoveGroup, '/move_action', callback_group=self.cb_group)
        self.gripper_client = ActionClient(self, GripperCommand, '/panda_hand_controller/gripper_cmd', callback_group=self.cb_group)
        
        # Subscriptions
        self.world_sub = self.create_subscription(WorldState, '/world_state', self.world_callback, 10, callback_group=self.cb_group)
        self.task_sub = self.create_subscription(TaskAction, '/task_plan', self.execute_task_callback, 10, callback_group=self.cb_group)
        
        # Parameters for 5cm colored cubes
        self.pre_grasp_offset = 0.10  # Hover above cube before descending
        self.lift_height = 0.15       # Lift height after grasp
        self.tcp_offset = 0.103       # Standard Panda EE offset
        
        # Initial status
        print("DEBUG: Waiting for MoveGroup action server...", flush=True)
        self.move_client.wait_for_server()
        print("DEBUG: MoveGroup server connected.", flush=True)
        
        print("DEBUG: Waiting for Gripper action server...", flush=True)
        self.gripper_client.wait_for_server()
        print("DEBUG: Gripper server connected. System fully initialized.", flush=True)

    def world_callback(self, msg):
        self.world_state = msg

    def get_object_pose(self, object_id):
        if self.world_state is None:
            return None
        for obj in self.world_state.objects:
            if obj.object_id == object_id:
                return obj.pose
        return None

    def execute_task_callback(self, msg):
        # Spawn a thread to handle the task so we don't block the spinning executor
        # and to avoid "Executor is already spinning" error.
        t = threading.Thread(target=self.run_task, args=(msg,))
        t.start()

    def run_task(self, msg):
        with self.lock:
            print(f"DEBUG: Received Action: {msg.action_type} for {msg.object_id}", flush=True)
            
            if msg.action_type == "PICK":
                self.execute_pick(msg.object_id)
            elif msg.action_type == "PLACE":
                self.execute_place(msg.object_id)

    def execute_pick(self, object_id):
        pose = self.get_object_pose(object_id)
        if pose is None:
            # Fall back to the pose encoded in the TaskAction message if perception lost the object
            print(f"DEBUG ERROR: Object {object_id} not in world_state. Skipping pick.", flush=True)
            return

        print(f"DEBUG: Executing PICK for {object_id} at {pose.position}", flush=True)
        
        # Orientation: gripper pointing straight down (Roll=180 deg → quaternion x=1, w=0)
        down = Quaternion(x=1.0, y=0.0, z=0.0, w=0.0)
        
        # 1. Open Gripper (5cm cube needs ~4cm finger gap)
        self.move_gripper(0.04, effort=0.0)
        time.sleep(1.0)
        
        # 2. Pre-Grasp: hover above cube centre
        pre_grasp = PoseStamped()
        pre_grasp.header.frame_id = "panda_link0"
        pre_grasp.pose.position = Point(
            x=pose.position.x,
            y=pose.position.y,
            z=pose.position.z + self.pre_grasp_offset + self.tcp_offset
        )
        pre_grasp.pose.orientation = down
        self.move_to_pose(pre_grasp)
        
        # 3. Descend to Grasp (just graze the top of the cube)
        grasp = PoseStamped()
        grasp.header.frame_id = "panda_link0"
        grasp.pose.position = Point(
            x=pose.position.x,
            y=pose.position.y,
            z=pose.position.z + self.tcp_offset
        )
        grasp.pose.orientation = down
        self.move_to_pose(grasp)
        
        # 4. Close Gripper around cube (cube is 5cm wide, grip at ~3cm)
        time.sleep(0.5)
        self.move_gripper(0.030, effort=50.0)
        time.sleep(2.0)
        
        # 5. Lift
        lift = PoseStamped()
        lift.header.frame_id = "panda_link0"
        lift.pose.position = Point(
            x=grasp.pose.position.x,
            y=grasp.pose.position.y,
            z=grasp.pose.position.z + self.lift_height
        )
        lift.pose.orientation = down
        self.move_to_pose(lift)

    def execute_place(self, object_id):
        print(f"DEBUG: Executing PLACE for {object_id}", flush=True)
        down = Quaternion(x=1.0, y=0.0, z=0.0, w=0.0)
        
        # Place target: fixed drop-zone to the side of the cubes
        place_x, place_y = 0.55, 0.15
        table_z_relative = -0.005  # Table surface relative to panda_link0
        
        # 1. Hover above drop-zone
        pre_place = PoseStamped()
        pre_place.header.frame_id = "panda_link0"
        pre_place.pose.position = Point(x=place_x, y=place_y, z=table_z_relative + 0.20 + self.tcp_offset)
        pre_place.pose.orientation = down
        self.move_to_pose(pre_place)
        
        # 2. Lower cube onto table
        place_down = PoseStamped()
        place_down.header.frame_id = "panda_link0"
        # Since fingertips hold the cube's center (2.5cm up from bottom), 
        # add 0.025 to table height so cube rests gently instead of smashing through.
        place_down.pose.position = Point(x=place_x, y=place_y, z=table_z_relative + 0.025 + self.tcp_offset)
        place_down.pose.orientation = down
        self.move_to_pose(place_down)
        
        # 3. Release cube
        time.sleep(0.5)
        self.move_gripper(0.04, effort=0.0)
        time.sleep(1.5)
        
        # 4. Retreat upward
        retreat = PoseStamped()
        retreat.header.frame_id = "panda_link0"
        retreat.pose.position = Point(x=place_x, y=place_y, z=table_z_relative + 0.20 + self.tcp_offset)
        retreat.pose.orientation = down
        self.move_to_pose(retreat)

    def move_to_pose(self, pose_stamped):
        print(f"DEBUG: Moving to pose {pose_stamped.pose.position}", flush=True)
        goal_msg = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = "panda_arm"
        req.num_planning_attempts = 10
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = 0.2
        req.max_acceleration_scaling_factor = 0.2
        
        c = Constraints()
        c.name = "goal"
        
        pos = PositionConstraint()
        pos.header = pose_stamped.header
        pos.link_name = "panda_link8"
        
        s = SolidPrimitive()
        s.type = SolidPrimitive.SPHERE
        s.dimensions = [0.01] 
        pos.constraint_region.primitives.append(s)
        pos.constraint_region.primitive_poses.append(pose_stamped.pose)
        pos.weight = 1.0
        
        ori = OrientationConstraint()
        ori.header = pose_stamped.header
        ori.link_name = "panda_link8"
        ori.orientation = pose_stamped.pose.orientation
        # Tightened tolerances (0.01 rad = ~0.5 degrees) enforce strictly
        # perpendicular vertical alignment, preventing "tilted" approaches.
        ori.absolute_x_axis_tolerance = 0.01
        ori.absolute_y_axis_tolerance = 0.01
        ori.absolute_z_axis_tolerance = 0.01
        ori.weight = 1.0
        
        c.position_constraints.append(pos)
        c.orientation_constraints.append(ori)
        
        req.goal_constraints.append(c)
        goal_msg.request = req
        goal_msg.planning_options.plan_only = False
        
        # Send Goal
        event = threading.Event()
        def goal_response_callback(future):
            event.set()

        future = self.move_client.send_goal_async(goal_msg)
        future.add_done_callback(goal_response_callback)
        
        if not event.wait(timeout=10.0):
            print("DEBUG: Goal response timeout", flush=True)
            return False
            
        handle = future.result()
        if handle is None or not handle.accepted:
            print("DEBUG: Goal rejected", flush=True)
            return False
            
        # Wait for result
        result_event = threading.Event()
        def result_callback(future):
            result_event.set()
            
        result_future = handle.get_result_async()
        result_future.add_done_callback(result_callback)
        
        if not result_event.wait(timeout=45.0):
            print("DEBUG: Result timeout", flush=True)
            return False
            
        print("DEBUG: Move completed", flush=True)
        return True

    def move_gripper(self, width, effort=0.0):
        print(f"DEBUG: Setting gripper to {width} with effort {effort}", flush=True)
        goal = GripperCommand.Goal()
        goal.command.position = float(width)
        goal.command.max_effort = float(effort)
        
        event = threading.Event()
        def callback(future):
            event.set()
            
        future = self.gripper_client.send_goal_async(goal)
        future.add_done_callback(callback)
        
        if not event.wait(timeout=10.0):
            print("DEBUG: Gripper goal SEND timeout — server may be down", flush=True)
            return False
            
        handle = future.result()
        if handle is None:
            print("DEBUG: Gripper goal handle is None — send failed entirely", flush=True)
            return False
        if not handle.accepted:
            print("DEBUG: Gripper goal REJECTED by server", flush=True)
            return False
            
        print(f"DEBUG: Gripper goal accepted, waiting for result...", flush=True)
        res_event = threading.Event()
        def res_callback(future):
            res_event.set()
        res_future = handle.get_result_async()
        res_future.add_done_callback(res_callback)
        
        if not res_event.wait(timeout=15.0):
            print("DEBUG: Gripper RESULT timeout (15s) — stall detection may be too slow", flush=True)
            return False
         
        res = res_future.result()
        if res:
            print(f"DEBUG: Gripper result status: {res.status} "
                  f"(4=SUCCEEDED, 5=CANCELED, 6=ABORTED)", flush=True)
        else:
            print("DEBUG: Gripper result is None", flush=True)
             
        print("DEBUG: Gripper command finished", flush=True)
        return True

def main(args=None):
    rclpy.init(args=args)
    node = ActionExecutor()
    # MultiThreadedExecutor lets the gripper goal be processed
    # concurrently while execute_pick sleeps
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
