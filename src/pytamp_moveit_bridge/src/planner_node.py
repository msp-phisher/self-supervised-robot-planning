#!/usr/bin/env python3
"""
PDDL-based Task Planner Node for TAMP Pick-and-Place.

Replaces PyTamp's MCTSPlannerNode with a pure-PDDL approach.
Uses pyperplan (pure Python BFS planner) — no C++ compilation required.

Flow:
  1. Subscribes to /world_state (live object perception from YOLO)
  2. Subscribes to /planning_goal (trigger string, e.g. "bolt")
  3. On goal: dynamically generates a PDDL Problem file from live scene
  4. Calls pyperplan BFS planner on domain.pddl + problem.pddl
  5. Parses the plan → TaskAction messages → publishes to /task_plan
"""

import os
import re
import tempfile
import logging

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory

from pytamp_moveit_bridge.msg import WorldState, TaskAction

# pyperplan API (direct, to bypass the validate-binary hang in search_plan wrapper)
from pyperplan.pddl.parser import Parser as PDDLParser
from pyperplan import grounding as pddl_grounding
from pyperplan.search import breadth_first_search

# Absolute path to the domain file (installed in share/pytamp_moveit_bridge/pddl/)
DOMAIN_FILE = os.path.join(
    get_package_share_directory('pytamp_moveit_bridge'),
    'pddl', 'domain.pddl'
)


class PDDLPlannerNode(Node):
    """ROS 2 node: listens for a goal → runs PDDL planner → publishes TaskAction plan."""

    def __init__(self):
        super().__init__('pddl_planner_node')
        self.get_logger().info("PDDL Planner Node starting (pyperplan backend)...")

        # Internal scene state: {object_id: ObjectState msg}
        self.scene_objects: dict = {}

        # Subscriptions
        self.world_sub = self.create_subscription(
            WorldState, '/world_state', self.world_callback, 10)
        self.goal_sub = self.create_subscription(
            String, '/planning_goal', self.goal_callback, 10)

        # Publisher
        self.plan_pub = self.create_publisher(TaskAction, '/task_plan', 10)

        self.get_logger().info("PDDL Planner node ready. Publish to /planning_goal to plan.")

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #

    def world_callback(self, msg: WorldState):
        """Update the internal scene state from perception."""
        self.scene_objects = {obj.object_id: obj for obj in msg.objects}
        self.get_logger().info(
            f"World state: {list(self.scene_objects.keys())}",
            throttle_duration_sec=2.0)

    def goal_callback(self, msg: String):
        """Triggered when a planning goal is published, e.g. 'bolt'."""
        target_class = msg.data.strip().lower()
        self.get_logger().info(f"Planning goal received: '{target_class}'")

        # 1. Identify the target object from perception
        target_obj = self._find_object(target_class)
        if target_obj is None:
            self.get_logger().warn(
                f"No '{target_class}' found in /world_state (z < 0.2). "
                "Check that perception is running.")
            return

        obj_id = target_obj.object_id
        self.get_logger().info(f"Target object identified: {obj_id}")

        # 2. Generate PDDL problem dynamically and solve
        plan = self._plan(obj_id)
        if plan is None:
            self.get_logger().error("PDDL planning failed (no plan found).")
            return

        # 3. Publish the plan as a sequence of TaskAction messages
        self._publish_plan(plan, obj_id)

    # ------------------------------------------------------------------ #
    # Planning Helpers
    # ------------------------------------------------------------------ #

    def _find_object(self, target_class: str):
        """Return first object matching target_class on the table (z > 1.0)."""
        for obj_id, obj in self.scene_objects.items():
            name_match  = target_class in obj_id.lower()
            class_match = hasattr(obj, 'class_name') and target_class in obj.class_name.lower()
            # Table is at z ~ 0.0 relative to panda_link0.
            if (name_match or class_match) and obj.pose.position.z < 0.2:
                return obj
        return None

    def _generate_problem_pddl(self, obj_id: str) -> str:
        """Generate a PDDL problem file content string for the current scene."""
        # Sanitise obj_id for use as a PDDL identifier
        safe_id = re.sub(r'[^a-z0-9_-]', '_', obj_id.lower())

        return f"""; Auto-generated PDDL problem — {obj_id}
(define (problem pick-and-place-{safe_id})
  (:domain pick-and-place)

  (:objects
    {safe_id} table target
  )

  (:init
    (on {safe_id} table)
    (gripper-empty)
  )

  (:goal
    (at-goal {safe_id})
  )
)
"""

    def _plan(self, obj_id: str):
        """Build a PDDL problem for obj_id, call pyperplan BFS, return solution or None."""
        problem_content = self._generate_problem_pddl(obj_id)

        # Write to a temporary file (pyperplan reads from disk)
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.pddl', delete=False, prefix='tamp_problem_'
        ) as f:
            f.write(problem_content)
            problem_file = f.name

        self.get_logger().info(f"Solving PDDL problem for '{obj_id}'...")

        try:
            # Use direct API: parse → ground → BFS search
            # (avoids the search_plan wrapper which hangs on optional 'validate' binary)
            parser = PDDLParser(DOMAIN_FILE, problem_file)
            domain  = parser.parse_domain()
            problem = parser.parse_problem(domain)
            task    = pddl_grounding.ground(
                problem,
                remove_statics_from_initial_state=True,
                remove_irrelevant_operators=True
            )
            solution = breadth_first_search(task)
        except Exception as e:
            self.get_logger().error(f"pyperplan error: {e}")
            solution = None
        finally:
            os.unlink(problem_file)  # clean up temp file

        if solution:
            plan_str = " → ".join(op.name for op in solution)
            self.get_logger().info(f"Plan found ({len(solution)} steps): {plan_str}")
        else:
            self.get_logger().warn("pyperplan BFS returned no solution.")

        return solution  # list of Operator objects, or None

    def _parse_action_name(self, op_name: str):
        """
        Parse a pyperplan operator name like '(pick bolt_1 table)' into
        (action_type, obj_id).  Returns (action_type_str, None) if no obj arg.
        """
        # Operator names look like: (pick bolt_1 table)
        parts = op_name.strip("() ").split()
        action_type = parts[0].upper()  # 'PICK' or 'PLACE'
        return action_type

    def _publish_plan(self, plan, obj_id: str):
        """Convert pyperplan solution to TaskAction messages and publish sequentially."""
        import time

        for operator in plan:
            action_type = self._parse_action_name(operator.name)

            task_msg = TaskAction()
            task_msg.action_type = action_type
            task_msg.object_id   = obj_id

            # Attach perceived pose so action_executor can use it directly
            if obj_id in self.scene_objects:
                task_msg.target_pose = self.scene_objects[obj_id].pose

            self.plan_pub.publish(task_msg)
            self.get_logger().info(f"Published: {action_type} {obj_id}")
            time.sleep(0.1)


def main(args=None):
    rclpy.init(args=args)
    node = PDDLPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
