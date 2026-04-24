#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import time
import numpy as np
import cv2
from typing import Dict, Any
from rclpy.node import Node

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task

from aic_control_interfaces.msg import (
    MotionUpdate,
    TrajectoryGenerationMode,
)

# OpenPI Client
from openpi_client import image_tools
from openpi_client import websocket_client_policy

class RunOpenPIBase_latest(Policy):
    """
    Refined policy for OpenPI VLA Base Models.
    Aligns with RunACT.py observations (3 cameras, 26-dim state).
    """
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        
        # 1. Initialize OpenPI Client
        # Adjust host/port if your TPU/GPU server is running elsewhere
        parent_node.declare_parameter("openpi_host", "openpi_server")
        parent_node.declare_parameter("openpi_port", 8000)

        host = parent_node.get_parameter("openpi_host").get_parameter_value().string_value
        port = parent_node.get_parameter("openpi_port").get_parameter_value().integer_value

        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.get_logger().info(f"OpenPI Base Policy initialized. Connected to VLA server at {host}:{port}")

        # Config
        self.image_size = 224      # Standard input size for pi0 models
        self.image_scaling = 0.25  # Match AICRobotAICControllerConfig scaling
        self.override_prompt = ""  # User can modify this for natural language control

    def _process_image(self, raw_img) -> np.ndarray:
        """Converts ROS Image -> Resized -> uint8 numpy array (H, W, C)."""
        if raw_img is None or raw_img.data is None:
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        # ROS Bytes to Numpy (H, W, C)
        img_np = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(
            raw_img.height, raw_img.width, 3
        )
        
        # Resize based on AIC controller scaling
        if self.image_scaling != 1.0:
            img_np = cv2.resize(
                img_np, None, fx=self.image_scaling, fy=self.image_scaling, interpolation=cv2.INTER_AREA
            )

        # OpenPI utility for consistent padding/resizing to model input size
        resized = image_tools.resize_with_pad(img_np, self.image_size, self.image_size)
        return image_tools.convert_to_uint8(resized)

    def prepare_observations(self, obs_msg: Observation, prompt: str) -> Dict[str, Any]:
        """Convert ROS Observation into ALOHA-compatible format for Base VLA."""
        
        # 1. Process 3 Cameras (Mapped to ALOHA EXPECTED_CAMERAS)
        obs = {
            "images": {
                "cam_high": self._process_image(obs_msg.center_image),
                "cam_left_wrist": self._process_image(obs_msg.left_image),
                "cam_right_wrist": self._process_image(obs_msg.right_image),
            }
        }

        # 2. Process Robot State (26 dimensions)
        tcp_pose = obs_msg.controller_state.tcp_pose
        tcp_vel = obs_msg.controller_state.tcp_velocity

        state = np.array([
            # TCP Position (3)
            tcp_pose.position.x, tcp_pose.position.y, tcp_pose.position.z,
            # TCP Orientation (4)
            tcp_pose.orientation.x, tcp_pose.orientation.y, tcp_pose.orientation.z, tcp_pose.orientation.w,
            # TCP Linear Vel (3)
            tcp_vel.linear.x, tcp_vel.linear.y, tcp_vel.linear.z,
            # TCP Angular Vel (3)
            tcp_vel.angular.x, tcp_vel.angular.y, tcp_vel.angular.z,
            # TCP Error (6)
            *obs_msg.controller_state.tcp_error,
            # Joint Positions (7)
            *obs_msg.joint_states.position[:7],
        ], dtype=np.float32)

        obs["state"] = state
        obs["prompt"] = prompt
        
        return obs

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ):
        self.get_logger().info(f"RunOpenPIBase_latest.execute_task() started. Task: {task.id}")
        
        # Natural Language Prompt
        prompt = self.override_prompt if self.override_prompt else (f"insert {task.cable_name} into {task.port_name}" if f"insert {task.cable_name} into {task.port_name}" else "perform task")

        while True:
            loop_start = time.time()

            # 1. Get & Process Observation
            observation_msg = get_observation()
            if observation_msg is None:
                continue

            openpi_obs = self.prepare_observations(observation_msg, prompt)

            # 2. Remote VLA Inference
            try:
                # Call server - returns action chunk
                result = self.client.infer(openpi_obs)
                action_chunk = result["actions"]
                
                # Use the first action from the horizon
                # Action is NOT a Twist - it's a raw control vector from the VLA
                action = action_chunk[0]
                
                self.get_logger().info(f"VLA Base Action received (dim {len(action)})")

                # 3. Command Robot
                motion_update = self.create_motion_command(action)
                move_robot(motion_update=motion_update)
                send_feedback(f"Executing: {prompt}")

            except Exception as e:
                self.get_logger().error(f"OpenPI Communication Error: {e}")
                time.sleep(1.0)
                continue

            # Control loop timing
            elapsed = time.time() - loop_start
            time.sleep(max(0, 0.1 - elapsed)) # 10Hz target

    def create_motion_command(self, action: np.ndarray) -> MotionUpdate:
        """
        Converts the raw VLA action vector into a ROS MotionUpdate.
        Modify based on whether your base model outputs Joints or Cartesian targets.
        """
        motion_update_msg = MotionUpdate()
        motion_update_msg.header.stamp = self.get_clock().now().to_msg()
        
        # EXAMPLE: If model outputs 7 joint targets + 1 gripper
        if len(action) >= 7:
            motion_update_msg.joint_positions = action[:7].tolist()
            motion_update_msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_JOINT_POSITION
        
        # EXAMPLE: If model outputs Cartesian velocity (Twist-like)
        # Note: The user confirmed Twist is an observation, not necessarily the VLA action.
        # motion_update_msg.velocity = Twist(...)
        # motion_update_msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY

        return motion_update_msg
