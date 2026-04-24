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
import logging
import numpy as np
import cv2
from typing import Dict, Any, List
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3

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
from geometry_msgs.msg import Wrench

# OpenPI Client
from openpi_client import image_tools
from openpi_client import websocket_client_policy

class RunOpenPI(Policy):
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        
        # 1. Initialize OpenPI Client
        # Adjust host/port if your server is running elsewhere
        self.client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
        self.get_logger().info("OpenPI Client initialized. Connecting to localhost:8000")

        # Config
        self.image_size = 224 # Standard for pi0 models
        self.override_prompt = "" # User can modify this via external parameter if needed

    def _process_image(self, raw_img) -> np.ndarray:
        """Converts ROS Image -> Resized -> uint8 numpy array."""
        # ROS Bytes to Numpy (H, W, C)
        img_np = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(
            raw_img.height, raw_img.width, 3
        )
        
        # Use OpenPI utilities for consistent preprocessing
        resized = image_tools.resize_with_pad(img_np, self.image_size, self.image_size)
        return image_tools.convert_to_uint8(resized)

    def prepare_observations(self, obs_msg: Observation, prompt: str) -> Dict[str, Any]:
        """Convert ROS Observation message into OpenPI compatible dictionary."""
        
        # DROID-based pi0 model expects:
        # - exterior_image_1_left: (224, 224, 3) uint8
        # - wrist_image_left: (224, 224, 3) uint8
        # - joint_position: (7,) float32
        # - gripper_position: (1,) float32
        
        # Based on RunACT mapping:
        # tcp_pose and tcp_vel are available but DROID pi0 primarily uses joint space + images
        
        obs = {
            "observation/exterior_image_1_left": self._process_image(obs_msg.center_image),
            "observation/wrist_image_left": self._process_image(obs_msg.left_image),
            "observation/joint_position": np.array(obs_msg.joint_states.position[:7], dtype=np.float32),
            "observation/gripper_position": np.array([obs_msg.joint_states.position[7]], dtype=np.float32) if len(obs_msg.joint_states.position) > 7 else np.array([0.0], dtype=np.float32),
            "prompt": prompt,
        }
        
        return obs

    def execute_task(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ):
        self.get_logger().info(f"RunOpenPI.execute_task() enter. Task: {task}")
        
        # Use overridden prompt if provided, otherwise use task description
        prompt = self.override_prompt if self.override_prompt else (task.task_description if task.task_description else "do something useful")

        while True:
            loop_start = time.time()

            # 1. Get Observation
            observation_msg = get_observation()
            if observation_msg is None:
                continue

            # 2. Prepare for OpenPI
            openpi_obs = self.prepare_observations(observation_msg, prompt)

            # 3. Remote Inference
            try:
                # Returns action chunk
                result = self.client.infer(openpi_obs)
                action_chunk = result["actions"]
                
                # OpenPI DROID output is typically (horizon, 8)
                # dims 0-6: joint velocities or positions
                # dim 7: gripper action
                action = action_chunk[0]
                
                self.get_logger().info(f"VLA Action: {action}")

                # 4. Command Robot
                # Map action vector to ROS MotionUpdate
                # Note: RunACT uses Cartesian Twist, but OpenPI DROID is often trained on Joint space.
                # If your robot accepts Joint Velocities:
                motion_update = self.create_joint_velocity_update(action)
                
                # Alternatively, if you must use Twist (dim 0-5):
                # twist = Twist(
                #     linear=Vector3(x=float(action[0]), y=float(action[1]), z=float(action[2])),
                #     angular=Vector3(x=float(action[3]), y=float(action[4]), z=float(action[5])),
                # )
                # motion_update = self.create_cartesian_update(twist)

                move_robot(motion_update=motion_update)
                send_feedback(f"Executing: {prompt}")

            except Exception as e:
                self.get_logger().error(f"OpenPI Inference failed: {e}")
                time.sleep(1.0)
                continue

            # Control loop timing (e.g., 10Hz)
            elapsed = time.time() - loop_start
            time.sleep(max(0, 0.1 - elapsed))

    def create_joint_velocity_update(self, action: np.ndarray):
        """Creates a MotionUpdate for joint velocity control."""
        motion_update_msg = MotionUpdate()
        
        # DROID action usually has 7 joint dims + 1 gripper dim
        # We assign the first 7 to joint velocities
        motion_update_msg.joint_velocities = action[:7].tolist()
        
        # Gripper action (action[7]) - depends on robot interface
        # Some use a separate service or a specific joint
        
        motion_update_msg.header.stamp = self.get_clock().now().to_msg()
        motion_update_msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_JOINT_VELOCITY
        
        return motion_update_msg

    def create_cartesian_update(self, twist: Twist, frame_id: str = "base_link"):
        """Creates a MotionUpdate for Cartesian velocity control."""
        motion_update_msg = MotionUpdate()
        motion_update_msg.velocity = twist
        motion_update_msg.header.frame_id = frame_id
        motion_update_msg.header.stamp = self.get_clock().now().to_msg()
        motion_update_msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
        
        # Default stiffness/damping
        motion_update_msg.target_stiffness = np.diag([100.0, 100.0, 100.0, 50.0, 50.0, 50.0]).flatten()
        motion_update_msg.target_damping = np.diag([40.0, 40.0, 40.0, 15.0, 15.0, 15.0]).flatten()
        
        return motion_update_msg
