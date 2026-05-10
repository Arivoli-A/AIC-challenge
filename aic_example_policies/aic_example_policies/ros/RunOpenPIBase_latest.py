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
    JointMotionUpdate,
    TrajectoryGenerationMode,
)

# OpenPI Client
from openpi_client import image_tools
from openpi_client import websocket_client_policy

class RunOpenPIBase_latest(Policy):
    """
    AIC adapter for OpenPI UR5e-style inference.

    Shapes the AIC observation into the UR5 example schema used by OpenPI:
    `state` + `image` + `image_mask` + `prompt`.
    """
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)

        parent_node.declare_parameter("openpi_host", "openpi_server")
        parent_node.declare_parameter("openpi_port", 8000)
        parent_node.declare_parameter("openpi_default_prompt", "")

        host = parent_node.get_parameter("openpi_host").get_parameter_value().string_value
        port = parent_node.get_parameter("openpi_port").get_parameter_value().integer_value
        self.override_prompt = (
            parent_node.get_parameter("openpi_default_prompt")
            .get_parameter_value()
            .string_value
        )

        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.get_logger().info(
            f"OpenPI UR5 adapter initialized. Connected to VLA server at {host}:{port}"
        )

        self.image_size = 224
        self.image_scaling = 0.25

    def _process_image(self, raw_img) -> np.ndarray:
        """Converts ROS Image -> Resized -> uint8 numpy array (H, W, C)."""
        if raw_img is None or not raw_img.data:
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

    @staticmethod
    def _pose_to_array(pose) -> np.ndarray:
        return np.asarray(
            [
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _twist_to_array(twist) -> np.ndarray:
        return np.asarray(
            [
                twist.linear.x,
                twist.linear.y,
                twist.linear.z,
                twist.angular.x,
                twist.angular.y,
                twist.angular.z,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _wrench_to_array(wrench) -> np.ndarray:
        return np.asarray(
            [
                wrench.force.x,
                wrench.force.y,
                wrench.force.z,
                wrench.torque.x,
                wrench.torque.y,
                wrench.torque.z,
            ],
            dtype=np.float32,
        )

    def prepare_observations(self, obs_msg: Observation, prompt: str) -> Dict[str, Any]:
        """Serialize a fuller AIC observation; OpenPI transforms pick required fields."""
        left_image = self._process_image(obs_msg.left_image)
        center_image = self._process_image(obs_msg.center_image)
        right_image = self._process_image(obs_msg.right_image)

        controller_state = obs_msg.controller_state

        return {
            "aic_observation": {
                "images": {
                    "left": left_image,
                    "center": center_image,
                    "right": right_image,
                },
                "joint_state": {
                    "position": np.asarray(obs_msg.joint_states.position, dtype=np.float32),
                    "velocity": np.asarray(obs_msg.joint_states.velocity, dtype=np.float32),
                    "effort": np.asarray(obs_msg.joint_states.effort, dtype=np.float32),
                },
                "controller_state": {
                    "tcp_pose": self._pose_to_array(controller_state.tcp_pose),
                    "tcp_velocity": self._twist_to_array(controller_state.tcp_velocity),
                    "reference_tcp_pose": self._pose_to_array(controller_state.reference_tcp_pose),
                    "tcp_error": np.asarray(controller_state.tcp_error, dtype=np.float32),
                    "reference_joint_positions": np.asarray(
                        controller_state.reference_joint_state.positions, dtype=np.float32
                    ),
                    "reference_joint_velocities": np.asarray(
                        controller_state.reference_joint_state.velocities, dtype=np.float32
                    ),
                    "reference_joint_accelerations": np.asarray(
                        controller_state.reference_joint_state.accelerations, dtype=np.float32
                    ),
                    "reference_joint_effort": np.asarray(
                        controller_state.reference_joint_state.effort, dtype=np.float32
                    ),
                },
                "wrist_wrench": self._wrench_to_array(obs_msg.wrist_wrench.wrench),
            },
            "prompt": prompt,
        }

    @staticmethod
    def _make_prompt(task: Task, override_prompt: str) -> str:
        if override_prompt:
            return override_prompt
        return (
            f"Insert the {task.cable_name} cable by guiding the {task.plug_name} "
            f"into port {task.port_name} on {task.target_module_name}."
        )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ):
        self.get_logger().info(f"RunOpenPIBase_latest.execute_task() started. Task: {task.id}")
        
        prompt = self._make_prompt(task, self.override_prompt)

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

                joint_motion_update = self.create_joint_motion_command(action)
                move_robot(joint_motion_update=joint_motion_update)
                send_feedback(f"Executing: {prompt}")

            except Exception as e:
                self.get_logger().error(f"OpenPI Communication Error: {e}")
                time.sleep(1.0)
                continue

            # Control loop timing
            elapsed = time.time() - loop_start
            time.sleep(max(0, 0.1 - elapsed)) # 10Hz target

    def create_joint_motion_command(self, action: np.ndarray) -> JointMotionUpdate:
        """Convert OpenPI UR5-style actions into an AIC joint-space command."""
        joint_targets = np.asarray(action[:6], dtype=np.float64)
        if joint_targets.shape[0] < 6:
            joint_targets = np.pad(joint_targets, (0, 6 - joint_targets.shape[0]))

        joint_motion_update = JointMotionUpdate(
            target_stiffness=[100.0, 100.0, 100.0, 50.0, 50.0, 50.0],
            target_damping=[40.0, 40.0, 40.0, 15.0, 15.0, 15.0],
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION
            ),
        )
        joint_motion_update.target_state.positions = joint_targets.tolist()
        return joint_motion_update
