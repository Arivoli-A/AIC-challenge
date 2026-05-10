import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model


@dataclasses.dataclass(frozen=True)
class UR5Inputs(transforms.DataTransformFn):
    """Inputs for single-arm UR5-style policies."""

    model_type: _model.ModelType = _model.ModelType.PI0

    @staticmethod
    def _extract_from_aic_observation(data: dict) -> dict:
        aic_observation = data["aic_observation"]
        images = aic_observation["images"]
        raw_joint_positions = np.asarray(aic_observation["joint_state"]["position"], dtype=np.float32)
        joint_positions = raw_joint_positions
        if joint_positions.shape[0] < 6:
            joint_positions = np.pad(joint_positions, (0, 6 - joint_positions.shape[0]))

        gripper_position = np.float32(0.0)
        if raw_joint_positions.shape[0] > 6:
            gripper_position = raw_joint_positions[6]

        base_image = np.asarray(images["center"])
        left_wrist_image = np.asarray(images["left"])
        right_wrist_image = np.asarray(images["right"])

        return {
            "state": np.concatenate(
                [joint_positions[:6], np.asarray([gripper_position], dtype=np.float32)]
            ),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if np.any(right_wrist_image) else np.False_,
            },
        }

    def __call__(self, data: dict) -> dict:
        if "aic_observation" in data:
            inputs = self._extract_from_aic_observation(data)
        else:
            base_image = np.asarray(data["image"]["base_0_rgb"])
            left_wrist_image = np.asarray(data["image"]["left_wrist_0_rgb"])
            right_wrist_image = np.asarray(data["image"]["right_wrist_0_rgb"])

            inputs = {
                "state": np.asarray(data["state"]),
                "image": {
                    "base_0_rgb": base_image,
                    "left_wrist_0_rgb": left_wrist_image,
                    "right_wrist_0_rgb": right_wrist_image,
                },
                "image_mask": {
                    "base_0_rgb": np.asarray(data["image_mask"]["base_0_rgb"]),
                    "left_wrist_0_rgb": np.asarray(data["image_mask"]["left_wrist_0_rgb"]),
                    "right_wrist_0_rgb": np.asarray(data["image_mask"]["right_wrist_0_rgb"]),
                },
            }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR5Outputs(transforms.DataTransformFn):
    """Outputs for single-arm UR5-style policies."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
