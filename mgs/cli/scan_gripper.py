# Copyright (c) 2025 Robert Bosch GmbH
# Author: Roman Freiberg
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from mgs.env.gripper_scan import GripperScanEnv
from mgs.gripper.base import MjScannableGripper
from mgs.gripper.selector import get_gripper
from mgs.util.file import generate_unique_hash


def scan(cfg: DictConfig):
    gripper = get_gripper(cfg.gripper)
    assert isinstance(gripper, MjScannableGripper)
    env = GripperScanEnv(gripper)

    state = cfg.gripper.get("state")
    if state is not None:
        state = np.array(state, dtype=np.float64)
        env.set_state(state)
    else:
        gripper_idxs = env.get_joint_idxs(gripper.get_actuator_joint_names())
        env.set_qpos(np.array(cfg.qpos), gripper_idxs)  # type: ignore

    images, extrinsics, image_masks, segmentation = env.scan(num_images=cfg.num_images)
    intrinsics = env.get_camera_intrinsics()
    return images, extrinsics, intrinsics, image_masks, segmentation


@hydra.main(config_path="config", config_name="scan_gripper")
def main(cfg: DictConfig):
    output_dir = os.getenv("MGS_OUTPUT_DIR")
    output_dir if output_dir is not None else "."
    output_dir = "."

    images, extrinsics, intrinsics, image_masks, segmentation = scan(cfg)

    # gather relevant segmentations
    segmentation_dict = OmegaConf.to_container(
        DictConfig.get(cfg, "gripper.segmentation", OmegaConf.create({}))
    )
    segmentation_dict = OmegaConf.to_container(cfg.gripper.segmentation)
    segments = {}
    for key in segmentation_dict.keys():
        mask_values = segmentation_dict[key]

        mask = segmentation == mask_values[0]
        for mask_value in mask_values:
            mask = ((segmentation == mask_value) & image_masks) | mask

        segments[key] = mask

    file_hash = generate_unique_hash()
    file_path = os.path.join(output_dir, f"{cfg.file_id}")  # type: ignore
    np.savez(
        file_path,
        **{
            "scans": images,
            "scan_extrinsics": extrinsics,
            "scan_intrinsics": intrinsics,
            "scan_masks": image_masks,
            "segments": segments if segments else np.array([]),
        },
    )


if __name__ == "__main__":
    main()
