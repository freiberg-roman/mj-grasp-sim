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
from copy import deepcopy

import hydra
import numpy as np
from omegaconf import DictConfig

from mgs.env.selector import get_env_from_dict
from mgs.gripper.base import MjScannableGripper
from mgs.gripper.selector import get_gripper


def scan(cfg: DictConfig, scene_def):
    gripper = get_gripper(cfg.gripper)
    assert isinstance(gripper, MjScannableGripper)
    env = get_env_from_dict(cfg.env, (deepcopy(scene_def)))
    images, extrinsics, image_masks, _ = env.scan(num_images=cfg.num_images)
    intrinsics = env.get_camera_intrinsics()
    return images, extrinsics, intrinsics, image_masks


@hydra.main(config_path="config", config_name="render_scene")
def main(cfg: DictConfig):
    output_dir = os.getenv("MGS_OUTPUT_DIR")
    assert output_dir is not None

    input_dir = os.getenv("MGS_INPUT_DIR")
    assert input_dir is not None

    all_files = [f for f in os.listdir(input_dir) if f.endswith(".npz")]

    for file in all_files:
        file_path = os.path.join(input_dir, file)
        scene = np.load(file_path, allow_pickle=True)
        scene_dict = scene["scene_definition"].item()
        valid_grasps = scene["valid_grasps"]
        images, extrinsics, intrinsics, image_masks = scan(
            deepcopy(cfg), deepcopy(scene_dict)
        )
        output_file_path = os.path.join(output_dir, file)
        np.savez(
            output_file_path,
            os.path.join(output_dir, file),
            **{
                "scene_definition": scene_dict,
                "valid_grasps": valid_grasps,
                "scans": images,
                "scan_extrinsics": extrinsics,
                "scan_intrinsics": intrinsics,
                "scan_masks": image_masks,
            },
        )
        print("Finished!")


if __name__ == "__main__":
    main()
