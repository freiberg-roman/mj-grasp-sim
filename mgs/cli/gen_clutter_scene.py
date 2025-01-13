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

from mgs.cli.stats.stat_check import is_graspable
from mgs.env.selector import get_env, get_env_from_dict
from mgs.gripper.selector import get_gripper
from mgs.obj.selector import get_objects
from mgs.util.file import generate_unique_hash
from mgs.util.geo.transforms import SE3Pose


def get_grasps(gripper_name, obj_id, gripper_type=""):
    grasp_path = os.path.join(  # type: ignore
        os.getenv("MGS_INPUT_DIR"),  # type: ignore
        gripper_name + "-" + gripper_type,
    )
    all_files = os.listdir(grasp_path)
    files_with_obj_id = [file for file in all_files if file.startswith(f"{obj_id}_")]
    if len(files_with_obj_id) == 0:
        return None

    grasp_vecs = []
    for file in files_with_obj_id:
        current_file_path = os.path.join(grasp_path, file)
        grasp_dict = np.load(current_file_path)
        grasps_vec = grasp_dict["grasps"]
        grasp_vecs.append(grasps_vec)

    merged_grasps = np.concatenate(grasp_vecs, axis=0)
    return SE3Pose.from_vec(merged_grasps, layout="pq", type="wxyz")


def gen_stable_scene(cfg: DictConfig):
    obj_list = get_objects(cfg.object)
    gripper = get_gripper(
        cfg.gripper,
        default_pose=SE3Pose(
            np.array([5.0, 5.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0]), type="wxyz"
        ),
    )

    env = get_env(cfg.env, gripper=deepcopy(gripper), obj_list=deepcopy(obj_list))
    env.gen_clutter()
    scene_dict = env.to_dict()

    if not env.is_stable():
        print("Scene unstable!")
        return None

    return scene_dict


def gen(cfg: DictConfig, scene_def):
    env = get_env_from_dict(cfg.env, (deepcopy(scene_def)))

    valid_grasps = []
    for obj_name, obj_id in zip(env.object_names, env.object_ids):
        is_graspable_res = is_graspable(cfg.gripper, obj_id, eta=cfg.gripper.eta)
        if not is_graspable_res:
            continue
        grasps = get_grasps(
            gripper_name=cfg.gripper.id,
            gripper_type=cfg.gripper.get("grasp_type", ""),
            obj_id=obj_id,
        )
        if grasps is None:
            continue

        o2w = env.get_obj_pose(obj_name)
        grasps = o2w @ grasps
        collision_free_mask = env.grasp_collision_mask(deepcopy(grasps))  # type: ignore
        grasps = grasps[collision_free_mask]  # type: ignore
        if len(grasps) == 0:
            continue

        if len(grasps) >= 1500:
            idx = np.random.randint(len(grasps), size=1500)
            grasps = grasps[idx]  # type: ignore

        stable_grasp_mask = env.grasp_stable_mask(
            deepcopy(grasps), deepcopy(scene_def["env_state"]["state"])
        )
        grasps = grasps[stable_grasp_mask]  # type: ignore

        if len(grasps) == 0:
            continue

        valid_grasps.append(
            {
                "object_name": obj_name,
                "object_id": obj_id,
                "grasps": grasps.to_vec(layout="pq", type="wxyz"),
            }
        )

    if len(valid_grasps) == 0:
        return None

    return valid_grasps


def scan(cfg: DictConfig, scene_def):
    env = get_env_from_dict(cfg.env, (deepcopy(scene_def)))
    images, extrinsics, image_masks, _ = env.scan(num_images=cfg.num_images)
    intrinsics = env.get_camera_intrinsics()

    return images, extrinsics, intrinsics, image_masks


@hydra.main(config_path="config", config_name="gen_clutter_scene")
def main(cfg: DictConfig):
    output_dir = os.getenv("MGS_OUTPUT_DIR")
    output_dir = output_dir if output_dir is not None else "."

    iteration = 0
    is_done = False
    while (not is_done) and (iteration < 10):
        iteration += 1
        scene_dict = gen_stable_scene(deepcopy(cfg))
        if scene_dict is None:
            continue

        if scene_dict is None:
            continue
        valid_grasps = gen(deepcopy(cfg), scene_dict)
        if valid_grasps is None:
            continue
        scene_hash = generate_unique_hash()

        file_path = os.path.join(output_dir, f"clutter_scene_{scene_hash}")

        np.savez(
            file_path,
            **{
                "scene_definition": scene_dict,
                "valid_grasps": valid_grasps,
            },
        )
        print("Finished!")
        is_done = True


if __name__ == "__main__":
    main()
