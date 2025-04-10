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

# --- File: mgs/cli/gen_grasps.py ---

import csv
import os
import time
from copy import deepcopy
from typing import List, Dict, Any  # Added List, Dict, Any

import hydra
import numpy as np
from omegaconf import DictConfig

# Assuming paths are correct relative to project structure
from mgs.cli.stats.stat_check import is_graspable
from mgs.env.gravityless_object_grasping import GravitylessObjectGrasping
from mgs.gripper.selector import get_gripper
from mgs.obj.selector import get_object
from mgs.sampler.antipodal import AntipodalGraspGenerator
from mgs.util.file import generate_unique_hash
from mgs.util.geo.transforms import SE3Pose  # Needed for SE3Pose operations


@hydra.main(config_path="config", config_name="gen_grasps")
def main(cfg: DictConfig):
    print(
        f"Generating grasps for object: {cfg.object_id} with gripper: {cfg.gripper.name}"
    )
    try:
        gripper = get_gripper(cfg.gripper)
        obj = get_object(cfg.object_id)
        sampler = AntipodalGraspGenerator(obj)
        env = GravitylessObjectGrasping(gripper, obj)
    except Exception as e:
        print(f"Error during initialization: {e}")
        import traceback

        traceback.print_exc()
        return  # Cannot proceed

    output_dir = os.getenv("MGS_OUTPUT_DIR")
    if output_dir is None:
        output_dir = "."
        print(
            f"Warning: MGS_OUTPUT_DIR not set. Saving results to current directory: {os.path.abspath(output_dir)}"
        )
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    # --- Grasp Generation and Filtering Loop ---
    num_target_grasps = cfg.num_grasps
    all_successful_grasps_poses: List[SE3Pose] = []
    all_successful_aux_info: Dict[str, List[np.ndarray]] = {
        "grasp_widths": [],
        "contact_points_one": [],
        "contact_points_two": [],
        "surface_normals_one": [],
        # Add other keys from aux_info if needed
    }
    num_generated_total = 0
    max_sampling_multiplier = (
        10  # Try sampling up to 10x the remaining needed grasps per iteration
    )
    max_total_samples = (
        num_target_grasps * max_sampling_multiplier * 5
    )  # Absolute max samples
    loop_start_time = time.time()

    while (
        len(all_successful_grasps_poses) < num_target_grasps
        and num_generated_total < max_total_samples
    ):
        num_remaining = num_target_grasps - len(all_successful_grasps_poses)
        # Sample more aggressively initially, reduce multiplier as we get closer
        sampling_multiplier = max(
            2, int(max_sampling_multiplier * (num_remaining / num_target_grasps))
        )
        num_to_sample = max(
            num_remaining * sampling_multiplier, 20
        )  # Sample at least 20
        num_generated_total += num_to_sample

        print(
            f"Sampling {num_to_sample} grasps... (Need {num_remaining} more, Total generated: {num_generated_total})"
        )
        try:
            # ---> Get poses AND auxiliary info from sampler <---
            grasps_batch_poses, aux_info_batch = sampler.generate_grasps(num_to_sample)
            aux_info_batch["grasp_widths"] = gripper._clamp_width(
                aux_info_batch["grasp_widths"]
            )
            print(aux_info_batch["grasp_widths"])
            # -----------------------------------------------------

            if len(grasps_batch_poses) == 0:
                print(
                    "Sampler returned 0 grasps in this batch. Trying again or stopping if limit reached."
                )
                if num_generated_total >= max_total_samples:
                    break
                continue  # Try sampling again

            # ---> Pass poses AND auxiliary info to evaluation <---
            evaluations_mask = env.grasp_stability_evaluation(
                deepcopy(grasps_batch_poses), deepcopy(aux_info_batch)
            )
            # ------------------------------------------------------

            num_valid_in_batch = np.sum(evaluations_mask)
            print(f"  Found {num_valid_in_batch} stable grasps in this batch.")

            if num_valid_in_batch > 0:
                # Filter poses
                valid_poses_batch = grasps_batch_poses[evaluations_mask]
                all_successful_grasps_poses.extend(
                    list(valid_poses_batch)
                )  # Store SE3Pose objects

                # Filter auxiliary info
                for key in all_successful_aux_info.keys():
                    if key in aux_info_batch:
                        filtered_data = aux_info_batch[key][evaluations_mask]
                        all_successful_aux_info[key].append(filtered_data)
                    else:
                        print(f"Warning: Key '{key}' not found in sampler aux_info.")

        except Exception as e:
            print(f"Error during sampling or evaluation batch: {e}")
            import traceback

            traceback.print_exc()
            # Decide whether to continue or stop
            # break # Option: Stop on error
            continue  # Option: Skip batch on error

    loop_end_time = time.time()
    print(
        f"Finished grasp generation loop in {loop_end_time - loop_start_time:.2f} seconds."
    )
    print(
        f"Generated {len(all_successful_grasps_poses)} stable grasps (target: {num_target_grasps})."
    )

    if not all_successful_grasps_poses:
        print("No successful grasps generated. Exiting.")
        return

    # --- Concatenate and Prepare Save Data ---
    # Convert list of SE3Pose objects to a stacked numpy array (7D vector)
    final_grasp_vecs = np.stack(
        [p.to_vec(layout="pq", type="wxyz") for p in all_successful_grasps_poses],
        axis=0,
    )

    # Concatenate auxiliary info arrays
    final_aux_info: Dict[str, np.ndarray] = {}
    for key, data_list in all_successful_aux_info.items():
        if data_list:  # Check if list is not empty
            final_aux_info[key] = np.concatenate(data_list, axis=0)
        else:
            # Handle case where no valid grasps were found for a specific aux key
            # Determine expected shape, e.g., (0, 3) for points, (0,) for widths
            example_shape = (
                aux_info_batch[key].shape[1:]
                if key in aux_info_batch and aux_info_batch[key].size > 0
                else ()
            )
            final_aux_info[key] = np.empty(
                (0, *example_shape), dtype=np.float32
            )  # Create empty array with correct dimensions

    # Limit to the exact number of target grasps if we overshot
    num_to_keep = min(len(final_grasp_vecs), num_target_grasps)
    final_grasp_vecs = final_grasp_vecs[:num_to_keep]
    for key in final_aux_info:
        final_aux_info[key] = final_aux_info[key][:num_to_keep]

    # --- Create Save Dictionary ---
    transform = gripper.base_to_contact_transform().to_vec(layout="pq", type="wxyz")
    state_dict = {
        "gripper_transform": transform,
        "grasps": final_grasp_vecs,  # Saved as 7D vectors
        "object_id": cfg.object_id,
        "gripper": cfg.gripper.name,
        # ---> Add auxiliary info <---
        **final_aux_info,  # Unpack the dict items
    }

    # --- Save Results ---
    # Use a unique hash for the filename part related to the object instance name from the sampler
    # obj.name might contain problematic characters, use hash instead or sanitize
    obj_instance_name_hash = (
        generate_unique_hash(length=8) if hasattr(obj, "name") else "unknown_obj"
    )
    save_filename = f"{cfg.object_id}_{obj_instance_name_hash}.npz"
    path = os.path.join(output_dir, save_filename)

    try:
        np.savez(path, **state_dict)
        print(f"Successfully saved {num_to_keep} grasps and auxiliary info to: {path}")
    except Exception as e:
        print(f"Error saving data to {path}: {e}")

    print("Done!")


if __name__ == "__main__":
    main()
