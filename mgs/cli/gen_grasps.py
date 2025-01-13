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

import csv
import os
import time
from copy import deepcopy

import hydra
import numpy as np
from omegaconf import DictConfig

from mgs.cli.stats.stat_check import is_graspable
from mgs.env.gravityless_object_grasping import GravitylessObjectGrasping
from mgs.gripper.selector import get_gripper
from mgs.obj.selector import get_object
from mgs.sampler.antipodal import AntipodalGraspGenerator
from mgs.util.file import generate_unique_hash


@hydra.main(config_path="config", config_name="gen_grasps")
def main(cfg: DictConfig):
    gripper = get_gripper(cfg.gripper)
    obj = get_object(cfg.object_id)
    sampler = AntipodalGraspGenerator(obj)
    env = GravitylessObjectGrasping(gripper, obj)
    output_dir = os.getenv("MGS_OUTPUT_DIR")

    if cfg.stats == True:
        start_time = time.time()
        grasps = sampler.generate_grasps(cfg.num_grasps)
        evaluations, pos_delta, rot_delta = env.grasp_stability_evaluation(
            deepcopy(grasps), stats=True
        )
        valid_grasps = grasps.to_vec(layout="pq", type="wxyz")[evaluations]
        end_time = time.time()
        total_time = end_time - start_time

        pos_delta = pos_delta[evaluations]
        rot_delta = rot_delta[evaluations]

        stats = {
            "name": obj.object_id,
            "number_sucesful_grasps": len(valid_grasps),
            "total_time": total_time,
            "pos_drift_under_005": len(pos_delta[pos_delta < 0.005]),
            "pos_drift_under_010": len(pos_delta[pos_delta < 0.01]),
            "pos_drift_under_015": len(pos_delta[pos_delta < 0.015]),
            "pos_drift_under_025": len(pos_delta[pos_delta < 0.025]),
            "rot_drift_under_010": len(rot_delta[rot_delta < 10.0]),
            "rot_drift_under_012": len(rot_delta[rot_delta < 12.0]),
            "rot_drift_under_015": len(rot_delta[rot_delta < 15.0]),
            "rot_drift_under_025": len(rot_delta[rot_delta < 25.0]),
            "rot_pos_setting_1": len(
                pos_delta[(pos_delta < 0.005) & (rot_delta < 10.0)]
            ),
            "rot_pos_setting_2": len(
                pos_delta[(pos_delta < 0.01) & (rot_delta < 12.0)]
            ),
            "rot_pos_setting_3": len(
                pos_delta[(pos_delta < 0.015) & (rot_delta < 15.0)]
            ),
            "rot_pos_setting_4": len(
                pos_delta[(pos_delta < 0.025) & (rot_delta < 25.0)]
            ),
        }

        path = os.path.join(output_dir, generate_unique_hash() + "_stat.csv")  # type: ignore
        with open(path, "a") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=stats.keys())
            if csvfile.tell() == 0:
                writer.writeheader()
            writer.writerow(stats)

        return

    # find out whether grasps will be computed in given time (estimates from prior stats)
    is_graspable_res = is_graspable(cfg.gripper, cfg.object_id, eta=cfg.gripper.eta)
    if not is_graspable_res:
        return  # we are done

    num_successful_grasps = 0
    all_successful_grasps = []

    while num_successful_grasps < cfg.num_grasps:
        grasps = sampler.generate_grasps(
            max([(cfg.num_grasps - num_successful_grasps) * 2, 0])
        )
        evaluations = env.grasp_stability_evaluation(deepcopy(grasps))
        valid_grasps = grasps.to_vec(layout="pq", type="wxyz")[evaluations]

        num_successful_grasps += len(valid_grasps)
        all_successful_grasps.append(valid_grasps)

    all_successful_grasps = np.concatenate(all_successful_grasps, axis=0)[
        : cfg.num_grasps
    ]

    transform = gripper.base_to_contact_transform().to_vec(layout="pq", type="wxyz")
    state_dict = {
        "gripper_transform": transform,
        "grasps": all_successful_grasps,
        "object_id": cfg.object_id,
        "gripper": cfg.gripper.name,
    }

    path = os.path.join(
        output_dir if output_dir is not None else ".",
        f"{cfg.object_id}_{obj.name}.npz",
    )

    np.savez(path, **state_dict)
    print("Done!")


if __name__ == "__main__":
    main()
