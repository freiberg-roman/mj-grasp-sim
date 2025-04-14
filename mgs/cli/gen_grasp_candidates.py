from tqdm import tqdm
import os

import hydra
import numpy as np
from omegaconf import DictConfig

from mgs.obj.selector import get_object
from mgs.sampler.contact import ContactBasedDiff
from mgs.sampler.antipodal import AntipodalGraspGenerator
from mgs.sampler.kin.shadow import ShadowKinematicsModel
from mgs.sampler.kin.leap import LeapHandKinematicsModel
from mgs.util.const import ASSET_PATH


@hydra.main(config_path="config", config_name="gen_grasp_candidates")
def main(cfg: DictConfig):
    print(f"Generating grasp candidates for gripper: {cfg.gripper.name}")
    object_id_file = os.path.join(
        ASSET_PATH, "mj-objects", "fast_eta_objects.txt")
    with open(object_id_file, "r") as file:
        all_object_ids = file.read().splitlines()

    obj = get_object(all_object_ids[0])
    cbd = ContactBasedDiff(obj) # required to not trigger recompilation for each object
    all_kins = {
            "ShadowHand": ShadowKinematicsModel(),
            "LeapGripper": LeapHandKinematicsModel(),
            "PandaGripper": None,
            "VXGripper": None,
        }

    for object_id in tqdm(all_object_ids):
        obj = get_object(object_id)
        sampler = (
            cbd.update_object(obj)
            if cfg.gripper.name in ["ShadowHand", "LeapGripper"]
            else AntipodalGraspGenerator(obj)
        )
        kin_model = all_kins[cfg.gripper.name]

        output_dir = os.getenv("MGS_OUTPUT_DIR")
        if output_dir is None:
            output_dir = "."
        output_dir = os.path.join(output_dir, cfg.gripper.name, object_id)
        print("Current output dir: ", output_dir)
        os.makedirs(output_dir, exist_ok=True)
        num_to_generate = cfg.get("num_grasps", 10000)

        if isinstance(sampler, AntipodalGraspGenerator):
            Hs_batch, aux_info_batch = sampler.generate_grasps(num_to_generate)
        elif isinstance(sampler, ContactBasedDiff):
            Hs_batch, aux_info_batch = sampler.generate_grasps(
                num_to_generate, kin_model
            )

        path = os.path.join(output_dir, "candidates.npz")

        state_dict = {"pose": Hs_batch, **aux_info_batch}
        np.savez(path, **state_dict)
        print("Done!")


if __name__ == "__main__":
    main()
