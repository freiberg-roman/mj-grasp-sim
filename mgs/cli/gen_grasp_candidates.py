from mgs.util.geo.transforms import SE3Pose
from tqdm import tqdm
import os

import hydra
import numpy as np
from omegaconf import DictConfig

from mgs.obj.selector import get_object
from mgs.sampler.antipodal import AntipodalGraspGenerator
from mgs.util.const import ASSET_PATH
from mgs.gripper.vx300 import GripperVX300
from mgs.gripper.panda import GripperPanda


@hydra.main(config_path="config", config_name="gen_grasp_candidates")
def main(cfg: DictConfig):
    print(f"Generating grasp candidates for gripper: {cfg.gripper.name}")
    object_id_file = os.path.join(ASSET_PATH, "mj-objects", "fast_eta_objects.txt")
    with open(object_id_file, "r") as file:
        all_object_ids = file.read().splitlines()

    all_gripper = {
        "ShadowHand": None,
        "LeapGripper": None,
        "PandaGripper": GripperPanda(SE3Pose.randn_se3(1)[0]),
        "VXGripper": GripperVX300(SE3Pose.randn_se3(1)[0]),
    }

    object_id = all_object_ids[int(cfg.id)]
    obj = get_object(object_id)
    sampler = None
    if cfg.gripper.name in ["ShadowHand", "LeapGripper", "AllegroGripper"]:
        from mgs.sampler.contact import ContactBasedDiff
        from mgs.sampler.kin.shadow import ShadowKinematicsModel
        from mgs.sampler.kin.leap import LeapHandKinematicsModel
        from mgs.sampler.kin.allegro import AllegroKinematicsModel
        sampler = ContactBasedDiff(obj)
        all_kins = {
            "ShadowHand": ShadowKinematicsModel(),
            "LeapGripper": LeapHandKinematicsModel(),
            "PandaGripper": None,
            "AllegroGripper": AllegroKinematicsModel(),
            "VXGripper": None,
        }
        kin_model = all_kins[cfg.gripper.name]
    else:
        sampler = AntipodalGraspGenerator(obj)

    output_dir = os.getenv("MGS_OUTPUT_DIR")
    if output_dir is None:
        output_dir = "."
    output_dir = os.path.join(output_dir, cfg.gripper.name, object_id)
    print("Current output dir: ", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    num_to_generate = cfg.get("num_grasps", 10000)

    all_hs = []
    all_joints = []
    for _ in range(1):
        if isinstance(sampler, AntipodalGraspGenerator):
            Hs_batch, aux_info_batch = sampler.generate_grasps(num_to_generate)
            gripper = all_gripper[cfg.gripper.name]
            width_clamped = gripper._clamp_width(aux_info_batch["width"])
            j1, j2 = gripper.width_to_joints(width_clamped)
            joints = np.stack([j1, j2], axis=-1)
            all_joints.append(joints)
            all_hs.append(Hs_batch)
            Hs_batch = np.concatenate(all_hs, axis=0)
            joints = np.concatenate(all_joints, axis=0)
        else:
            import jax.numpy as jnp
            Hs_batch, aux_info_batch = sampler.generate_grasps(
                num_to_generate, kin_model
            )
            all_joints.append(aux_info_batch["joints"])
            all_hs.append(Hs_batch)
            Hs_batch = jnp.concatenate(all_hs, axis=0)
            joints = jnp.concatenate(all_joints, axis=0)

    path = os.path.join(output_dir, "candidates.npz")

    state_dict = {"pose": Hs_batch, "joints": joints}
    np.savez(path, **state_dict)
    print("Done!")


if __name__ == "__main__":
    main()
