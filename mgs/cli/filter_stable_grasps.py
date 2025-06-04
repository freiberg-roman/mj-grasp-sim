import os

import hydra
import numpy as np
from omegaconf import DictConfig

from mgs.env.gravityless_object_grasping import GravitylessObjectGrasping
from mgs.gripper.selector import get_gripper
from mgs.obj.selector import get_object
from mgs.util.geo.transforms import SE3Pose  # Needed for SE3Pose operations
from mgs.util.const import ASSET_PATH


@hydra.main(config_path="config", config_name="filter_stable_grasps")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.to_absolute_path("."))
    gripper = get_gripper(cfg.gripper)
    object_id_file = os.path.join(ASSET_PATH, "mj-objects", "fast_eta_objects.txt")
    with open(object_id_file, "r") as file:
        all_object_ids = file.read().splitlines()

    object_id = all_object_ids[int(cfg.id)]
    obj = get_object(object_id)
    env = GravitylessObjectGrasping(gripper, obj)

    # load in grasp candidates
    file_dir = os.getenv("MGS_INPUT_DIR")
    if file_dir is None:
        file_dir = "."
    file_dir = "/home/ngh2rng/project/mj-grasp-sim/AllegroGripper/007_tuna_fish_can"
    file_path = "/home/ngh2rng/project/mj-grasp-sim/AllegroGripper/007_tuna_fish_can/candidates_collision_free.npz"

    grasps = np.load(file_path)
    poses = grasps["pose"]
    poses = SE3Pose.from_mat(poses, type="wxyz")
    joints = grasps["joints"]
    mask = env.grasp_stability_evaluation_from_joints(poses, joints)
    print(sum(mask))

    poses_stable = poses[mask].to_mat()
    joints_stable = joints[mask]


    file_ouput_path = os.path.join(file_dir, "stable_grasps.npz")
    np.savez(
        file_ouput_path,
        **{
            "pose": poses_stable,
            "joints": joints_stable,
        },
    )


if __name__ == "__main__":
    main()
