import os

import hydra
import numpy as np
from omegaconf import DictConfig

from mgs.env.gravityless_object_grasping import GravitylessObjectGrasping
from mgs.gripper.selector import get_gripper
from mgs.obj.selector import get_object
from mgs.util.geo.transforms import SE3Pose  # Needed for SE3Pose operations
from mgs.util.const import ASSET_PATH


@hydra.main(config_path="config", config_name="filter_to_stable")
def main(cfg: DictConfig):
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
    file_dir = os.path.abspath(os.path.join(file_dir, cfg.gripper.name, object_id))
    file_path = os.path.join(file_dir, "candidates.npz")

    grasps = np.load(file_path)
    poses = grasps["pose"]
    poses = SE3Pose.from_mat(poses, type="wxyz")
    joints = grasps["joints"]

    mask = env.grasp_collision_mask(poses, joints)
    poses_collision_free = poses[mask]
    joints_collision_free = joints[mask]
    print(sum(mask))

    mask_of_mask = env.grasp_stability_evaluation_from_joints(
        poses_collision_free, joints_collision_free
    )
    poses_stable = poses_collision_free[mask_of_mask]
    joints_stable = joints_collision_free[mask_of_mask]
    print(sum(mask_of_mask))

    file_ouput_path = os.path.join(file_dir, "candidates_collision_free.npz")
    np.savez(
        file_ouput_path,
        **{
            "pose": poses_collision_free.to_mat(),
            "joints": joints_collision_free,
        },
    )
    file_ouput_path = os.path.join(file_dir, "stable_grasps.npz")
    np.savez(
        file_ouput_path,
        **{
            "pose": poses_stable.to_mat(),
            "joints": joints_stable,
        },
    )


if __name__ == "__main__":
    main()
