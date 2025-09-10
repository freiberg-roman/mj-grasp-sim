import os

import hydra
import numpy as np
from omegaconf import DictConfig

os.environ.setdefault("MUJOCO_GL", "glfw")
from mgs.env.gravityless_object_grasping import GravitylessObjectGrasping
from mgs.gripper.panda import GripperPanda
from mgs.obj.selector import get_object
from mgs.sampler.antipodal import AntipodalGraspGenerator
from mgs.util.const import ASSET_PATH
from mgs.util.geo.transforms import SE3Pose


@hydra.main(
    version_base="1.3.2", config_path="config", config_name="gen_gripper_object_grasps"
)
def main(cfg: DictConfig):
    object_id_file = os.path.join(ASSET_PATH, "mj-objects", "fast_eta_objects.txt")
    with open(object_id_file, "r") as file:
        all_object_ids = file.read().splitlines()

    object_id = all_object_ids[int(cfg.object_id)]
    obj = get_object(object_id)
    sampler = (
        AntipodalGraspGenerator(obj)
        if cfg.gripper.grasp_sampler == "Antipodal"
        else None
    )
    assert sampler is not None
    gripper = GripperPanda(
        SE3Pose.from_vec(
            np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), type="wxyz", layout="pq"
        )
    )

    print(
        f"Generating grasp candidates using gripper: {cfg.gripper.name}"
        f"\nfor object {object_id} using {cfg.gripper.grasp_sampler} sampler"
    )

    output_dir = os.getenv("MGS_OUTPUT_DIR")
    output_dir = os.path.join(output_dir, cfg.gripper.name, object_id)
    os.makedirs(output_dir, exist_ok=True)

    # load in all .npz files already present in output dir
    all_already_present_files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".npz")
    ]
    # load in pose

    num_total_grasps = 0
    for file in all_already_present_files:
        num_total_grasps += len(np.load(file)["poses"])

    grasps_to_generate = cfg.target_grasps - num_total_grasps
    if grasps_to_generate <= 0:
        return
    print(f"Grasps to generate: {grasps_to_generate}")
    print(f"Generating {grasps_to_generate} new grasps.")

    env = GravitylessObjectGrasping(gripper, obj)

    stable_poses, stable_joints = [], []

    collision_free_poses, collision_free_joints = [], []
    while sum([len(p) for p in collision_free_poses]) < cfg.collect_grasps_till_eval:
        poses, aux_info = sampler.generate_grasps(num=cfg.sample_grasps)
        if len(poses) == 0:
            continue
        padding = 0.01
        joints = gripper.width_to_joints(aux_info["width"] + padding)

        # filter out collisions
        poses = SE3Pose.from_mat(poses)
        collision_mask = env.grasp_collision_mask(poses, joints, with_padding=0.002)
        collision_free_poses.append(poses.to_mat()[collision_mask])
        collision_free_joints.append(joints[collision_mask])
        break

    collision_free_poses = SE3Pose.from_mat(
        np.concatenate(collision_free_poses, axis=0)
    )
    collision_free_joints = np.concatenate(collision_free_joints, axis=0)
    # eval
    stable_mask = env.grasp_stability_evaluation_from_joints(
        collision_free_poses, collision_free_joints
    )

    # # Save the generated grasps
    # output_filename = os.path.join(
    #     output_dir, f"{len(all_already_present_files):04d}.npz"
    # )
    # np.savez(
    #     output_filename,
    #     poses=generated_grasps,
    #     qualities=generated_qualities,
    # )
    # print(f"Saved {len(generated_grasps)} grasps to {output_filename}")
    #     return
    #


if __name__ == "__main__":
    main()
