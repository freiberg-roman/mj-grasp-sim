import os
from copy import deepcopy

import hydra
import numpy as np
from omegaconf import DictConfig

from mgs.env.selector import get_env_from_dict
from mgs.util.geo.transforms import SE3Pose

# In-bound XY limits (replicated from cleaning logic)
_XY_MIN = -0.20
_XY_MAX = 0.20


def _collision_mask(scene_def, poses, joints):
    """Return boolean mask: True where grasp is collision-free."""

    class cfg:
        name: str = "ClutterTable"

    env = get_env_from_dict(cfg(), deepcopy(scene_def))
    mask = env.grasp_collision_mask(
        SE3Pose.from_mat(deepcopy(poses), type="wxyz"),
        deepcopy(joints),
    )
    return np.asarray(mask, dtype=bool)


def _stability_mask(scene_def, poses, joints):
    """Return boolean mask: True where grasp is stable (implies collision-free)."""

    class cfg:
        name: str = "ClutterTable"

    env = get_env_from_dict(cfg(), deepcopy(scene_def))
    mask = env.grasp_stable_mask(
        SE3Pose.from_mat(deepcopy(poses), type="wxyz"),
        joints,
        deepcopy(scene_def["env_state"]["state"]),
        show_progress=True,
    )
    return np.asarray(mask, dtype=bool)


def filter_grasps(cfg: DictConfig, scene_def):
    env = get_env_from_dict(cfg.env, (deepcopy(scene_def)))

    all_grasps = []
    for obj_name, obj_id in zip(env.object_names, env.object_ids):
        poses, joints = get_grasps(
            gripper_name=cfg.gripper.name,
            obj_id=obj_id,
        )
        if poses is None or joints is None:
            continue  # skip objects with missing grasp data
        o2w = env.get_obj_pose(obj_name)
        se3_pose = SE3Pose.from_mat(deepcopy(poses))
        grasp_pose = o2w @ se3_pose
        all_grasps.append(
            (
                grasp_pose.to_mat(),
                joints,
                obj_name,
                obj_id,
            )
        )

    all_poses = []
    all_joints = []
    obj_indices = []
    obj_map = []

    for idx, (
        collision_free_poses,
        collision_free_joints,
        obj_name,
        obj_id,
    ) in enumerate(all_grasps):
        pose_count = len(collision_free_poses)
        if pose_count > 0:
            all_poses.append(collision_free_poses)
            all_joints.append(collision_free_joints)
            obj_indices.append(np.full(pose_count, idx, dtype=np.int32))
            obj_map.append((obj_name, obj_id))

    if len(all_poses) == 0:
        raise ValueError("No grasps loaded")
    all_poses = np.concatenate(all_poses, axis=0)
    all_joints = np.concatenate(all_joints, axis=0)
    obj_indices = np.concatenate(obj_indices, axis=0)

    # In-bound filtering (before collision check) for select grippers
    in_bound_mask = _compute_in_bound_mask(all_poses, all_joints, cfg.gripper.name)
    if in_bound_mask.sum() == 0:
        raise ValueError("No in-bound grasps")
    all_poses = all_poses[in_bound_mask]
    all_joints = all_joints[in_bound_mask]
    obj_indices = obj_indices[in_bound_mask]

    collision_free_mask = env.grasp_collision_mask(
        SE3Pose.from_mat(deepcopy(all_poses), type="wxyz"),
        deepcopy(all_joints),
        with_padding=0.002,
    )

    if sum(collision_free_mask) <= 0:
        raise ValueError(
            f"Not enough collision free grasps! Only: {sum(collision_free_mask)}"
        )

    collision_free_poses = all_poses[collision_free_mask]
    collision_free_joints = all_joints[collision_free_mask]
    collision_free_obj_indices = obj_indices[collision_free_mask]

    collision_poses = all_poses[~collision_free_mask]
    collision_joints = all_joints[~collision_free_mask]
    collision_obj_indices = obj_indices[~collision_free_mask]

    if not cfg.only_collision_free:
        order = fps_rank_grasps(
            collision_free_poses,
            k=None,  # keep ordering for all; set to an int to subsample if desired
            rot_weight=getattr(cfg, "fps_rot_weight", 0.1),
            seed=getattr(cfg, "fps_seed", None),
        )
        collision_free_poses = collision_free_poses[order]
        collision_free_joints = collision_free_joints[order]
        collision_free_obj_indices = collision_free_obj_indices[order]

        if sum(collision_free_mask) < cfg.min_stable:
            raise ValueError(
                f"Not enough collision free grasps! Only: {sum(collision_free_mask)}"
            )

        stable_grasp_mask = env.grasp_stable_mask(
            SE3Pose.from_mat(deepcopy(collision_free_poses), type="wxyz"),
            deepcopy(collision_free_joints),
            deepcopy(scene_def["env_state"]["state"]),
            enough_stable=cfg.enough_stable,
        )
        if sum(stable_grasp_mask) < cfg.min_stable:
            raise ValueError(
                f"Not enough stable grasps! Only: {sum(stable_grasp_mask)}"
            )

        result_poses = collision_free_poses[stable_grasp_mask]
        result_joints = collision_free_joints[stable_grasp_mask]
        result_obj_indices = collision_free_obj_indices[stable_grasp_mask]
    else:
        result_poses = collision_free_poses
        result_joints = collision_free_joints
        result_obj_indices = collision_free_obj_indices

    result = []
    neg_result = []
    for obj_idx in np.unique(result_obj_indices):
        mask = result_obj_indices == obj_idx
        if sum(mask) == 0:
            continue
        obj_name, obj_id = obj_map[obj_idx]
        result.append(
            {
                "object_id": obj_id,
                "object_name": obj_name,
                "pose": result_poses[mask],
                "joints": result_joints[mask],
            }
        )
        if cfg.save_collision_grasps:
            collision_mask = collision_obj_indices == obj_idx
            if sum(collision_mask) > 0:
                neg_result.append(
                    {
                        "object_id": obj_id,
                        "object_name": obj_name,
                        "pose": collision_poses[collision_mask],
                        "joints": collision_joints[collision_mask],
                    }
                )
    return result, neg_result


@hydra.main(config_path="config", config_name="label")
def main(cfg: DictConfig):
    input_dir = os.getenv("MGS_INPUT_DIR")
    assert input_dir is not None, "No ouput_dir defined!"

    all_scenes = os.listdir(os.path.join(input_dir, cfg.gripper.name))
    all_scenes.sort()  # deterministic order
    scene_dir = os.path.join(input_dir, cfg.gripper.name, all_scenes[cfg.scene_id])
    print(f"Scene directory: {scene_dir}")

    scene_path = os.path.join(scene_dir, "scene.npz")
    scene_def = np.load(scene_path, allow_pickle=True)["scene_definition"].item()

    grasp_path = os.path.join(scene_dir, cfg.grasp_file)
    label_path = os.path.join(scene_dir, "label.npz")
    grasps = np.load(grasp_path)

    poses = grasps["pose"]
    joints = grasps["joints"]
    cf_mask = _collision_mask(scene_def, poses, joints)
    cf_rate = np.sum(cf_mask) / cf_mask.shape[0]
    print(f"(Collision free rate: {cf_rate:.2%})")
    stable_mask = _stability_mask(scene_def, poses, joints)
    stable_rate = np.sum(stable_mask) / stable_mask.shape[0]
    print(f"(Stable rate: {stable_rate:.2%})")

    label = cf_mask & stable_mask

    np.savez_compressed(
        label_path,
        valid=label,
        collision=cf_mask,
        stable=stable_mask,
    )


if __name__ == "__main__":
    main()
