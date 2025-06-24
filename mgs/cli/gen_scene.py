import os
from copy import deepcopy

import hydra
import numpy as np
from omegaconf import DictConfig

from mgs.env.selector import get_env, get_env_from_dict
from mgs.gripper.selector import get_gripper
from mgs.obj.selector import get_objects
from mgs.util.geo.transforms import SE3Pose
from mgs.util.file import generate_unique_hash


def get_grasps(gripper_name, obj_id, gripper_type=""):
    grasp_path = os.path.join(  # type: ignore
        os.getenv("MGS_INPUT_DIR"),  # type: ignore
        gripper_name,
        obj_id,
        "stable_grasps.npz",
    )
    grasp_dict = np.load(grasp_path)
    poses = grasp_dict["pose"]
    joints = grasp_dict["joints"]
    return poses, joints


def gen_stable_scene(cfg: DictConfig):
    obj_list = get_objects(cfg.object)
    gripper = get_gripper(
        cfg.gripper,
        default_pose=SE3Pose(
            np.array([5.0, 5.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0]), type="wxyz"
        ),
    )

    env = get_env(cfg.env, gripper=deepcopy(
        gripper), obj_list=deepcopy(obj_list))
    env.gen_clutter()
    scene_dict = env.to_dict()

    if not env.is_stable():
        raise ValueError("Scene unstable")

    return scene_dict


def filter_grasps(cfg: DictConfig, scene_def):
    env = get_env_from_dict(cfg.env, (deepcopy(scene_def)))

    all_grasps = []
    for obj_name, obj_id in zip(env.object_names, env.object_ids):
        poses, joints = get_grasps(
            gripper_name=cfg.gripper.name,
            obj_id=obj_id,
        )

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
        raise ValueError("No collision free grasps")
    all_poses = np.concatenate(all_poses, axis=0)
    all_joints = np.concatenate(all_joints, axis=0)
    obj_indices = np.concatenate(obj_indices, axis=0)

    collision_free_mask = env.grasp_collision_mask(
        SE3Pose.from_mat(deepcopy(all_poses), type="wxyz"),
        deepcopy(all_joints),
    )
    enough_collision_free = 128
    if sum(collision_free_mask) < enough_collision_free:
        raise ValueError("Not enough collision free grasps!")
    collision_free_poses = all_poses[collision_free_mask]
    collision_free_joints = all_joints[collision_free_mask]
    collision_free_obj_indices = obj_indices[collision_free_mask]

    collision_poses = all_poses[~collision_free_mask]
    collision_joints = all_joints[~collision_free_mask]
    collision_obj_indices = obj_indices[~collision_free_mask]

    if not cfg.only_collision_free:
        shuffle_indices = np.random.permutation(len(collision_free_poses))
        collision_free_poses = collision_free_poses[shuffle_indices]
        collision_free_joints = collision_free_joints[shuffle_indices]
        collision_free_obj_indices == collision_free_obj_indices[shuffle_indices]

        enough_stable = min(128, cfg.num_objects * 32)
        stable_grasp_mask = env.grasp_stable_mask(
            SE3Pose.from_mat(deepcopy(collision_free_poses), type="wxyz"),
            deepcopy(collision_free_joints),
            deepcopy(scene_def["env_state"]["state"]),
            enough_stable=enough_stable,
        )
        if sum(stable_grasp_mask) < enough_stable:
            raise ValueError("Not enough stable grasps!")

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


@hydra.main(config_path="config", config_name="gen_scene")
def main(cfg: DictConfig):
    output_dir = os.getenv("MGS_OUTPUT_DIR")
    input_dir = os.getenv("MGS_INPUT_DIR")
    assert output_dir is not None, "No ouput_dir defined!"
    assert input_dir is not None, "No input_dir defined!"

    output_dir = os.path.join(
        output_dir,
        cfg.gripper.name,
        generate_unique_hash(16),
    )

    try:
        scene_dict = gen_stable_scene(cfg)
        valid_grasps, invalid_grasps = filter_grasps(cfg, scene_dict)
        scene_path = os.path.join(output_dir, "scene")
        os.makedirs(output_dir, exist_ok=True)
        np.savez(
            scene_path,
            **{
                "scene_definition": scene_dict,
            },
        )
        for grasps in valid_grasps:
            obj_id, obj_name = grasps["object_id"], grasps["object_name"]
            object_path = os.path.join(output_dir, obj_id + "_" + obj_name)
            np.savez(
                object_path,
                **{
                    "pose": grasps["pose"],
                    "joints": grasps["joints"],
                },
            )
        for grasps in invalid_grasps:
            obj_id, obj_name = grasps["object_id"], grasps["object_name"]
            object_path = os.path.join(output_dir, obj_id + "_" + obj_name + "_" + "collision")
            np.savez(
                object_path,
                **{
                    "pose": grasps["pose"],
                    "joints": grasps["joints"],
                },
            )

    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
