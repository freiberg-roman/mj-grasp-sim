import os
import json
from copy import deepcopy

import hydra
import numpy as np
from omegaconf import DictConfig

from mgs.env.selector import get_env_from_dict
from mgs.util.geo.transforms import SE3Pose


def eval_grasps(cfg: DictConfig, scene_def, grasps):
    env = get_env_from_dict(cfg.env, (deepcopy(scene_def)))
    b2c = env.gripper.base_to_contact_transform().inverse().to_mat()

    pose, joints = grasps
    pose = np.einsum("nij,jk->nik", pose, b2c)

    collision_free_mask = env.grasp_collision_mask(
        SE3Pose.from_mat(deepcopy(pose), type="wxyz"),
        deepcopy(joints),
    )

    pose_collision_free = pose[collision_free_mask]
    joints_collision_free = joints[collision_free_mask]

    stable_grasp_mask = env.grasp_stable_mask(
        SE3Pose.from_mat(deepcopy(pose_collision_free), type="wxyz"),
        deepcopy(joints_collision_free),
        deepcopy(scene_def["env_state"]["state"]),
    )

    num_total = float(len(pose))
    return (
        sum(stable_grasp_mask) / num_total,
        {"num_objects": len(env.object_names)},
    )


@hydra.main(config_path="config", config_name="eval_grasps")
def main(cfg: DictConfig):
    output_dir = os.getenv("MGS_OUTPUT_DIR")
    input_dir = os.getenv("MGS_INPUT_DIR")
    assert output_dir is not None, "No ouput_dir defined!"
    assert input_dir is not None, "No input_dir defined!"

    all_scene_dir = os.path.join(
        output_dir,
        cfg.gripper.name,
    )
    all_scenes = os.listdir(all_scene_dir)
    scene = all_scenes.sort()[cfg.id]
    scene_dir = os.path.join(all_scene_dir, scene)

    scene_path = os.path.join(scene_dir, "scene.npz")
    scene_dict = np.load(scene_path)["scene_definition"].item()

    grasp_path = os.path.join(scene_dir, "inference_grasps.npz")
    grasps = np.load(grasp_path)
    pose, joints = grasps["pose"], grasps["joints"]
    success_rate, aux = eval_grasps(cfg, scene_dict, (pose, joints))

    # store as json
    results = {
        "success_rate": float(success_rate),
        "num_objects": aux["num_objects"],
        "scene_id": scene,
    }

    results_path = os.path.join(output_dir, "grasp_evaluation.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation complete: {success_rate:.2%} success rate")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
