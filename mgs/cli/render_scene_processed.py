import os
import jax.numpy as jnp
from copy import deepcopy

import hydra
import numpy as np
from omegaconf import DictConfig

from mgs.env.selector import get_env_from_dict
from mgs.gripper.base import MjScannableGripper
from mgs.gripper.selector import get_gripper
from mgs.util.img_proc import rgbd_to_pcd, voxel_downsample_pcd
from mgs.sampler.kin.jax_util import farthest_point_sampling


def scan(cfg: DictConfig, scene_def):
    gripper = get_gripper(cfg.gripper)
    assert isinstance(gripper, MjScannableGripper)
    env = get_env_from_dict(cfg.env, (deepcopy(scene_def)))
    images, extrinsics, image_masks, _ = env.scan(num_images=cfg.num_images)
    intrinsics = env.get_camera_intrinsics()
    return images, extrinsics, intrinsics, image_masks


@hydra.main(config_path="config", config_name="render_scene_proc")
def main(cfg: DictConfig):
    output_dir = os.getenv("MGS_OUTPUT_DIR")
    input_dir = os.getenv("MGS_INPUT_DIR")

    assert output_dir is not None
    assert input_dir is not None
    input_dir = os.path.join(input_dir, cfg.gripper.name)
    scene_dir = [
        d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))
    ][cfg.id]
    input_dir = os.path.join(input_dir, scene_dir)
    print("Scene dir: ", input_dir)

    scene_path = os.path.join(input_dir, "scene.npz")
    scene = np.load(scene_path, allow_pickle=True)
    scene_dict = scene["scene_definition"].item()
    images, extrinsics, intrinsics, image_masks = scan(
        deepcopy(cfg), deepcopy(scene_dict)
    )
    pcd, feature = rgbd_to_pcd(images, intrinsics, extrinsics)
    pcd = pcd[image_masks]
    feature = feature[image_masks]

    region_mask = np.all(
        (pcd < np.array([[0.25, 0.25, 1.0]]))
        & (pcd > np.array([[-0.25, -0.25, -0.01]])),
        axis=-1,
    )
    pcd = pcd[region_mask]
    feature = feature[region_mask]

    pcd, feature = voxel_downsample_pcd(pcd, feature, voxel_size=0.002)
    idx = farthest_point_sampling(
        jnp.asarray(pcd, dtype=jnp.float32), num_samples=15000
    )
    pcd = pcd[idx]
    feature = feature[idx]

    output_dir = os.path.join(output_dir, cfg.gripper.name, scene_dir)
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        os.path.join(output_dir, "scene_pcd"),
        **{
            "points": np.asarray(pcd, dtype=np.float32),
            "colors": np.asarray(feature, dtype=np.float32),
        },
    )
    print("Finished!")


if __name__ == "__main__":
    main()
