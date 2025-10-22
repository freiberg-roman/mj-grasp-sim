import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import jax.numpy as jnp
import numpy as np
from flax import nnx
from tqdm import tqdm

from mgs.sampler.kin.allegro import AllegroKinematicsModel
from mgs.sampler.kin.base import KinematicsModel
from mgs.sampler.kin.dexee import DexeeKinematicsModel
from mgs.sampler.kin.op import forward_kinematic_point_transform
from mgs.sampler.kin.shadow import ShadowKinematicsModel

# Bounding square limits (XY) replicated from viz scripts
_XY_MIN = -0.20
_XY_MAX = 0.20


@dataclass
class FileStats:
    file_path: str
    before: int
    after: int
    removed: int


@dataclass
class SceneStats:
    scene_dir: str
    files: List[FileStats]
    buffer_excluded: bool = False
    low_grasp_deleted: bool = False

    @property
    def removed(self) -> int:
        return sum(f.removed for f in self.files)

    @property
    def before(self) -> int:
        return sum(f.before for f in self.files)

    @property
    def after(self) -> int:
        return sum(f.after for f in self.files)


def _compute_in_bound_mask(
    poses: np.ndarray, joints: np.ndarray, kin: KinematicsModel
) -> np.ndarray:
    """Return boolean mask of in-bound grasps.

    Mirrors logic from viz_*_grasp_centers scripts.
    Args:
        poses: (B,4,4)
        joints: (B,J)
        kin: kinematics model with `.fingertip_idx` attribute.
    Returns:
        mask: (B,) boolean array.
    """
    if poses.size == 0:
        return np.zeros((0,), dtype=bool)

    N, DOF = joints.shape[0], joints.shape[1]
    padding = jnp.zeros(shape=(1500 - N, DOF))  # max number of grasps
    padded_joints = jnp.concatenate([joints, padding], axis=0)
    g, s = nnx.split(kin)
    fingertip_idx = kin.fingertip_idx.value  # (K,)
    num_fingertips = fingertip_idx.shape[0]
    local_points = jnp.zeros((num_fingertips, 3), dtype=jnp.float32)

    transformed_local = nnx.vmap(  # over grasps
        nnx.vmap(  # over fingertip indices
            forward_kinematic_point_transform,
            in_axes=(None, 0, 0, None, None),
        ),
        in_axes=(0, None, None, None, None),
    )(jnp.asarray(padded_joints, dtype=jnp.float32), local_points, fingertip_idx, g, s)[
        :N
    ]

    world_pts = (
        jnp.einsum("bij,bkj->bki", jnp.asarray(poses)[:, :3, :3], transformed_local)
        + jnp.asarray(poses)[:, :3, 3][:, None, :]
    )  # (B,K,3)
    centers = jnp.mean(world_pts, axis=1)  # (B,3)
    in_bound = (
        (centers[:, 0] < _XY_MAX)
        & (centers[:, 0] > _XY_MIN)
        & (centers[:, 1] < _XY_MAX)
        & (centers[:, 1] > _XY_MIN)
    )
    return np.asarray(in_bound, dtype=bool)


def _iter_scene_dirs(gripper_dir: str) -> Iterable[str]:
    for entry in sorted(os.listdir(gripper_dir)):
        scene_path = os.path.join(gripper_dir, entry)
        if not os.path.isdir(scene_path):
            continue
        if not os.path.exists(os.path.join(scene_path, "scene.npz")):
            continue
        yield scene_path


def _atomic_save_npz(out_path: str, arrays: Dict[str, Any]) -> None:
    """Atomic write helper for npz files.

    Writes to a temporary file in the same directory with a guaranteed `.npz` suffix,
    then replaces the destination. Avoids numpy auto-appending issues and is safe
    across concurrent readers.
    """
    directory = os.path.dirname(out_path)
    base = os.path.basename(out_path)
    fd, tmp_path = tempfile.mkstemp(prefix=base + ".", suffix=".npz", dir=directory)
    try:
        os.close(fd)
        np.savez(tmp_path, **arrays)
        os.replace(tmp_path, out_path)
    except Exception:
        # Best effort cleanup if something fails before replace.
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise


def _filter_object_file(path: str, kin: KinematicsModel) -> FileStats:
    data = np.load(path, allow_pickle=True)
    if "pose" not in data or "joints" not in data:
        return FileStats(path, 0, 0, 0)
    poses = data["pose"]
    joints = data["joints"]
    before = poses.shape[0]
    if before == 0:
        return FileStats(path, 0, 0, 0)
    mask = _compute_in_bound_mask(poses, joints, kin)
    after = int(mask.sum())
    removed = before - after

    if after == 0:
        # Remove file entirely if no valid grasps remain.
        os.remove(path)
        return FileStats(path, before, 0, removed)

    if removed > 0:
        filtered_poses = poses[mask]
        filtered_joints = joints[mask]
        # Preserve other keys in original file (e.g., scores, metadata) if present.
        out_dict: Dict[str, Any] = {k: data[k] for k in data.files}
        out_dict["pose"] = filtered_poses
        out_dict["joints"] = filtered_joints
        _atomic_save_npz(path, out_dict)
    return FileStats(path, before, after, removed)


def _get_kinematics(gripper_name: str) -> KinematicsModel:
    if gripper_name == "AllegroGripper":
        return AllegroKinematicsModel()
    if gripper_name == "DexeeGripper":
        return DexeeKinematicsModel()
    if gripper_name == "ShadowHand":
        return ShadowKinematicsModel()
    raise ValueError(f"Unsupported gripper: {gripper_name}")


def clean_dataset(
    gripper_arg: str, dry_run: bool = False, min_scene_grasps: int = 50
) -> List[SceneStats]:
    root = os.getenv("MGS_INPUT_DIR")
    if root is None:
        raise EnvironmentError("MGS_INPUT_DIR not set")
    gripper_dir = os.path.join(root, gripper_arg)
    if not os.path.isdir(gripper_dir):
        raise FileNotFoundError(f"Gripper directory not found: {gripper_dir}")

    kin = _get_kinematics(gripper_arg)

    scene_stats: List[SceneStats] = []
    scenes = list(_iter_scene_dirs(gripper_dir))

    pbar = tqdm(scenes, desc=f"Cleaning {gripper_arg}", unit="scene")
    total_removed = 0
    total_before = 0
    total_after = 0
    buffer_excluded_count = 0
    low_grasp_deleted_count = 0

    for scene_dir in pbar:
        # Scene-level exclusion: buffer region (0.20, 0.225] on either |x| or |y| for any object center.
        exclude_scene = False
        file_stats: List[FileStats] = []
        # Only compute object file stats if not buffer excluded (real run) or if dry_run (for reporting)
        if (not exclude_scene) or dry_run:
            for fname in os.listdir(scene_dir):
                if not fname.endswith(".npz"):
                    continue
                if fname.startswith("scene"):
                    continue
                if fname.endswith("collision.npz"):
                    continue
                fpath = os.path.join(scene_dir, fname)
                if dry_run:
                    data = np.load(fpath, allow_pickle=True)
                    if "pose" not in data or "joints" not in data:
                        continue
                    poses = data["pose"]
                    joints = data["joints"]
                    before = poses.shape[0]
                    if before == 0:
                        file_stats.append(FileStats(fpath, 0, 0, 0))
                        continue
                    mask = _compute_in_bound_mask(poses, joints, kin)
                    after = int(mask.sum())
                    removed = before - after
                    file_stats.append(FileStats(fpath, before, after, removed))
                else:
                    file_stats.append(_filter_object_file(fpath, kin))

        stats = SceneStats(
            scene_dir=scene_dir, files=file_stats, buffer_excluded=exclude_scene
        )

        # Low-grasp deletion logic (only if not buffer excluded)
        if not exclude_scene:
            if stats.after < min_scene_grasps:
                if dry_run:
                    stats.low_grasp_deleted = True
                else:
                    # Delete all .npz files (including scene definition)
                    for fname_del in os.listdir(scene_dir):
                        if fname_del.endswith(".npz"):
                            try:
                                os.remove(os.path.join(scene_dir, fname_del))
                            except OSError:
                                pass
                    # Mark all grasps removed
                    for f in stats.files:
                        f.removed = f.before
                        f.after = 0
                    stats.low_grasp_deleted = True

        scene_stats.append(stats)

        # Aggregate global counters
        total_removed += stats.removed
        total_before += stats.before
        total_after += stats.after
        if stats.buffer_excluded:
            buffer_excluded_count += 1
        if stats.low_grasp_deleted:
            low_grasp_deleted_count += 1

        pbar.set_postfix(
            removed=total_removed,
            kept=total_after,
            before=total_before,
            pct_removed=(100.0 * total_removed / total_before) if total_before else 0.0,
            buffer_excluded=stats.buffer_excluded,
            low_grasp_deleted=stats.low_grasp_deleted,
        )
    pbar.close()
    return scene_stats


def _summarize(scene_stats: List[SceneStats]) -> None:
    total_removed = sum(s.removed for s in scene_stats)
    total_before = sum(s.before for s in scene_stats)
    # Effective after counts exclude deleted scenes (their after already zeroed)
    total_after = sum(s.after for s in scene_stats)
    pct = (100.0 * total_removed / total_before) if total_before else 0.0
    buffer_excluded = sum(1 for s in scene_stats if s.buffer_excluded)
    low_grasp_deleted = sum(1 for s in scene_stats if s.low_grasp_deleted)
    kept_scenes = len(scene_stats) - buffer_excluded - low_grasp_deleted
    print("\nSummary:")
    print(f"  Scenes total             : {len(scene_stats)}")
    print(f"  Scenes buffer-excluded   : {buffer_excluded}")
    print(f"  Scenes low-grasp deleted : {low_grasp_deleted}")
    print(f"  Scenes kept              : {kept_scenes}")
    print(f"  Total grasps before      : {total_before}")
    print(f"  Total grasps after       : {total_after}")
    print(f"  Removed grasps           : {total_removed} ({pct:.2f}%)")


import hydra
from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="clean_grasp_centers")
def main(cfg: DictConfig):  # type: ignore
    stats = clean_dataset(
        cfg.gripper.name, dry_run=cfg.dry_run, min_scene_grasps=cfg.min_scene_grasps
    )
    _summarize(stats)


if __name__ == "__main__":  # pragma: no cover
    main()
