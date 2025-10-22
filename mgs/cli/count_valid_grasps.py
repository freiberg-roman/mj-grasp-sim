import os
from pathlib import Path
from typing import List, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig

from mgs.gripper.selector import get_gripper

SCENE_FILE = "scene.npz"
RENDER_FILE = "scene_pcd.npz"


def _count_scenes_and_grasps(
    gripper_name: str, input_root: Path
) -> Tuple[int, int, int, int, List[int]]:
    """Count scenes (generated + rendered) and grasps, plus per-rendered-scene valid grasp counts.

    Scene directories are produced by `gen_scene.py` under
    `<MGS_INPUT_DIR>/<gripper_name>/<scene_hash>/`.

    Each directory may contain multiple grasp `.npz` files named
    `<obj_id>_<obj_name>` (valid) and optionally `<obj_id>_<obj_name>_collision`
    (invalid) plus the scene file `scene.npz`. After rendering, a
    `scene_pcd.npz` file appears.

    Returns
    -------
    total_scenes : int
        Directories containing a `scene.npz` file.
    rendered_scenes : int
        Subset of total having a `scene_pcd.npz` file.
    total_valid_grasps : int
        Sum over all valid grasp records (rows in `pose` arrays of valid files).
    total_invalid_grasps : int
        Sum over all invalid grasp records (rows in `pose` arrays of *_collision files).
    rendered_scene_valid_counts : List[int]
        For each rendered scene, the number of valid grasp records in that scene.
    """
    gripper_dir = input_root / gripper_name
    if not gripper_dir.exists():
        return 0, 0, 0, 0, []

    total_scenes = 0
    rendered_scenes = 0
    total_valid = 0
    total_invalid = 0
    rendered_scene_valid_counts: List[int] = []

    for scene_dir in gripper_dir.iterdir():
        if not scene_dir.is_dir():
            continue
        scene_path = scene_dir / SCENE_FILE
        if not scene_path.exists():
            continue
        total_scenes += 1
        scene_valid = 0
        scene_invalid = 0

        # Iterate grasp files (exclude scene + rendered pcd)
        for f in scene_dir.iterdir():
            if f.name in {SCENE_FILE, RENDER_FILE}:
                continue
            if not f.name.endswith(".npz"):
                continue
            try:
                data = np.load(f, allow_pickle=True)
                if "pose" in data:
                    pose_arr = data["pose"]
                    if f.name.endswith("_collision.npz"):
                        rows = pose_arr.shape[0]
                        scene_invalid += rows
                        total_invalid += rows
                    else:
                        rows = pose_arr.shape[0]
                        scene_valid += rows
                        total_valid += rows
            except Exception:
                # Ignore unreadable/malformed grasp files
                pass

        # Only record per-scene counts if rendered scene exists (per requirement)
        if (scene_dir / RENDER_FILE).exists():
            rendered_scenes += 1
            rendered_scene_valid_counts.append(scene_valid)

    return (
        total_scenes,
        rendered_scenes,
        total_valid,
        total_invalid,
        rendered_scene_valid_counts,
    )


def _print_threshold_table(valid_counts: List[int]) -> None:
    """Print table of number of rendered scenes exceeding grasp count thresholds.

    Thresholds: 50, 100, 200, 300, ..., 1500 (inclusive).
    """
    if not valid_counts:
        print("No rendered scenes with valid grasps to summarize.")
        return

    thresholds = [50, 100] + list(range(200, 1600, 100))  # 200..1500 step 100
    header = f"{'Threshold':>10} | {'Scenes > Threshold':>18}"
    print("\nRendered Scene Valid Grasp Count Summary")
    print(header)
    print("-" * len(header))
    for t in thresholds:
        count = sum(1 for c in valid_counts if c >= t)
        print(f"{t:10d} | {count:18d}")


@hydra.main(config_path="config", config_name="count_valid_grasps")
def main(cfg: DictConfig) -> None:  # pragma: no cover - CLI side effects
    input_dir = os.getenv("MGS_INPUT_DIR")
    assert input_dir is not None, "MGS_INPUT_DIR not set"

    # Validate gripper selection (ensures cfg.gripper has 'name')
    gripper = get_gripper(cfg.gripper)  # noqa: F841 - unused, just validation

    (
        total_scenes,
        rendered_scenes,
        total_valid,
        total_invalid,
        rendered_scene_valid_counts,
    ) = _count_scenes_and_grasps(cfg.gripper.name, Path(input_dir))

    print(f"Input dir {input_dir}")
    print(f"Gripper: {cfg.gripper.name}")
    print(f"Generated scenes: {total_scenes}")
    print(f"Rendered scenes: {rendered_scenes}")
    if total_scenes > 0:
        pct = 100.0 * rendered_scenes / total_scenes
        print(f"Percent rendered: {pct:.2f}%")
    print(f"Valid grasps (total rows): {total_valid}")
    print(f"Invalid grasps (collision rows): {total_invalid}")

    _print_threshold_table(rendered_scene_valid_counts)


if __name__ == "__main__":
    main()
