from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple
import shutil

import hydra
import numpy as np
from omegaconf import DictConfig

SCENE_FILE = "scene.npz"
RENDER_FILE = "scene_pcd.npz"


def _count_valid_grasps_in_scene(scene_dir: Path) -> int:
    """Count valid grasp rows in a single rendered scene directory.

    Valid grasp files: ``<obj_id>_<obj_name>.npz`` containing a ``pose`` array.
    Collision grasp files (ending with ``_collision.npz``) are ignored for
    selection ranking (only valid grasp count considered).
    """
    count = 0
    for f in scene_dir.iterdir():
        if f.name in {SCENE_FILE, RENDER_FILE}:
            continue
        if not f.name.endswith(".npz"):
            continue
        if f.name.endswith("_collision.npz"):
            continue
        try:
            data = np.load(f, allow_pickle=True)
            if "pose" in data:
                pose_arr = data["pose"]
                if pose_arr.ndim >= 2:  # rows are grasps (N, ...)
                    count += pose_arr.shape[0]
        except Exception:
            # Ignore unreadable/malformed files
            pass
    return count


def _gather_rendered_scenes(source_root: Path) -> List[Tuple[int, str, Path]]:
    """Return list of (valid_grasp_count, gripper_name, scene_path) for rendered scenes.

    A rendered scene is any directory under ``<source_root>/<gripper>/<scene_hash>``
    containing both ``scene.npz`` and ``scene_pcd.npz``.
    """
    results: List[Tuple[int, str, Path]] = []
    if not source_root.exists():
        return results
    for gripper_dir in source_root.iterdir():
        if not gripper_dir.is_dir():
            continue
        gripper_name = gripper_dir.name
        for scene_dir in gripper_dir.iterdir():
            if not scene_dir.is_dir():
                continue
            scene_path = scene_dir / SCENE_FILE
            render_path = scene_dir / RENDER_FILE
            if not (scene_path.exists() and render_path.exists()):
                continue  # skip non-rendered or incomplete
            valid_count = _count_valid_grasps_in_scene(scene_dir)
            results.append((valid_count, gripper_name, scene_dir))
    return results


@hydra.main(config_path="config", config_name="select_top_scenes")
def main(cfg: DictConfig) -> None:  # pragma: no cover - CLI side effects
    """Select top-N rendered scenes for a SINGLE specified gripper.

    The gripper is specified via the Hydra gripper config (``gripper: <name>``)
    matching existing CLI patterns. Only scenes under that gripper directory are
    considered. Renders (``scene.npz`` + ``scene_pcd.npz``) are ranked by valid
    grasp count (sum of rows in non-collision grasp files) and the top
    ``num_scenes`` are copied or moved into the output root.

    Required env vars:
    - ``MGS_OUTPUT_DIR``: target root to receive selected scenes.

    Config keys (``select_top_scenes.yaml``):
    - ``gripper``: composed config providing ``name`` (e.g. AllegroGripper) and id.
    - ``num_scenes`` (int): number of scenes to select for this gripper.
    - ``source_root`` (str): path whose child is the gripper directory. If empty,
      defaults to ``MGS_INPUT_DIR``.
    - ``move`` (bool): if True, move directories; else copy (source remains intact).

    Source hierarchy::

        <source_root>/<gripper_id>/<scene_hash>/
            scene.npz
            scene_pcd.npz
            <obj grasp files>.npz

    Destination hierarchy::

        <MGS_OUTPUT_DIR>/<gripper_id>/<scene_hash>/

    ``gripper.id`` is used as the directory name to stay consistent with other
    configs (e.g. ``allegro``). Source directories are not modified when
    ``move = False`` (default).
    """
    output_root = os.getenv("MGS_OUTPUT_DIR")
    assert output_root is not None, "MGS_OUTPUT_DIR not set"

    source_root_cfg = cfg.get("source_root", "")
    if source_root_cfg:
        source_root = Path(source_root_cfg).expanduser().resolve()
    else:
        input_root_env = os.getenv("MGS_INPUT_DIR")
        assert input_root_env is not None, "source_root empty and MGS_INPUT_DIR not set"
        source_root = Path(input_root_env).resolve()

    num_scenes = int(cfg.num_scenes)
    assert num_scenes > 0, "num_scenes must be > 0"
    move_flag = bool(getattr(cfg, "move", False))

    # Resolve gripper directory using cfg.gripper.name (disk layout uses .name)
    grip_cfg = cfg.gripper
    grip_dir_name = getattr(grip_cfg, "name")
    assert grip_dir_name, "gripper config must provide name"

    gripper_root = source_root / grip_dir_name
    if not gripper_root.exists():
        print(f"No directory for gripper '{grip_dir_name}' at {gripper_root}")
        return

    # Collect rendered scenes for this gripper only
    scenes: List[Tuple[int, Path]] = []  # (valid_count, scene_path)
    for scene_dir in gripper_root.iterdir():
        if not scene_dir.is_dir():
            continue
        if not (scene_dir / SCENE_FILE).exists() or not (scene_dir / RENDER_FILE).exists():
            continue
        count = _count_valid_grasps_in_scene(scene_dir)
        scenes.append((count, scene_dir))

    if not scenes:
        print(f"No rendered scenes found for gripper '{grip_dir_name}' under {gripper_root}")
        return

    scenes.sort(key=lambda x: x[0], reverse=True)
    chosen = scenes[:num_scenes]

    print(f"Source root: {source_root}")
    print(f"Output root: {output_root}")
    print(f"Gripper: {grip_cfg.name} (dir='{grip_dir_name}')")
    print(f"Rendered scenes available: {len(scenes)}")
    print(f"Selecting {len(chosen)} (top by valid grasp count)")

    for rank, (count, scene_path) in enumerate(chosen, start=1):
        rel_scene = scene_path.name
        dest_dir = Path(output_root) / grip_dir_name / rel_scene
        if dest_dir.exists():
            print(f"[skip] Destination exists: {dest_dir}")
            continue
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        if move_flag:
            shutil.move(str(scene_path), str(dest_dir))
            action = "moved"
        else:
            shutil.copytree(scene_path, dest_dir)
            action = "copied"
        print(f"[{rank}] {action} {grip_dir_name}/{rel_scene} (valid_grasps={count})")

    print("Done.")


if __name__ == "__main__":  # pragma: no cover
    main()
