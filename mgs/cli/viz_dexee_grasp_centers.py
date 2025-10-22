import os
from typing import Tuple

import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from flax import nnx

from mgs.sampler.helper import farthest_point_sampling
from mgs.sampler.kin.base import KinematicsModel
from mgs.sampler.kin.dexee import DexeeKinematicsModel
from mgs.sampler.kin.op import forward_kinematic_point_transform
from mgs.sampler.kin.seg_op import kinematic_transform, point_transform

# Ordered segmentation keys expected in the Dexee gripper npz file
SEGMENTATION_KEYS_ORDERED = [
    "f0j0",
    "f0j1",
    "f0j2",
    "f0j3",
    "f1j0",
    "f1j1",
    "f1j2",
    "f1j3",
    "f2j0",
    "f2j1",
    "f2j2",
    "f2j3",
]


def compute_fingertip_positions(
    poses: jnp.ndarray, joints: jnp.ndarray, kinematics
) -> jnp.ndarray:
    """Return fingertip world positions (using distal joint origins) for each grasp.

    Args:
        poses: (B,4,4) grasp base poses.
        joints: (B,J) joint angles.
        kinematics: Kinematics model instance.
    Returns:
        Fingertip world positions (B,K,3)
    """
    g, s = nnx.split(kinematics)
    num_fingertips = kinematics.fingertip_idx.value.shape[0]
    local_points = jnp.zeros((num_fingertips, 3), dtype=jnp.float32)
    joint_indices = kinematics.fingertip_idx.value
    transformed_local = nnx.vmap(
        nnx.vmap(forward_kinematic_point_transform, in_axes=(None, 0, 0, None, None)),
        in_axes=(0, None, None, None, None),
    )(
        joints, local_points, joint_indices, g, s
    )  # (B,K,3)
    world = (
        jnp.einsum("bij,bkj->bki", poses[:, :3, :3], transformed_local)
        + poses[:, :3, 3][:, None, :]
    )
    return world


def compute_grasp_centers(
    poses: jnp.ndarray, joints: jnp.ndarray, kinematics
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute centers, in-bound mask, and fingertip positions.

    Returns:
        centers: (B,3)
        in_bound: (B,) boolean mask for centers lying inside fixed XY square.
        fingertip_world: (B,K,3)
    """
    fingertip_world = compute_fingertip_positions(poses, joints, kinematics)
    centers = jnp.mean(fingertip_world, axis=1)
    in_bound = (
        (centers[:, 0] < 0.20)
        & (centers[:, 0] > -0.20)
        & (centers[:, 1] < 0.20)
        & (centers[:, 1] > -0.20)
    )
    return centers, in_bound, fingertip_world


def gather_scene_grasps(scene_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load all grasp pose/joint arrays in a scene directory (excluding scene & collision files)."""
    assert os.path.isdir(scene_dir), f"Scene dir not found: {scene_dir}"
    pose_list = []
    joint_list = []
    for fname in os.listdir(scene_dir):
        if fname.startswith("scene"):
            continue
        if fname.endswith("collision.npz"):
            continue
        if not fname.endswith(".npz"):
            continue
        path = os.path.join(scene_dir, fname)
        data = np.load(path)
        if "pose" not in data or "joints" not in data:
            continue
        pose_list.append(data["pose"])
        joint_list.append(data["joints"])
    if len(pose_list) == 0:
        raise RuntimeError("No grasp files found in scene directory.")
    poses = np.concatenate(pose_list, axis=0)
    joints = np.concatenate(joint_list, axis=0)
    return poses, joints


def maybe_load_scene_pcd(scene_dir: str):
    """Optionally load scene point cloud (points, colors) if present."""
    pcd_path = os.path.join(scene_dir, "scene_pcd.npz")
    if not os.path.exists(pcd_path):
        return None, None
    data = np.load(pcd_path)
    points = data.get("points")
    colors = data.get("colors")
    return points, colors


def load_dexee_gripper_pcd(path: str):
    """Load Dexee gripper point cloud and segmentation masks from npz file."""
    if not os.path.exists(path):
        print(f"Warning: dexee gripper npz not found: {path}")
        return None, None
    raw = np.load(path, allow_pickle=True)
    pcd = raw.get("pcd_point")
    if pcd is None:
        print("Warning: missing 'pcd_point' in gripper npz")
        return None, None
    segs = []
    for key in SEGMENTATION_KEYS_ORDERED:
        arr = raw.get(key)
        if arr is None:
            print(f"Warning: missing segmentation key {key}")
            return pcd, None
        segs.append(arr.astype(bool))
    segmentation = np.stack(segs, axis=0)  # (D,N)
    return pcd, segmentation


def transform_gripper_cloud(
    pcd, segmentation, pose: np.ndarray, joints: np.ndarray, kin: DexeeKinematicsModel
):
    """Transform gripper point cloud (with segmentation) to world frame for a single grasp."""
    if pcd is None or segmentation is None:
        return None
    g, s = nnx.split(kin)
    pcd_j = jnp.asarray(pcd, dtype=jnp.float32)
    seg_j = jnp.asarray(segmentation, dtype=jnp.bool_)
    joints_j = jnp.asarray(joints, dtype=jnp.float32)
    transformed = kinematic_transform(point_transform, pcd_j, joints_j, seg_j, g, s)
    world = jnp.einsum("ij,nj->ni", pose[:3, :3], transformed) + pose[:3, 3]
    return np.asarray(world)


def _fps_indices(points: np.ndarray | None, max_points: int | None):
    if max_points is None or points is None:
        return None
    n = points.shape[0]
    if max_points <= 0 or n <= max_points:
        return None  # no downsampling
    pts_j = jnp.asarray(points, dtype=jnp.float32)
    idx = farthest_point_sampling(pts_j, int(max_points))
    return np.asarray(idx)


def visualize(
    scene_dir: str,
    gripper_pcd_path: str,
    max_scene_points: int | None,
    max_gripper_points: int | None,
    max_centers: int | None,
):
    """Main visualization entry: loads grasps + optional scene PCD; displays centers & two example gripper clouds."""
    poses_np, joints_np = gather_scene_grasps(scene_dir)
    print(f"Loaded grasps: poses {poses_np.shape}, joints {joints_np.shape}")

    kin = DexeeKinematicsModel()

    centers, in_bound, fingertip_world = compute_grasp_centers(
        jnp.asarray(poses_np, dtype=jnp.float32),
        jnp.asarray(joints_np, dtype=jnp.float32),
        kin,
    )
    centers_np = np.asarray(centers)
    in_bound_np = np.asarray(in_bound)
    fingertip_np = np.asarray(fingertip_world)
    print(
        f"In-bound grasps: {in_bound_np.sum()} / {len(in_bound_np)} ({in_bound_np.mean()*100:.2f}%)"
    )

    # Downsample centers for visualization only (keep originals for selection)
    centers_vis = centers_np
    colors_vis = np.where(in_bound_np, "green", "red")
    if max_centers is not None and centers_np.shape[0] > max_centers:
        idx_centers = _fps_indices(centers_np, max_centers)
        if idx_centers is not None:
            centers_vis = centers_np[idx_centers]
            colors_vis = colors_vis[idx_centers]
            print(
                f"Downsampled grasp centers: {centers_np.shape[0]} -> {centers_vis.shape[0]}"
            )

    # Select one green (in-bound) and one red (out-of-bound) example from full set
    green_indices = np.where(in_bound_np)[0]
    red_indices = np.where(~in_bound_np)[0]
    green_idx = int(green_indices[0]) if len(green_indices) > 0 else 0
    red_idx = int(red_indices[0]) if len(red_indices) > 0 else green_idx

    gripper_pcd, gripper_seg = load_dexee_gripper_pcd(gripper_pcd_path)

    # Downsample gripper point cloud + segmentation before transforming
    if gripper_pcd is not None and gripper_seg is not None:
        idx_gripper = _fps_indices(gripper_pcd, max_gripper_points)
        if idx_gripper is not None:
            gripper_pcd = gripper_pcd[idx_gripper]
            gripper_seg = gripper_seg[:, idx_gripper]
            print(
                f"Downsampled gripper cloud: segmentation dims {gripper_seg.shape}, points {gripper_pcd.shape}"
            )

    gripper_green_world = transform_gripper_cloud(
        gripper_pcd, gripper_seg, poses_np[green_idx], joints_np[green_idx], kin
    )
    gripper_red_world = (
        transform_gripper_cloud(
            gripper_pcd, gripper_seg, poses_np[red_idx], joints_np[red_idx], kin
        )
        if red_idx != green_idx
        else None
    )

    scene_points, scene_colors = maybe_load_scene_pcd(scene_dir)
    if scene_points is not None:
        idx_scene = _fps_indices(scene_points, max_scene_points)
        if idx_scene is not None:
            scene_points = scene_points[idx_scene]
            if scene_colors is not None and scene_colors.shape[0] == idx_scene.shape[0]:
                pass
            elif scene_colors is not None:
                scene_colors = scene_colors[idx_scene]
            print(
                f"Downsampled scene points: original centers -> {scene_points.shape[0]} points"
            )

    fig = go.Figure()

    # Scene point cloud
    if scene_points is not None:
        fig.add_trace(
            go.Scatter3d(
                x=scene_points[:, 0],
                y=scene_points[:, 1],
                z=scene_points[:, 2],
                mode="markers",
                marker=dict(size=2, color=scene_colors, opacity=0.8),
                name="Scene",
            )
        )

    # Grasp centers (possibly downsampled)
    fig.add_trace(
        go.Scatter3d(
            x=centers_vis[:, 0],
            y=centers_vis[:, 1],
            z=centers_vis[:, 2],
            mode="markers",
            marker=dict(size=3, color=colors_vis, opacity=0.9),
            name="Grasp Centers",
        )
    )

    # Fingertip spheres for selected grasps
    fig.add_trace(
        go.Scatter3d(
            x=fingertip_np[green_idx, :, 0],
            y=fingertip_np[green_idx, :, 1],
            z=fingertip_np[green_idx, :, 2],
            mode="markers",
            marker=dict(size=6, color="lime", opacity=1.0),
            name="Fingertips (in-bound)",
        )
    )
    if red_idx != green_idx:
        fig.add_trace(
            go.Scatter3d(
                x=fingertip_np[red_idx, :, 0],
                y=fingertip_np[red_idx, :, 1],
                z=fingertip_np[red_idx, :, 2],
                mode="markers",
                marker=dict(size=6, color="orange", opacity=1.0),
                name="Fingertips (out-of-bound)",
            )
        )

    # Gripper point clouds for selected examples
    if gripper_green_world is not None:
        fig.add_trace(
            go.Scatter3d(
                x=gripper_green_world[:, 0],
                y=gripper_green_world[:, 1],
                z=gripper_green_world[:, 2],
                mode="markers",
                marker=dict(size=2, color="cyan", opacity=0.8),
                name="Gripper (in-bound)",
            )
        )
    if gripper_red_world is not None:
        fig.add_trace(
            go.Scatter3d(
                x=gripper_red_world[:, 0],
                y=gripper_red_world[:, 1],
                z=gripper_red_world[:, 2],
                mode="markers",
                marker=dict(size=2, color="magenta", opacity=0.8),
                name="Gripper (out-of-bound)",
            )
        )

    # Bounding square at minimum Z of centers
    z_plane = float(np.min(centers_np[:, 2])) if centers_np.size > 0 else 0.0
    square = np.array(
        [
            [-0.20, -0.20, z_plane],
            [0.20, -0.20, z_plane],
            [0.20, 0.20, z_plane],
            [-0.20, 0.20, z_plane],
            [-0.20, -0.20, z_plane],
        ]
    )
    fig.add_trace(
        go.Scatter3d(
            x=square[:, 0],
            y=square[:, 1],
            z=square[:, 2],
            mode="lines",
            line=dict(color="blue", width=4),
            name="In-Bound Region",
        )
    )

    fig.update_layout(
        title=f"Dexee Grasp Centers + Samples\n{scene_dir}",
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(itemsizing="constant"),
    )
    fig.show()


def main():
    # Example usage: update paths as needed before running.
    visualize(
        "/home/frr2rng/projects/kinematics-flow/data/train/DexeeGripper/00922ca107f017a40e12f4636657c2cf",
        "/home/frr2rng/projects/kinematics-flow/data/gripper_dexee.npz",
        2000,
        1000,
        500,
    )


if __name__ == "__main__":
    main()
