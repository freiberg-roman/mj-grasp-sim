import os

import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go

from mgs.sampler.helper import farthest_point_sampling

# Panda gripper visualization:
# - No kinematics model used.
# - Each grasp pose (4x4) is provided in scene grasp npz files.
# - We approximate the contact "center" between the fingers by shifting the pose
#   origin along local -Z by 0.102m (do NOT apply any additional origin hacks).
# - We also visualize the gripper point cloud (if provided) for one in-bound and one out-of-bound grasp.

LOCAL_Z_OFFSET = (
    +0.102
)  # meters, applied in local frame to approximate contact point between fingers


def gather_scene_grasps(scene_dir: str) -> np.ndarray:
    """Load all grasp pose arrays in a scene directory (excluding scene & collision files).
    Returns:
        poses: (B,4,4) array of grasp base poses.
    """
    assert os.path.isdir(scene_dir), f"Scene dir not found: {scene_dir}"
    pose_list = []
    for fname in os.listdir(scene_dir):
        if fname.startswith("scene"):
            continue
        if fname.endswith("collision.npz"):
            continue
        if not fname.endswith(".npz"):
            continue
        path = os.path.join(scene_dir, fname)
        data = np.load(path)
        if "pose" not in data:
            continue
        pose_list.append(data["pose"])  # (K,4,4) or (1,4,4)
    if len(pose_list) == 0:
        raise RuntimeError("No grasp files with 'pose' found in scene directory.")
    poses = np.concatenate(pose_list, axis=0)
    return poses


def maybe_load_scene_pcd(scene_dir: str):
    """Optionally load scene point cloud (points, colors) if present."""
    pcd_path = os.path.join(scene_dir, "scene_pcd.npz")
    if not os.path.exists(pcd_path):
        return None, None
    data = np.load(pcd_path)
    points = data.get("points")
    colors = data.get("colors")
    return points, colors


def transform_gripper_cloud(pcd: np.ndarray, pose: np.ndarray):
    """Transform local gripper point cloud to world frame using grasp pose."""
    if pcd is None:
        return None
    rot = pose[:3, :3]
    pos = pose[:3, 3]
    world = (rot @ pcd.T).T + pos
    return world


def _fps_indices(points: np.ndarray | None, max_points: int | None):
    if max_points is None or points is None:
        return None
    n = points.shape[0]
    if max_points <= 0 or n <= max_points:
        return None  # no downsampling
    pts_j = jnp.asarray(points, dtype=jnp.float32)
    idx = farthest_point_sampling(pts_j, int(max_points))
    return np.asarray(idx)


def compute_contact_centers(poses: np.ndarray) -> np.ndarray:
    """Apply fixed local -Z offset to each pose origin to approximate grasp contact center.

    Args:
        poses: (B,4,4) grasp base poses.
    Returns:
        centers: (B,3) array of shifted contact centers in world coordinates.
    """
    rot = poses[:, :3, :3]  # (B,3,3)
    pos = poses[:, :3, 3]  # (B,3)
    # Local -Z offset: multiply +Z axis by negative value to move along -Z
    centers = pos
    return centers


def visualize(
    scene_dir: str,
    gripper_pcd_path: str,
    max_scene_points: int | None,
    max_gripper_points: int | None,
    max_centers: int | None,
):
    """Visualize Panda grasp centers with fixed local -Z offset and gripper PCD examples."""
    poses_np = gather_scene_grasps(scene_dir)
    print(f"Loaded grasp poses: {poses_np.shape}")

    centers_np = compute_contact_centers(poses_np)
    print(f"Computed contact centers shape: {centers_np.shape}")

    # In-bound check using same XY box as other visualizers
    in_bound_np = (
        (centers_np[:, 0] < 0.20)
        & (centers_np[:, 0] > -0.20)
        & (centers_np[:, 1] < 0.20)
        & (centers_np[:, 1] > -0.20)
    )
    print(
        f"In-bound centers: {in_bound_np.sum()} / {len(in_bound_np)} ({in_bound_np.mean()*100:.2f}%)"
    )

    # Downsample centers for visualization only
    centers_vis = centers_np
    colors_vis = np.where(in_bound_np, "green", "red")
    if max_centers is not None and centers_np.shape[0] > max_centers:
        idx_centers = _fps_indices(centers_np, max_centers)
        if idx_centers is not None:
            centers_vis = centers_np[idx_centers]
            colors_vis = colors_vis[idx_centers]
            print(
                f"Downsampled centers: {centers_np.shape[0]} -> {centers_vis.shape[0]}"
            )

    # Select indices for examples (one in-bound, one out-of-bound if available)
    green_indices = np.where(in_bound_np)[0]
    red_indices = np.where(~in_bound_np)[0]
    green_idx = int(green_indices[0]) if len(green_indices) > 0 else 0
    red_idx = int(red_indices[0]) if len(red_indices) > 0 else green_idx

    # Scene point cloud (optional)
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
                f"Downsampled scene points: original -> {scene_points.shape[0]} points"
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

    # Grasp centers
    fig.add_trace(
        go.Scatter3d(
            x=centers_vis[:, 0],
            y=centers_vis[:, 1],
            z=centers_vis[:, 2],
            mode="markers",
            marker=dict(size=3, color=colors_vis, opacity=0.9),
            name="Grasp Contact Centers",
        )
    )

    # Example contact points (just centers for selected grasps)
    example_points = np.array(
        [
            centers_np[green_idx],
            centers_np[red_idx] if red_idx != green_idx else centers_np[green_idx],
        ]
    )
    fig.add_trace(
        go.Scatter3d(
            x=example_points[:, 0],
            y=example_points[:, 1],
            z=example_points[:, 2],
            mode="markers",
            marker=dict(size=6, color=["cyan", "magenta"], opacity=1.0),
            name="Example Centers",
        )
    )

    # Bounding square at min Z of centers
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
        title=f"Panda Grasp Contact Centers + Gripper Examples (Fixed Local -Z Offset)\n{scene_dir}",
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(itemsizing="constant"),
    )
    fig.show()


def main():
    # Example usage paths (update before running)
    visualize(
        "/home/frr2rng/projects/kinematics-flow/data/train/PandaGripper/0018b3ef0b828bb4e684ab3d6ab4fdf1/",  # scene directory containing grasp npz files
        "/home/frr2rng/projects/kinematics-flow/data/gripper_panda.npz",  # gripper npz path
        2000,  # max scene points
        1500,  # max gripper points
        500,  # max centers
    )


if __name__ == "__main__":
    main()
