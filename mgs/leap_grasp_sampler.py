# Research Code - Visualize Initial Pose, Targets, and Initial Fingertips

from mgs.operations import rotation_6d_to_matrix
from flax import nnx
import json
import os
import sys
import numpy as np
import trimesh
import jax
import jax.numpy as jnp
from jax.lax import fori_loop
import plotly.graph_objects as go
from scipy.spatial import KDTree
from itertools import permutations
from mgs.leap_kin import (
    LeapHandKinematicsModel,
    KinematicsModel,
    kinematic_pcd_transform,
)
from mgs.util.const import ASSET_PATH

FINGERTIP_INFO = {
    "fingertip": 3,  # Index finger tip joint index
    "fingertip_2": 7,  # Middle finger tip joint index
    "fingertip_3": 11,  # Ring finger tip joint index
    "thumb_fingertip": 15,  # Thumb finger tip joint index
}


@jax.jit
def compute_l2_distance_matrix(
    points_a: jnp.ndarray, points_b: jnp.ndarray
) -> jnp.ndarray:
    """Computes the pairwise L2 distance matrix between two sets of points.
    Args:
        points_a: (N, 3) JAX array.
        points_b: (M, 3) JAX array.
    Returns:
        dist_matrix: (N, M) JAX array where dist_matrix[i, j] is ||points_a[i] - points_b[j]||.
    """
    diff = points_a[:, None, :] - points_b[None, :, :]  # Shape (N, M, 3)
    dist_sq = jnp.sum(diff**2, axis=-1)  # Shape (N, M)
    return jnp.sqrt(dist_sq)


def find_best_assignment_and_reorder_targets(
    # Shape (4, 3) - Current fingertip positions
    initial_fingertips_world: jnp.ndarray,
    # Shape (4, 3) - Potential target positions
    offset_target_points: jnp.ndarray,
):
    """
    Finds the optimal assignment of targets to fingertips to minimize initial L2 distance
    and reorders the target points accordingly.

    Args:
        initial_fingertips_world: Current world positions of the 4 fingertips.
        offset_target_points: The 4 potential target world positions.

    Returns:
        A tuple containing:
        - assigned_target_points: The offset_target_points reordered according to the best assignment.
                                  assigned_target_points[i] is the target for fingertip i. (Shape: 4, 3)
        - best_permutation_indices: The indices showing how targets were reordered. (Shape: 4,)
        - min_initial_loss: The sum of L2 distances for the best assignment.
    """
    num_points = initial_fingertips_world.shape[0]
    dist_matrix = compute_l2_distance_matrix(
        initial_fingertips_world, offset_target_points
    )
    target_indices = np.arange(num_points)
    # Use itertools.permutations and convert to NumPy array for indexing
    all_perms_np = np.array(
        list(permutations(target_indices)), dtype=np.int32
    )  # Shape (24, 4)

    # 3. Calculate total distance for each permutation using the distance matrix
    # Convert JAX dist_matrix to NumPy for indexing compatibility with NumPy permutation array
    dist_matrix_np = np.array(dist_matrix)
    finger_row_indices = np.arange(num_points)  # [0, 1, 2, 3]
    # Use advanced indexing: dist_matrix_np[ [0,1,2,3], perm ] for each perm
    # Shape (24, 4)
    perm_distances = dist_matrix_np[finger_row_indices, all_perms_np]
    perm_loss = np.sum(perm_distances, axis=1)  # Shape (24,)

    # 4. Find the best permutation (minimum total distance)
    best_perm_index = np.argmin(perm_loss)
    best_permutation_indices = all_perms_np[
        best_perm_index
    ]  # Shape (4,) - indices of targets
    min_initial_loss = perm_loss[best_perm_index]

    # 5. Reorder the offset target points according to the best permutation
    # Use JAX array for potential further JAX operations
    assigned_target_points = offset_target_points[best_permutation_indices]

    return assigned_target_points, best_permutation_indices, float(min_initial_loss)


def farthest_point_sampling(x, num_samples):
    # ... (implementation as before) ...
    x = jax.lax.stop_gradient(x)
    farthest_points_idx = jnp.zeros(num_samples, dtype=jnp.int32)
    farthest_points_idx = farthest_points_idx.at[0].set(0)
    distances = jnp.full(x.shape[0], jnp.inf)

    def sampling_fn(i, val):
        farthest_points_idx, distances = val
        latest_point_idx = farthest_points_idx[i - 1]
        latest_point = x[latest_point_idx]
        new_dr = x - latest_point
        new_distances = jnp.sum(new_dr**2, axis=-1)
        distances = jnp.minimum(distances, new_distances)
        farthest_point_idx = jnp.argmax(distances)
        farthest_points_idx = farthest_points_idx.at[i].set(farthest_point_idx)
        return farthest_points_idx, distances

    farthest_points_idx, _ = fori_loop(
        1, num_samples, sampling_fn, (farthest_points_idx, distances)
    )
    return farthest_points_idx


# --- Configuration ---
NUM_SURFACE_SAMPLES = 10000
NUM_FPS_SEEDS = 256
LOCAL_REGION_RADIUS = 0.10  # 10 cm
NUM_TARGET_POINTS = 4
TARGET_OFFSET_DISTANCE = 0.02  # 3 cm offset along normal
POSE_OFFSET_DISTANCE = 0.08  # 8 cm offset for base pose along normal
FRAME_VIS_LENGTH = 0.03
CURRENT = os.path.dirname(os.path.abspath(__file__))
GRIPPER_NPZ_FILE = os.path.join(CURRENT, "gripper_leap.npz")
CONTACT_JSON_FILE = os.path.join(CURRENT, "contact_candidates.json")
NUM_POINTS_VIS = 2000


# --- Helper Functions ---
def normalize_vector(v, axis=-1, epsilon=1e-8):
    """Normalizes a vector or batch of vectors."""
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / (norm + epsilon)


def create_frame_lines(origin, x_axis, y_axis, z_axis, length):
    """Generates line segments for Plotly representing a coordinate frame."""
    # ... (implementation as before) ...
    lines_x, lines_y, lines_z = [], [], []
    p = origin
    ax, ay, az = x_axis * length, y_axis * length, z_axis * length
    lines_x.extend(
        [p[0], p[0] + ax[0], None, p[0], p[0] +
            ay[0], None, p[0], p[0] + az[0], None]
    )
    lines_y.extend(
        [p[1], p[1] + ax[1], None, p[1], p[1] +
            ay[1], None, p[1], p[1] + az[1], None]
    )
    lines_z.extend(
        [p[2], p[2] + az[2], None, p[2], p[2] +
            ay[2], None, p[2], p[2] + az[2], None]
    )
    return lines_x, lines_y, lines_z


# --- LEAP Hand Constants ---
LEAP_QPOS_OPEN = np.array(
    [
        1.57 / 2.0,
        0.0,
        0.0,
        0.0,
        1.57 / 2.0,
        0.0,
        0.0,
        0.0,
        1.57 / 2.0,
        0.0,
        0.0,
        0.0,
        1.57 / 2.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=jnp.float32,
)


# --- Main Logic ---
def visualize_grasp_initialization(obj_mesh_path: str):
    """Loads mesh, samples, calculates initial pose & targets, visualizes."""

    # 1. Load Mesh, Sample Points+Normals, FPS (NumPy)
    print(f"Loading mesh from: {obj_mesh_path}")
    mesh = trimesh.load_mesh(obj_mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("Not a Trimesh.")
    print("Sampling points and normals...")
    points_np, face_indices = trimesh.sample.sample_surface(
        mesh, NUM_SURFACE_SAMPLES)
    normals_np = mesh.face_normals[face_indices]
    normals_np = normalize_vector(normals_np)  # Ensure unit normals
    print("Performing FPS...")
    points_jax = jnp.array(points_np, dtype=jnp.float32)
    if points_jax.shape[0] < NUM_FPS_SEEDS:
        fps_indices_np = np.arange(points_jax.shape[0])
    else:
        fps_indices_jax = farthest_point_sampling(points_jax, NUM_FPS_SEEDS)
        fps_indices_np = np.array(fps_indices_jax)
    seed_points_np = points_np[fps_indices_np]
    seed_normals_np = normals_np[fps_indices_np]
    print(f"Selected {len(seed_points_np)} FPS seeds.")

    # 2. Select Seed, Find Local Region, Select Targets
    random_seed_index = np.random.randint(0, len(seed_points_np))
    selected_seed_point = seed_points_np[random_seed_index]
    selected_seed_normal = seed_normals_np[random_seed_index]
    print(f"Selected seed point index: {random_seed_index}")

    distances_sq = np.sum((seed_points_np - selected_seed_point) ** 2, axis=1)
    in_region_mask = distances_sq < (LOCAL_REGION_RADIUS**2)
    indices_in_region = np.where(in_region_mask)[0]
    num_in_region = len(indices_in_region)
    print(f"Found {num_in_region} points within radius.")

    if num_in_region < NUM_TARGET_POINTS:
        print(
            f"ERROR: Only found {num_in_region} points in local region, need {NUM_TARGET_POINTS}. Cannot proceed."
        )
        return

    # Select 4 random points from the region
    target_indices = np.random.choice(
        indices_in_region, NUM_TARGET_POINTS, replace=False
    )
    local_target_points = seed_points_np[target_indices]
    local_target_normals = seed_normals_np[target_indices]

    # 3. Offset Target Points
    offset_target_points = (
        local_target_points + local_target_normals * TARGET_OFFSET_DISTANCE
    )
    print("Calculated offset target points.")

    # 4. Calculate Initial Base Pose Frame
    # Use the originally selected seed point and normal for pose calculation
    seed_point_for_pose = selected_seed_point
    normal_for_pose = selected_seed_normal

    # Base Position: Offset from seed point along the POSITIVE surface normal
    initial_base_translation = (
        seed_point_for_pose + normal_for_pose * POSE_OFFSET_DISTANCE
    )

    # Base Rotation: -Z should align with surface normal (pointing towards surface)
    z_axis = normal_for_pose  # Gripper approach direction

    other_seed_indices = np.delete(
        np.arange(len(seed_points_np)), random_seed_index)
    if len(other_seed_indices) > 0:
        kdtree = KDTree(seed_points_np[other_seed_indices])
        _, nearest_neighbor_rel_idx = kdtree.query(seed_point_for_pose, k=1)
        nearest_neighbor_abs_idx = other_seed_indices[nearest_neighbor_rel_idx]
        next_point = seed_points_np[nearest_neighbor_abs_idx]
        x_axis_raw = next_point - seed_point_for_pose
    else:  # Fallback if only one seed point exists
        x_axis_raw = np.array([1.0, 0.0, 0.0])  # Default X

    x_axis = normalize_vector(x_axis_raw)

    # Orthogonalize X relative to Z
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    x_axis = normalize_vector(x_axis)

    # Calculate Y = Z x X
    y_axis = np.cross(z_axis, x_axis)
    # No need to normalize y_axis if z and x are orthogonal unit vectors

    initial_base_rotation_mat = np.stack([x_axis, y_axis, z_axis], axis=-1)
    print("Calculated initial base pose.")

    # 5. Calculate Initial Fingertip Positions for Visualization
    print("Calculating initial fingertip positions...")
    raw = np.load(GRIPPER_NPZ_FILE, allow_pickle=True)
    points_full = raw["pcd_point"]
    print(f"  Loaded {len(points_full)} points.")

    random_idx = np.random.choice(
        points_full.shape[0], size=NUM_POINTS_VIS, replace=False
    )
    points_vis = points_full[random_idx]
    print(f"  Sampled {len(points_vis)} points for visualization.")

    print(f"Loading contact candidates from: {CONTACT_JSON_FILE}")
    with open(CONTACT_JSON_FILE, "r") as f:
        contact_candidates_data = json.load(f)
    print("  Loaded contact candidates.")
    finger_tips = []
    for link_name, link_joint_index in FINGERTIP_INFO.items():
        local_points = contact_candidates_data.get(link_name, [])
        if local_points:
            local_points_jax = jnp.array(local_points, dtype=jnp.float32)
            # Select only the FIRST contact point for this link for visualization
            if local_points_jax.shape[0] > 0:
                first_contact_point_local = local_points_jax[0, :]
                finger_tips.append(first_contact_point_local)
    from mgs.leap_sampler_proty import Trainer, forward_kinematic_point_transform

    kin_model = LeapHandKinematicsModel()  # Initialize our JAX model
    trainer = Trainer(
        kin=kin_model,
        init_rotation=jnp.array(initial_base_rotation_mat),
        init_translation=jnp.array(initial_base_translation),
    )

    local_finger_tips = np.stack(finger_tips, axis=0)
    current_theta = trainer.theta
    transformed_points = nnx.vmap(
        forward_kinematic_point_transform, in_axes=(None, 0, 0, None)
    )(
        current_theta.theta.value,
        local_finger_tips,
        jnp.array([3, 7, 11, 15], dtype=jnp.int32),
        kin_model,
    )
    transformed_points = (
        jnp.einsum(
            "ij,...j->...i",
            rotation_6d_to_matrix(current_theta.rot.value),
            transformed_points,
        )
        + current_theta.translation.value
    )

    transformed_finger_tips = transformed_points
    offset_target_points = jnp.array(offset_target_points)

    target_points, _, _ = find_best_assignment_and_reorder_targets(
        transformed_finger_tips, offset_target_points
    )

    for i in range(150):
        loss = trainer.train_step(
            local_finger_tips, (target_points, normal_for_pose))
        print(loss)
    _, _, opt_state = nnx.merge(trainer.train_graph, trainer.train_state)
    rot = rotation_6d_to_matrix(opt_state.rot.value)
    trans = opt_state.translation.value
    theta = opt_state.theta.value

    segmentations = np.stack(
        [
            raw["if_mcp"][random_idx],
            raw["if_rot"][random_idx],
            raw["if_pip"][random_idx],
            raw["if_dip"][random_idx],
            raw["mf_mcp"][random_idx],
            raw["mf_rot"][random_idx],
            raw["mf_pip"][random_idx],
            raw["mf_dip"][random_idx],
            raw["rf_mcp"][random_idx],
            raw["rf_rot"][random_idx],
            raw["rf_pip"][random_idx],
            raw["rf_dip"][random_idx],
            raw["th_cmc"][random_idx],
            raw["th_axl"][random_idx],
            raw["th_mcp"][random_idx],
            raw["th_ipl"][random_idx],
            # raw["palm"][random_idx],
        ]
    )
    points_vis = kinematic_pcd_transform(
        points_vis,
        theta,
        segmentations,
        kin_model,
    )
    points_vis = jnp.einsum("ij,nj->ni", rot, points_vis) + trans

    # 6. Prepare Plotly Visualization
    print("Preparing visualization...")
    plot_traces = []

    # Object Seed points
    plot_traces.append(
        go.Scatter3d(
            x=seed_points_np[:, 0],
            y=seed_points_np[:, 1],
            z=seed_points_np[:, 2],
            mode="markers",
            marker=dict(size=3, color="grey", opacity=0.5),
            name="Object Seeds",
        )
    )
    # Selected Seed Point
    plot_traces.append(
        go.Scatter3d(
            x=[seed_point_for_pose[0]],
            y=[seed_point_for_pose[1]],
            z=[seed_point_for_pose[2]],
            mode="markers",
            marker=dict(size=5, color="black", symbol="x"),
            name="Pose Seed Point",
        )
    )
    # Offset Target Points (where fingertips should land)
    plot_traces.append(
        go.Scatter3d(
            x=offset_target_points[:, 0],
            y=offset_target_points[:, 1],
            z=offset_target_points[:, 2],
            mode="markers",
            marker=dict(size=5, color="red"),
            name="Offset Target Points",
        )
    )
    # Initial Hand Base Position and Frame
    plot_traces.append(
        go.Scatter3d(
            x=[initial_base_translation[0]],
            y=[initial_base_translation[1]],
            z=[initial_base_translation[2]],
            mode="markers",
            marker=dict(size=6, color="magenta"),
            name="Initial Base Pos",
        )
    )
    frame_x, frame_y, frame_z = create_frame_lines(
        initial_base_translation, x_axis, y_axis, z_axis, FRAME_VIS_LENGTH
    )
    plot_traces.append(
        go.Scatter3d(
            x=frame_x,
            y=frame_y,
            z=frame_z,
            mode="lines",
            line=dict(color="purple", width=4),
            name="Base Frame",
        )
    )
    # Initial Fingertip positions (where they actually are with the initial pose/qpos)
    plot_traces.append(
        go.Scatter3d(
            x=points_vis[:, 0],
            y=points_vis[:, 1],
            z=points_vis[:, 2],
            mode="markers",
            marker=dict(size=1, color="grey", symbol="diamond"),
            name="Initial Fingertips",
        )
    )

    # ------------------------------------------

    # 7. Show Plot
    fig = go.Figure(data=plot_traces)
    fig.update_layout(
        title=f"Grasp Initialization Check (Seed {random_seed_index})",
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    print("Showing plot...")
    fig.show()


# --- Example Usage ---
if __name__ == "__main__":
    # Example using a YCB object
    object_id_to_load = "011_banana"
    obj_mesh_dir = os.path.join(
        ASSET_PATH, "mj-objects", "YCB", object_id_to_load)
    collision_mesh_path = os.path.join(obj_mesh_dir, "collision.obj")
    visual_mesh_path = os.path.join(obj_mesh_dir, "textured.obj")

    if os.path.exists(collision_mesh_path):
        obj_mesh_file_path_to_use = collision_mesh_path
    elif os.path.exists(visual_mesh_path):
        obj_mesh_file_path_to_use = visual_mesh_path
        print(f"Warning: Using visual mesh {visual_mesh_path}")
    else:
        print(
            f"ERROR: Could not find mesh for {object_id_to_load} in {obj_mesh_dir}")
        sys.exit(1)

    visualize_grasp_initialization(obj_mesh_file_path_to_use)
