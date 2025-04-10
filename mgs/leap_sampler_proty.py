# Minimal single file prototype for visualizing LEAP hand point cloud
# and **transformed** fingertip contact candidates (at zero joint config).
# Research Code - Bare minimum for visualization

import os
import json
import numpy as np
import plotly.graph_objects as go
import jax
import jax.numpy as jnp

from mgs.leap_kin import LeapHandKinematicsModel


# --- Configuration ---
# Adjust these paths based on your actual project structure relative to where you run the script
CURRENT = os.path.dirname(os.path.abspath(__file__)) # Get directory of the script
# Assume data files are in the same directory as the script for this prototype
GRIPPER_NPZ_FILE = os.path.join(CURRENT, "gripper_leap.npz")
CONTACT_JSON_FILE = os.path.join(CURRENT, "contact_candidates.json")
NUM_POINTS_VIS = 2000

# Fingertip link names and their corresponding FINAL joint indices in the chain
# This mapping is crucial for push_to_link_frame
FINGERTIP_INFO = {
    "fingertip": 3,        # Index finger tip joint index
    "fingertip_2": 7,      # Middle finger tip joint index
    "fingertip_3": 11,     # Ring finger tip joint index
    "thumb_fingertip": 15, # Thumb finger tip joint index
}
# --- Main Visualization Logic ---

@jax.jit
def transform_single_contact_set(
    contact_points_local: jnp.ndarray, # Points for ONE link (N, 3)
    link_joint_index: int,             # The index of the joint defining this link's frame
    theta: jnp.ndarray,                # Full joint state (16,)
    gripper_kinematics: LeapHandKinematicsModel
) -> jnp.ndarray:
    """Transforms contact points from local link frame to world frame using FK."""
    link_transforms = {}
    identity_tf = jnp.array([1.0, 0, 0, 0, 0, 0, 0], dtype=jnp.float32)
    link_transforms[-1] = identity_tf # Base frame

    parent_map = {}
    for chain in gripper_kinematics.kinematics_graph:
         parent_map[chain[0]] = -1
         for i in range(len(chain) - 1):
              parent_map[chain[i+1]] = chain[i]

    target_transform_world = None
    all_link_transforms = {}
    all_link_transforms[-1] = identity_tf

    for index in range(gripper_kinematics.num_dofs):
        parent_index = parent_map[index]
        T_world_parent = all_link_transforms[parent_index]
        T_parent_jointStatic = gripper_kinematics.kinematics_transforms.value[index] # Access NNX variable
        joint_theta = theta[index]
        translation = gripper_kinematics.joint_transforms.value[index,:3] * joint_theta
        joint_axis = gripper_kinematics.joint_transforms.value[index,3:]
        rotation_quat = quaternion_from_axis_angle(joint_axis, joint_theta)
        T_jointStatic_jointDynamic = jnp.concatenate([rotation_quat, translation], axis=-1)
        T_world_joint = se3_raw_mupltiply(
            se3_raw_mupltiply(T_world_parent, T_parent_jointStatic),
            T_jointStatic_jointDynamic
        )
        all_link_transforms[index] = T_world_joint

    target_transform_world = all_link_transforms[link_joint_index]
    transformed_points = transform_points_jax(contact_points_local, target_transform_world) # T should be (7,)
    return transformed_points


def visualize_gripper_and_contacts():
    """Loads and visualizes the gripper point cloud and transformed fingertip contact points."""

    # 1. Load Gripper Point Cloud Data
    print(f"Loading gripper point cloud from: {GRIPPER_NPZ_FILE}")
    raw = np.load(GRIPPER_NPZ_FILE, allow_pickle=True)
    points_full = raw["pcd_point"]
    print(f"  Loaded {len(points_full)} points.")

    random_idx = np.random.choice(points_full.shape[0], size=NUM_POINTS_VIS, replace=False)
    points_vis = points_full[random_idx]
    print(f"  Sampled {len(points_vis)} points for visualization.")

    # 2. Load Contact Candidates (local coordinates)
    print(f"Loading contact candidates from: {CONTACT_JSON_FILE}")
    with open(CONTACT_JSON_FILE, "r") as f:
        contact_candidates_data = json.load(f)
    print("  Loaded contact candidates.")

    # 3. Initialize Kinematics and Target State
    kin_model = LeapHandKinematicsModel() # Initialize our JAX model
    # Target joint state: all zeros
    theta_zero = jnp.zeros(kin_model.num_dofs, dtype=jnp.float32)

    # 4. Transform Contact Points
    transformed_contacts_list = []
    print("Transforming contact points using FK (theta=0)...")
    for link_name, link_joint_index in FINGERTIP_INFO.items():
        local_points = contact_candidates_data.get(link_name, [])
        if local_points:
            local_points_jax = jnp.array(local_points, dtype=jnp.float32)
            # Select only the FIRST contact point for this link for visualization
            if local_points_jax.shape[0] > 0:
                first_contact_point_local = local_points_jax[0:1, :] # Keep batch dim (1, 3)

                # --- Get World Transform of Link ---
                # Reuse FK logic to get the transform T_world_link
                all_link_transforms = {}
                identity_tf = jnp.array([1.0, 0, 0, 0, 0, 0, 0], dtype=jnp.float32)
                all_link_transforms[-1] = identity_tf
                parent_map = {}
                for chain in kin_model.kinematics_graph:
                    parent_map[chain[0]] = -1
                    for i in range(len(chain) - 1):
                        parent_map[chain[i+1]] = chain[i]
                for index in range(kin_model.num_dofs):
                    parent_index = parent_map[index]
                    T_world_parent = all_link_transforms[parent_index]
                    T_parent_jointStatic = kin_model.kinematics_transforms.value[index]
                    joint_theta = theta_zero[index]
                    translation = kin_model.joint_transforms.value[index,:3] * joint_theta
                    joint_axis = kin_model.joint_transforms.value[index,3:]
                    rotation_quat = quaternion_from_axis_angle(joint_axis, joint_theta)
                    T_jointStatic_jointDynamic = jnp.concatenate([rotation_quat, translation], axis=-1)
                    T_world_joint = se3_raw_mupltiply(
                        se3_raw_mupltiply(T_world_parent, T_parent_jointStatic),
                        T_jointStatic_jointDynamic
                    )
                    all_link_transforms[index] = T_world_joint
                target_transform_world = all_link_transforms[link_joint_index]
                # ---------------------------------

                # Transform the single point
                transformed_point = transform_points_jax(first_contact_point_local, target_transform_world) # Input (1,3), (7,) -> Output (1,3)
                transformed_contacts_list.append(np.array(transformed_point[0])) # Append NumPy array (3,)

    contact_points_np = np.array(transformed_contacts_list) # Shape (NumFingersWithContacts, 3)
    print(f"  Transformed {len(contact_points_np)} contact points.")

    # 5. Prepare Plotly Data
    plot_traces = []

    # Gripper cloud trace
    gripper_trace = go.Scatter3d(
        x=points_vis[:, 0], y=points_vis[:, 1], z=points_vis[:, 2],
        mode="markers", marker=dict(size=2, color="blue", opacity=0.6),
        name="Gripper Cloud (theta=0)"
    )
    plot_traces.append(gripper_trace)

    # Transformed contact points trace
    if contact_points_np.shape[0] > 0:
        contact_trace = go.Scatter3d(
            x=contact_points_np[:, 0], y=contact_points_np[:, 1], z=contact_points_np[:, 2],
            mode="markers", marker=dict(size=5, color="red", opacity=1.0),
            name="Transformed Contact Points (theta=0)"
        )
        plot_traces.append(contact_trace)

    # 6. Create and Show Plot
    fig = go.Figure(data=plot_traces)
    fig.update_layout(
        title="LEAP Hand (theta=0) + Transformed Contact Points",
        scene=dict(aspectmode="data")
    )
    print("Showing plot...")
    fig.show()


if __name__ == "__main__":
    if not os.path.exists(GRIPPER_NPZ_FILE):
        print(f"ERROR: Cannot find {GRIPPER_NPZ_FILE}")
    elif not os.path.exists(CONTACT_JSON_FILE):
        print(f"ERROR: Cannot find {CONTACT_JSON_FILE}")
    else:
        # Need to import SE3 helper functions
        # This assumes leap_kin.py imports operations correctly
        try:
            from mgs.operations import (
                 quaternion_from_axis_angle,
                 se3_raw_mupltiply,
                 transform_points_jax
                 )
            visualize_gripper_and_contacts()
        except ImportError as e:
            print(f"ERROR: Failed to import necessary JAX operations: {e}")
            print("Ensure mgs/operations.py exists and is accessible.")