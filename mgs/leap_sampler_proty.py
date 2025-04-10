import optax
import os
import json
import numpy as np
import plotly.graph_objects as go
from flax import nnx
import jax.numpy as jnp

from mgs.leap_kin import LeapHandKinematicsModel, kinematic_pcd_transform


CURRENT = os.path.dirname(os.path.abspath(__file__))
GRIPPER_NPZ_FILE = os.path.join(CURRENT, "gripper_leap.npz")
CONTACT_JSON_FILE = os.path.join(CURRENT, "contact_candidates.json")
NUM_POINTS_VIS = 2000

FINGERTIP_INFO = {
    "fingertip": 3,  # Index finger tip joint index
    "fingertip_2": 7,  # Middle finger tip joint index
    "fingertip_3": 11,  # Ring finger tip joint index
    "thumb_fingertip": 15,  # Thumb finger tip joint index
}
# --- Main Visualization Logic ---


@nnx.jit
def forward_kinematic_point_transform(
    theta: jnp.ndarray,
    local_point: jnp.ndarray,
    joint_idx: jnp.ndarray,
    kin_model: LeapHandKinematicsModel,
):
    all_link_transforms = jnp.zeros(shape=(kin_model.num_dofs + 1, 7))
    identity_tf = jnp.array([1.0, 0, 0, 0, 0, 0, 0], dtype=jnp.float32)
    all_link_transforms = all_link_transforms.at[0].set(identity_tf)
    parent_map = {}
    for chain in kin_model.kinematics_graph:
        parent_map[chain[0]] = -1
        for i in range(len(chain) - 1):
            parent_map[chain[i + 1]] = chain[i]
    for index in range(kin_model.num_dofs):
        parent_index = parent_map[index]
        T_world_parent = all_link_transforms[parent_index + 1]
        T_parent_jointStatic = kin_model.kinematics_transforms.value[index]
        joint_theta = theta[index]
        translation = kin_model.joint_transforms.value[index, :3] * joint_theta
        joint_axis = kin_model.joint_transforms.value[index, 3:]
        rotation_quat = quaternion_from_axis_angle(joint_axis, joint_theta)
        T_jointStatic_jointDynamic = jnp.concatenate(
            [rotation_quat, translation], axis=-1
        )
        T_world_joint = se3_raw_mupltiply(
            se3_raw_mupltiply(T_world_parent, T_parent_jointStatic),
            T_jointStatic_jointDynamic,
        )
        all_link_transforms = all_link_transforms.at[index + 1].set(
            T_world_joint)
    target_transform_world = all_link_transforms[joint_idx + 1]
    transformed_point = transform_points_jax(
        local_point, target_transform_world)
    return transformed_point


@nnx.jit
def update(
    graph,
    state,
    gripper_local_contact_positions,
    target_surface_positions,
):
    kin, optimizer, theta = nnx.merge(graph, state)

    def loss_fn(current_theta):
        transformed_points = nnx.vmap(
            forward_kinematic_point_transform, in_axes=(None, 0, 0, None)
        )(
            current_theta.theta.value,
            gripper_local_contact_positions,
            jnp.array([3, 7, 11, 15], dtype=jnp.int32),
            kin,
        )
        loss = jnp.mean((target_surface_positions - transformed_points) ** 2)
        return loss

    loss, grad = nnx.value_and_grad(loss_fn)(theta)
    optimizer.update(grad)

    return loss, nnx.state((kin, optimizer, theta))


class JointState(nnx.Module):
    def __init__(self, init_theta):
        self.theta = nnx.Param(init_theta)


class Trainer:
    def __init__(self, kin):
        self.kin = kin
        tx = optax.adamw(0.005)
        self.theta = JointState(
            init_theta=jnp.array(
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
        )
        self.optimizer = nnx.Optimizer(self.theta, tx)
        self.train_graph, self.train_state = nnx.split(
            (self.kin, self.optimizer, self.theta)
        )

    def train_step(self, input, target: jnp.ndarray):
        loss, self.train_state = update(
            self.train_graph, self.train_state, input, target
        )
        return loss


def example_optimization():
    kin = LeapHandKinematicsModel()
    trainer = Trainer(kin)

    local_positions = jnp.zeros(shape=(4, 3), dtype=jnp.float32)
    target_positions = jnp.zeros(shape=(4, 3), dtype=jnp.float32)
    target_positions = target_positions.at[:, 2].set(-0.08)

    for _ in range(1000):
        loss = trainer.train_step(local_positions, target_positions)
        print("Loss:", loss)
    _, _, theta = nnx.merge(trainer.train_graph, trainer.train_state)
    return theta.theta.value


def visualize_gripper_and_contacts(theta):
    """Loads and visualizes the gripper point cloud and transformed fingertip contact points."""

    # 1. Load Gripper Point Cloud Data
    print(f"Loading gripper point cloud from: {GRIPPER_NPZ_FILE}")
    raw = np.load(GRIPPER_NPZ_FILE, allow_pickle=True)
    points_full = raw["pcd_point"]
    print(f"  Loaded {len(points_full)} points.")

    random_idx = np.random.choice(
        points_full.shape[0], size=NUM_POINTS_VIS, replace=False
    )
    points_vis = points_full[random_idx]
    print(f"  Sampled {len(points_vis)} points for visualization.")

    # 2. Load Contact Candidates (local coordinates)
    print(f"Loading contact candidates from: {CONTACT_JSON_FILE}")
    with open(CONTACT_JSON_FILE, "r") as f:
        contact_candidates_data = json.load(f)
    print("  Loaded contact candidates.")

    kin_model = LeapHandKinematicsModel()  # Initialize our JAX model
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

    # 4. Transform Contact Points
    transformed_contacts_list = []
    for link_name, link_joint_index in FINGERTIP_INFO.items():
        local_points = contact_candidates_data.get(link_name, [])
        if local_points:
            local_points_jax = jnp.array(local_points, dtype=jnp.float32)
            # Select only the FIRST contact point for this link for visualization
            if local_points_jax.shape[0] > 0:
                first_contact_point_local = local_points_jax[
                    0:1, :
                ]  # Keep batch dim (1, 3)
                transformed_point = forward_kinematic_point_transform(
                    theta,
                    first_contact_point_local,
                    jnp.asarray(link_joint_index),
                    kin_model,
                )

                transformed_contacts_list.append(
                    np.array(transformed_point[0])
                )  # Append NumPy array (3,)

    contact_points_np = np.array(
        transformed_contacts_list
    )  # Shape (NumFingersWithContacts, 3)
    print(f"  Transformed {len(contact_points_np)} contact points.")

    # 5. Prepare Plotly Data
    plot_traces = []

    # Gripper cloud trace
    gripper_trace = go.Scatter3d(
        x=points_vis[:, 0],
        y=points_vis[:, 1],
        z=points_vis[:, 2],
        mode="markers",
        marker=dict(size=2, color="blue", opacity=0.6),
        name="Gripper Cloud (theta=0)",
    )
    plot_traces.append(gripper_trace)

    # Transformed contact points trace
    if contact_points_np.shape[0] > 0:
        contact_trace = go.Scatter3d(
            x=contact_points_np[:, 0],
            y=contact_points_np[:, 1],
            z=contact_points_np[:, 2],
            mode="markers",
            marker=dict(size=5, color="red", opacity=1.0),
            name="Transformed Contact Points (theta=0)",
        )
        plot_traces.append(contact_trace)

    # 6. Create and Show Plot
    fig = go.Figure(data=plot_traces)
    fig.update_layout(
        title="LEAP Hand (theta=0) + Transformed Contact Points",
        scene=dict(aspectmode="data"),
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
                transform_points_jax,
            )

            # visualize_gripper_and_contacts()
            theta = example_optimization()
            visualize_gripper_and_contacts(theta)
        except ImportError as e:
            print(f"ERROR: Failed to import necessary JAX operations: {e}")
            print("Ensure mgs/operations.py exists and is accessible.")
