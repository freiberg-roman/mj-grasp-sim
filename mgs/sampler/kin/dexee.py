import jax.numpy as jnp
from flax import nnx

from mgs.sampler.kin.base import KinematicsModel
from mgs.sampler.kin.seg_op import (
    kinematic_frames,
    kinematic_transform,
    point_transform,
)


def _quat_mul(a, b):
    """Hamilton product, (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return jnp.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=jnp.float32,
    )


def _quat_rotate(q, v):
    """Rotate vector v by unit quaternion q=(w,x,y,z)."""
    w, x, y, z = q
    qvec = jnp.array([x, y, z], dtype=jnp.float32)
    uv = jnp.cross(qvec, v)
    uuv = jnp.cross(qvec, uv)
    return v + 2.0 * (w * uv + uuv)


def _normalize(q):
    return q / jnp.linalg.norm(q)


class DexeeKinematicsModel(nnx.Module, KinematicsModel):
    """
    Kinematics for the 3‑finger Dexee hand matching your MuJoCo XML.

    Joint order (num_dofs = 12):
      [F0/J0, F0/J1, F0/J2, F0/J3,
       F1/J0, F1/J1, F1/J2, F1/J3,
       F2/J0, F2/J1, F2/J2, F2/J3]
    """

    def __init__(self):
        # DoFs and graph (three chains of length 4).
        self.num_dofs = 12
        self.num_extra_dofs = 0
        self.kinematics_graph = [
            [0, 1, 2, 3],  # F0
            [4, 5, 6, 7],  # F1
            [8, 9, 10, 11],  # F2
        ]

        # As requested: identity. TCP / base offsets are handled in the XML, not here.
        self.base_to_contact = nnx.Variable(
            jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
        )

        # A reasonable initial alignment for approach frames. This rotates the model
        # so that the local -Y finger directions tend to align with the object normal.
        # Feel free to tweak R or t to taste for your seeding strategy.
        Rx90 = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=jnp.float32,
        )
        self.align_to_approach = nnx.Variable(
            (Rx90, jnp.array([0.0, 0.0, 0.07], dtype=jnp.float32))
        )

        # ----------------------------------------------------------------------
        # Static body-to-body transforms from XML, folded per joint as [qw, qx, qy, qz, tx, ty, tz].
        # For J0, we compose the finger base (F{0,1,2}/) with the knuckle body pose.
        # For J1..J3, we use the per-segment offsets as-is (including distal's fixed quaternion).
        # ----------------------------------------------------------------------

        # Common knuckle pose relative to finger_base: pos=(0, 0.015, 0.17902), euler X = -1.0472 rad.
        q_knuckle = jnp.array(
            [jnp.cos(-1.0472 / 2.0), jnp.sin(-1.0472 / 2.0), 0.0, 0.0],
            dtype=jnp.float32,
        )
        t_knuckle = jnp.array([0.0, 0.015, 0.17902], dtype=jnp.float32)

        # Distal fixed frame quaternion on J3: "quat='0 0 -1 1'" (normalized).
        q_distal_raw = jnp.array([0.0, 0.0, -1.0, 1.0], dtype=jnp.float32)
        q_distal = _normalize(q_distal_raw)

        # Finger bases from XML (relative to dexee_gripper).
        # F0: pos="0 0.05 0.017", quat="1 0 0 0"
        q_F0 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
        t_F0 = jnp.array([0.0, 0.05, 0.017], dtype=jnp.float32)

        # F1: pos="0.039 -0.029 0.017", quat="-0.16212752892551119 0 0 0.98676981326168844"
        # Sign does not matter; keep as-is.
        q_F1 = jnp.array([-0.16212753, 0.0, 0.0, 0.9867698], dtype=jnp.float32)
        t_F1 = jnp.array([0.039, -0.029, 0.017], dtype=jnp.float32)

        # F2: pos="-0.039 -0.029 0.017", quat="0.16212752892551119 0 0 0.98676981326168844"
        q_F2 = jnp.array([0.16212753, 0.0, 0.0, 0.9867698], dtype=jnp.float32)
        t_F2 = jnp.array([-0.039, -0.029, 0.017], dtype=jnp.float32)

        def j0_static(q_base, t_base):
            q = _quat_mul(q_base, q_knuckle)
            t = t_base + _quat_rotate(q_base, t_knuckle)
            return jnp.concatenate([q, t], axis=0)

        # Segment offsets down the chain (shared across fingers).
        T_J1 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, -0.03, 0.0], dtype=jnp.float32)
        T_J2 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, -0.05, 0.0], dtype=jnp.float32)
        T_J3 = jnp.concatenate(
            [q_distal, jnp.array([0.0, -0.035, 0.0], dtype=jnp.float32)], axis=0
        )

        # Assemble per-finger chains.
        T_F0 = [j0_static(q_F0, t_F0), T_J1, T_J2, T_J3]
        T_F1 = [j0_static(q_F1, t_F1), T_J1, T_J2, T_J3]
        T_F2 = [j0_static(q_F2, t_F2), T_J1, T_J2, T_J3]

        self.kinematics_transforms = nnx.Variable(jnp.stack(T_F0 + T_F1 + T_F2, axis=0))

        # ----------------------------------------------------------------------
        # Joint motions: [tx_per_rad, ty_per_rad, tz_per_rad, ax_x, ax_y, ax_z].
        # All joints are revolute; no translation per angle, axes copied from XML.
        # ----------------------------------------------------------------------
        JT = []
        # F0
        JT += [[0.0, 0.0, 0.0, 0.0, 0.0, -1.0]]  # F0/J0  axis="0 0 -1"
        JT += [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]  # F0/J1  axis="1 0 0"
        JT += [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]  # F0/J2  axis="1 0 0"
        JT += [[0.0, 0.0, 0.0, -1.0, 0.0, 0.0]]  # F0/J3  axis="-1 0 0"
        # F1
        JT += [[0.0, 0.0, 0.0, 0.0, 0.0, -1.0]]
        JT += [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
        JT += [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
        JT += [[0.0, 0.0, 0.0, -1.0, 0.0, 0.0]]
        # F2
        JT += [[0.0, 0.0, 0.0, 0.0, 0.0, -1.0]]
        JT += [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
        JT += [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
        JT += [[0.0, 0.0, 0.0, -1.0, 0.0, 0.0]]
        self.joint_transforms = nnx.Variable(jnp.array(JT, dtype=jnp.float32))

        # Joint ranges (copied from XML for all three fingers).
        JR = []
        # F0
        JR += [[-0.8727, 0.8727]]  # J0
        JR += [[-1.3963, 0.7854]]  # J1
        JR += [[0.0, 1.3963]]  # J2
        JR += [[-0.5236, 1.4835]]  # J3
        # F1
        JR += [[-0.8727, 0.8727]]
        JR += [[-1.3963, 0.7854]]
        JR += [[0.0, 1.3963]]
        JR += [[-0.5236, 1.4835]]
        # F2
        JR += [[-0.8727, 0.8727]]
        JR += [[-1.3963, 0.7854]]
        JR += [[0.0, 1.3963]]
        JR += [[-0.5236, 1.4835]]
        self.joint_ranges = nnx.Variable(jnp.array(JR, dtype=jnp.float32))

        # Fingertip outward directions in each distal joint's local frame.
        # The finger link chain advances along local -Y; these normals follow that.
        self.fingertip_normals = nnx.Variable(
            jnp.array(
                [
                    [0.0, -1.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, -1.0, 0.0],
                ],
                dtype=jnp.float32,
            )
        )

        # J indices of distal links for F0, F1, F2.
        self.fingertip_idx = nnx.Variable(jnp.array([3, 7, 11], dtype=jnp.int32))

        # Local fingertip contact positions (in the distal joint frames).
        # First point sits near your XML 'fingertip_site'; two small neighbors help assignment.
        self.local_fingertip_contact_positions = nnx.Variable(
            jnp.array(
                [
                    [
                        [0, 0.007, 0.03],
                        [0.005, 0.007, 0.03],
                        [-0.005, 0.007, 0.03],
                        [0.0, 0.007, 0.025],
                        [0.005, 0.007, 0.025],
                        [-0.005, 0.007, 0.025],
                    ],
                    [
                        [0, 0.007, 0.03],
                        [0.005, 0.007, 0.03],
                        [-0.005, 0.007, 0.03],
                        [0.0, 0.007, 0.025],
                        [0.005, 0.007, 0.025],
                        [-0.005, 0.007, 0.025],
                    ],
                    [
                        [0, 0.007, 0.03],
                        [0.005, 0.007, 0.03],
                        [-0.005, 0.007, 0.03],
                        [0.0, 0.007, 0.025],
                        [0.005, 0.007, 0.025],
                        [-0.005, 0.007, 0.025],
                    ],
                ],
                dtype=jnp.float32,
            )
        )

        # A gentle pre‑grasp. Tune as you like; ranges enforce safety during optimization.
        self.init_pregrasp_joint = nnx.Variable(
            # jnp.zeros_like(
            jnp.array(
                [
                    0,
                    -1.0,
                    0,
                    0,
                    0,
                    -1.0,
                    0,
                    0,
                    0,
                    -1.0,
                    0,
                    0,
                ],
                dtype=jnp.float32,
            )
            # )
        )


# --- Assume necessary imports are handled by the user ---
# Need:
# - ShadowKinematicsModel class definition from mgs.sampler.shadow_kin
# - kinematic_pcd_transform function from mgs.sampler.kin.jax_util
# - forward_kinematic_point_transform function from mgs.sampler.kin.jax_util
# - DATA_PATH variable pointing to the dex-grasp-net data directory
# ---
# --- Configuration ---
try:
    import os
    import sys

    import jax
    import numpy as np
    import plotly.graph_objects as go

    from mgs.sampler.kin.seg_op import kinematic_transform, point_transform

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root_mj = os.path.abspath(os.path.join(_script_dir, "../../.."))
    if _project_root_mj not in sys.path:
        sys.path.insert(0, _project_root_mj)
    from mgs.util.const import ASSET_PATH  # Use ASSET_PATH convention

    # Derive DATA_PATH relative to ASSET_PATH or project root if needed
    _project_root_dex = os.path.abspath(
        os.path.join(_project_root_mj, "../dex-grasp-net/DexGraspNet2")
    )  # Adjust if needed
    if os.path.exists(os.path.join(_project_root_dex, "data")):
        DATA_PATH = os.path.join(_project_root_dex, "data")
    else:
        DATA_PATH = "./data"
    print(f"Using DATA_PATH: {DATA_PATH}")
except ImportError:
    print("ERROR: Could not import ASSET_PATH. Ensure mgs/util/const.py exists.")
    DATA_PATH = "./data"
    print(f"Warning: Using default DATA_PATH: {DATA_PATH}")
except Exception as e:
    DATA_PATH = "./data"
    print(f"Warning: Path detection failed {e}. Using default DATA_PATH: {DATA_PATH}")


SHADOW_NPZ_FILE = "gripper_dexee.npz"
NUM_POINTS_VIS = 2000
NORMAL_VIS_LENGTH = 0.02

# --- Helper Functions ---


def normalize_vector(v, axis=-1, epsilon=1e-8):
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / (norm + epsilon)


# --- Main Visualization Logic ---


def visualize_shadow_initial_contacts_normals():
    """Loads Shadow Hand cloud, uses YOUR FK functions to show initial contacts and normals."""

    # 1. Load Gripper Point Cloud Data and Segmentation
    print(f"Loading gripper point cloud from: {SHADOW_NPZ_FILE}")
    if not os.path.exists(SHADOW_NPZ_FILE):
        print(f"ERROR: Not found {SHADOW_NPZ_FILE}")
        return
    raw = np.load(SHADOW_NPZ_FILE, allow_pickle=True)
    points_full = raw["pcd_point"]
    print(f"  Loaded {len(points_full)} points.")
    if len(points_full) < NUM_POINTS_VIS:
        random_idx = np.arange(len(points_full))
    else:
        random_idx = np.random.choice(
            points_full.shape[0], size=NUM_POINTS_VIS, replace=False
        )
    points_vis_np = points_full[random_idx]
    print(f"  Sampled {len(points_vis_np)} points.")
    segmentation_keys_ordered = [
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
    segmentations_np = np.stack(
        [raw[key][random_idx] for key in segmentation_keys_ordered]
    )
    print(f"  Loaded segmentations, shape: {segmentations_np.shape}")

    # 2. Initialize Kinematics Model and Get Initial State
    print("Initializing Shadow Kinematics Model...")
    # These imports need to be resolvable
    kin_model = DexeeKinematicsModel()
    # --- Ensure using JAX array for theta ---
    initial_pose_jax = jnp.array(
        kin_model.init_pregrasp_joint.value
    )  # Get initial pose
    print("  Using initial pre-grasp joint configuration.")

    # Convert inputs to JAX arrays
    points_vis_jax = jnp.array(points_vis_np)
    segmentations_jax = jnp.array(segmentations_np)

    # 3. Transform Visualization Point Cloud using YOUR function
    print(
        "Transforming visualization point cloud using YOUR kinematic_pcd_transform..."
    )
    # Ensure kin_model is passed correctly (might need graph/state if using nnx.split elsewhere)
    # Assuming kinematic_pcd_transform can take the model instance directly
    # points_vis_transformed_jax = kinematic_pcd_transform(
    #     points_vis_jax, initial_pose_jax, segmentations_jax, kin_model
    # )
    points_vis_transformed_jax = kinematic_transform(
        point_transform,
        points_vis_jax,
        initial_pose_jax,
        segmentations_jax,
        *nnx.split(kin_model),
    )
    points_vis_transformed_np = np.array(points_vis_transformed_jax)
    print("  Transformed visualization point cloud.")

    # 4. Transform Contact Points and Calculate Normals using YOUR FK function
    print("Transforming contact points and calculating normals using YOUR FK...")
    local_contacts = kin_model.local_fingertip_contact_positions.value[
        :, 0, :
    ]  # (5, 3)
    local_normals = kin_model.fingertip_normals.value
    # Local origin for normal calculation
    local_origin = jnp.zeros((5, 3), dtype=jnp.float32)
    fingertip_joint_indices = kin_model.fingertip_idx.value  # (5,)

    world_contact_points_list = []
    world_normal_vectors_list = []

    # Use vmap for transforming points associated with each fingertip
    # We need to transform 3 points per fingertip: contact point, origin, point along normal

    for i in range(len(fingertip_joint_indices)):
        joint_idx = fingertip_joint_indices[i][None]
        local_point_contact = local_contacts[i]

        # Transform these three points to world frame
        frame_rot, frame_pos = kinematic_frames(initial_pose_jax, *nnx.split(kin_model))
        world_contact = (
            frame_rot[joint_idx] @ local_point_contact + frame_pos[joint_idx]
        )
        world_normal = frame_rot[joint_idx] @ local_normals[i]

        world_contact_points_list.append(np.array(world_contact))
        world_normal_vectors_list.append(np.array(world_normal))

    contact_points_np = np.concatenate(
        world_contact_points_list, axis=0
    )  # Shape (5, 3)
    contact_normals_np = np.concatenate(
        world_normal_vectors_list, axis=0
    )  # Shape (5, 3)
    # Normalize the calculated world normals
    contact_normals_np = normalize_vector(contact_normals_np)

    # Normalize the calculated world normals
    contact_normals_np = normalize_vector(contact_normals_np)
    print(
        f"  Transformed {len(contact_points_np)} contact points and normals using YOUR functions."
    )

    # 5. Prepare Plotly Visualization
    print("Preparing visualization...")
    plot_traces = []

    # Trace for the transformed gripper point cloud
    plot_traces.append(
        go.Scatter3d(
            x=points_vis_transformed_np[:, 0],
            y=points_vis_transformed_np[:, 1],
            z=points_vis_transformed_np[:, 2],
            mode="markers",
            marker=dict(size=2, color="grey", opacity=0.5),
            name="Gripper Cloud (Initial Pose)",
        )
    )
    # Trace for the transformed contact points
    if contact_points_np.shape[0] > 0:
        plot_traces.append(
            go.Scatter3d(
                x=contact_points_np[:, 0],
                y=contact_points_np[:, 1],
                z=contact_points_np[:, 2],
                mode="markers",
                marker=dict(size=5, color="blue", opacity=1.0),
                name="Contact Points",
            )
        )
        # Traces for the normals (as lines)
        lines_x, lines_y, lines_z = [], [], []
        for i in range(len(contact_points_np)):
            p0 = contact_points_np[i]
            p1 = p0 + contact_normals_np[i] * NORMAL_VIS_LENGTH
            lines_x.extend([p0[0], p1[0], None])
            lines_y.extend([p0[1], p1[1], None])
            lines_z.extend([p0[2], p1[2], None])
        plot_traces.append(
            go.Scatter3d(
                x=lines_x,
                y=lines_y,
                z=lines_z,
                mode="lines",
                line=dict(color="red", width=3),
                name="Contact Normals",
            )
        )

    # 6. Show Plot
    fig = go.Figure(data=plot_traces)
    fig.update_layout(
        title="Shadow Hand (Initial Pose) + Contact Points & Normals (Using Imported FK)",
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    print("Showing plot...")
    fig.show()


if __name__ == "__main__":
    visualize_shadow_initial_contacts_normals()
