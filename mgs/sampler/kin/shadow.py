from mgs.sampler.kin.base import (
    kinematic_pcd_transform,
    forward_kinematic_point_transform,
)
from flax import nnx  # Assuming nnx is used
import sys
from mgs.sampler.kin.jax_util import normalize_vector
import plotly.graph_objects as go
import numpy as np
import os
import jax.numpy as jnp
import math
import jax
from mgs.sampler.kin.base import KinematicsModel


class ShadowKinematicsModel(nnx.Module, KinematicsModel):
    def __init__(self, lmax=2, num_emb_channels=64):
        self.lmax = lmax
        self.num_channels = num_emb_channels
        self.num_dofs = 22
        self.num_extra_dofs = 0
        self.embedding = nnx.Param(
            jax.random.normal(
                nnx.Rngs(0)(), shape=(self.num_dofs, (lmax + 1) ** 2, num_emb_channels)
            )
        )
        self.kinematics_graph = [
            [0, 1, 2, 3],  # rh_FF
            [4, 5, 6, 7],  # rh_MF
            [8, 9, 10, 11],  # rh_RF
            [12, 13, 14, 15, 16],  # rh_LF
            [17, 18, 19, 20, 21],  # rh_TH
        ]
        self.base_to_contact = nnx.Variable(
            jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )

        self.kinematics_transforms = nnx.Variable(
            jnp.array(
                [
                    # rh_FF 4 (palm offset)
                    [1.0, 0.0, 0.0, 0.0, 0.033, 0, 0.095 + 0.034],
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.0],  # rh_FF 3
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.045],  # rh_FF 2
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.025],  # rh_FF 1
                    # rh_MF 4 (palm offset)
                    [1.0, 0.0, 0.0, 0.0, 0.011, 0, 0.099 + 0.034],
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.0],  # rh_MF 3
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.045],  # rh_MF 2
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.025],  # rh_MF 1
                    # rh_RF 4 (palm offset)
                    [1.0, 0.0, 0.0, 0.0, -0.011, 0, 0.095 + 0.034],
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.0],  # rh_RF 3
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.045],  # rh_RF 2
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.025],  # rh_RF 1
                    # rh_LF 5 (palm offset)
                    [1.0, 0.0, 0.0, 0.0, -0.033, 0, 0.02071 + 0.034],
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.06579],  # rh_LF 4
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.0],  # rh_RF 3
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.045],  # rh_LF 2
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.025],  # rh_LF 1
                    # rh_TH 5 (palm offset)
                    [0.92388, 0, 0.382683, 0, 0.034, -0.00858, 0.029 + 0.034],
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0],  # rh_TH 4
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.038],  # rh_TH 3
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0],  # rh_TH 2
                    [
                        1.0 / math.sqrt(2),
                        0.0,
                        0.0,
                        -1.0 / math.sqrt(2),
                        0,
                        0,
                        0.032,
                    ],  # rh_TH 1
                ]
            )
        )
        self.joint_transforms = nnx.Variable(
            jnp.array(
                [
                    # FF
                    [0, 0, 0, 0, -1, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    # MF
                    [0, 0, 0, 0, -1, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    # RF
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    # LF
                    [0, 0, 0, 0.573576, 0, 0.819152],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    # TH
                    [0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, -1, 0],
                    [0, 0, 0, 1, 0, 0],
                ],
                dtype=jnp.float32,
            )
        )
        self.joint_ranges = nnx.Variable(
            jnp.array(
                [
                    # FF
                    [-0.349066, 0.349066],
                    [-0.261799, 1.5708],
                    [0, 1.5708],
                    [0, 1.5708],
                    # MF
                    [-0.349066, 0.349066],
                    [-0.261799, 1.5708],
                    [0, 1.5708],
                    [0, 1.5708],
                    # RF
                    [-0.349066, 0.349066],
                    [-0.261799, 1.5708],
                    [0, 1.5708],
                    [0, 1.5708],
                    # LF
                    [0, 0.785398],
                    [-0.349066, 0.349066],
                    [-0.261799, 1.5708],
                    [0, 1.5708],
                    [0, 1.5708],
                    # TH
                    [-1.0472, 1.0472],
                    [0, 1.22173],
                    [-0.20944, 0.20944],
                    [-0.698132, 0.698132],
                    [-0.261799, 1.5708],
                ]
            )
        )

        self.fingertip_normals = nnx.Variable(
            jnp.array(
                [
                    [0.0, 1.0, 0],
                    [0.0, 1.0, 0],
                    [0.0, 1.0, 0],
                    [0.0, 1.0, 0],
                    [0.0, 1.0, 0],
                ]
            )
        )
        self.fingertip_idx = nnx.Variable(
            jnp.array([3, 7, 11, 16, 21], dtype=jnp.int32)
        )
        self.local_fingertip_contact_positions = nnx.Variable(
            jnp.array(
                [
                    [
                        [0, -0.01, 0],
                        [0, -0.01, 0.01],
                        [0, -0.01, -0.01],
                    ],
                    [
                        [0, -0.01, 0],
                        [0, -0.01, 0.01],
                        [0, -0.01, -0.01],
                    ],
                    [
                        [0, -0.01, 0],
                        [0, -0.01, 0.01],
                        [0, -0.01, -0.01],
                    ],
                    [
                        [0, -0.01, 0],
                        [0, -0.01, 0.01],
                        [0, -0.01, -0.01],
                    ],
                    [
                        [0, -0.01, 0],
                        [0, -0.01, 0.01],
                        [0, -0.01, -0.01],
                    ],
                ]
            )
        )
        self.init_pregrasp_joint = nnx.Variable(
            jnp.array(
                [
                    -0.350,
                    0.425,
                    0.015,
                    0.005,
                    -0.095,
                    0.415,
                    0.010,
                    0.0,
                    -0.075,
                    0.435,
                    0.015,
                    0.005,
                    0.0,
                    -0.220,
                    0.255,
                    0.0,
                    0.0,
                    -0.480,
                    1.05,
                    -0.19,
                    -0.080,
                    0.45,
                ]
            )
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


SHADOW_NPZ_FILE = "gripper_shadow.npz"
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
        "ff_j4",
        "ff_j3",
        "ff_j2",
        "ff_j1",
        "mf_j4",
        "mf_j3",
        "mf_j2",
        "mf_j1",
        "rf_j4",
        "rf_j3",
        "rf_j2",
        "rf_j1",
        "lf_j5",
        "lf_j4",
        "lf_j3",
        "lf_j2",
        "lf_j1",
        "th_j5",
        "th_j4",
        "th_j3",
        "th_j2",
        "th_j1",
    ]
    segmentations_np = np.stack(
        [raw[key][random_idx] for key in segmentation_keys_ordered]
    )
    print(f"  Loaded segmentations, shape: {segmentations_np.shape}")

    # 2. Initialize Kinematics Model and Get Initial State
    print("Initializing Shadow Kinematics Model...")
    # These imports need to be resolvable
    kin_model = ShadowKinematicsModel()
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
    points_vis_transformed_jax = kinematic_pcd_transform(
        points_vis_jax, initial_pose_jax, segmentations_jax, kin_model
    )
    points_vis_transformed_np = np.array(points_vis_transformed_jax)
    print("  Transformed visualization point cloud.")

    # 4. Transform Contact Points and Calculate Normals using YOUR FK function
    print("Transforming contact points and calculating normals using YOUR FK...")
    local_contacts = kin_model.local_fingertip_contact_positions.value.squeeze(
        1
    )  # (5, 3)
    # (5, 3)
    local_normals = kin_model.fingertip_normals.value
    # Local origin for normal calculation
    local_origin = jnp.zeros((5, 3), dtype=jnp.float32)
    fingertip_joint_indices = kin_model.fingertip_idx.value  # (5,)

    world_contact_points_list = []
    world_normal_vectors_list = []

    # Use vmap for transforming points associated with each fingertip
    # We need to transform 3 points per fingertip: contact point, origin, point along normal
    @nnx.jit
    def get_world_pts_for_link(theta, local_pts, joint_idx, model):
        # Vmap over the points (contact, origin, point_on_normal) for a single link
        return jax.vmap(
            forward_kinematic_point_transform, in_axes=(None, 0, None, None)
        )(theta, local_pts, joint_idx, model)

    for i in range(len(fingertip_joint_indices)):
        joint_idx = fingertip_joint_indices[i]
        local_point_contact = local_contacts[i]
        local_point_origin = local_origin[i]
        # Point along local normal from origin
        local_point_on_normal = local_origin[i] + local_normals[i]

        # Pack the three local points for this link
        local_pts_for_link = jnp.stack(
            [local_point_contact, local_point_origin, local_point_on_normal], axis=0
        )  # Shape (3, 3)

        # Transform these three points to world frame
        world_pts = get_world_pts_for_link(
            initial_pose_jax, local_pts_for_link, joint_idx, kin_model
        )
        world_contact = world_pts[0]
        world_origin = world_pts[1]
        world_point_on_normal = world_pts[2]

        # Calculate world normal vector
        world_normal = world_point_on_normal - world_origin

        world_contact_points_list.append(np.array(world_contact))
        world_normal_vectors_list.append(np.array(world_normal))

    contact_points_np = np.array(world_contact_points_list)  # Shape (5, 3)
    contact_normals_np = np.array(world_normal_vectors_list)  # Shape (5, 3)
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
