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
from mgs.util.geo.transforms import SE3Pose



class AllegroKinematicsModel(nnx.Module, KinematicsModel):
    def __init__(self, lmax=2, num_emb_channels=64):
        self.lmax = lmax
        self.num_channels = num_emb_channels
        self.num_dofs = 16
        self.num_extra_dofs = 0
        self.embedding = nnx.Param(
            jax.random.normal(
                nnx.Rngs(0)(), shape=(self.num_dofs, (lmax + 1) ** 2, num_emb_channels)
            )
        )
        self.kinematics_graph = [
            [0, 1, 2, 3],  # ffa
            [4, 5, 6, 7],  # mfa
            [8, 9, 10, 11],  # rfa
            [12, 13, 14, 15],  # tha
        ]
        self.base_to_contact = nnx.Variable(
            jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )

        # Rotate around y axis pi/2 
        self.align_to_approach = nnx.Variable(
            (
                jnp.array([[np.cos(-np.pi / 2), 0, -np.sin(-np.pi /2)], 
                           [0, 1.0, 0], 
                           [np.sin(-np.pi / 2), 0, np.cos(-np.pi / 2)]]),
                jnp.array([+0.01, 0.0, +0.08]),
            )
        )


        self.kinematics_transforms = nnx.Variable(
            jnp.array(
                [
                    # FF
                    # ffj0 is the base of ffj1, ffj1 is the base of ffj2, etc.
                    [0.999048, -0.0436194, 0, 0, 0, 0.0435, -0.001542], # ffj0
                    [1, 0, 0, 0, 0, 0, 0.0164],  # ffj1  
                    [1, 0, 0, 0, 0, 0, 0.054],  # ffj2
                    [1, 0, 0, 0, 0, 0, 0.0384],  # ffj3

                    # MF
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.0007],  #mfj0
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.0164],  # mfj1
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.054],  # mfj2
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.0384],  # mfj3
                    
                    # RF
                    [0.999048, 0.0436194, 0, 0, 0, -0.0435, -0.001542], # rfj0
                    [1, 0, 0, 0, 0, 0, 0.0164],  # rfj1
                    [1, 0, 0, 0, 0, 0, 0.054],  # rfj2
                    [1, 0, 0, 0, 0, 0, 0.0384],  # rfj3
                    
                    # TH
                    [0.477714, -0.521334, -0.521334, -0.477714, -0.0182, 0.019333, -0.045987],  # thj0
                    [1, 0 ,0 ,0 ,-0.027, 0.005, 0.0399],  # thj1
                    [1, 0, 0, 0, 0, 0, 0.0177],  # thj2
                    [1, 0 ,0, 0, 0, 0, 0.0514],  # thj3
                ]
            )
        )
        # a six-dimensional vector with a translational direction and rotation axis
        self.joint_transforms = nnx.Variable(
            jnp.array(
                [
                    # FF
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    # MF
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    # RF
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                 
                    # TH
                    [0, 0, 0, -1, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                ],
                dtype=jnp.float32,
            )
        )
        self.joint_ranges = nnx.Variable(
            jnp.array(
                [
                    # FF
                    [-0.47, 0.47],
                    [-0.196, 1.61],
                    [-0.174, 1.709],
                    [-0.227, 1.618],
                    # MF
                    [-0.47, 0.47],
                    [-0.196, 1.61],
                    [-0.174, 1.709],
                    [-0.227, 1.618],
                    # RF
                    [-0.47, 0.47],
                    [-0.196, 1.61],
                    [-0.174, 1.709],
                    [-0.227, 1.618],
        
                    # TH
                    [0.263, 1.396],
                    [-0.105, 1.163],
                    [-0.189, 1.644],
                    [-0.162, 1.719],
                ]
            )
        )

        # ask
        self.fingertip_normals = nnx.Variable(
            jnp.array(
                [
                    [-1.0, 0, 0],
                    [-1.0, 0, 0],
                    [-1.0, 0.0, 0],
                    [-1.0, 0.0, 0.0],
                ]
            )
        )

        self.fingertip_idx = nnx.Variable(
            jnp.array([3, 7, 11, 15], dtype=jnp.int32)
        )

        # ask!!
        self.local_fingertip_contact_positions = nnx.Variable(
            jnp.array(
                [
                    [
                        [0.0, 0, 0.023],
                        [0.002, 0, 0.02],
                        [0, 0.002, 0.02],
                    ],
                    [
                        [0, 0, 0.023],
                        [0.002, 0, 0.02],
                        [0.0, 0.002, 0.02],
                    ],
                    [
                        [0, 0, 0.023],
                        [0.002, 0, 0.02],
                        [0.0, 0.002, 0.02],
                    ],
                    [
                        [0, 0, 0.035],
                        [0.002, 0, 0.032],
                        [0.0, 0.002, 0.032],
                    ],
                ]
            )
        )
        self.init_pregrasp_joint = nnx.Variable(
            jnp.array(
                 [
                -0.08,
                0.297,
                0.710,
                0.95,
                0,
                0.319,
                0.71,
                0.67,
                0.08,
                0.454,
                0.710,
                0.95,
                1.06,
                0.358,
                0.251,
                0.318,
                ]
            )
        )


ALLEGRO_NPZ_FILE = "./allegro_hand.npz"
NUM_POINTS_VIS = 2000
NORMAL_VIS_LENGTH = 0.02

# --- Helper Functions ---


def normalize_vector(v, axis=-1, epsilon=1e-8):
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / (norm + epsilon)


# --- Main Visualization Logic ---


def visualize_shadow_initial_contacts_normals():
    """Loads Shadow Hand cloud, uses YOUR FK functions to show initial contacts and normals."""
    print(f"Loading gripper point cloud from: {ALLEGRO_NPZ_FILE}")
    if not os.path.exists(ALLEGRO_NPZ_FILE):
        print(f"ERROR: Not found {ALLEGRO_NPZ_FILE}")
        return
    raw = np.load(ALLEGRO_NPZ_FILE, allow_pickle=True)
    points_full = raw["pcd_point"]
    print(f"  Loaded {len(points_full)} points.")
    np.random.seed(100)
    if len(points_full) < NUM_POINTS_VIS:
        random_idx = np.arange(len(points_full))
    else:
        random_idx = np.random.choice(
            points_full.shape[0], size=NUM_POINTS_VIS, replace=False
        )
    points_vis_np = points_full[random_idx]
    print(f"  Sampled {len(points_vis_np)} points.")
    segmentation_keys_ordered = [
        "ffj0",
        "ffj1",
        "ffj2",
        "ffj3",
        "mfj0", 
        "mfj1", 
        "mfj2",
        "mfj3",
        "rfj0",
        "rfj1",
        "rfj2",
        "rfj3", 
        "thj0",
        "thj1",
        "thj2",
        "thj3" 
    ]
    segmentations_np = np.stack(
        [raw[key][random_idx] for key in segmentation_keys_ordered]
    )
    print(f"  Loaded segmentations, shape: {segmentations_np.shape}")

    # 2. Initialize Kinematics Model and Get Initial State
    print("Initializing Shadow Kinematics Model...")
    # These imports need to be resolvable
    kin_model = AllegroKinematicsModel()
    # --- Ensure using JAX array for theta ---
    # Use the GUI to compare some initial joint values
    
    initial_pose_jax = jnp.array(
        kin_model.init_pregrasp_joint.value
    )  # Get initial pose
    print("  Using initial pre-grasp joint configuration.")

    # initial_pose_jax = jnp.zeros((16,))

    # Convert inputs to JAX arrays
    points_vis_jax = jnp.array(points_vis_np)
    segmentations_jax = jnp.array(segmentations_np)

    # 3. Transform Visualization Point Cloud using YOUR function
    print(
        "Transforming visualization point cloud using YOUR kinematic_pcd_transform..."
    )
    # Ensure kin_model is passed correctly (might need graph/state if using nnx.split elsewhere)
    # Assuming kinematic_pcd_transform can take the model instance directly

    #TODO: understand kinematic_pcd_transform doing
    points_vis_transformed_jax = kinematic_pcd_transform(
        points_vis_jax, initial_pose_jax, segmentations_jax, kin_model
    )
    points_vis_transformed_np = np.array(points_vis_transformed_jax)
    print("  Transformed visualization point cloud.")


    # 4. Transform Contact Points and Calculate Normals using YOUR FK function
    print("Transforming contact points and calculating normals using YOUR FK...")
    local_contacts = kin_model.local_fingertip_contact_positions.value[:, 0, :]   # (4, 3)
    # (4, 3)
    local_normals = kin_model.fingertip_normals.value
    # Local origin for normal calculation
    local_origin = jnp.zeros((5, 3), dtype=jnp.float32)
    fingertip_joint_indices = kin_model.fingertip_idx.value  # (4,)

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


    #TODO : visual finger tip points and their normals
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

    #Trace for the transformed contact points
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
        title="Allegro hand (Initial Pose) + Contact Points & Normals (Using Imported FK)",
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    print("Showing plot...")
    fig.show()


if __name__ == "__main__":
    visualize_shadow_initial_contacts_normals()