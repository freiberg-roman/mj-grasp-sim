import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative
from mgs.sampler.kin.allegro import AllegroKinematicsModel
from mgs.sampler.kin.base import KinematicsModel, kinematic_pcd_transform

 
from mgs.sampler.kin.jax_util import (
    quaternion_apply_jax,
    quaternion_from_axis_angle,
    se3_raw_mupltiply,
    similarity_transform,
)


def quat_to_rot_mat(q: jnp.ndarray) -> jnp.ndarray:
    q = q / jnp.linalg.norm(q)
    w, x, y, z = q[0], q[1], q[2], q[3]
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
 
    R = jnp.array(
        [
            [ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz],
        ]
    )
    return R
 
 
def point_transform(
    points: jnp.ndarray,
    Ts: jnp.ndarray,
    _: jnp.ndarray,
):
    q, t = Ts[..., :4], Ts[..., 4:]
    return quaternion_apply_jax(q, points) + t
 
 
 
def id_transform(
    _: jnp.ndarray,
    Ts: jnp.ndarray,
    Ts_world: jnp.ndarray,
):
    return se3_raw_mupltiply(Ts, Ts_world)
 
 
def kinematic_transform(
    transform,
    data: jnp.ndarray,
    theta: jnp.ndarray,
    segmentation: jnp.ndarray,
    gripper_kinematics : KinematicsModel
):
    # unsqueeze segmentation to the data dimensions
    additional_data_shapes = len(data.shape) - 1
    segmentation = jnp.expand_dims(
        segmentation,
        axis=[
            i
            for i in range(
                -1,
                -additional_data_shapes - 1,
                -1,
            )
        ],
    )

    for chain in gripper_kinematics.kinematics_graph:
        current_transform = gripper_kinematics.base_to_contact
        current_delta = jnp.array([1.0, 0, 0, 0, 0, 0, 0])
 
        for index in chain:
            current_transform = se3_raw_mupltiply(
                se3_raw_mupltiply(current_transform, current_delta),
                gripper_kinematics.kinematics_transforms[index],
            )
            joint_theta = theta[index]
            translation = gripper_kinematics.joint_transforms[index][:3] * joint_theta
 
            joint_axis = gripper_kinematics.joint_transforms[index][3:]
            axis_norm = jnp.linalg.norm(joint_axis)
            rotation_quat = jnp.where(
                axis_norm > 0,
                quaternion_from_axis_angle(joint_axis, joint_theta),
                jnp.array([1.0, 0.0, 0.0, 0.0]),
            )
            current_delta = jnp.concatenate([rotation_quat, translation], axis=-1)
            mask = segmentation[index]
            Ts = similarity_transform(current_transform, current_delta)
            data = jnp.where(
                mask,
                transform(data, Ts, current_transform),
                data,
            )
 
    return data
 

def joint_segmentation(kin):
    segmentation = jnp.full(
        shape=(kin.num_dofs, kin.num_dofs),
        fill_value=False,
    )
    counter = 0
    for chain in kin.kinematics_graph:
        for i in range(len(chain)):
            segmentation = segmentation.at[counter, chain[i:]].set(True)
            counter += 1
    return segmentation
 
 
def kinematic_frames(theta, kin):
    """
    Returns
        R     : (num_dofs, 3, 3)  world-space rotation matrix per joint
        origin: (num_dofs, 3)     world-space position per joint
    """
    seg = joint_segmentation(kin)
    data = jnp.zeros(shape=(len(seg), 7))
    Ts = kinematic_transform(id_transform, data, theta, seg, kin)
    quats = Ts[..., :4]
    joints = Ts[..., 4:]
    R = jax.vmap(quat_to_rot_mat)(quats)
    return R, joints

def viz_point_clouds(
    point_clouds,
    frames=None,
    scale=0.01,
    color_scheme="Plotly",
    show=True,
):
    """
    Visualise multiple point-clouds (each possibly of different size)
    and an optional set of coordinate frames.
 
    Parameters
    ----------
    point_clouds : Sequence[np.ndarray]
        Iterable of (Ni, 3) arrays.
    frames : Sequence[tuple[np.ndarray, np.ndarray]] | None
        Iterable of (R, p) pairs, where
            R : (3, 3) rotation / orientation matrix
            p : (3,)   position of the frame origin
        If None, no frames are shown.
    scale : float, default 0.01
        Half-length of the axis arrows that depict each frame.
    color_scheme : str | Sequence[str], default "Plotly"
        Either the name of a Plotly **qualitative** palette
        (e.g. "Plotly", "D3", "Dark24"), or a list of HEX/RGB strings.
    show : bool, default True
        Whether to call ``fig.show()`` before returning.
 
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The figure, so you can further tweak or save it.
    """
    # ── set up colours (one per cloud) ──────────────────────────────
    if isinstance(color_scheme, str):
        palette = qualitative.__dict__.get(color_scheme, qualitative.Plotly)
    else:
        palette = color_scheme
    if len(palette) < len(point_clouds):
        # repeat colours if the palette is shorter than the number of clouds
        palette = (palette * (len(point_clouds) // len(palette) + 1))[
            : len(point_clouds)
        ]
 
    fig = go.Figure()
 
    #── plot each point-cloud ───────────────────────────────────────
    for idx, cloud in enumerate(point_clouds):
        cloud = np.asarray(cloud)
        fig.add_trace(
            go.Scatter3d(
                x=cloud[:, 0],
                y=cloud[:, 1],
                z=cloud[:, 2],
                mode="markers",
                marker=dict(size=3, color=palette[idx], opacity=0.85),
                name=f"cloud {idx}",
            )
        )
 
    # ── plot frames (tiny RGB arrows) ───────────────────────────────
    if frames is not None:
        for R, p in frames:
            R = np.asarray(R)
            p = np.asarray(p)
            if R.shape != (3, 3) or p.shape != (3,):
                raise ValueError("Each frame must be (R (3×3), p (3,))")
            axes = [
                ("red", R[:, 0]),  # x
                ("green", R[:, 1]),  # y
                ("blue", R[:, 2]),
            ]  # z
            for colour, axis_vec in axes:
                q = p + axis_vec * scale
                fig.add_trace(
                    go.Scatter3d(
                        x=[p[0], q[0]],
                        y=[p[1], q[1]],
                        z=[p[2], q[2]],
                        mode="lines",
                        line=dict(color=colour, width=4),
                        showlegend=False,
                    )
                )
 
    # ── layout & return ────────────────────────────────────────────
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )
    if show:
        fig.show()
 
    return fig


def visualize_frames_and_point_cloud():
    kin = AllegroKinematicsModel()
    theta = jnp.array(
        kin.init_pregrasp_joint.value
    )

    R, joints = kinematic_frames(theta, kin)
    assert(R.shape[0] == joints.shape[0])

    frames = [(R[i], joints[i]) for i in range(R.shape[0])]


    ALLEGRO_NPZ_FILE = "./mgs/sampler/kin/allegro.npz"
    NUM_POINTS_VIS = 2000

    print(f"Loading gripper point cloud from: {ALLEGRO_NPZ_FILE}")
    raw = np.load(ALLEGRO_NPZ_FILE, allow_pickle=True)
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

    points_vis_jax = jnp.array(points_vis_np)
    segmentations_jax = jnp.array(segmentations_np)

    points_vis_transformed_jax = kinematic_pcd_transform(
        points_vis_jax, theta, segmentations_jax, kin
    )

    points_vis_transformed_np = np.array(points_vis_transformed_jax)

    viz_point_clouds([points_vis_transformed_np], frames)    


if __name__=="__main__":
    visualize_frames_and_point_cloud()




    
