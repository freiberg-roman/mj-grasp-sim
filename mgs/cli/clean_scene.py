import jax
import jax.numpy as jnp
from flax import nnx

from mgs.sampler.kin.base import KinematicsModel
from mgs.sampler.kin.op import forward_kinematic_point_transform


def compute_grasp_centers(
    poses: jnp.ndarray, joints: jnp.ndarray, kinematics: KinematicsModel
):
    """Compute grasp centers + in-bound mask using fingertip joint world positions.

    We mirror the logic in `check_grasps_in_bounds` (clean_scene.py) but return
    the centers explicitly. Fingertip local points are taken as the joint
    origin (zero vector) for each fingertip DOF.
    """
    g, s = nnx.split(kinematics)
    num_fingertips = kinematics.local_fingertip_contact_positions.value.shape[0]
    # One point (origin) per fingertip joint to approximate fingertip position
    fingertip_local_points = jnp.zeros((num_fingertips, 3), dtype=jnp.float32)
    fingertip_joint_indices = kinematics.fingertip_idx.value  # (K,)

    # Transform fingertip local points for every grasp (batch over joints & points)
    transformed = nnx.vmap(  # batch over grasps
        nnx.vmap(  # batch over fingertip points
            forward_kinematic_point_transform,
            in_axes=(None, 0, 0, None, None),
        ),
        in_axes=(0, None, None, None, None),
    )(
        joints, fingertip_local_points, fingertip_joint_indices, g, s
    )  # (B,K,3)

    # Apply SE3 pose (rotation + translation) to points
    transformed_world = (
        jnp.einsum("bij,bnj->bni", poses[:, :3, :3], transformed)
        + poses[:, :3, 3][..., None, :]
    )  # (B,K,3)

    centers = jnp.mean(transformed_world, axis=1)  # (B,3)
    in_bound = (
        (centers[:, 0] < 0.20)
        & (centers[:, 0] > -0.20)
        & (centers[:, 1] < 0.20)
        & (centers[:, 1] > -0.20)
    )  # (B,)
    return in_bound
