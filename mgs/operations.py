from flax import nnx
import jax.numpy as jnp
import numpy as np


def theta_norm_to_denorm(
    theta_norm: jnp.ndarray, min_ranges: jnp.ndarray, max_ranges: jnp.ndarray
):
    theta_denorm = (theta_norm * 0.5 + 0.5) * (max_ranges - min_ranges) - min_ranges
    return theta_denorm


def theta_denorm_to_norm(
    theta_denorm: jnp.ndarray, min_ranges: jnp.ndarray, max_ranges: jnp.ndarray
):
    theta_norm = (theta_denorm + min_ranges) / (max_ranges - min_ranges) * 2 - 1
    return theta_norm


def quaternion_raw_multiply_jax(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.
    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.
    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = jnp.unstack(a, axis=-1)
    bw, bx, by, bz = jnp.unstack(b, axis=-1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return jnp.stack((ow, ox, oy, oz), -1)


def quaternion_invert_jax(quaternion: jnp.ndarray) -> jnp.ndarray:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.
    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).
    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    scaling = jnp.array([1, -1, -1, -1])
    return quaternion * scaling


def axis_angle_to_quaternion_jax(axis_angle: jnp.ndarray) -> jnp.ndarray:
    angles = jnp.linalg.norm(axis_angle, axis=-1, keepdims=True)  # Shape (..., 1)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = jnp.abs(angles) < eps  # Shape (..., 1), bool

    angles_squeezed = angles.squeeze(-1)  # Shape (...)
    half_angles_squeezed = half_angles.squeeze(-1)  # Shape (...)
    small_angles_squeezed = small_angles.squeeze(-1)  # Shape (...)

    # Compute sin_half_angles_over_angles avoiding division by zero
    sin_half_angles_over_angles = jnp.where(
        small_angles_squeezed,
        0.5 - (angles_squeezed**2) / 48,
        jnp.sin(half_angles_squeezed) / angles_squeezed,
    )  # Shape (...)

    # Expand back to shape (..., 1)
    sin_half_angles_over_angles = sin_half_angles_over_angles[..., None]

    quaternions = jnp.concatenate(
        [jnp.cos(half_angles), axis_angle * sin_half_angles_over_angles], axis=-1
    )
    return quaternions


def quaternion_apply_jax(quaternion: jnp.ndarray, point: jnp.ndarray) -> jnp.ndarray:
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.
    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).
    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.shape[-1] != 3:
        raise ValueError(f"Points are not in 3D, {point.shape}.")
    real_parts = jnp.zeros(point.shape[:-1] + (1,))
    point_as_quaternion = jnp.concatenate((real_parts, point), -1)
    out = quaternion_raw_multiply_jax(
        quaternion_raw_multiply_jax(quaternion, point_as_quaternion),
        quaternion_invert_jax(quaternion),
    )
    return out[..., 1:]


def transform_points_jax(points: jnp.ndarray, Ts: jnp.ndarray) -> jnp.ndarray:
    q, t = Ts[..., :4], Ts[..., 4:]
    points = quaternion_apply_jax(q, points) + t
    return points


def se3_raw_mupltiply(Ts_one: jnp.ndarray, Ts_two: jnp.ndarray) -> jnp.ndarray:
    q_one, _ = Ts_one[..., :4], Ts_one[..., 4:]
    q_two, t_two = Ts_two[..., :4], Ts_two[..., 4:]
    q = quaternion_raw_multiply_jax(q_one, q_two)
    t = transform_points_jax(t_two, Ts_one)
    return jnp.concatenate([q, t], axis=-1)


def se3_invert(Ts: jnp.ndarray) -> jnp.ndarray:
    q, t = Ts[..., :4], Ts[..., 4:]
    q_inv = quaternion_invert_jax(q)
    t = -quaternion_apply_jax(q_inv, t)
    return jnp.concatenate([q_inv, t], axis=-1)


def similarity_transform(Ts_sim: jnp.ndarray, Ts: jnp.ndarray) -> jnp.ndarray:
    return se3_raw_mupltiply(se3_raw_mupltiply(Ts_sim, Ts), se3_invert(Ts_sim))


def quaternion_from_axis_angle(axis: jnp.ndarray, angle: jnp.ndarray) -> jnp.ndarray:
    axis = axis / jnp.linalg.norm(axis)
    half_angle = angle / 2.0
    s = jnp.sin(half_angle)
    q = jnp.array([jnp.cos(half_angle), axis[0] * s, axis[1] * s, axis[2] * s])
    return q


def rotation_translation_vector_field(
    gripper_pcd: jnp.ndarray, segmentation: jnp.ndarray, gripper_type, dofs, dof_type
):
    translation_field = jnp.zeros_like(gripper_pcd)
    rotation_field = jnp.zeros_like(gripper_pcd)
    if gripper_type == 0:
        translation_field = jnp.where(
            segmentation[1][..., None], jnp.array([[4.0, 0, 0]]), translation_field
        )  # left finger
        translation_field = jnp.where(
            segmentation[2][..., None], jnp.array([[-4.0, 0, 0]]), translation_field
        )  # right finger

    return rotation_field, translation_field


def rotation_6d_to_matrix(d6: jnp.ndarray) -> jnp.ndarray:
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    b1 = a1 / jnp.linalg.norm(a1, axis=-1, keepdims=True)
    dot_product = jnp.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot_product * b1
    b2 = b2 / jnp.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = jnp.cross(b1, b2)
    return jnp.stack((b1, b2, b3), axis=-2)


def matrix_to_rotation_6d(matrix: jnp.ndarray) -> jnp.ndarray:
    batch_shape = matrix.shape[:-2]
    return matrix[..., :2, :].reshape(*batch_shape, 6)
