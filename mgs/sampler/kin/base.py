import jax
import jax.numpy as jnp
from flax import nnx
from abc import ABC
from typing import List
from mgs.sampler.kin.jax_util import (
    quaternion_apply_jax,
    se3_raw_mupltiply,
    quaternion_from_axis_angle,
    similarity_transform,
    transform_points_jax,
)


class KinematicsModel(ABC):
    num_dofs: int
    num_extra_dofs: int
    embedding: nnx.Param
    kinematic_graph: List[List[int]]
    base_to_contact: nnx.Variable
    kinematics_transforms: nnx.Variable
    joint_transforms: nnx.Variable
    joint_ranges: nnx.Variable
    keypoint_offset: nnx.Variable
    lmax: int
    num_channels: int
    fingertip_idx: nnx.Variable
    local_fingertip_contact_positions: nnx.Variable
    fingertip_normals: nnx.Variable
    init_pregrasp_joint: nnx.Variable


@jax.jit
def masked_feature_pcd_transform(
    points: jnp.ndarray,
    Ts: jnp.ndarray,
    mask: jnp.ndarray,
):
    q, t = Ts[..., :4], Ts[..., 4:]
    points = jnp.where(mask[..., None], quaternion_apply_jax(q, points) + t, points)
    return points


@nnx.jit
def kinematic_pcd_transform(
    points: jnp.ndarray,
    theta: jnp.ndarray,
    segmentation: jnp.ndarray,
    gripper_kinematics: KinematicsModel,
):
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
            points = masked_feature_pcd_transform(points, Ts, mask)

    return points


@nnx.jit
def forward_kinematic_point_transform(
    theta: jnp.ndarray,
    local_point: jnp.ndarray,
    joint_idx: jnp.ndarray,
    kin_model: KinematicsModel,
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
        all_link_transforms = all_link_transforms.at[index + 1].set(T_world_joint)
    target_transform_world = all_link_transforms[joint_idx + 1]
    transformed_point = transform_points_jax(local_point, target_transform_world)
    return transformed_point
