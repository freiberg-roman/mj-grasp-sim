import jax
import jax.numpy as jnp
from typing import List
from flax import nnx
from abc import ABC
from mgs.operations import (
    se3_raw_mupltiply,
    quaternion_from_axis_angle,
    quaternion_apply_jax,
    similarity_transform,
)

# This constant is required for jax GPU acceleration
MAX_DOF = 22


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


class LeapHandKinematicsModel(nnx.Module, KinematicsModel):
    def __init__(self, lmax=2, num_emb_channels=64):
        self.lmax = lmax
        self.num_channels = num_emb_channels
        self.num_dofs = 16
        self.num_extra_dofs = 0
        self.embedding = nnx.Param(
            jax.random.normal(
                nnx.Rngs(0)(),
                shape=(self.num_dofs, (lmax + 1) ** 2, num_emb_channels),
            )
        )
        self.kinematics_graph = [
            [0, 1, 2, 3],  # index finger
            [4, 5, 6, 7],  # middle finger
            [8, 9, 10, 11],  # ring finger
            [12, 13, 14, 15],  # thumb finger
        ]
        self.base_to_contact = nnx.Variable(
            jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )

        self.kinematics_transforms = nnx.Variable(
            jnp.array(
                [
                    # index finger
                    [0.500003, 0.5, 0.5, -0.499997, -0.007, 0.023, -0.0187],
                    [0.500003, -0.5, -0.499997, 0.5, -0.0122, 0.0381, 0.0145],
                    [0.500003, 0.5, -0.5, 0.499997, 0.015, 0.0143, -0.013],
                    [1, 0, 0, 0, 0, -0.0361, 0.0002],
                    # middle finger
                    [0.500003, 0.5, 0.5, -0.499997, -0.0071, -0.0224, -0.0187],
                    [0.500003, -0.5, -0.499997, 0.5, -0.0122, 0.0381, 0.0145],
                    [0.500003, 0.5, -0.5, 0.499997, 0.015, 0.0143, -0.013],
                    [1, 0, 0, 0, 0, -0.0361, 0.0002],
                    # ring finger
                    [0.500003, 0.5, 0.5, -0.499997, -0.00709, -0.0678, -0.0187],
                    [0.500003, -0.5, -0.499997, 0.5, -0.0122, 0.0381, 0.0145],
                    [0.500003, 0.5, -0.5, 0.499997, 0.015, 0.0143, -0.013],
                    [1.0, 0, 0, 0, 0, -0.03609, 0.0002],
                    # thumb
                    [0.707109, 0, 0.707105, 0, -0.0693, -0.0012, -0.0216],
                    [0.500003, 0.5, -0.5, 0.499997, 0, 0.0143, -0.013],
                    [0.707109, -0.707105, 0, 0, 0, 0.0145, -0.017],
                    [0, 0, 0, 1, 0, 0.0466, 0.0002],
                ]
            )
        )
        self.keypoint_offset = nnx.Variable(
            jnp.array(
                [
                    # index finger
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    # middle finger
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    # ring finger
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    # thumb
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            )
        )
        # a six-dimensional vector with a translational direction and rotation axis
        self.joint_transforms = nnx.Variable(
            jnp.array(
                [
                    [0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, -1],
                ],
                dtype=jnp.float32,
            )
        )
        # note these ranges are fo the leap hand in dex
        # MuJoCo ranges differ for the thumb
        self.joint_ranges = nnx.Variable(
            jnp.array(
                [
                    [-0.314, 2.23],  # mcp
                    [-1.047, 1.047],  # rot
                    [-0.506, 1.885],  # pip
                    [-0.366, 2.042],  # dip
                    [-0.314, 2.23],
                    [-1.047, 1.047],
                    [-0.506, 1.885],
                    [-0.366, 2.042],
                    [-0.314, 2.23],
                    [-1.047, 1.047],
                    [-0.506, 1.885],
                    [-0.366, 2.042],
                    # thumb
                    [-0.349, 2.094],
                    [-0.47, 2.443],
                    [-1.2, 1.9],
                    [-1.34, 1.88],
                ]
            )
        )


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
