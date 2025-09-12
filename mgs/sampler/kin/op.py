import jax
import jax.numpy as jnp
from flax import nnx

from mgs.sampler.kin.se3 import (
    quaternion_from_axis_angle,
    se3_raw_mupltiply,
    transform_points_jax,
)


@jax.jit
def forward_kinematic_point_transform(
    theta: jnp.ndarray,
    local_point: jnp.ndarray,
    joint_idx: jnp.ndarray,
    kin_g,
    kin_s,
) -> jnp.ndarray:
    """
    Compat version (old semantics):
    - theta: (D,) joint angles
    - local_point: (3,)
    - joint_idx: () scalar int – index of the link the point is attached to
    - kin_g, kin_s: nnx.split(kinematics)

    Returns the local_point transformed to world using the world transform of joint_idx.
    """
    kin = nnx.merge(kin_g, kin_s)

    # Build parent map once (works with jit: static-size loops over D)
    D = kin.joint_transforms.shape[0]
    all_link_T = jnp.zeros((D + 1, 7), dtype=jnp.float32)
    identity_tf = jnp.array([1.0, 0, 0, 0, 0, 0, 0], dtype=jnp.float32)
    all_link_T = all_link_T.at[0].set(identity_tf)

    # parent map from kinematic_graph
    parent = jnp.full((D,), -1, dtype=jnp.int32)

    # (while loops are awkward in jit; do a static python loop – D is small)
    def _build_parent(parent_init):
        p = parent_init
        for chain in kin.kinematics_graph:
            p = p.at[chain[0]].set(-1)
            for a, b in zip(chain[:-1], chain[1:]):
                p = p.at[b].set(a)
        return p

    parent = _build_parent(parent)

    # Forward chain – compute world transform for each joint
    def body_fun(i, carry):
        allT = carry
        par = parent[i]
        T_world_parent = jax.lax.cond(par < 0, lambda: allT[0], lambda: allT[par + 1])
        T_parent_static = kin.kinematics_transforms[i]
        jt = theta[i]
        trans = kin.joint_transforms[i, :3] * jt
        axis = kin.joint_transforms[i, 3:]
        q = quaternion_from_axis_angle(axis, jt)
        T_dyn = jnp.concatenate([q, trans], axis=-1)
        T_world_joint = se3_raw_mupltiply(
            se3_raw_mupltiply(T_world_parent, T_parent_static), T_dyn
        )
        allT = allT.at[i + 1].set(T_world_joint)
        return allT

    all_link_T = jax.lax.fori_loop(0, D, body_fun, all_link_T)

    # Transform local point by the selected joint’s world transform
    T_world = all_link_T[joint_idx + 1]
    return transform_points_jax(local_point, T_world)
