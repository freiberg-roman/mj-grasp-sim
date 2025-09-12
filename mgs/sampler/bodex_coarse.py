from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from mgs.obj.base import CollisionMeshObject
from mgs.sampler.base import GraspGenerator

# Reuse your helpers if already in scope
from mgs.sampler.helper import (
    farthest_point_sampling,
    find_best_assignment_and_reorder_targets,
    matrix_to_rotation_6d,
    normalize_vector,
    rotation_6d_to_matrix,
)
from mgs.sampler.kin.base import KinematicsModel
from mgs.util.geo.transforms import SE3Pose

# ---------------------------------------------------------------------------
# Geometry constants & knobs (feel free to tweak)
CONTACT_SPHERE_R = 0.01  # 2 cm fingertip "contact sphere"
CAPSULE_R = 0.01  # 2 cm link capsules (pill radius)
PAD = 0.01  # 1 cm pre-grasp padding, like BODex
LOCAL_REGION_RADIUS = 0.10  # 10 cm local region around each seed (unchanged)
POSE_OFFSET_DISTANCE = 0.05  # 5 cm base-offset along normal (unchanged)
TARGET_OFFSET_DISTANCE = 0.02  # 2 cm target offset along normal (unchanged)
COLLISION_SUBSAMPLE = 256  # #points per seed for capsule clearance
OPT_ITERS = 200  # coarse-stage iterations (BODex coarse uses 300)
LR = 5e-3  # AdamW LR
W_CONTACT = 1.0  # weight: padded contact fit
W_ALIGN = 1e-2  # weight: finger normal vs object normal
W_CAPSULE = 10.0  # weight: capsule clearance
W_JLIM = 1e-3  # weight: joint-limit soft penalty

# ---------------------------------------------------------------------------
# Vectorized geometry helpers


def _pairwise_capsule_edges(kin: KinematicsModel) -> jnp.ndarray:
    """Return consecutive joint pairs as edges, shape (E, 2) with joint indices."""
    # Each chain contributes (len(chain)-1) edges of consecutive joints
    edges = []
    for chain in kin.kinematics_graph:  # Shadow: 5 chains
        for i in range(len(chain) - 1):
            edges.append([chain[i], chain[i + 1]])
    return jnp.asarray(edges, dtype=jnp.int32)


def _point_segment_dist(p: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Distance from points p[K,3] to segments (a,b) broadcast to (K,M)."""
    # Shapes: p: (K,3), a: (M,3), b: (M,3)
    ab = b - a  # (M,3)
    ap = p[:, None, :] - a[None, :, :]  # (K,M,3)
    ab2 = jnp.sum(ab * ab, axis=-1)  # (M,)
    # Avoid 0-length segments
    ab2 = jnp.maximum(ab2, 1e-12)
    t = jnp.sum(ap * ab[None, :, :], axis=-1) / ab2[None, :]  # (K,M)
    t = jnp.clip(t, 0.0, 1.0)
    closest = a[None, :, :] + t[..., None] * ab[None, :, :]  # (K,M,3)
    d = jnp.linalg.norm(p[:, None, :] - closest, axis=-1)  # (K,M)
    return d


def _capsule_clearance_min(
    p_local: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray, r: float
) -> jnp.ndarray:
    """Minimum signed clearance of each capsule (a,b) to a point cloud p_local[K,3].
    Returns (M,) with min over K of (||p - seg|| - r)."""
    d = _point_segment_dist(p_local, a, b)  # (K,M)
    min_over_points = jnp.min(d, axis=0)  # (M,)
    return min_over_points - r  # (M,)


def _relu_sq(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.square(jnp.maximum(0.0, x))


# ---------------------------------------------------------------------------
# Core batched loss


def build_local_collision_clouds(
    points_all: jnp.ndarray,
    seeds: jnp.ndarray,
    k_sub: int = COLLISION_SUBSAMPLE,
    radius: float = LOCAL_REGION_RADIUS,
) -> jnp.ndarray:
    """For each seed[b], take k_sub nearest points from points_all into a local cloud."""

    # points_all: (N,3), seeds: (B,3) -> out: (B, k_sub, 3)
    # Compute squared distances BxN
    # For stability, we use top_k on negative distances to get the k smallest
    def one_seed_cloud(seed):
        d2 = jnp.sum((points_all - seed) ** 2, axis=-1)  # (N,)
        # prioritize points within radius, but always take k_sub
        # we penalize farther points less because the capsule margin won't trigger
        vals, idx = jax.lax.top_k(-d2, k=k_sub)
        return points_all[idx]

    return jax.vmap(one_seed_cloud)(seeds)


@nnx.jit
def coarse_loss_and_grads(
    kin_g,
    kin_s,
    rot6d: jnp.ndarray,
    pos: jnp.ndarray,
    joints: jnp.ndarray,
    target_pts: jnp.ndarray,
    target_normals: jnp.ndarray,
    finger_local_pts: jnp.ndarray,
    fingertip_idx: jnp.ndarray,
    finger_dir_local: jnp.ndarray,
    edges: jnp.ndarray,
    clouds_local: jnp.ndarray,
):
    """
    Args:
      rot6d: (B,6), pos: (B,3), joints: (B,D)
      target_pts: (B,F,3), target_normals: (B,F,3)
      finger_local_pts: (F,3) local point of each fingertip (use the first sphere)
      fingertip_idx: (F,) joint indices of fingertip links
      finger_dir_local: (F,3) local "approach" vectors for alignment
      edges: (E,2) capsule joint index pairs
      clouds_local: (B,K,3) local clouds for capsule clearance per seed

    Returns:
      loss scalar, grads for rot6d/pos/joints
    """
    kin = nnx.merge(kin_g, kin_s)
    B = rot6d.shape[0]
    D = kin.num_dofs
    F = fingertip_idx.shape[0]
    E = edges.shape[0]

    base_R = rotation_6d_to_matrix(rot6d)  # (B,3,3)

    # Get per-batch world joint positions by transforming each joint origin [0,0,0]
    joint_origin_local = jnp.zeros((D, 3), dtype=jnp.float32)
    joint_indices_all = jnp.arange(D, dtype=jnp.int32)

    def fk_points(theta):
        g, s = nnx.split(kin)
        return nnx.vmap(
            # forward_kinematic_point_transform(theta, data, joint_indices, g, s)
            # data: (N,3), joint_indices: (N,)
            nnx.partial(
                kin.forward_kinematic_point_transform
            ),  # expose if method is attached
            in_axes=(None, 0, 0, None, None),
        )(theta, joint_origin_local, joint_indices_all, g, s)

    # Some repos attach FK as a free function rather than a method; fall back if needed.
    def fk_points_fallback(theta):
        from mgs.sampler.kin.op import forward_kinematic_point_transform as FK

        g, s = nnx.split(kin)
        return nnx.vmap(FK, in_axes=(None, 0, 0, None, None))(
            theta, joint_origin_local, joint_indices_all, g, s
        )

    try:
        joints_local = nnx.vmap(fk_points, in_axes=(0,))(joints)  # (B,D,3)
    except AttributeError:
        joints_local = nnx.vmap(fk_points_fallback, in_axes=(0,))(joints)

    # World joint positions: apply base transform
    joints_world = (
        jnp.einsum("bij,bdj->bdi", base_R, joints_local) + pos[:, None, :]
    )  # (B,D,3)

    # Fingertip world points (contact spheres)
    R_tip_local = (
        joints_world[jnp.arange(B)[:, None], fingertip_idx[None, :], :] * 0.0
    )  # dummy to carry shape
    # Using local points per fingertip attached to the fingertip joint
    tip_local = finger_local_pts  # (F,3)
    tip_local_b = jnp.broadcast_to(tip_local[None, :, :], (B, F, 3))

    # For fingertip link rotation, approximate by finite difference on transformed normals:
    # We can reuse FK to transform a "normal anchor" point per fingertip, like you do in your current code.
    finger_dir_anchor_local = finger_dir_local  # (F,3)

    # Transform fingertip local points and the anchor points
    def fk_fid(theta, data, idx_vec):
        g, s = nnx.split(kin)
        from mgs.sampler.kin.op import forward_kinematic_point_transform as FK

        return FK(theta, data, idx_vec, g, s)

    # Transform tip points
    g, s = nnx.split(kin)
    tip_local_repeated = jnp.repeat(tip_local[None, :, :], B, axis=0)  # (B,F,3)
    tip_world_local = nnx.vmap(
        nnx.vmap(fk_fid, in_axes=(None, 0, 0)), in_axes=(0, 0, None)  # over F
    )(joints, tip_local_repeated, fingertip_idx)
    # Transform finger-dir anchor points
    dir_local_repeated = jnp.repeat(finger_dir_anchor_local[None, :, :], B, axis=0)
    dir_world_anchor_local = nnx.vmap(
        nnx.vmap(fk_fid, in_axes=(None, 0, 0)), in_axes=(0, 0, None)
    )(joints, dir_local_repeated, fingertip_idx)

    # Bring both to world with base_R,pos
    tip_world = jnp.einsum("bij,bfj->bfi", base_R, tip_world_local) + pos[:, None, :]
    dir_world_anchor = (
        jnp.einsum("bij,bfj->bfi", base_R, dir_world_anchor_local) + pos[:, None, :]
    )
    # Approximate fingertip direction in world as (anchor - joint origin world)
    joint_origin_world = joints_world[jnp.arange(B)[:, None], fingertip_idx[None, :], :]
    finger_dir_world = dir_world_anchor - joint_origin_world  # (B,F,3)
    finger_dir_world = finger_dir_world / (
        jnp.linalg.norm(finger_dir_world, axis=-1, keepdims=True) + 1e-9
    )

    # --- Loss 1: padded contact distance for fingertip spheres
    dist_to_target = jnp.linalg.norm(tip_world - target_pts, axis=-1)  # (B,F)
    residual = dist_to_target - (CONTACT_SPHERE_R + PAD)
    loss_contact = jnp.mean(jnp.square(residual))

    # --- Loss 2: alignment of fingertip directions and object normals
    obj_norm = target_normals / (
        jnp.linalg.norm(target_normals, axis=-1, keepdims=True) + 1e-9
    )
    cos_sim = jnp.sum(obj_norm * finger_dir_world, axis=-1)  # (B,F)
    loss_align = jnp.mean(0.5 * (1.0 - cos_sim))

    # --- Loss 3: capsule clearance to local object clouds
    # Build capsule endpoints in world for each edge (a,b): shape (B,E,3)
    a_idx, b_idx = edges[:, 0], edges[:, 1]
    a_world = joints_world[:, a_idx, :]  # (B,E,3)
    b_world = joints_world[:, b_idx, :]  # (B,E,3)

    def batch_capsule_loss(p_cloud, a, b):
        # p_cloud: (K,3), a: (E,3), b: (E,3)
        dmin = _capsule_clearance_min(p_cloud, a, b, CAPSULE_R)  # (E,)
        # want dmin >= PAD  -> penalize (PAD - dmin)+
        return jnp.mean(_relu_sq(PAD - dmin))

    loss_capsule = jnp.mean(
        jax.vmap(batch_capsule_loss)(clouds_local, a_world, b_world)
    )

    # --- Loss 4: soft joint-limit penalty (keeps joints away from bounds)
    jmin = kin.joint_ranges[:, 0]
    jmax = kin.joint_ranges[:, 1]
    # quadratic walls
    low_violation = jnp.maximum(0.0, jmin[None, :] - joints)
    high_violation = jnp.maximum(0.0, joints - jmax[None, :])
    loss_jlim = jnp.mean(jnp.square(low_violation) + jnp.square(high_violation))

    total = (
        W_CONTACT * loss_contact
        + W_ALIGN * loss_align
        + W_CAPSULE * loss_capsule
        + W_JLIM * loss_jlim
    )
    return total, (loss_contact, loss_align, loss_capsule, loss_jlim)


class _CoarseOptState(nnx.Module):
    def __init__(self, init_rot, init_pos, kin: KinematicsModel, batch: int):
        # base pose
        self.rot = nnx.Param(matrix_to_rotation_6d(init_rot))  # (B,6)
        self.pos = nnx.Param(init_pos)  # (B,3)
        # joint dofs (pre-grasp)
        joints = jnp.copy(kin.init_pregrasp_joint.value)
        self.joints = nnx.Param(
            jnp.broadcast_to(joints[None, :], (batch, joints.shape[0]))
        )


class BODexCoarsePregrasp(GraspGenerator):
    """
    Coarse-stage, padded pre-grasp synthesis:
      - one fingertip sphere per finger targets radius+pad from selected surface points
      - capsule clearance penalty keeps all links >= pad from object
      - optional alignment between fingertip directions and target normals
    """

    def __init__(self, obj: CollisionMeshObject):
        super().__init__(obj)
        import trimesh  # used only to sample surface points/normals; optimization is pure JAX

        self.mesh = trimesh.load_mesh(self.mesh_file_path)

    def update_object(self, obj: CollisionMeshObject):
        import trimesh

        self.mesh = trimesh.load_mesh(obj.obj_file_path)
        return self

    def _edges(self, kin: KinematicsModel):
        if not hasattr(self, "_edges_cache"):
            self._edges_cache = _pairwise_capsule_edges(kin)
        return self._edges_cache

    def generate_grasps(
        self, num: int, gripper: KinematicsModel
    ) -> Tuple[SE3Pose, Dict[str, Any]]:
        import trimesh

        # ---- 1) Sample surface points/normals for seeds/targets (same as your current code)
        NUM_SURFACE_SAMPLES = max(30000, num * 3)
        points, face_idx = trimesh.sample.sample_surface(self.mesh, NUM_SURFACE_SAMPLES)
        normals = self.mesh.face_normals[face_idx]
        points = jnp.array(points, dtype=jnp.float32)
        normals = normalize_vector(jnp.array(normals, dtype=jnp.float32))

        # Seeds
        fps_idx = farthest_point_sampling(points, num)
        seeds = points[fps_idx]  # (B,3)
        seed_normals = normals[fps_idx]  # (B,3)

        # Build target sets: pick F points near each seed
        F = len(gripper.fingertip_idx.value)
        dists = jnp.linalg.norm(seeds[:, None, :] - seeds[None, :, :], axis=-1)
        admissible = dists < LOCAL_REGION_RADIUS
        rng_key = jax.random.PRNGKey(0)
        rand_vals = jax.random.uniform(rng_key, shape=(seeds.shape[0], seeds.shape[0]))
        rand_vals = jnp.where(admissible, rand_vals, -jnp.inf)
        nn_idx = jnp.argsort(rand_vals, axis=1)[:, -F:]
        target_pts = jnp.take(seeds, nn_idx, axis=0)  # (B,F,3)
        target_normals = jnp.take(seed_normals, nn_idx, axis=0)  # (B,F,3)
        target_pts = (
            target_pts + TARGET_OFFSET_DISTANCE * target_normals
        )  # 2cm off along normal

        # ---- 2) Initial base pose (same as your alignment trick)
        z_axis = seed_normals
        sorted_indices = jnp.argsort(dists, axis=1)
        nearest_neighbor_idx = sorted_indices[:, 1]
        x_axis = seeds[nearest_neighbor_idx] - seeds
        x_axis = normalize_vector(x_axis)
        y_axis = jnp.cross(z_axis, x_axis)
        initial_rot = jnp.stack([x_axis, y_axis, z_axis], axis=-1)  # (B,3,3)

        align_rot, align_pos_local = gripper.align_to_approach.value
        align_pos = jnp.einsum("bij,j->bi", initial_rot, align_pos_local)
        initial_rot = jnp.einsum("bij,jk->bik", initial_rot, align_rot)
        initial_pos = seeds + POSE_OFFSET_DISTANCE * seed_normals + align_pos  # (B,3)

        # ---- 3) Build local collision clouds for capsule clearance (pure JAX)
        clouds_local = build_local_collision_clouds(
            points, seeds, COLLISION_SUBSAMPLE, LOCAL_REGION_RADIUS
        )  # (B,K,3)

        # ---- 4) Prepare optimizer state
        B = seeds.shape[0]
        opt_vars = _CoarseOptState(initial_rot, initial_pos, gripper, B)
        tx = optax.adamw(LR)
        opt = nnx.Optimizer(opt_vars, tx)
        kin_g, kin_s = nnx.split(gripper)

        # fixed fingertip model: first local point as the coarse "contact sphere" center
        F_idx = gripper.fingertip_idx.value
        finger_local_pts = gripper.local_fingertip_contact_positions.value[
            jnp.arange(F), 0, :
        ]  # (F,3)
        finger_dir_local = gripper.fingertip_normals.value  # (F,3)
        edges = self._edges(gripper)  # (E,2)

        # utility to clamp joints after each step
        jmin = gripper.joint_ranges.value[:, 0]
        jmax = gripper.joint_ranges.value[:, 1]

        # jit'ed step
        @nnx.jit
        def train_step(
            vars_g,
            vars_s,
            opt_vars: _CoarseOptState,
            target_pts,
            target_normals,
            finger_local_pts,
            F_idx,
            finger_dir_local,
            edges,
            clouds_local,
        ):
            def loss_fn(ov: _CoarseOptState):
                total, parts = coarse_loss_and_grads(
                    vars_g,
                    vars_s,
                    ov.rot.value,
                    ov.pos.value,
                    ov.joints.value,
                    target_pts,
                    target_normals,
                    finger_local_pts,
                    F_idx,
                    finger_dir_local,
                    edges,
                    clouds_local,
                )
                return total, parts

            (loss, parts), grads = nnx.value_and_grad(loss_fn, has_aux=True)(opt_vars)
            return loss, parts, grads

        # ---- 5) Optimize
        hist = []
        for _ in range(OPT_ITERS):
            loss, parts, grads = train_step(
                kin_g,
                kin_s,
                opt_vars,
                target_pts,
                target_normals,
                finger_local_pts,
                F_idx,
                finger_dir_local,
                edges,
                clouds_local,
            )
            opt.update(grads)
            # hard clamp to limits
            opt_vars.joints.value = jnp.clip(
                opt_vars.joints.value, jmin[None, :], jmax[None, :]
            )
            hist.append(loss)

        # ---- 6) Export poses (pre-grasp)
        rot = rotation_6d_to_matrix(opt_vars.rot.value)  # (B,3,3)
        trans = opt_vars.pos.value  # (B,3)
        trans = trans[:, :, None]
        Hs_3x4 = jnp.concatenate([rot, trans], axis=-1)  # (B,3,4)
        last_row = jnp.tile(jnp.array([0, 0, 0, 1])[None, None, :], (B, 1, 1))
        Hs = jnp.concatenate([Hs_3x4, last_row], axis=1)  # (B,4,4)

        # Optional squeeze pose if you later want execution:
        #   x_s = 2*x - x_p  -> not used here, but trivial to compute outside.
        aux_info = {
            "joints": np.array(opt_vars.joints.value),
            "loss_history": np.array([float(l) for l in hist]),
            "loss_parts_last": tuple(float(x) for x in parts),
        }
        return Hs, aux_info
