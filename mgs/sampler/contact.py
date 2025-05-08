import time
import numpy as np
import jax
import optax
from typing import Tuple, Dict, Any
import trimesh

from mgs.obj.base import CollisionMeshObject
from mgs.sampler.base import GraspGenerator
from mgs.util.geo.transforms import SE3Pose
from itertools import permutations
from flax import nnx
from mgs.sampler.kin.jax_util import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    normalize_vector,
    farthest_point_sampling,
    find_best_assignment_and_reorder_targets,
)
from mgs.sampler.kin.base import KinematicsModel, forward_kinematic_point_transform
import jax.numpy as jnp
import plotly.graph_objects as go

NUM_SURFACE_SAMPLES = 30000
LOCAL_REGION_RADIUS = 0.10  # 10 cm
TARGET_OFFSET_DISTANCE = 0.02  # 3 cm offset along normal
POSE_OFFSET_DISTANCE = 0.05  # 8 cm offset for base pose along normal


class OptState(nnx.Module):
    def __init__(
        self,
        init_rot,
        init_pos,
        kin: KinematicsModel,
    ):
        rot = matrix_to_rotation_6d(init_rot)
        pos = init_pos

        joints = jnp.copy(kin.init_pregrasp_joint.value)
        batch_size = rot.shape[0]
        joints_batch = jnp.broadcast_to(
            joints[None, ...],
            shape=(batch_size, joints.shape[0]),
        )

        self.rot = nnx.Param(rot)
        self.pos = nnx.Param(pos)
        self.joints = nnx.Param(joints_batch)


class Trainer:
    def __init__(self, kin, init_rot_batch, init_pos_batch):
        self.kin = kin
        self.tx = optax.adamw(0.005)
        self.to_opt = OptState(
            init_rot=init_rot_batch,
            init_pos=init_pos_batch,
            kin=kin,
        )
        self.optimizer = nnx.Optimizer(self.to_opt, self.tx)
        self.train_graph, self.train_state = nnx.split(
            (self.kin, self.optimizer, self.to_opt)
        )
        self.permutation_idx = jnp.asarray(
            list(permutations([i for i in range(len(kin.fingertip_idx))]))
        )

    def train_step(self, input, target: jnp.ndarray):
        loss, self.train_state = update(
            self.train_graph, self.train_state, input, target, self.permutation_idx
        )
        return loss

    def reset(self, init_rot_batch, init_pos_batch):
        self.to_opt.pos.value = init_pos_batch
        self.to_opt.rot.value = matrix_to_rotation_6d(init_rot_batch)

        joints = jnp.copy(self.kin.init_pregrasp_joint.value)
        batch_size = init_rot_batch.shape[0]
        joints_batch = jnp.broadcast_to(
            joints[None, ...],
            shape=(batch_size, joints.shape[0]),
        )
        self.to_opt.joints.value = joints_batch
        self.train_graph, self.train_state = nnx.split(
            (self.kin, self.optimizer, self.to_opt)
        )


@nnx.jit
def update(
    graph,
    state,
    gripper_contact_positions,
    target_surface_positions,
    permutations,
):
    kin, optimizer, to_opt = nnx.merge(graph, state)
    kin: KinematicsModel = kin
    batch_surface_points, batch_surface_normals = target_surface_positions
    joint_origin = jnp.zeros_like(gripper_contact_positions)
    contact_neg_normal = kin.fingertip_normals.value

    contact_idx = kin.fingertip_idx.value
    forward_kin = jnp.stack(
        [gripper_contact_positions, joint_origin, contact_neg_normal], axis=0
    )


    def loss_fn(to_opt, i):
        transformed_values = nnx.vmap(
            forward_kinematic_point_transform,
            in_axes=(None, 0, None, None),  # over stack
        )(
            to_opt.joints.value,
            forward_kin,
            contact_idx,
            kin,
        )
        transformed_values = (
            jnp.einsum(
                "ij,sdj->sdi",
                rotation_6d_to_matrix(to_opt.rot.value),
                transformed_values,
            )
            + to_opt.pos.value[None, None, :]
        )
        positions = transformed_values[0, ...]
        joint_origin = transformed_values[1, ...]
        positions_normals = transformed_values[2, ...]
        finger_normals = positions_normals - joint_origin

        cos_sim = jnp.sum(batch_surface_normals[i] * finger_normals, axis=-1)
        loss_cos = jnp.mean(0.5 * (1 - cos_sim))

        target_points = find_best_assignment_and_reorder_targets(
            positions,
            batch_surface_points[i],
            permutations,
        )

        loss = jnp.mean((target_points - positions) ** 2) + 0.001 * loss_cos
        return loss

    grad_fn = nnx.value_and_grad(loss_fn)
    losses, grads = nnx.vmap(grad_fn)(
        to_opt,
        jnp.arange(
            batch_surface_points.shape[0],
        ),
    )
    optimizer.update(grads)
    to_opt.joints.value = jnp.clip(
        to_opt.joints.value,
        min=kin.joint_ranges[..., 0],
        max=kin.joint_ranges[..., 1],
    )

    return losses, nnx.state((kin, optimizer, to_opt))


class ContactBasedDiff(GraspGenerator):
    """
    Generates antipodal grasps by sampling points on the object surface,
    finding opposing points via ray casting, and calculating the required
    gripper width and pose.
    """

    def __init__(self, object: CollisionMeshObject):
        super().__init__(object)
        self.trainer = None
        self.mesh: trimesh.Trimesh = trimesh.load_mesh(self.mesh_file_path)

    def update_object(self, object: CollisionMeshObject):
        self.mesh = trimesh.load_mesh(object.obj_file_path)
        return self

    def generate_grasps(
        self, num: int, gripper: KinematicsModel
    ) -> Tuple[SE3Pose, Dict[str, Any]]:
        points, face_idx = trimesh.sample.sample_surface(
            self.mesh,
            max(NUM_SURFACE_SAMPLES, num * 3),
        )
        normals = self.mesh.face_normals[face_idx]

        points = jnp.array(points)
        normals = jnp.array(normals)
        normals = normalize_vector(normals)

        # select seed poitns
        fps_idx = farthest_point_sampling(points, num)
        seeds = points[fps_idx]
        seed_normals = normals[fps_idx]

        dists = jnp.linalg.norm(seeds[:, None, :] - seeds[None, :, :], axis=-1)
        admissable_target_positions = dists < LOCAL_REGION_RADIUS
        num_contact_points = len(gripper.fingertip_idx)
        rng_key = jax.random.PRNGKey(0)

        rand_vals = jax.random.uniform(
            rng_key, shape=(seeds.shape[0], seeds.shape[0]))
        rand_vals = jnp.where(admissable_target_positions, rand_vals, -jnp.inf)
        random_selected_idx = jnp.argsort(rand_vals, axis=1)[
            :, -num_contact_points:]
        contact_points_for_seeds = jnp.take(seeds, random_selected_idx, axis=0)
        contact_points_normals = jnp.take(
            seed_normals, random_selected_idx, axis=0)
        contact_points_for_seeds_offset = (
            contact_points_for_seeds + TARGET_OFFSET_DISTANCE * contact_points_normals
        )

        z_axis = seed_normals
        sorted_indices = jnp.argsort(dists, axis=1)
        nearest_neighbor_idx = sorted_indices[:, 1]
        x_axis = seeds[nearest_neighbor_idx] - seeds
        x_axis = normalize_vector(x_axis)
        y_axis = jnp.cross(z_axis, x_axis)

        (align_rot, align_pos) = gripper.align_to_approach.value
        initial_rotations = jnp.stack([x_axis, y_axis, z_axis], axis=-1)
        align_pos = jnp.einsum("...ij,j->...i", initial_rotations, align_pos)
        initial_rotations = jnp.einsum(
            "...ij,jk->...ik", initial_rotations, align_rot)
        initial_positions = seeds + POSE_OFFSET_DISTANCE * seed_normals
        initial_positions = initial_positions + align_pos

        # Optimization
        if self.trainer is None:
            self.trainer = Trainer(
                gripper,
                init_rot_batch=initial_rotations,
                init_pos_batch=initial_positions,
            )
        self.trainer.reset(initial_rotations, initial_positions)
        trainer = self.trainer

        num_possible_fingertips = gripper.local_fingertip_contact_positions.value.shape[
            1
        ]
        idx = jax.random.randint(
            rng_key,
            shape=(num_contact_points,),
            minval=0,
            maxval=num_possible_fingertips,
        )
        transformed_points = nnx.vmap(
            nnx.vmap(forward_kinematic_point_transform,
                     in_axes=(None, 0, 0, None)),
            in_axes=(0, None, None, None),
        )(
            trainer.to_opt.joints.value,
            gripper.local_fingertip_contact_positions[
                jnp.arange(num_contact_points), idx, :
            ],
            gripper.fingertip_idx,
            gripper,
        )
        transformed_points = (
            jnp.einsum("bij, bnj -> bni", initial_rotations,
                       transformed_points)
            + initial_positions[:, None, :]
        )

        permutation_idx = list(
            permutations([i for i in range(len(gripper.fingertip_idx))])
        )
        permutation_idx = jnp.asarray(permutation_idx)
        target_points = nnx.vmap(
            find_best_assignment_and_reorder_targets, in_axes=(0, 0, None)
        )(
            transformed_points,
            contact_points_for_seeds_offset,
            permutation_idx,
        )

        #TODO visual target contact points on the object
        # intial poses + joints related to the object

        # Visualize target contact points on the object (before optimization)
        # Create figure
        fig = go.Figure()
        
        # Add mesh as wireframe
        mesh_vertices = np.array(self.mesh.vertices)
        mesh_faces = np.array(self.mesh.faces)
        x, y, z = mesh_vertices.T
        i, j, k = mesh_faces.T
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, 
                                color='lightgray', opacity=0.5))
        
        # Add target contact points
        print(contact_points_for_seeds_offset.shape)
        contact_pts = contact_points_for_seeds_offset[0]  # For the first grasp
        fig.add_trace(go.Scatter3d(x=target_points[0, :, 0], 
                                    y=target_points[0, :, 1], 
                                    z=target_points[0, :, 2],
                                    mode='markers',
                                    marker=dict(size=8, color='blue'),
                                    name='Target Contact Points'))
        
        # Add normals at contact points
        normals = contact_points_normals[0]
        scale = 0.03  # Scale for normal visualization
        for i, (pt, norm) in enumerate(zip(contact_pts, normals)):
            end_pt = pt + scale * norm
            fig.add_trace(go.Scatter3d(
                x=[pt[0], end_pt[0]], 
                y=[pt[1], end_pt[1]], 
                z=[pt[2], end_pt[2]],
                mode='lines',
                line=dict(color='red', width=4),
                name=f'Target Normal {i}'
            ))
        
        # Add initial finger contact points
        init_contacts = transformed_points[0]
        fig.add_trace(go.Scatter3d(
            x=init_contacts[:, 0],
            y=init_contacts[:, 1],
            z=init_contacts[:, 2],
            mode='markers',
            marker=dict(size=8, color='green'),
            name='Initial Finger Positions'
        ))
        
        
        
        for i in range(150):
            trainer.train_step(
                gripper.local_fingertip_contact_positions[
                    jnp.arange(num_contact_points), idx, :
                ],
                (target_points, contact_points_normals),
            )

        _, _ , opt_state = nnx.merge(trainer.train_graph, trainer.train_state)

        #TODO visualize the same after optimization
    
        transformed_points = nnx.vmap(
            nnx.vmap(forward_kinematic_point_transform,
                     in_axes=(None, 0, 0, None)),
            in_axes=(0, None, None, None),
        )(
            opt_state.joints.value,
            gripper.local_fingertip_contact_positions[
                jnp.arange(num_contact_points), idx, :
            ],
            gripper.fingertip_idx,
            gripper,
        )
        transformed_points = (
            jnp.einsum("bij, bnj -> bni", rotation_6d_to_matrix(opt_state.rot.value),
                       transformed_points)
            + opt_state.pos.value[:, None, :]
        )

        transformed_normals = nnx.vmap(
            nnx.vmap(forward_kinematic_point_transform,
                     in_axes=(None, 0, 0, None)),
            in_axes=(0, None, None, None),
        )(
            opt_state.joints.value,
            gripper.fingertip_normals.value,
            gripper.fingertip_idx,
            gripper,
        )
        transformed_normals = (
            jnp.einsum("bij, bnj -> bni", rotation_6d_to_matrix(opt_state.rot.value),
                       transformed_normals)
            + opt_state.pos.value[:, None, :]
        )

        fig.add_trace(go.Scatter3d(x=transformed_points[0, :, 0], 
                                    y=transformed_points[0, :, 1], 
                                    z=transformed_points[0, :, 2],
                                    mode='markers',
                                    marker=dict(size=8, color='purple'),
                                    name='Optimized Contact Points'))
        
        for i, (pt, norm) in enumerate(zip(transformed_points[0], transformed_normals[0])):
            end_pt = pt + scale * norm
            fig.add_trace(go.Scatter3d(
                x=[pt[0], end_pt[0]], 
                y=[pt[1], end_pt[1]], 
                z=[pt[2], end_pt[2]],
                mode='lines',
                line=dict(color='pink', width=4),
                name=f'Optimized Contact Normal {i}'
        ))
        
        
        fig.show()

        

        rot = rotation_6d_to_matrix(opt_state.rot.value)
        trans = opt_state.pos.value
        joints = np.array(opt_state.joints.value)
        trans = trans[:, :, None]  # reshape to (num, 3, 1)
        Hs_3x4 = jnp.concatenate([rot, trans], axis=-1)

        last_row = jnp.tile(jnp.array([0, 0, 0, 1])[
                            None, None, :], (num, 1, 1))

        Hs = jnp.concatenate([Hs_3x4, last_row], axis=1)
        aux_info = {"joints": joints}
        return Hs, aux_info


