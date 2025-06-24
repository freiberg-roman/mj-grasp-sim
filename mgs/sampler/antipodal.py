# Copyright (c) 2025 Robert Bosch GmbH
# Author: Roman Freiberg
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import Tuple, Dict, Any

import numpy as np
import trimesh
from scipy.stats import vonmises_fisher

from mgs.obj.base import CollisionMeshObject
from mgs.sampler.base import GraspGenerator
from mgs.util.geo.transforms import SE3Pose


class AntipodalGraspGenerator(GraspGenerator):
    """
    Generates antipodal grasps by sampling points on the object surface,
    finding opposing points via ray casting, and calculating the required
    gripper width and pose.
    """

    def __init__(self, object: CollisionMeshObject):
        super().__init__(object)
        self.mesh: trimesh.Trimesh = None
        self.scale: float = 1.0
        self.offset: np.ndarray = np.array([0.0, 0.0, 0.0])

    def denormalize_points(self, points: np.ndarray) -> np.ndarray:
        """Denormalizes points from the mesh's local frame to the original frame."""
        if points.ndim == 1:
            return (points - self.offset) * self.scale
        return (points - self.offset[np.newaxis, :]) * self.scale

    def denorm_grasp_pose(self, Hs: np.ndarray) -> np.ndarray:
        """Denormalizes the grasp pose transformation matrix."""
        # Denormalize the translation (center point)
        position = Hs[..., :3, 3]
        position = self.denormalize_points(position)
        Hs[..., :3, 3] = position
        # Rotation is not affected by uniform scale or translation offset
        return Hs

    def normalize_load(self):
        """Loads and normalizes the mesh (unit scale, centered at origin)."""
        mesh = trimesh.load_mesh(self.mesh_file_path)
        if not isinstance(mesh, trimesh.Trimesh):
            # If load_mesh returns a Scene, try to extract the first Trimesh geometry
            if isinstance(mesh, trimesh.Scene):
                geometries = list(mesh.geometry.values())
                trimesh_geoms = [
                    g for g in geometries if isinstance(g, trimesh.Trimesh)
                ]
                if not trimesh_geoms:
                    raise TypeError(
                        f"Loaded mesh scene from {self.mesh_file_path} contains no trimesh.Trimesh objects"
                    )
                mesh = trimesh_geoms[0]  # Use the first mesh found
                print(
                    f"Warning: Loaded a scene, using the first Trimesh geometry found: {list(mesh.geometry.keys())[0]}"
                )
            else:
                raise TypeError(
                    f"Loaded mesh from {self.mesh_file_path} is not a trimesh.Trimesh or Scene object, but {type(mesh)}"
                )

        # Store original scale and centroid-offset for denormalization
        self.scale = float(mesh.scale)
        self.offset = (
            -mesh.centroid
        )  # Note: offset is defined as the vector TO ADD to normalized coords

        # Apply normalization
        mesh.apply_scale(1.0 / self.scale)
        transform_matrix = np.eye(4)
        transform_matrix[
            :3, 3
        ] = -mesh.centroid  # Translate mesh so its centroid is at origin
        mesh.apply_transform(transform_matrix)

        self.mesh = mesh


    def generate_grasps(
        self, num: int, kappa: float = 10.0, eps: float = 1e-5
    ) -> Tuple[SE3Pose, Dict[str, Any]]:
        self.normalize_load()
        surface_points, face_idx = trimesh.sample.sample_surface(
            self.mesh, 5 * num)
        normals = self.mesh.face_normals[face_idx]
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        random_dirs = np.empty_like(surface_points)

        for i in range(len(surface_points)):
            sample_dir = vonmises_fisher.rvs(
                mu=-normals[i], kappa=kappa, size=1)[0]
            random_dirs[i] = sample_dir / np.linalg.norm(sample_dir)

        contact_pairs_one = []
        contact_pairs_two = []
        num_points = len(surface_points)

        for i in range(num_points):
            if len(contact_pairs_one) >= num:
                break

            origin = surface_points[i]  # First contact point

            # Create two ray directions: one as the sampled direction, one as its negative.
            directions = [random_dirs[i], -random_dirs[i]]
            origins_for_rays = [origin, origin]
            # Compute intersections (for both rays) without wrapping in any try/except block
            locations, index_ray, _ = self.mesh.ray.intersects_location(
                ray_origins=origins_for_rays,
                ray_directions=directions,
            )

            valid_candidates = []
            if locations.size > 0:
                # Compute distance from the origin for each hit point
                dists = np.linalg.norm(locations - origin, axis=1)
                # Filter out intersection points closer than eps.
                for loc, d in zip(locations, dists):
                    if d >= eps:
                        valid_candidates.append(loc)

            # If at least one valid intersection was found, choose one at random.
            # Otherwise, select a fallback second contact: a random point within a 10cm-cube.
            if valid_candidates:
                chosen_loc = valid_candidates[np.random.randint(
                    len(valid_candidates))]
            else:
                # Random offset in each dimension from uniform distribution over [-0.05, 0.05]
                # (10cm cube centered on the origin point)
                random_offset = np.random.uniform(-0.05, 0.05, size=3)
                chosen_loc = origin + random_offset

            contact_pairs_one.append(origin)
            contact_pairs_two.append(chosen_loc)

        # In case the above pass did not yield enough pairs, fill the remainder with
        # pairs using random fallback contacts.
        while len(contact_pairs_one) < num:
            idx = np.random.randint(num_points)
            origin = surface_points[idx]
            random_offset = np.random.uniform(-0.05, 0.05, size=3)
            fallback_second = origin + random_offset
            contact_pairs_one.append(origin)
            contact_pairs_two.append(fallback_second)

        # Compute gripper poses (in normalized space) from the contact pairs.
        Hs_norm = AntipodalGraspGenerator.define_gripper_pose(
            np.array(contact_pairs_one), np.array(contact_pairs_two)
        )
        # Convert poses back to the original object's coordinate system.
        Hs_denorm = self.denorm_grasp_pose(Hs_norm)

        # Compute gripper widths (and clamp negative values to 0).
        widths = np.linalg.norm(
            np.array(contact_pairs_two) - np.array(contact_pairs_one), axis=1
        )
        widths = np.maximum(widths, 0)

        # Scale widths back to the original object dimensions.
        aux_info = {"width": widths * self.scale}

        return Hs_denorm, aux_info

    @classmethod
    def define_gripper_pose(
        cls, contact_one: np.ndarray, contact_two: np.ndarray
    ) -> np.ndarray:
        """
        Defines gripper pose transformation matrices based on contact point pairs.
        Assumes input points are in a normalized space.
        """
        assert len(contact_one) == len(contact_two)
        if contact_one.ndim == 1:  # Handle single grasp case
            contact_one = contact_one[np.newaxis, :]
            contact_two = contact_two[np.newaxis, :]

        num_grasps = len(contact_one)

        center = (contact_two + contact_one) / 2.0
        # Gripper x-axis (approach direction for fingers) points from contact_one to contact_two
        antipodal_direction = contact_two - contact_one
        norm = np.linalg.norm(antipodal_direction, axis=1, keepdims=True)

        # Avoid division by zero if contacts are coincident (should be filtered earlier)
        valid_norm = ~np.isclose(norm, 0.0)
        if not np.all(valid_norm):
            print(
                "Warning: Coincident contact points found in define_gripper_pose. Normals/Poses might be invalid."
            )
            # Handle invalid norms, e.g., set direction to a default or skip
            # For now, normalize where possible, others might remain NaN/Inf
            antipodal_direction[valid_norm.flatten()] /= norm[valid_norm]
            # Set invalid ones to a default like [1, 0, 0] to avoid crashes downstream
            antipodal_direction[~valid_norm.flatten()] = np.array([
                1.0, 0.0, 0.0])
        else:
            antipodal_direction /= norm

        # Gripper z-axis (gripper approach direction) - random vector orthogonal to x-axis
        random_vectors = np.random.randn(num_grasps, 3)

        # Calculate cross product for z-axis (approach)
        approach_direction = np.cross(antipodal_direction, random_vectors)
        approach_norm = np.linalg.norm(
            approach_direction, axis=1, keepdims=True)

        # Handle cases where random vector is parallel to antipodal_direction
        invalid_approach = np.isclose(approach_norm, 0.0).flatten()
        attempts = 0
        max_attempts_ortho = 10
        while np.any(invalid_approach) and attempts < max_attempts_ortho:
            attempts += 1
            print(f"Warning: Regenerating approach vector, attempt {attempts}")
            # Regenerate random vectors only for the failed ones
            num_invalid = np.sum(invalid_approach)
            random_vectors[invalid_approach] = np.random.rand(num_invalid, 3)

            # Recalculate cross product and norm for invalid ones
            approach_direction[invalid_approach] = np.cross(
                antipodal_direction[invalid_approach], random_vectors[invalid_approach]
            )
            approach_norm[invalid_approach] = np.linalg.norm(
                approach_direction[invalid_approach], axis=1, keepdims=True
            )
            # Update mask
            invalid_approach = np.isclose(approach_norm, 0.0).flatten()

        if np.any(invalid_approach):
            print(
                "Warning: Failed to find orthogonal approach vector after multiple attempts. Using default Z=[0,0,1] cross X."
            )
            # Fallback: Try crossing with Z-axis, unless antipodal is Z
            z_axis = np.array([0.0, 0.0, 1.0])
            fallback_approach = np.cross(
                antipodal_direction[invalid_approach], z_axis)
            fallback_norm = np.linalg.norm(
                fallback_approach, axis=1, keepdims=True)
            # Check if antipodal was Z-axis
            parallel_to_z = np.isclose(fallback_norm, 0.0).flatten()
            if np.any(parallel_to_z):
                y_axis = np.array([0.0, 1.0, 0.0])
                fallback_approach[parallel_to_z] = np.cross(
                    antipodal_direction[invalid_approach][parallel_to_z], y_axis
                )
                fallback_norm[parallel_to_z] = np.linalg.norm(
                    fallback_approach[parallel_to_z], axis=1, keepdims=True
                )

            valid_fallback = ~np.isclose(fallback_norm, 0.0).flatten()
            if np.any(valid_fallback):
                approach_direction[invalid_approach][valid_fallback] /= fallback_norm[
                    valid_fallback
                ]
                # Mark as valid
                approach_norm[invalid_approach][valid_fallback] = 1.0
            # Remaining invalid approaches will be handled by normalization check below

        # Normalize valid approach directions
        valid_approach_norm = ~np.isclose(approach_norm, 0.0)
        if not np.all(valid_approach_norm):
            print(
                "Warning: Could not determine valid approach direction for some grasps."
            )
            approach_direction[~valid_approach_norm.flatten()] = np.array(
                [0.0, 0.0, 1.0]
            )  # Default if still invalid
        else:
            approach_direction /= approach_norm

        # Gripper y-axis (orthogonal to x and z)
        co_direction = np.cross(approach_direction, antipodal_direction)
        # No need to normalize y, as x and z are orthogonal unit vectors

        # Construct transformation matrices
        Hs = np.zeros((num_grasps, 4, 4))
        Hs[..., :3, 0] = antipodal_direction  # x-axis
        Hs[..., :3, 1] = co_direction  # y-axis
        Hs[..., :3, 2] = approach_direction  # z-axis
        Hs[..., :3, 3] = center
        Hs[..., 3, 3] = 1.0
        return Hs
