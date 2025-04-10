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
        self,
        num: int,
        kappa: float = 15.0,
        min_width_threshold: float = 0.003,
        width_offset: float = 0.01,
    ) -> Tuple[SE3Pose, Dict[str, Any]]:
        if self.mesh is None:
            self.normalize_load()

        # List to store tuples: (p1, p2, dist, normal1)
        valid_contacts_info = []

        # Keep sampling until we have enough valid grasps
        samples_needed = num
        max_attempts = 5  # Limit attempts to prevent infinite loops on difficult meshes
        attempts = 0
        while len(valid_contacts_info) < num and attempts < max_attempts:
            attempts += 1
            num_to_sample = (
                samples_needed * 5
            )  # Oversample to increase chance of finding valid pairs
            try:
                surface_points, face_indices = trimesh.sample.sample_surface(
                    self.mesh, num_to_sample
                )
            except Exception as e:
                print(
                    f"Warning: Trimesh sampling failed on attempt {attempts}: {e}. Skipping object or trying again."
                )
                if attempts >= max_attempts:
                    # Return empty if sampling consistently fails
                    return SE3Pose(np.empty((0, 3)), np.empty((0, 4)), type="wxyz"), {}
                continue  # Try sampling again

            if len(surface_points) == 0:
                print(
                    f"Warning: Trimesh sampling returned 0 points on attempt {attempts}. Skipping object or trying again."
                )
                if attempts >= max_attempts:
                    return SE3Pose(np.empty((0, 3)), np.empty((0, 4)), type="wxyz"), {}
                continue

            # Use face normals corresponding to the sampled points
            normals = self.mesh.face_normals[face_indices]

            # Sample approach directions (~negative normals) from von Mises-Fisher distribution
            random_approaches = np.zeros((len(surface_points), 3))
            for i in range(len(surface_points)):
                # Ensure mu is a unit vector
                mu_norm = np.linalg.norm(normals[i, :])
                if np.isclose(mu_norm, 0):
                    # Handle zero norm case (e.g., degenerate triangle) - skip this point
                    continue
                mu = -normals[i, :] / mu_norm
                try:
                    # Sometimes rvs fails if mu is not perfectly normalized due to float issues
                    random_approaches[i, :] = vonmises_fisher.rvs(
                        mu=mu, kappa=kappa, size=1
                    )[0]
                except ValueError as e:
                    print(
                        f"Warning: vonmises_fisher failed for normal {normals[i, :]} (mu={mu}): {e}. Skipping point."
                    )
                    random_approaches[i, :] = np.nan  # Mark as invalid

            # Prepare for ray casting
            valid_indices = ~np.isnan(
                random_approaches[:, 0]
            )  # Indices of points where vMF worked
            origins = surface_points[valid_indices]
            directions = random_approaches[valid_indices]
            # Keep corresponding normals
            original_normals = normals[valid_indices]

            if len(origins) == 0:
                continue  # No valid approach directions generated

            # Perform ray intersection tests
            # Note: intersects_location returns locations, index_ray, index_tri

            # Sample intersections; note that we now capture the index_ray output.
            locations, index_ray, _ = self.mesh.ray.intersects_location(
                ray_origins=origins,
                ray_directions=directions,
                multiple_hits=True,  # Important: Get all hits along the ray
            )

            # Process intersections to find valid antipodal pairs
            for i in range(len(origins)):  # Iterate through each original sampled point
                if len(valid_contacts_info) >= num:
                    break  # Stop if we have enough grasps

                p1 = origins[i]
                normal1 = original_normals[i]
                # Use index_ray to filter locations corresponding to the i-th ray.
                hits_for_p1 = locations[index_ray == i]

                if len(hits_for_p1) > 0:
                    # Calculate distances from p1 to all hit points for this ray
                    distances = np.linalg.norm(hits_for_p1 - p1, axis=1)

                    # Find hits that satisfy the minimum width threshold
                    valid_hit_indices = np.where(distances >= min_width_threshold)[0]

                    if len(valid_hit_indices) > 0:
                        random_valid_idx = np.random.choice(valid_hit_indices)
                        p2 = hits_for_p1[random_valid_idx]
                        dist = distances[random_valid_idx]

                        valid_contacts_info.append((p1, p2, dist, normal1))
                        samples_needed -= 1  # Decrement needed count

            if samples_needed <= 0:
                break  # Exit outer while loop

        if len(valid_contacts_info) < num:
            print(
                f"Warning: Could only generate {len(valid_contacts_info)}/{num} valid grasps after {attempts} attempts."
            )
            if not valid_contacts_info:
                # Return empty if no grasps could be generated
                return SE3Pose(np.empty((0, 3)), np.empty((0, 4)), type="wxyz"), {}

        # Unpack the results up to the required number
        final_contacts_info = valid_contacts_info[:num]
        contact_pairs_one = np.array([item[0] for item in final_contacts_info])
        contact_pairs_two = np.array([item[1] for item in final_contacts_info])
        contact_distances = np.array([item[2] for item in final_contacts_info])
        surface_normals_one = np.array([item[3] for item in final_contacts_info])

        # --- Calculate Widths ---
        # Width in normalized space, including offset
        norm_widths = contact_distances
        # Final width after scaling
        final_widths = norm_widths * self.scale

        # --- Generate Poses ---
        # Poses are calculated in normalized space first
        Hs_norm = AntipodalGraspGenerator.define_gripper_pose(
            contact_pairs_one, contact_pairs_two
        )
        # Denormalize the poses
        Hs_denorm = self.denorm_grasp_pose(Hs_norm)
        grasp_poses = SE3Pose.from_mat(Hs_denorm, type="wxyz")  # Assuming wxyz output

        # --- Denormalize Contact Points for Auxiliary Info ---
        denorm_points_one = self.denormalize_points(contact_pairs_one)
        denorm_points_two = self.denormalize_points(contact_pairs_two)

        # --- Prepare Auxiliary Information ---
        aux_info = {
            "contact_points_one": denorm_points_one,
            "contact_points_two": denorm_points_two,
            "surface_normals_one": surface_normals_one,  # From normalized mesh
            "grasp_widths": final_widths,
        }

        return grasp_poses, aux_info

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
            antipodal_direction[~valid_norm.flatten()] = np.array([1.0, 0.0, 0.0])
        else:
            antipodal_direction /= norm

        # Gripper z-axis (gripper approach direction) - random vector orthogonal to x-axis
        random_vectors = np.random.randn(num_grasps, 3)

        # Calculate cross product for z-axis (approach)
        approach_direction = np.cross(antipodal_direction, random_vectors)
        approach_norm = np.linalg.norm(approach_direction, axis=1, keepdims=True)

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
            fallback_approach = np.cross(antipodal_direction[invalid_approach], z_axis)
            fallback_norm = np.linalg.norm(fallback_approach, axis=1, keepdims=True)
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
