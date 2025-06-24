import numpy as np
from typing import Tuple


def voxel_downsample_pcd(
    points: np.ndarray, features: np.ndarray, voxel_size: float
) -> Tuple[np.ndarray, np.ndarray]:
    mins = np.min(points, axis=0)  # Shape: (3,)
    vox_idx = np.floor_divide(points - mins, voxel_size).astype(
        np.int64
    )  # Shape: (N, 3)
    shape = np.max(vox_idx, axis=0) + 1  # Shape: (3,)
    raveled_idx = np.ravel_multi_index(vox_idx.T, shape)  # Shape: (N,)
    n_voxels = np.prod(shape)
    n_pts_per_vox = np.bincount(raveled_idx, minlength=n_voxels)  # Shape: (n_voxels,)
    nonzero_vox = np.nonzero(n_pts_per_vox)[0]  # Shape: (num_nonzero_voxels,)
    # Shape: (num_nonzero_voxels,)
    n_pts_per_vox_nonzero = n_pts_per_vox[nonzero_vox]
    feature_sum = np.zeros(
        (n_voxels, features.shape[1]), dtype=features.dtype
    )  # Shape: (n_voxels, C)
    np.add.at(feature_sum, raveled_idx, features)
    feature_vox = feature_sum[nonzero_vox]  # Shape: (num_nonzero_voxels, C)
    coord_sum = np.zeros(
        (n_voxels, points.shape[1]), dtype=points.dtype
    )  # Shape: (n_voxels, 3)
    np.add.at(coord_sum, raveled_idx, points)
    coord_vox = coord_sum[nonzero_vox]  # Shape: (num_nonzero_voxels, 3)
    n_pts_per_vox_nonzero = n_pts_per_vox_nonzero[
        :, np.newaxis
    ]  # Shape: (num_nonzero_voxels, 1)
    feature_vox /= n_pts_per_vox_nonzero
    coord_vox /= n_pts_per_vox_nonzero

    return coord_vox, feature_vox


def rgbd_to_pcd(rgbd, intrinsics, extrinsics) -> Tuple[np.ndarray, np.ndarray]:
    """
    rgbd: numpy array of shape (N, height, width, k) where the last channel is the depth value
    intrinsics: numpy array of shape (3, 3) representing the camera intrinsics matrix
    extrinsics: numpy array of shape (N, 4, 4) representing the camera extrinsics matrix
    """
    width, height = rgbd.shape[1], rgbd.shape[2]
    fx, fy, cx, cy = (
        intrinsics[0, 0],
        intrinsics[1, 1],
        intrinsics[0, 2],
        intrinsics[1, 2],
    )
    z = rgbd[..., -1]
    u = np.arange(width) - cx
    v = np.arange(height) - cy
    x = (z * u) / fx
    y = np.transpose((np.transpose(z, axes=[0, 2, 1]) * v), axes=[0, 2, 1]) / fy

    points = np.stack((x, y, z), axis=-1)
    points_homo = np.concatenate([points, np.ones((*points.shape[:-1], 1))], axis=-1)
    points_homo = np.einsum("nij,nhwj->nhwi", extrinsics, points_homo)
    points = points_homo[..., :3]
    color = rgbd[..., :-1]
    return (points, color)
