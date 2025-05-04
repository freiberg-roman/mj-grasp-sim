from mgs.sampler.kin.base import (
    kinematic_pcd_transform,
    forward_kinematic_point_transform,
)
from flax import nnx  # Assuming nnx is used
import sys
from mgs.sampler.kin.jax_util import normalize_vector
import plotly.graph_objects as go
import numpy as np
import os
import jax.numpy as jnp
import math
import jax
from mgs.sampler.kin.base import KinematicsModel


class AllegroKinematicsModel(nnx.Module, KinematicsModel):
    def __init__(self, lmax=2, num_emb_channels=64):
        self.lmax = lmax
        self.num_channels = num_emb_channels
        self.num_dofs = 16
        self.num_extra_dofs = 0
        self.embedding = nnx.Param(
            jax.random.normal(
                nnx.Rngs(0)(), shape=(self.num_dofs, (lmax + 1) ** 2, num_emb_channels)
            )
        )
        self.kinematics_graph = [
            [0, 1, 2, 3],  # ffa
            [4, 5, 6, 7],  # mfa
            [8, 9, 10, 11],  # rfa
            [12, 13, 14, 15],  # tha
        ]
        self.base_to_contact = nnx.Variable(
            jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )

        # Ask
        self.align_to_approach = nnx.Variable(
            (
                jnp.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]),
                jnp.array([0, 0, 0.0]),
            )
        )


        self.kinematics_transforms = nnx.Variable(
            jnp.array(
                [
                    # FF
                    # ffj0 is the base of ffj1, ffj1 is the base of ffj2, etc.
                    [0.999048, -0.0436194, 0, 0, 0, 0.0435, -0.001542], # ffj0
                    [1, 0, 0, 0, 0, 0, 0.0164],  # ffj1  
                    [1, 0, 0, 0, 0, 0, 0.054],  # ffj2
                    [1, 0, 0, 0, 0, 0, 0.0384],  # ffj3

                    # MF
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.0007],  #mfj0
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.0164],  # mfj1
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.054],  # mfj2
                    [1.0, 0.0, 0.0, 0.0, 0, 0, 0.0384],  # mfj3
                    
                    # RF
                    [0.999048, 0.0436194, 0, 0, 0, -0.0435, -0.001542], # rfj0
                    [1, 0, 0, 0, 0, 0, 0.0164],  # rfj1
                    [1, 0, 0, 0, 0, 0, 0.054],  # rfj2
                    [1, 0, 0, 0, 0, 0, 0.0384],  # rfj3
                    
                    # TH
                    [0.477714, -0.521334, -0.521334, -0.477714, -0.0182, 0.019333, -0.045987],  # thj0
                    [1, 0 ,0 ,0 ,-0.027, 0.005, 0.0399],  # thj1
                    [1, 0, 0, 0, 0, 0, 0.0177],  # thj2
                    [1, 0 ,0, 0, 0, 0, 0.0514],  # thj3
                ]
            )
        )
        # a six-dimensional vector with a translational direction and rotation axis
        self.joint_transforms = nnx.Variable(
            jnp.array(
                [
                    # FF
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                    # MF
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                    # RF
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                 
                    # TH
                    [0, 0, 0, -1, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, -1, 0, 0],
                    [0, 0, 0, -1, 0, 0],
                ],
                dtype=jnp.float32,
            )
        )
        self.joint_ranges = nnx.Variable(
            jnp.array(
                [
                    # FF
                    [-0.47, 0.47],
                    [-0.196, 1.61],
                    [-0.174, 1.709],
                    [-0.227, 1.618],
                    # MF
                    [-0.47, 0.47],
                    [-0.196, 1.61],
                    [-0.174, 1.709],
                    [-0.227, 1.618],
                    # RF
                    [-0.47, 0.47],
                    [-0.196, 1.61],
                    [-0.174, 1.709],
                    [-0.227, 1.618],
        
                    # TH
                    [0.263, 1.396],
                    [-0.105, 1.163],
                    [-0.189, 1.644],
                    [-0.162, 1.719],
                ]
            )
        )

        # ask
        self.fingertip_normals = nnx.Variable(
            jnp.array(
                [
                    [0.0, 1.0, 0],
                    [0.0, 1.0, 0],
                    [0.0, 1.0, 0],
                    [0.0, 1.0, 0],
                ]
            )
        )

        self.fingertip_idx = nnx.Variable(
            jnp.array([3, 7, 11, 15], dtype=jnp.int32)
        )

        # ask!!
        self.local_fingertip_contact_positions = nnx.Variable(
            jnp.array(
                [
                    [
                        [0, -0.01, 0],
                        [0, -0.01, 0.01],
                        [0, -0.01, -0.01],
                    ],
                    [
                        [0, -0.01, 0],
                        [0, -0.01, 0.01],
                        [0, -0.01, -0.01],
                    ],
                    [
                        [0, -0.01, 0],
                        [0, -0.01, 0.01],
                        [0, -0.01, -0.01],
                    ],
                    [
                        [0, -0.01, 0],
                        [0, -0.01, 0.01],
                        [0, -0.01, -0.01],
                    ],
                ]
            )
        )
        self.init_pregrasp_joint = nnx.Variable(
            jnp.array(
                 [
                -0.08,
                0.715,
                0.710,
                0.95,
                0,
                0.8,
                0.71,
                0.67,
                0.08,
                0.715,
                0.710,
                0.95,
                1.4,
                0.55,
                -0.19,
                1.45,
                ]
            )
        )

