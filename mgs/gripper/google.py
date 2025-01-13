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

import os
from typing import Any, Dict, List, Tuple

import mujoco
import numpy as np

from mgs.core.simualtion import MjSimulation
from mgs.util.const import ASSET_PATH
from mgs.util.geo.transforms import SE3Pose

from .base import MjScannable, MjShakableOpenCloseGripper

# The XML template string is derived from MuJoCo Menagerie
# (https://github.com/google-deepmind/mujoco_menagerie/tree/469893211c41d5da9c314f5ab58059fa17c8e360)
# Copyright 2024 Google DeepMind License: Apache-2.0
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
XML = """
<asset>
    <texture type="2d" name="robot_texture" file="robot_texture.png"/>
    <texture type="2d" name="finger_base_texture" file="finger_base_texture.png"/>
    <texture type="2d" name="finger_tip_texture" file="finger_tip_texture.png"/>
    <material name="robot_mtl" texture="robot_texture"/>
    <material name="finger_base_mtl" texture="finger_base_texture"/>
    <material name="finger_tip_mtl" texture="finger_tip_texture"/>

    <mesh file="link_forearm.stl"/>
    <mesh file="link_wrist.stl"/>
    <mesh file="link_gripper.stl"/>
    <mesh file="link_head_pan.stl"/>
    <mesh file="link_head_tilt.stl"/>
    <mesh file="link_finger_base.stl"/>
    <mesh file="link_finger_tip.stl"/>

    <!-- Visual meshes -->
    <mesh class="visual" file="link_wrist_v.obj"/>
    <mesh class="visual" file="link_gripper_v.obj"/>
    <mesh class="visual" file="link_finger_base_v.obj"/>
    <mesh class="visual" file="link_finger_tip_v.obj"/>
  </asset>

  <default>
    <default class="robot">
      <joint damping="10.0" frictionloss="1" armature=".1"/>
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2" material="robot_mtl"/>
      </default>
      <default class="collision">
        <geom type="mesh" rgba="1 1 1 1" group="3" contype="1" conaffinity="1"/>
          <default class="finger_base">
            <geom type="capsule" rgba="1 1 1 1" size="0.015 0.01" quat="1 0 1 0" mass="0.0133245" pos="0 -0.005 0.03"/>
          </default>
          <default class="finger_tip">
            <geom type="capsule" rgba="1 1 1 1" size="0.01 0.008" quat="1 0 1 0" mass="0.0161862"/>
          </default>
        </default>
    </default>
  </default>

  <worldbody>
    <body name="mocap" mocap="true" pos="{position}" quat="{quaternion}"/>
    <body name="link_gripper" pos="{position}" quat="{quaternion}">
      <freejoint name="freejoint"/>
      <site name="gripper" pos="0 0 0.13" group="5"/>
      <inertial pos="-3.20302e-05 0.000371384 0.0387883" quat="0.997934 0.0574504 -0.0278948 -0.00699571" mass="0.41474" diaginertia="0.000327698 0.000308611 0.000272392"/>
      <geom class="visual" mesh="link_gripper_v"/>
      <geom class="collision" name="gripper" quat="1 0 0 -1" type="mesh" mesh="link_gripper"/>
      <!-- right finger -->
      <body name="link_finger_right" pos="0 0.025 0.05886" quat="0.879969 -0.475032 0 0" gravcomp="1">
        <joint name="joint_finger_right" axis="1 0 0" range="0.01 1.3" damping="2.0" frictionloss="1" armature=".1"/>
        <geom class="visual" mesh="link_finger_base_v" material="finger_base_mtl"/>
        <geom class="collision" mesh="link_finger_base" mass="0.0333245"/>
        <geom class="finger_base"/>
        <geom class="finger_base" pos="0 -0.005 0.04"/>
        <geom class="finger_base" pos="0 -0.005 0.05"/>
        <body name="link_finger_tip_right" pos="0 -0.0103567 0.0641556" quat="0.995004 0.0998334 0 0" gravcomp="1">
          <geom class="visual" mesh="link_finger_tip_v" material="finger_tip_mtl"/>
          <geom class="collision" mesh="link_finger_tip_v" mass="0.0161862"/>
          <geom class="finger_tip"/>
          <geom class="finger_tip" pos="0 0 0.01"/>
          <geom class="finger_tip" pos="0 0 0.02"/>
          <geom class="finger_tip" pos="0 0 0.03"/>
          <geom class="finger_tip" pos="0 0 0.04"/>
        </body>
      </body>
      <!-- left finger -->
      <body name="link_finger_left" pos="0 -0.025 0.05886" quat="0 0 -0.475032 0.879969" gravcomp="1">
        <joint name="joint_finger_left" axis="1 0 0" range="0.01 1.3" damping="2.0" frictionloss="1" armature=".1"/>
        <geom class="visual" mesh="link_finger_base_v" material="finger_base_mtl"/>
        <geom class="collision" mesh="link_finger_base_v" mass="0.0333245" />
          <geom class="finger_base"/>
          <geom class="finger_base" pos="0 -0.005 0.04"/>
          <geom class="finger_base" pos="0 -0.005 0.05"/>
        <body name="link_finger_tip_left" pos="0 -0.0103567 0.0641556" quat="0.995004 0.0998334 0 0" gravcomp="1">
          <geom class="visual" mesh="link_finger_tip_v" material="finger_tip_mtl"/>
          <geom class="collision" mesh="link_finger_tip" mass="0.0161862"/>
          <geom class="finger_tip"/>
          <geom class="finger_tip" pos="0 0 0.01"/>
          <geom class="finger_tip" pos="0 0 0.02"/>
          <geom class="finger_tip" pos="0 0 0.03"/>
          <geom class="finger_tip" pos="0 0 0.04"/>
        </body>
      </body>
    </body>  <!-- gripper -->
  </worldbody>
  <equality>
    <weld body1="mocap" body2="link_gripper"/>
  </equality>

   <actuator>
     <position joint="joint_finger_right" kp="20" ctrlrange="0.01 1.3"   forcerange="-30 30"/>
     <position joint="joint_finger_left"  kp="20" ctrlrange="0.01 1.3"   forcerange="-30 30"/>
   </actuator>
"""


class GripperGoogle(MjShakableOpenCloseGripper, MjScannable):
    def __init__(self, pose: SE3Pose):
        super().__init__(pose, "link_gripper")

    def base_to_contact_transform(self) -> SE3Pose:
        return SE3Pose(
            np.array([0, 0, -0.15]),
            np.array([0.707106781, 0.0, 0.0, 0.707106781]),
            type="wxyz",
        )

    def open_gripper(self, sim: MjSimulation):
        sim.data.ctrl[:] = np.array([0.01, 0.01])

    def close_gripper_at(self, sim: MjSimulation, pose: SE3Pose):
        self.set_pose(sim, pose)
        sim.data.ctrl[:] = np.array([1.3, 1.3])
        mujoco.mj_step(sim.model, sim.data, 3000)  # type: ignore

    def to_xml(self) -> Tuple[str, Dict[str, Any]]:
        pos = "{} {} {}".format(self.pos[0], self.pos[1], self.pos[2])
        quat = "{} {} {} {}".format(
            self.quat[0], self.quat[1], self.quat[2], self.quat[3]
        )
        xml = XML.format(
            **{
                "position": pos,
                "quaternion": quat,
            }
        )

        ASSETS = dict()
        base_path = os.path.join(ASSET_PATH, "google")
        for file_name in os.listdir(base_path):
            path = os.path.join(base_path, file_name)
            with open(path, "rb") as f:
                ASSETS[file_name] = f.read()

        return (xml, ASSETS)

    def get_actuator_joint_names(self) -> List[str]:
        return ["joint_finger_right", "joint_finger_left"]

    def get_freejoint_idxs(self, sim: MjSimulation) -> List[int]:
        start_idx = sim.get_joint_idxs(["freejoint"])[0]
        return list(range(start_idx, start_idx + 7))
