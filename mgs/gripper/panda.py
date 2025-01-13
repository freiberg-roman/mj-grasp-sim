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
# Copyright 2024 Franka Robotics License: Apache-2.0
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
XML = """
  <default>
    <default class="panda">
      <material specular="0.5" shininess="0.25"/>
      <joint armature="0.1" damping="1" axis="0 0 1" range="-2.8973 2.8973"/>
      <general dyntype="none" biastype="affine" ctrlrange="-2.8973 2.8973" forcerange="-87 87"/>

      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom type="mesh" group="3"/>
        <default class="fingertip_pad_collision_1">
          <geom type="box" size="0.0085 0.004 0.0085" pos="0 0.0055 0.0445" friction="2.8 0.3 0.1"/>
        </default>
        <default class="fingertip_pad_collision_2">
          <geom type="box" size="0.003 0.002 0.003" pos="0.0055 0.002 0.05" friction="2.8 0.3 0.1"/>
        </default>
        <default class="fingertip_pad_collision_3">
          <geom type="box" size="0.003 0.002 0.003" pos="-0.0055 0.002 0.05" friction="2.8 0.3 0.1"/>
        </default>
        <default class="fingertip_pad_collision_4">
          <geom type="box" size="0.003 0.002 0.0035" pos="0.0055 0.002 0.0395" friction="2.8 0.3 0.1"/>
        </default>
        <default class="fingertip_pad_collision_5">
          <geom type="box" size="0.003 0.002 0.0035" pos="-0.0055 0.002 0.0395" friction="2.8 0.3 0.1"/>
        </default>
      </default>
    </default>
  </default>

  <asset>
    <material class="panda" name="white" rgba="1 1 1 1"/>
    <material class="panda" name="off_white" rgba="0.901961 0.921569 0.929412 1"/>
    <material class="panda" name="black" rgba="0.25 0.25 0.25 1"/>

    <!-- Collision meshes -->
    <mesh name="hand_c" file="hand.stl"/>

    <!-- Visual meshes -->
    <mesh file="hand_0.obj"/>
    <mesh file="hand_1.obj"/>
    <mesh file="hand_2.obj"/>
    <mesh file="hand_3.obj"/>
    <mesh file="hand_4.obj"/>
    <mesh file="finger_0.obj"/>
    <mesh file="finger_1.obj"/>
  </asset>

  <worldbody>
    <body name="mocap" mocap="true" pos="{position}" quat="{quaternion}"/>
    <body name="hand" childclass="panda" quat="{quaternion}" pos="{position}">
      <freejoint name="freejoint"/>
      <inertial mass="0.73" pos="-0.01 0 0.03" diaginertia="0.001 0.0025 0.0017"/>
      <geom mesh="hand_0" material="off_white" class="visual"/>
      <geom mesh="hand_1" material="black" class="visual"/>
      <geom mesh="hand_2" material="black" class="visual"/>
      <geom mesh="hand_3" material="white" class="visual"/>
      <geom mesh="hand_4" material="off_white" class="visual"/>
      <geom mesh="hand_c" class="collision"/>
      <body name="left_finger" pos="0 0 0.0584">
        <inertial mass="0.015" pos="0 0 0" diaginertia="2.375e-6 2.375e-6 7.5e-7"/>
        <joint name="finger_joint1" pos="0 0 0" axis="0 1 0" type="slide" limited="true" range="0.0 0.04" damping="100" armature="1.0" frictionloss="1.0"/>
        <geom mesh="finger_0" material="off_white" class="visual"/>
        <geom mesh="finger_1" material="black" class="visual"/>
        <geom mesh="finger_0" class="collision" name="panda_col_1" />
        <geom class="fingertip_pad_collision_1" name="panda_col_2" />
        <geom class="fingertip_pad_collision_2" name="panda_col_3" />
        <geom class="fingertip_pad_collision_3" name="panda_col_4"/>
        <geom class="fingertip_pad_collision_4" name="panda_col_5"/>
        <geom class="fingertip_pad_collision_5" name="panda_col_6"/>
      </body>
      <body name="right_finger" pos="0 -0.04 0.0584" quat="0 0 0 1">
        <inertial mass="0.015" pos="0 0 0" diaginertia="2.375e-6 2.375e-6 7.5e-7"/>
        <joint name="finger_joint2" pos="0 0 0" axis="0 1 0" type="slide" limited="true" range="-0.04 0.0" damping="100" armature="1.0" frictionloss="1.0"/>
        <geom mesh="finger_0" material="off_white" class="visual"/>
        <geom mesh="finger_1" material="black" class="visual"/>
        <geom mesh="finger_0" class="collision" name="panda_col_7" />
        <geom class="fingertip_pad_collision_1" name="panda_col_8" />
        <geom class="fingertip_pad_collision_2" name="panda_col_9" />
        <geom class="fingertip_pad_collision_3" name="panda_col_10" />
        <geom class="fingertip_pad_collision_4" name="panda_col_11" />
        <geom class="fingertip_pad_collision_5" name="panda_col_12" />
      </body>
    </body>
  </worldbody>

  <contact>
    <exclude body1="hand" body2="left_finger"/>
    <exclude body1="hand" body2="right_finger"/>
  </contact>


  <tendon>
    <fixed name="split">
      <joint joint="finger_joint1" coef="0.5"/>
      <joint joint="finger_joint2" coef="0.5"/>
    </fixed>
  </tendon>

  <equality>
    <weld body1="mocap" body2="hand"/>
  </equality>

  <actuator>
    <position ctrllimited="true" ctrlrange="0.0 0.04" joint="finger_joint1" kp="1000" name="gripper_finger_joint1" forcelimited="true" forcerange="-15 15"/>
    <position ctrllimited="true" ctrlrange="-0.04 0.0" joint="finger_joint2" kp="1000" name="gripper_finger_joint2" forcelimited="true" forcerange="-15 15"/>
  </actuator>
"""


class GripperPanda(MjShakableOpenCloseGripper, MjScannable):
    def __init__(self, pose: SE3Pose):
        super().__init__(pose, "hand")

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
        base_path = os.path.join(ASSET_PATH, "panda")
        for file_name in os.listdir(base_path):
            path = os.path.join(base_path, file_name)
            with open(path, "rb") as f:
                ASSETS[file_name] = f.read()

        return (xml, ASSETS)

    def base_to_contact_transform(self) -> SE3Pose:
        pos = np.array([0, 0, -0.102])
        quat = np.array([0.707106781, 0.0, 0.0, 0.707106781])
        return SE3Pose(pos, quat, type="wxyz")

    def open_gripper(self, sim: MjSimulation):
        idxs = sim.get_joint_idxs(self.get_actuator_joint_names())
        sim.set_qpos(np.array([0.04, 0.0]), idxs)
        sim.data.ctrl[:] = np.array([0.04, 0.0])

    def close_gripper_at(self, sim: MjSimulation, pose: SE3Pose):
        sim.data.mocap_pos = np.copy(pose.pos)
        sim.data.mocap_quat = np.copy(pose.quat)
        sim.data.ctrl[:] = np.array([0.0, -0.04])
        mujoco.mj_step(sim.model, sim.data, 3000)  # type: ignore

    def get_actuator_joint_names(self) -> List[str]:
        return ["finger_joint1", "finger_joint2"]

    def get_freejoint_idxs(self, sim: MjSimulation) -> List[int]:
        start_idx = sim.get_joint_idxs(["freejoint"])[0]
        return list(range(start_idx, start_idx + 7))

    def get_render_options(self):
        options = mujoco.MjvOption()  # type: ignore
        return options
