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
from mgs.gripper.base import MjScannable, MjShakableOpenCloseGripper
from mgs.util.const import ASSET_PATH
from mgs.util.geo.transforms import SE3Pose

# The XML template string is derived from MuJoCo Menagerie
# (https://github.com/google-deepmind/mujoco_menagerie/tree/469893211c41d5da9c314f5ab58059fa17c8e360)
# Copyright (c) 2023, Trossen Robotics License: BSD-3-Clause
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
XML = """
  <option cone="elliptic" impratio="10"/>

  <asset>
    <texture type="2d" file="interbotix_black.png"/>
    <material name="black" texture="interbotix_black"/>

    <mesh file="vx300s_1_base.stl" scale="0.001 0.001 0.001"/>
    <mesh file="vx300s_2_shoulder.stl" scale="0.001 0.001 0.001"/>
    <mesh file="vx300s_3_upper_arm.stl" scale="0.001 0.001 0.001"/>
    <mesh file="vx300s_4_upper_forearm.stl" scale="0.001 0.001 0.001"/>
    <mesh file="vx300s_5_lower_forearm.stl" scale="0.001 0.001 0.001"/>
    <mesh file="vx300s_6_wrist.stl" scale="0.001 0.001 0.001"/>
    <mesh file="vx300s_7_gripper.stl" scale="0.001 0.001 0.001"/>
    <mesh file="vx300s_8_gripper_prop.stl" scale="0.001 0.001 0.001"/>
    <mesh file="vx300s_9_gripper_bar.stl" scale="0.001 0.001 0.001"/>
    <mesh file="vx300s_10_gripper_finger.stl" scale="0.001 0.001 0.001"/>
  </asset>

  <default>
    <default class="vx300s">
      <joint axis="0 1 0"/>
      <position forcerange="-35 35"/>
      <default class="waist">
        <joint axis="0 0 1" range="-3.14158 3.14158" damping="2.86"/>
        <position ctrlrange="-3.14158 3.14158" kp="25"/>
      </default>
      <default class="shoulder">
        <joint range="-1.85005 1.25664" armature="0.004" frictionloss="0.06" damping="6.25"/>
        <position ctrlrange="-1.85005 1.25664" kp="76" forcerange="-57 57"/>
      </default>
      <default class="elbow">
        <joint range="-1.76278 1.6057" armature="0.072" frictionloss="1.74" damping="8.15"/>
        <position ctrlrange="-1.76278 1.6057" kp="106" forcerange="-25 25"/>
      </default>
      <default class="forearm_roll">
        <joint axis="1 0 0" range="-3.14158 3.14158" armature="0.060" damping="3.07"/>
        <position ctrlrange="-3.14158 3.14158" kp="35" forcerange="-10 10"/>
      </default>
      <default class="wrist_angle">
        <joint range="-1.8675 2.23402" damping="1.18"/>
        <position ctrlrange="-1.8675 2.23402" kp="8"/>
      </default>
      <default class="wrist_rotate">
        <joint axis="1 0 0" range="-3.14158 3.14158" damping="0.78"/>
        <position ctrlrange="-3.14158 3.14158" kp="7"/>
      </default>
      <default class="finger_left">
        <position ctrlrange="0.021 0.057" kp="300"/>
      </default>
      <default class="finger_right">
        <position ctrlrange="-0.057 -0.021 " kp="300"/>
      </default>
      <default class="finger">
        <joint type="slide" armature="0.251" damping="10"/>
        <position ctrlrange="0.021 0.057" kp="300"/>
        <default class="left_finger">
          <joint range="0.021 0.057"/>
        </default>
        <default class="right_finger">
          <joint range="-0.057 -0.021"/>
        </default>
      </default>
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" density="0" group="2" material="black"/>
      </default>
      <default class="collision">
        <geom group="3" type="mesh"/>
        <default class="finger_collision">
          <geom condim="4" solimp="2 1 0.01" solref="0.01 1" friction="1 0.005 0.0001"/>
        </default>
      </default>
    </default>
  </default>

  <worldbody>
    <body name="mocap" mocap="true" pos="{position}" quat="{quaternion}"/>
    <body name="gripper_link" pos="{position}" quat="{quaternion}">
      <freejoint name="freejoint" />
      <inertial pos="0.0395662 -2.56311e-07 0.00400649" quat="0.62033 0.619916 -0.339682 0.339869"
        mass="0.251652" diaginertia="0.000689546 0.000650316 0.000468142"/>
      <geom pos="-0.02 0 0" quat="1 0 0 1" class="visual" mesh="vx300s_7_gripper"/>
      <geom pos="-0.02 0 0" quat="1 0 0 1" class="collision" mesh="vx300s_7_gripper"/>
      <geom pos="-0.020175 0 0" quat="1 0 0 1" class="visual" mesh="vx300s_9_gripper_bar"/>
      <geom pos="-0.020175 0 0" quat="1 0 0 1" class="collision" mesh="vx300s_9_gripper_bar"/>
      <site name="pinch" pos="0.1 0 0" size="0.005" rgba="0.6 0.3 0.3 1" group="5"/>
      <body name="gripper_prop_link" pos="0.0485 0 0">
        <inertial pos="0.002378 2.85e-08 0" quat="0 0 0.897698 0.440611" mass="0.008009"
          diaginertia="4.2979e-06 2.8868e-06 1.5314e-06"/>
        <geom pos="-0.0685 0 0" quat="1 0 0 1" class="visual" mesh="vx300s_8_gripper_prop"/>
        <geom pos="-0.0685 0 0" quat="1 0 0 1" class="collision" mesh="vx300s_8_gripper_prop"/>
      </body>
      <body name="left_finger_link" pos="0.0687 0 0">
        <inertial pos="0.017344 -0.0060692 0" quat="0.449364 0.449364 -0.54596 -0.54596" mass="0.034796"
          diaginertia="2.48003e-05 1.417e-05 1.20797e-05"/>
        <joint name="left_finger" class="left_finger"/>
        <geom pos="-0.0404 -0.0575 0" quat="-1 1 -1 1" class="visual" mesh="vx300s_10_gripper_finger"/>
        <geom class="finger_collision" type="box" name="left_finger_pad_0" size="0.01405 0.01405 0.001"
          pos="0.0478 -0.0125 0.0106" quat="0.65 0.65 -0.27 0.27"/>
        <geom class="finger_collision" type="box" name="left_finger_pad_1" size="0.01405 0.01405 0.001"
          pos="0.0478 -0.0125 -0.0106" quat="0.65 0.65 -0.27 0.27"/>
        <geom class="finger_collision" type="box" name="left_finger_pad_2" size="0.01058 0.01058 0.001"
          pos="0.0571 -0.0125 0.0" quat="1 1 0 0"/>
        <geom class="finger_collision" type="box" name="left_finger_pad_3" size="0.01 0.0105 0.001"
          pos="0.0378 -0.0125 0.0" quat="1 1 0 0"/>
        <geom class="finger_collision" type="box" name="left_finger_pad_4" size="0.015 0.0105 0.001"
          pos="0.0128 -0.0125 0.0" quat="1 1 0 0"/>
        <geom class="finger_collision" type="box" name="left_finger_pad_5" size="0.01 0.0105 0.001"
          pos="0.0378 -0.0125 0.02" quat="1 1 0 0"/>
        <geom class="finger_collision" type="box" name="left_finger_pad_6" size="0.015 0.0105 0.001"
          pos="0.0128 -0.0125 0.02" quat="1 1 0 0"/>
        <geom class="finger_collision" type="box" name="left_finger_pad_7" size="0.01 0.0105 0.001"
          pos="0.0378 -0.0125 -0.02" quat="1 1 0 0"/>
        <geom class="finger_collision" type="box" name="left_finger_pad_8" size="0.015 0.0105 0.001"
          pos="0.0128 -0.0125 -0.02" quat="1 1 0 0"/>
      </body>
      <body name="right_finger_link" pos="0.0687 0 0">
        <inertial pos="0.017344 0.0060692 0" quat="0.44937 -0.44937 0.545955 -0.545955" mass="0.034796"
          diaginertia="2.48002e-05 1.417e-05 1.20798e-05"/>
        <joint name="right_finger" class="right_finger"/>
        <geom pos="-0.0404 0.0575 0" quat="1 1 1 1" class="visual" mesh="vx300s_10_gripper_finger"/>
        <geom class="finger_collision" type="box" name="right_finger_pad_0" size="0.01405 0.01405 0.001"
          pos="0.0478 0.0125 0.0106" quat="0.65 0.65 -0.27 0.27"/>
        <geom class="finger_collision" type="box" name="right_finger_pad_1" size="0.01405 0.01405 0.001"
          pos="0.0478 0.0125 -0.0106" quat="0.65 0.65 -0.27 0.27"/>
        <geom class="finger_collision" type="box" name="right_finger_pad_2" size="0.01058 0.01058 0.001"
          pos="0.0571 0.0125 0.0" quat="1 1 0 0"/>
        <geom class="finger_collision" type="box" name="right_finger_pad_3" size="0.01 0.0105 0.001"
          pos="0.0378 0.0125 0.0" quat="1 1 0 0"/>
        <geom class="finger_collision" type="box" name="right_finger_pad_4" size="0.015 0.0105 0.001"
          pos="0.0128 0.0125 0.0" quat="1 1 0 0"/>
        <geom class="finger_collision" type="box" name="right_finger_pad_5" size="0.01 0.0105 0.001"
          pos="0.0378 0.0125 0.02" quat="1 1 0 0"/>
        <geom class="finger_collision" type="box" name="right_finger_pad_6" size="0.015 0.0105 0.001"
          pos="0.0128 0.0125 0.02" quat="1 1 0 0"/>
        <geom class="finger_collision" type="box" name="right_finger_pad_7" size="0.01 0.0105 0.001"
          pos="0.0378 0.0125 -0.02" quat="1 1 0 0"/>
        <geom class="finger_collision" type="box" name="right_finger_pad_8" size="0.015 0.0105 0.001"
          pos="0.0128 0.0125 -0.02" quat="1 1 0 0"/>
      </body>
    </body>
  </worldbody>

  <equality>
    <weld body1="mocap" body2="gripper_link"/>
  </equality>

  <actuator>
    <position class="finger_left" name="act_left" joint="left_finger"/>
    <position class="finger_right" name="act_right" joint="right_finger"/>
  </actuator>
"""


class GripperVX300(MjShakableOpenCloseGripper, MjScannable):
    def __init__(self, pose: SE3Pose):
        super().__init__(pose, "gripper_link")

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
        base_path = os.path.join(ASSET_PATH, "vx300")
        for file_name in os.listdir(base_path):
            path = os.path.join(base_path, file_name)
            with open(path, "rb") as f:
                ASSETS[file_name] = f.read()

        return (xml, ASSETS)

    def base_to_contact_transform(self) -> SE3Pose:
        # rot_around_y = SE3Pose(
        #     np.array([0, 0, 0]),
        #     np.array([0.707106781, 0, -0.707106781, 0]),
        #     type="wxyz",
        # )
        # rot_around_z = SE3Pose(
        #     np.array([0, 0, 0]),
        #     np.array([0.707106781, 0, 0.0, 0.707106781]),
        #     type="wxyz",
        # )

        # rot = rot_around_z @ rot_around_y
        # rot.pos = np.array([0, 0, -0.12])
        # return rot

        pos = np.array([0, 0, 0.])
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        return SE3Pose(pos, quat, type="wxyz")

    def open_gripper(self, sim: MjSimulation):
        sim.data.qpos[7:9] = np.array([0.057, -0.057])

    def close_gripper_at(self, sim: MjSimulation, pose: SE3Pose):
        sim.data.mocap_pos = np.copy(pose.pos)
        sim.data.mocap_quat = np.copy(pose.quat)
        sim.data.ctrl[:] = np.array([0.0021])
        mujoco.mj_step(sim.model, sim.data, 3000)  # type: ignore

    def get_freejoint_idxs(self, sim: MjSimulation) -> List[int]:
        start_idx = sim.get_joint_idxs(["freejoint"])[0]
        return list(range(start_idx, start_idx + 7))

    def get_actuator_joint_names(self) -> List[str]:
        return ["left_finger", "right_finger"]
