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

import glob
import os
from typing import Any, Dict, List, Tuple

import mujoco
import numpy as np

from mgs.core.simualtion import MjSimulation
from mgs.gripper.base import MjScannable, MjShakableOpenCloseGripper
from mgs.util.const import ASSET_PATH
from mgs.util.geo.transforms import SE3Pose

# The XML template string is derived from robosuite
# (https://github.com/ARISE-Initiative/robosuite/releases/tag/v1.5.0)
# Copyright (c) 2022 Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab License: MIT
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
XML = """
    <asset>
        <mesh name="standard_narrow" file="standard_narrow.stl" />
        <mesh name="half_round_tip" file="half_round_tip.stl" />
	<material name="Material_001.001" specular="0.5" shininess="0.25" rgba="0.640000 0.000000 0.000000 1.000000"/>
	<material name="Material_002" specular="0.5" shininess="0.25" rgba="0.640000 0.640000 0.640000 1.000000"/>
	<mesh name="electric_gripper_base_0" file="electric_gripper_base_0.obj"/>
	<mesh name="electric_gripper_base_1" file="electric_gripper_base_1.obj"/>
	<material name="finger_mat" specular="0.5" shininess="0.25" rgba="0.000000 0.000000 0.000000 1.000000"/>
	<material name="Material_001" specular="0.5" shininess="0.25" rgba="0.001651 0.001651 0.001651 1.000000"/>
	<mesh name="connector_plate" file="connector_plate.obj"/>
    </asset>
    <actuator>
        <position ctrllimited="true" ctrlrange="-0.0115 0.020833" joint="r_finger_joint" kp="1000" name="gripper_r_finger_joint" forcelimited="true" forcerange="-20 20"/>
        <position ctrllimited="true" ctrlrange="-0.020833 0.0115" joint="l_finger_joint" kp="1000" name="gripper_l_finger_joint" forcelimited="true" forcerange="-20 20"/>
    </actuator>
    <default>
      <default class="visual">
	<geom contype="0" conaffinity="0" group="1" type="mesh"/>
      </default>
    </default>
    <worldbody>
        <body name="mocap" mocap="true" pos="{position}" quat="{quaternion}"/>
        <body name="gripper_base" pos="{position}" quat="{quaternion}">
            <freejoint name="rethink:base"/>
            <site name="ft_frame" pos="0 0 0" size="0.01 0.01 0.01" rgba="1 0 0 1" type="sphere" group="1"/>
            <inertial pos="0 0 0" quat="-0.5 0.5 0.5 0.5" mass="0.3" diaginertia="3e-08 2e-08 2e-08" />
	    <geom material="Material_001" mesh="connector_plate" pos="0 0 0.0018" quat="0.7071068 0 0 0.7071068" class="visual"/>
	    <geom mesh="electric_gripper_base_0" material="Material_002" pos="0 0 0.0194" quat="0.7071068 0 0 0.7071068" class="visual"/>
	    <geom mesh="electric_gripper_base_1" material="Material_001.001" pos="0 0 0.0194" quat="0.7071068 0 0 0.7071068" class="visual"/>

            <geom size="0.029 0.05" quat="0 0 0.707107 0.707107" type="cylinder" group="0" name="gripper_base_col" pos="0.004 0.0 0.04"/>
            <!-- This site was added for visualization. -->
            <body name="eef" pos="0 0 0.109" quat="0.707105 0 0 -0.707105">
                <site name="grip_site" pos="0 0 0" size="0.01 0.01 0.01" rgba="1 0 0 0.5" type="sphere" group="1"/>
                <site name="ee_x" pos="0.1 0 0" size="0.005 .1"  quat="0.707105  0 0.707108 0 " rgba="1 0 0 0" type="cylinder" group="1"/>
                <site name="ee_y" pos="0 0.1 0" size="0.005 .1" quat="0.707105 0.707108 0 0" rgba="0 1 0 0" type="cylinder" group="1"/>
                <site name="ee_z" pos="0 0 0.1" size="0.005 .1" quat="1 0 0 0" rgba="0 0 1 0" type="cylinder" group="1"/>
                <!-- This site was added for visualization. -->
                <site name="grip_site_cylinder" pos="0 0 0" size="0.005 10" rgba="0 1 0 0.3" type="cylinder" group="1"/>
            </body>
            <body name="l_finger" pos="0 0.01 0.0444">
                <inertial pos="0 0 0" quat="0 0 0 -1" mass="0.02" diaginertia="0.01 0.01 0.01" />
                <joint name="l_finger_joint" pos="0 0 0" axis="0 1 0" type="slide" limited="true" range="-0.0115 0.020833" damping="100" armature="1.0" frictionloss="1.0"/>
                <geom name="l_finger" quat="0 0 0 -1" type="mesh" contype="0" conaffinity="0" group="1" mesh="standard_narrow" material="finger_mat"/>
                <geom size="0.005 0.00675 0.0375" pos="0 0.01725 0.04" quat="0 0 0 -1" type="box" group="0" conaffinity="1" contype="0" name="l_finger_g0" friction="0 0 0"/>
                <geom size="0.005 0.025 0.0085" pos="-0.005 -0.003 0.0083" quat="0 0 0 -1" type="box" group="0" conaffinity="1" contype="0" name="l_finger_g1" friction="0 0 0"/>
                <body name="l_finger_tip" pos="0 0.01725 0.075">
                    <inertial pos="0 0 0" quat="0 0 0 1" mass="0.01" diaginertia="0.01 0.01 0.01" />
                    <geom name="l_fingertip_g0_vis" quat="0 0 0 1" type="mesh" contype="0" conaffinity="0" group="1" mesh="half_round_tip" material="finger_mat"/>

                    <geom size="0.004 0.004 0.0185" pos="0 -0.0045 -0.015" quat="0 0 0 -1" type="box" group="0"  conaffinity="1" contype="0" name="l_fingertip_g0" friction="0 0 0"/>
                    <geom size="0.0035 0.004 0.0165" pos="0 -0.0047 -0.017" type="box"  conaffinity="1" contype="0" name="l_fingerpad_g0" friction="0 0 0"/>
                </body>
            </body>
            <body name="r_finger" pos="0 -0.01 0.0444">
                <inertial pos="0 0 0" mass="0.02" diaginertia="0.01 0.01 0.01" />
                <joint name="r_finger_joint" pos="0 0 0" axis="0 1 0" type="slide" limited="true" range="-0.020833 0.0115" damping="100" armature="1.0" frictionloss="1.0"/>
                <geom name="r_finger" type="mesh" contype="0" conaffinity="0" group="1" mesh="standard_narrow" material="finger_mat"/>
                <geom size="0.005 0.00675 0.0375" pos="0 -0.01725 0.04" type="box" group="0" conaffinity="1" contype="0" name="r_finger_g0" friction="0 0 0"/>
                <geom size="0.005 0.025 0.0085" pos="0.005 0.003 0.0083" type="box" group="0" conaffinity="1" contype="0" name="r_finger_g1" friction="0 0 0"/>
                <body name="r_finger_tip" pos="0 -0.01725 0.075">
                    <inertial pos="0 0 0" mass="0.01" diaginertia="0.01 0.01 0.01" />
                    <geom name="r_fingertip_g0_vis" type="mesh" contype="0" conaffinity="0" group="1" mesh="half_round_tip" material="finger_mat"/>
                    <geom size="0.004 0.004 0.0185" pos="0 0.0045 -0.015" type="box" group="0" conaffinity="1" contype="0" name="r_fingertip_g0" friction="0 0 0"/>
                    <geom size="0.0035 0.004 0.0165" pos="0 0.0047 -0.017" type="box"  conaffinity="1" contype="0" name="r_fingerpad_g0" friction="0 0 0"/>
                </body>
            </body>
        </body>
    </worldbody>
  <equality>
    <weld body1="mocap" body2="gripper_base"/>
  </equality>
"""


class GripperRethink(MjShakableOpenCloseGripper, MjScannable):
    def __init__(self, pose: SE3Pose):
        super().__init__(pose, "gripper_base")

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
        base_path = os.path.join(ASSET_PATH, "rethink")
        for file_path in glob.glob(os.path.join(base_path, "**"), recursive=True):
            if os.path.isfile(file_path):
                with open(file_path, "rb") as f:
                    ASSETS[os.path.basename(file_path)] = f.read()

        return (xml, ASSETS)

    def base_to_contact_transform(self) -> SE3Pose:
        pos = np.array([0, 0, -0.11])
        quat = np.array([np.sqrt(2) / 2.0, 0, 0, np.sqrt(2) / 2.0])
        return SE3Pose(pos, quat, type="wxyz")

    def open_gripper(self, sim: MjSimulation):
        sim.data.ctrl[:] = np.array([-0.0115, 0.0115])

    def close_gripper_at(self, sim: MjSimulation, pose: SE3Pose):
        sim.data.mocap_pos = np.copy(pose.pos)
        sim.data.mocap_quat = np.copy(pose.quat)
        sim.data.ctrl[:] = np.array([0.020883, -0.020883])
        mujoco.mj_step(sim.model, sim.data, 3000)  # type: ignore

    def get_actuator_joint_names(self) -> List[str]:
        return [
            "l_finger_joint",
            "r_finger_joint",
        ]

    def get_freejoint_idxs(self, sim: MjSimulation) -> List[int]:
        start_idx = sim.get_joint_idxs(["rethink:base"])[0]
        return list(range(start_idx, start_idx + 7))

    def get_render_options(self):
        options = mujoco.MjvOption()  # type: ignore
        options.geomgroup[0] = 0.0
        options.sitegroup[1] = 0.0
        return options
