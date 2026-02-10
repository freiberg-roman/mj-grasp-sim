"""
These models are used to derive the differentiable surrogates.
See ../../sampler/kin/shadow_constraint/dip_pip_decoder.py
"""

import os
from typing import Any, Dict, List, Tuple

import mujoco
import numpy as np

from mgs.core.simualtion import MjSimulation
from mgs.gripper.base import MjScannable, MjShakableOpenCloseGripper
from mgs.util.const import ASSET_PATH
from mgs.util.geo.transforms import SE3Pose

XML = """
  <compiler angle="radian" meshdir="assets" autolimits="true"/>

  <option cone="elliptic" impratio="10"/>
  <default>
    <default class="right_hand">
      <mesh scale="0.001 0.001 0.001"/>
      <default class="xflip">
        <mesh scale="-0.001 0.001 0.001"/>
      </default>
      <joint axis="1 0 0" damping="0.05" armature="0.001" frictionloss="0.001" margin="0.01"/>
      <position forcerange="-1 1"/>
      <default class="wrist">
        <joint damping="0.5"/>
        <default class="wrist_y">
          <joint axis="0 1 0" range="-0.523599 0.174533"/>
          <position kp="10" ctrlrange="-0.523599 0.174533" forcerange="-10 10"/>
        </default>
        <default class="wrist_x">
          <joint range="-0.698132 0.488692"/>
          <position kp="8" ctrlrange="-0.698132 0.488692" forcerange="-5 5"/>
        </default>
      </default>
      <default class="thumb">
        <default class="thbase">
          <joint axis="0 0 -1" range="-1.0472 1.0472"/>
          <position kp="0.4" ctrlrange="-1.0472 1.0472" forcerange="-3 3"/>
        </default>
        <default class="thproximal">
          <joint range="0 1.22173"/>
          <position ctrlrange="0 1.22173" forcerange="-2 2"/>
        </default>
        <default class="thhub">
          <joint range="-0.20944 0.20944"/>
          <position kp="0.5" ctrlrange="-0.20944 0.20944"/>
        </default>
        <default class="thmiddle">
          <joint axis="0 -1 0" range="-0.698132 0.698132"/>
          <position kp="1.5" ctrlrange="-0.698132 0.698132"/>
        </default>
        <default class="thdistal">
          <joint range="-0.261799 1.5708"/>
          <position ctrlrange="-0.261799 1.5708"/>
        </default>
      </default>
      <default class="metacarpal">
        <joint axis="0.573576 0 0.819152" range="0 0.785398"/>
        <position ctrlrange="0 0.785398"/>
      </default>
      <default class="knuckle">
        <joint axis="0 -1 0" range="-0.349066 0.349066"/>
        <position ctrlrange="-0.349066 0.349066"/>
      </default>
      <default class="proximal">
        <joint range="-0.261799 1.5708"/>
        <position kp="0.5" ctrlrange="-0.261799 1.5708"/>
      </default>
      <default class="middle_distal">
        <joint range="0 1.5708" damping="0.15"/>
        <position kp="2" ctrlrange="0 3.1415" forcerange="-0.465773 0.465773"/>
      </default>
      <default class="coupling_tendon">
        <geom type="cylinder" fromto="-0.01 0 0 0.01 0 0" rgba="0.8 0.2 0.2 0.4" contype="0" conaffinity="0" group="4"/>
        <site type="sphere" size="0.002" group="4"/>
      </default>
      <default class="plastic">
        <geom solimp="0.5 0.99 0.0001" solref="0.005 1"/>
        <default class="plastic_visual">
          <geom type="mesh" material="black" contype="0" conaffinity="0" group="2"/>
        </default>
        <default class="plastic_collision">
          <geom group="3"/>
        </default>
      </default>
    </default>
  </default>

  <asset>
    <material name="black" shininess="0.25" rgba="0.16355 0.16355 0.16355 1"/>
    <material name="gray" specular="0" shininess="0.25" rgba="0.80848 0.80848 0.80848 1"/>
    <material name="metallic" specular="0" shininess="0.25" rgba="0.9 0.9 0.9 1"/>
    <mesh class="right_hand" file="forearm_0.obj"/>
    <mesh class="right_hand" file="forearm_1.obj"/>
    <mesh class="right_hand" file="wrist.obj"/>
    <mesh class="right_hand" file="palm.obj"/>
    <mesh class="right_hand" file="f_knuckle.obj"/>
    <mesh class="right_hand" file="f_proximal.obj"/>
    <mesh class="right_hand" file="f_middle.obj"/>
    <mesh class="right_hand" file="f_distal_pst.obj"/>
    <mesh class="right_hand" file="lf_metacarpal.obj"/>
    <mesh class="right_hand" file="th_proximal.obj"/>
    <mesh class="right_hand" file="th_middle.obj"/>
    <mesh class="right_hand" file="th_distal_pst.obj"/>
  </asset>

  <worldbody>
    <body name="mocap" mocap="true" pos="{position}" quat="{quaternion}"/>
      <!-- Wrist -->
    <body name="rh_wrist" pos="{position}" quat="{quaternion}">
      <freejoint name="freejoint"/>
      <inertial mass="0.1" pos="0 0 0.029" quat="1 1 1 1" diaginertia="6.4e-05 4.38e-05 3.5e-05"/>
      <!-- Palm -->
      <body name="rh_palm" pos="0 0 0.034">
        <inertial mass="0.3" pos="0 0 0.035" quat="1 0 0 1" diaginertia="0.0005287 0.0003581 0.000191"/>
          <body name="rh_mfproximal">
            <inertial mass="0.03" pos="0 0 0.0225" quat="1 0 0 1" diaginertia="1e-05 9.8e-06 1.8e-06"/>
            <geom class="plastic_visual" mesh="f_proximal"/>
            <site class="coupling_tendon" name="coupling_MFJ3_tendon_end" pos="0 -0.007 0.023"/>
            <body name="rh_mfmiddle" pos="0 0 0.045">
              <inertial mass="0.017" pos="0 0 0.0125" quat="1 0 0 1" diaginertia="2.7e-06 2.6e-06 8.7e-07"/>
              <joint name="rh_MFJ2" class="middle_distal"/>
              <geom class="plastic_visual" mesh="f_middle"/>
              <geom class="coupling_tendon" name="coupling_MFJ2_pulley" size="0.01125"/>
              <site class="coupling_tendon" name="coupling_MFJ2_pulley_inside" pos="0 -0.011 0.011"/>
              <site class="coupling_tendon" name="coupling_MFJ2_between_pulleys" pos="0 -0.005 0.008"/>
              <body name="rh_mfdistal" pos="0 0 0.025">
                <inertial mass="0.013" pos="0 0 0.0130769" quat="1 0 0 1"
                  diaginertia="1.28092e-06 1.12092e-06 5.3e-07"/>
                <joint name="rh_MFJ1" class="middle_distal"/>
                <geom class="plastic_visual" mesh="f_distal_pst"/>
                <geom class="coupling_tendon" name="coupling_MFJ1_pulley" size="0.008929"/>
                <site class="coupling_tendon" name="coupling_MFJ1_pulley_outside" pos="0 0.010 -0.01"/>
                <site class="coupling_tendon" name="coupling_MFJ1_tendon_end" pos="0 0.007 0.015"/>
              </body>
            </body>
          </body>
      </body>
    </body>
  </worldbody>

  <tendon>
    <fixed name="rh_MFT0" range="0 1.5707" group="4">
      <joint coef="1" joint="rh_MFJ2"/>
      <joint coef="-1" joint="rh_MFJ1"/>
    </fixed>
    <fixed name="rh_MFT1" range="0 3.14159" group="4">
      <joint coef="1" joint="rh_MFJ2"/>
      <joint coef="1" joint="rh_MFJ1"/>
    </fixed>
    <spatial name="rh_MFT2" range="0 0.080876" springlength="0.059349" stiffness="4023.08" width="0.0005">
      <site site="coupling_MFJ1_tendon_end"/>
      <geom geom="coupling_MFJ1_pulley" sidesite="coupling_MFJ1_pulley_outside"/>
      <site site="coupling_MFJ2_between_pulleys"/>
      <geom geom="coupling_MFJ2_pulley" sidesite="coupling_MFJ2_pulley_inside"/>
      <site site="coupling_MFJ3_tendon_end"/>
    </spatial>
  </tendon>

  <actuator>
    <!-- Middle finger -->
    <position name="rh_MFJ0" tendon="rh_MFT1" class="middle_distal" forcerange="-0.616938 0.616938"/>
  </actuator>
  <equality>
    <weld body1="mocap" body2="rh_wrist"/>
  </equality>

"""


class ShadowRightMF(MjShakableOpenCloseGripper, MjScannable):
    def __init__(self, pose: SE3Pose, grasp_type=None):
        super().__init__(pose, "rh_wrist")

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
        base_path = os.path.join(ASSET_PATH, "shadow-constraint")
        for file_name in os.listdir(base_path):
            path = os.path.join(base_path, file_name)
            with open(path, "rb") as f:
                ASSETS[file_name] = f.read()

        return (xml, ASSETS)

    def base_to_contact_transform(self) -> SE3Pose:
        pos = np.array([0, 0, 0.0])
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        return SE3Pose(pos, quat, type="wxyz")

    def open_gripper(self, sim: MjSimulation):
        gripper_idxs = sim.get_joint_idxs(self.get_actuator_joint_names())
        open_pose = np.zeros(shape=(22,))
        sim.set_qpos(open_pose, gripper_idxs)  # type: ignore
        sim.data.ctrl[:] = self._qpos_to_qacc(np.copy(open_pose))  # type: ignore

    def close_gripper_at(self, sim: MjSimulation, pose: SE3Pose):
        pass

    def get_freejoint_idxs(self, sim: MjSimulation) -> List[int]:
        start_idx = sim.get_joint_idxs(["freejoint"])[0]
        return list(range(start_idx, start_idx + 7))

    def get_actuator_joint_names(
        self,
    ) -> List[str]:
        return [
            "rh_MFJ2",  # 0
            "rh_MFJ1",  # 1
        ]
