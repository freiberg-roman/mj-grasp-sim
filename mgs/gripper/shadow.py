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
# Copyright 2022 Shadow Robot Company Ltd License: Apache-2.0
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
XML = """
  <compiler angle="radian" meshdir="assets" autolimits="true"/>

  <option cone="elliptic" impratio="10"/>

  <default>
    <default class="right_hand">
      <mesh scale="0.001 0.001 0.001"/>
      <joint axis="1 0 0" damping="0.05" armature="0.0002" frictionloss="0.01"/>
      <position forcerange="-10 10"/>

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
          <position kp="0.4" ctrlrange="-1.0472 1.0472" forcerange="-30 30"/>
        </default>
        <default class="thproximal">
          <joint range="0 1.22173"/>
          <position ctrlrange="0 1.22173" forcerange="-20 20"/>
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
        <position ctrlrange="-0.261799 1.5708"/>
      </default>
      <default class="middle_distal">
        <joint range="0 1.5708"/>
        <position kp="0.5" ctrlrange="0 3.1415"/>
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
    <material name="black" specular="0.5" shininess="0.25" rgba="0.16355 0.16355 0.16355 1"/>
    <material name="gray" specular="0.0" shininess="0.25" rgba="0.80848 0.80848 0.80848 1"/>
    <material name="metallic" specular="0" shininess="0.25" rgba="0.9 0.9 0.9 1"/>

    <mesh class="right_hand" file="forearm_0.obj"/>
    <mesh class="right_hand" file="forearm_1.obj"/>
    <mesh class="right_hand" file="forearm_collision.obj"/>
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
    <body name="rh_forearm" childclass="right_hand" pos="{position}" quat="{quaternion}">
      <freejoint name="freejoint"/>
      <inertial mass="3" pos="0 0 0.09" diaginertia="0.0138 0.0138 0.00744"/>
      <geom class="plastic_visual" mesh="forearm_0" material="gray"/>
      <geom class="plastic_visual" mesh="forearm_1"/>
      <geom class="plastic_collision" type="mesh" mesh="forearm_collision" friction="4.0"/>
      <geom class="plastic_collision" size="0.035 0.035 0.035" pos="0.01 0.0 0.181" quat="0.380188 0.924909 0 0" friction="4.0" type="box"/>
      <body name="rh_wrist" pos="0.01 0 0.21301" quat="1 0 0 1">
        <inertial mass="0.1" pos="0 0 0.029" quat="0.5 0.5 0.5 0.5" diaginertia="6.4e-05 4.38e-05 3.5e-05"/>
        <joint class="wrist_y" name="rh_WRJ2"/>
        <geom class="plastic_visual" mesh="wrist" material="metallic"/>
        <geom size="0.0135 0.015" quat="0.499998 0.5 0.5 -0.500002" type="cylinder" class="plastic_collision" friction="4.0"/>
        <geom size="0.011 0.005" pos="-0.026 0 0.034" quat="1 0 1 0" type="cylinder" class="plastic_collision" friction="4.0"/>
        <geom size="0.011 0.005" pos="0.031 0 0.034" quat="1 0 1 0" type="cylinder" class="plastic_collision" friction="4.0"/>
        <geom size="0.0135 0.009 0.005" pos="-0.021 0 0.011" quat="0.923879 0 0.382684 0" type="box"
          class="plastic_collision" friction="4.0"/>
        <geom size="0.0135 0.009 0.005" pos="0.026 0 0.01" quat="0.923879 0 -0.382684 0" type="box"
          class="plastic_collision" friction="4.0"/>
        <body name="rh_palm" pos="0 0 0.034">
          <inertial mass="0.3" pos="0 0 0.035" quat="1 0 0 1" diaginertia="0.0005287 0.0003581 0.000191"/>
          <joint class="wrist_x" name="rh_WRJ1"/>
          <site name="grasp_site" pos="0 -.035 0.09" group="4"/>
          <geom class="plastic_visual" mesh="palm"/>
          <geom size="0.031 0.0035 0.049" pos="0.011 0.0085 0.038" type="box" class="plastic_collision" friction="4.0"/>
          <geom size="0.018 0.0085 0.049" pos="-0.002 -0.0035 0.038" type="box" class="plastic_collision" friction="4.0"/>
          <geom size="0.013 0.0085 0.005" pos="0.029 -0.0035 0.082" type="box" class="plastic_collision" friction="4.0" />
          <geom size="0.013 0.007 0.009" pos="0.0265 -0.001 0.07" quat="0.987241 0.0990545 0.0124467 0.124052"
            type="box" class="plastic_collision" friction="4.0"/>
          <geom size="0.0105 0.0135 0.012" pos="0.0315 -0.0085 0.001" type="box" class="plastic_collision" friction="4.0"/>
          <geom size="0.011 0.0025 0.015" pos="0.0125 -0.015 0.004" quat="0.971338 0 0 -0.237703" type="box"
            class="plastic_collision" friction="4.0"/>
          <geom size="0.009 0.012 0.002" pos="0.011 0 0.089" type="box" class="plastic_collision" friction="4.0"/>
          <geom size="0.01 0.012 0.02" pos="-0.03 0 0.009" type="box" class="plastic_collision" friction="4.0"/>
          <body name="rh_ffknuckle" pos="0.033 0 0.095">
            <inertial mass="0.008" pos="0 0 0" quat="0.5 0.5 -0.5 0.5" diaginertia="3.2e-07 2.6e-07 2.6e-07"/>
            <joint name="rh_FFJ4" class="knuckle"/>
            <geom pos="0 0 0.0005" class="plastic_visual" mesh="f_knuckle" material="metallic"/>
            <geom size="0.009 0.009" quat="1 0 1 0" type="cylinder" class="plastic_collision" friction="4.0"/>
            <body name="rh_ffproximal">
              <inertial mass="0.03" pos="0 0 0.0225" quat="1 0 0 1" diaginertia="1e-05 9.8e-06 1.8e-06"/>
              <joint name="rh_FFJ3" class="proximal"/>
              <geom class="plastic_visual" mesh="f_proximal"/>
              <geom size="0.009 0.02" pos="0 0 0.025" type="capsule" class="plastic_collision" friction="4.0"/>
              <body name="rh_ffmiddle" pos="0 0 0.045">
                <inertial mass="0.017" pos="0 0 0.0125" quat="1 0 0 1" diaginertia="2.7e-06 2.6e-06 8.7e-07"/>
                <joint name="rh_FFJ2" class="middle_distal"/>
                <geom class="plastic_visual" mesh="f_middle"/>
                <geom size="0.009 0.0125" pos="0 0 0.0125" type="capsule" class="plastic_collision" friction="4.0"/>
                <body name="rh_ffdistal" pos="0 0 0.025">
                  <inertial mass="0.013" pos="0 0 0.0130769" quat="1 0 0 1"
                    diaginertia="1.28092e-06 1.12092e-06 5.3e-07"/>
                  <joint name="rh_FFJ1" class="middle_distal"/>
                  <geom class="plastic_visual" mesh="f_distal_pst"/>
                  <geom class="plastic_collision" type="mesh" mesh="f_distal_pst" friction="4.0"/>
                </body>
              </body>
            </body>
          </body>
          <body name="rh_mfknuckle" pos="0.011 0 0.099">
            <inertial mass="0.008" pos="0 0 0" quat="0.5 0.5 -0.5 0.5" diaginertia="3.2e-07 2.6e-07 2.6e-07"/>
            <joint name="rh_MFJ4" class="knuckle"/>
            <geom pos="0 0 0.0005" class="plastic_visual" mesh="f_knuckle" material="metallic"/>
            <geom size="0.009 0.009" quat="1 0 1 0" type="cylinder" class="plastic_collision" friction="4.0"/>
            <body name="rh_mfproximal">
              <inertial mass="0.03" pos="0 0 0.0225" quat="1 0 0 1" diaginertia="1e-05 9.8e-06 1.8e-06"/>
              <joint name="rh_MFJ3" class="proximal"/>
              <geom class="plastic_visual" mesh="f_proximal"/>
              <geom size="0.009 0.02" pos="0 0 0.025" type="capsule" class="plastic_collision" friction="4.0"/>
              <body name="rh_mfmiddle" pos="0 0 0.045">
                <inertial mass="0.017" pos="0 0 0.0125" quat="1 0 0 1" diaginertia="2.7e-06 2.6e-06 8.7e-07"/>
                <joint name="rh_MFJ2" class="middle_distal"/>
                <geom class="plastic_visual" mesh="f_middle"/>
                <geom size="0.009 0.0125" pos="0 0 0.0125" type="capsule" class="plastic_collision" friction="4.0"/>
                <body name="rh_mfdistal" pos="0 0 0.025">
                  <inertial mass="0.013" pos="0 0 0.0130769" quat="1 0 0 1"
                    diaginertia="1.28092e-06 1.12092e-06 5.3e-07"/>
                  <joint name="rh_MFJ1" class="middle_distal"/>
                  <geom class="plastic_visual" mesh="f_distal_pst"/>
                  <geom class="plastic_collision" type="mesh" mesh="f_distal_pst" friction="4.0"/>
                </body>
              </body>
            </body>
          </body>
          <body name="rh_rfknuckle" pos="-0.011 0 0.095">
            <inertial mass="0.008" pos="0 0 0" quat="0.5 0.5 -0.5 0.5" diaginertia="3.2e-07 2.6e-07 2.6e-07"/>
            <joint name="rh_RFJ4" class="knuckle" axis="0 1 0"/>
            <geom pos="0 0 0.0005" class="plastic_visual" mesh="f_knuckle" material="metallic"/>
            <geom size="0.009 0.009" quat="1 0 1 0" type="cylinder" class="plastic_collision" friction="4.0"/>
            <body name="rh_rfproximal">
              <inertial mass="0.03" pos="0 0 0.0225" quat="1 0 0 1" diaginertia="1e-05 9.8e-06 1.8e-06"/>
              <joint name="rh_RFJ3" class="proximal"/>
              <geom class="plastic_visual" mesh="f_proximal"/>
              <geom size="0.009 0.02" pos="0 0 0.025" type="capsule" class="plastic_collision" friction="4.0"/>
              <body name="rh_rfmiddle" pos="0 0 0.045">
                <inertial mass="0.017" pos="0 0 0.0125" quat="1 0 0 1" diaginertia="2.7e-06 2.6e-06 8.7e-07"/>
                <joint name="rh_RFJ2" class="middle_distal"/>
                <geom class="plastic_visual" mesh="f_middle"/>
                <geom size="0.009 0.0125" pos="0 0 0.0125" type="capsule" class="plastic_collision" friction="4.0"/>
                <body name="rh_rfdistal" pos="0 0 0.025">
                  <inertial mass="0.013" pos="0 0 0.0130769" quat="1 0 0 1"
                    diaginertia="1.28092e-06 1.12092e-06 5.3e-07"/>
                  <joint name="rh_RFJ1" class="middle_distal"/>
                  <geom class="plastic_visual" mesh="f_distal_pst"/>
                  <geom class="plastic_collision" type="mesh" mesh="f_distal_pst" friction="4.0"/>
                </body>
              </body>
            </body>
          </body>
          <body name="rh_lfmetacarpal" pos="-0.033 0 0.02071">
            <inertial mass="0.03" pos="0 0 0.04" quat="1 0 0 1" diaginertia="1.638e-05 1.45e-05 4.272e-06"/>
            <joint name="rh_LFJ5" class="metacarpal"/>
            <geom class="plastic_visual" mesh="lf_metacarpal"/>
            <geom size="0.011 0.012 0.025" pos="0.002 0 0.033" type="box" class="plastic_collision" friction="4.0"/>
            <body name="rh_lfknuckle" pos="0 0 0.06579">
              <inertial mass="0.008" pos="0 0 0" quat="0.5 0.5 -0.5 0.5" diaginertia="3.2e-07 2.6e-07 2.6e-07"/>
              <joint name="rh_LFJ4" class="knuckle" axis="0 1 0"/>
              <geom pos="0 0 0.0005" class="plastic_visual" mesh="f_knuckle" material="metallic"/>
              <geom size="0.009 0.009" quat="1 0 1 0" type="cylinder" class="plastic_collision" friction="4.0"/>
              <body name="rh_lfproximal">
                <inertial mass="0.03" pos="0 0 0.0225" quat="1 0 0 1" diaginertia="1e-05 9.8e-06 1.8e-06"/>
                <joint name="rh_LFJ3" class="proximal"/>
                <geom class="plastic_visual" mesh="f_proximal"/>
                <geom size="0.009 0.02" pos="0 0 0.025" type="capsule" class="plastic_collision" friction="4.0"/>
                <body name="rh_lfmiddle" pos="0 0 0.045">
                  <inertial mass="0.017" pos="0 0 0.0125" quat="1 0 0 1" diaginertia="2.7e-06 2.6e-06 8.7e-07"/>
                  <joint name="rh_LFJ2" class="middle_distal"/>
                  <geom class="plastic_visual" mesh="f_middle"/>
                  <geom size="0.009 0.0125" pos="0 0 0.0125" type="capsule" class="plastic_collision" friction="4.0"/>
                  <body name="rh_lfdistal" pos="0 0 0.025">
                    <inertial mass="0.013" pos="0 0 0.0130769" quat="1 0 0 1"
                      diaginertia="1.28092e-06 1.12092e-06 5.3e-07"/>
                    <joint name="rh_LFJ1" class="middle_distal"/>
                    <geom class="plastic_visual" mesh="f_distal_pst"/>
                    <geom class="plastic_collision" type="mesh" mesh="f_distal_pst" friction="4.0"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
          <body name="rh_thbase" pos="0.034 -0.00858 0.029" quat="0.92388 0 0.382683 0">
            <inertial mass="0.01" pos="0 0 0" diaginertia="1.6e-07 1.6e-07 1.6e-07"/>
            <joint name="rh_THJ5" class="thbase"/>
            <geom class="plastic_collision" size="0.013" friction="4.0"/>
            <body name="rh_thproximal">
              <inertial mass="0.04" pos="0 0 0.019" diaginertia="1.36e-05 1.36e-05 3.13e-06"/>
              <joint name="rh_THJ4" class="thproximal"/>
              <geom class="plastic_visual" mesh="th_proximal"/>
              <geom class="plastic_collision" size="0.0105 0.009" pos="0 0 0.02" type="capsule" friction="4.0"/>
              <body name="rh_thhub" pos="0 0 0.038">
                <inertial mass="0.005" pos="0 0 0" diaginertia="1e-06 1e-06 3e-07"/>
                <joint name="rh_THJ3" class="thhub"/>
                <geom size="0.011" class="plastic_collision" friction="4.0"/>
                <body name="rh_thmiddle">
                  <inertial mass="0.02" pos="0 0 0.016" diaginertia="5.1e-06 5.1e-06 1.21e-06"/>
                  <joint name="rh_THJ2" class="thmiddle"/>
                  <geom class="plastic_visual" mesh="th_middle"/>
                  <geom size="0.009 0.009" pos="0 0 0.012" type="capsule" class="plastic_collision" friction="4.0"/>
                  <geom size="0.01" pos="0 0 0.03" class="plastic_collision" friction="4.0"/>
                  <body name="rh_thdistal" pos="0 0 0.032" quat="1 0 0 -1">
                    <inertial mass="0.017" pos="0 0 0.0145588" quat="1 0 0 1"
                      diaginertia="2.37794e-06 2.27794e-06 1e-06"/>
                    <joint name="rh_THJ1" class="thdistal"/>
                    <geom class="plastic_visual" mesh="th_distal_pst"/>
                    <geom class="plastic_collision" type="mesh" mesh="th_distal_pst" friction="4.0"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <contact>
    <exclude body1="rh_wrist" body2="rh_forearm"/>
    <exclude body1="rh_thproximal" body2="rh_thmiddle"/>
  </contact>

  <tendon>
    <fixed name="rh_FFJ0">
      <joint joint="rh_FFJ2" coef="1"/>
      <joint joint="rh_FFJ1" coef="1"/>
    </fixed>
    <fixed name="rh_MFJ0">
      <joint joint="rh_MFJ2" coef="1"/>
      <joint joint="rh_MFJ1" coef="1"/>
    </fixed>
    <fixed name="rh_RFJ0">
      <joint joint="rh_RFJ2" coef="1"/>
      <joint joint="rh_RFJ1" coef="1"/>
    </fixed>
    <fixed name="rh_LFJ0">
      <joint joint="rh_LFJ2" coef="1"/>
      <joint joint="rh_LFJ1" coef="1"/>
    </fixed>
  </tendon>

  <actuator>
    <position name="rh_A_WRJ2" joint="rh_WRJ2" class="wrist_y"/>
    <position name="rh_A_WRJ1" joint="rh_WRJ1" class="wrist_x"/>
    <position name="rh_A_THJ5" joint="rh_THJ5" class="thbase"/>
    <position name="rh_A_THJ4" joint="rh_THJ4" class="thproximal"/>
    <position name="rh_A_THJ3" joint="rh_THJ3" class="thhub"/>
    <position name="rh_A_THJ2" joint="rh_THJ2" class="thmiddle"/>
    <position name="rh_A_THJ1" joint="rh_THJ1" class="thdistal"/>
    <position name="rh_A_FFJ4" joint="rh_FFJ4" class="knuckle"/>
    <position name="rh_A_FFJ3" joint="rh_FFJ3" class="proximal"/>
    <position name="rh_A_FFJ0" tendon="rh_FFJ0" class="middle_distal"/>
    <position name="rh_A_MFJ4" joint="rh_MFJ4" class="knuckle"/>
    <position name="rh_A_MFJ3" joint="rh_MFJ3" class="proximal"/>
    <position name="rh_A_MFJ0" tendon="rh_MFJ0" class="middle_distal"/>
    <position name="rh_A_RFJ4" joint="rh_RFJ4" class="knuckle"/>
    <position name="rh_A_RFJ3" joint="rh_RFJ3" class="proximal"/>
    <position name="rh_A_RFJ0" tendon="rh_RFJ0" class="middle_distal"/>
    <position name="rh_A_LFJ5" joint="rh_LFJ5" class="metacarpal"/>
    <position name="rh_A_LFJ4" joint="rh_LFJ4" class="knuckle"/>
    <position name="rh_A_LFJ3" joint="rh_LFJ3" class="proximal"/>
    <position name="rh_A_LFJ0" tendon="rh_LFJ0" class="middle_distal"/>
  </actuator>
  <equality>
    <weld body1="mocap" body2="rh_forearm"/>
  </equality>
"""


class GripperShadowRight(MjShakableOpenCloseGripper, MjScannable):
    def __init__(self, pose: SE3Pose, grasp_type=None):
        super().__init__(pose, "rh_wrist")

        self.type = grasp_type
        self.close_poses = {
            "two_finger_pinch": np.array(
                [
                    0.0188,
                    -0.2686,
                    -0.3013,
                    0.9774,
                    1.045,
                    0.1665,
                    0.1255,
                    0.165,
                    0.3459,
                    0.0005741,
                    -0.1349,
                    0.4096,
                    0.04252,
                    0.03161,
                    0.1938,
                    -0.2039,
                    0.05253,
                    0.4297,
                    0.009526,
                    0.03082,
                    1.08,
                    0.1604,
                    0.5839,
                    0.3069,
                ]
            ),
            "three_finger_pinch": np.array(
                [
                    -0.01977,
                    -0.5255,
                    -0.3464,
                    1.253,
                    0.7836,
                    -0.001106,
                    0.01103,
                    1.475,
                    0.6181,
                    0.0155,
                    -0.2083,
                    0.3328,
                    0.07129,
                    0.02873,
                    0.1829,
                    -0.2676,
                    0.05465,
                    0.3892,
                    0.008468,
                    0.07708,
                    1.21,
                    0.2023,
                    0.6614,
                    0.0102,
                ]
            ),
            "grasp_hard": np.array(
                [
                    -0.06713,
                    -0.5377,
                    -0.3099,
                    1.128,
                    0.5067,
                    0.001659,
                    -0.1352,
                    0.8652,
                    0.93,
                    0.01263,
                    0.1724,
                    1.165,
                    0.2726,
                    0.034,
                    0.3116,
                    -0.354,
                    0.9553,
                    0.7928,
                    0.01958,
                    0.05755,
                    1.182,
                    0.1947,
                    0.5578,
                    0.2482,
                ]
            ),
            "full_grasp": np.array(
                [
                    0,
                    0,
                    0,
                    1.571,
                    1.571,
                    1.571,
                    0,
                    1.571,
                    1.571,
                    1.571,
                    0,
                    1.571,
                    1.571,
                    1.571,
                    0,
                    0,
                    1.571,
                    1.571,
                    1.571,
                    0.17,
                    1.2,
                    0,
                    0.61,
                    0.52,
                ]
            ),
        }
        self.open_poses = {
            "two_finger_pinch": np.array(
                [
                    0.0188,
                    -0.2686,
                    -0.3013,
                    -0.262,
                    1.045,
                    0.1665,
                    0.1255,
                    0.165,
                    0.3459,
                    0.0005741,
                    -0.1349,
                    0.4096,
                    0.04252,
                    0.03161,
                    0.1938,
                    -0.2039,
                    0.05253,
                    0.4297,
                    0.009526,
                    -0.5,
                    1.08,
                    0.1604,
                    0.5839,
                    0.3069,
                ]
            ),
            "three_finger_pinch": np.array(
                [
                    -0.01977,
                    -0.5255,
                    -0.3464,
                    0.645,
                    0.7836,
                    -0.001106,
                    0.01103,
                    0.81,
                    0.6181,
                    0.0155,
                    -0.2083,
                    0.3328,
                    0.07129,
                    0.02873,
                    0.1829,
                    -0.2676,
                    0.05465,
                    0.3892,
                    0.008468,
                    -0.7,
                    1.21,
                    0.2023,
                    0.6614,
                    0.0102,
                ]
            ),
            "grasp_hard": np.array(
                [
                    -0.03896,
                    -0.5694,
                    -0.3497,
                    0.1074,
                    0.1424,
                    -0.008296,
                    -0.05406,
                    -0.07509,
                    0.4658,
                    0.01033,
                    -0.1062,
                    0.2738,
                    0.001251,
                    0.02586,
                    0.1551,
                    -0.3593,
                    0.1445,
                    -0.00691,
                    -0.001588,
                    -0.1413,
                    0.8383,
                    0.199,
                    0.4945,
                    0.1291,
                ]
            ),
            "full_grasp": np.array([0.0] * 24),
        }

        self.offset = {
            "two_finger_pinch": np.array([-0.065, -0.035, -0.346]),
            "three_finger_pinch": np.array([-0.05, -0.02, -0.35]),
            "grasp_hard": np.array([-0.02, -0.0, -0.36]),
            "full_grasp": np.array([-0.02, -0.0, -0.36]),
        }

        self.rotation = {
            "two_finger_pinch": np.pi * (5.2 / 3.0),
            "three_finger_pinch": np.pi * (5.2 / 3.0),
            "grasp_hard": np.pi * (5.2 / 3.0),
            "full_grasp": np.pi * (5.2 / 3.0),
        }

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
        base_path = os.path.join(ASSET_PATH, "shadow")
        for file_name in os.listdir(base_path):
            path = os.path.join(base_path, file_name)
            with open(path, "rb") as f:
                ASSETS[file_name] = f.read()

        return (xml, ASSETS)

    def base_to_contact_transform(self) -> SE3Pose:
        theta = self.rotation[self.type]  # type: ignore
        rot_offset = SE3Pose(np.array([0, 0, 0]), np.array([np.cos(theta / 2.0), 0.0, np.sin(theta / 2.0), 0.0]), type="wxyz")  # type: ignore
        offset = rot_offset @ SE3Pose(
            self.offset[self.type], np.array([1.0, 0, 0, 0]), type="wxyz"
        )
        return SE3Pose(offset.pos, np.array([np.cos(theta / 2.0), 0.0, np.sin(theta / 2.0), 0.0]), type="wxyz")  # type: ignore

    def open_gripper(self, sim: MjSimulation):
        gripper_idxs = sim.get_joint_idxs(self.get_actuator_joint_names())
        sim.set_qpos(np.array(self.open_poses[self.type]), gripper_idxs)  # type: ignore
        sim.data.ctrl[:] = self._qpos_to_qacc(np.copy(self.open_poses[self.type]))  # type: ignore

    def close_gripper_at(self, sim: MjSimulation, pose: SE3Pose):
        sim.data.mocap_pos = pose.pos
        sim.data.mocap_quat = pose.quat
        sim.data.ctrl[:] = self._qpos_to_qacc(self.close_poses[self.type])  # type: ignore
        mujoco.mj_step(sim.model, sim.data, 3000)  # type: ignore

    def get_freejoint_idxs(self, sim: MjSimulation) -> List[int]:
        start_idx = sim.get_joint_idxs(["freejoint"])[0]
        return list(range(start_idx, start_idx + 7))

    def get_actuator_joint_names(
        self,
    ) -> List[str]:
        return [
            "rh_WRJ2",
            "rh_WRJ1",
            "rh_FFJ4",
            "rh_FFJ3",
            "rh_FFJ2",
            "rh_FFJ1",
            "rh_MFJ4",
            "rh_MFJ3",
            "rh_MFJ2",
            "rh_MFJ1",
            "rh_RFJ4",
            "rh_RFJ3",
            "rh_RFJ2",
            "rh_RFJ1",
            "rh_LFJ5",
            "rh_LFJ4",
            "rh_LFJ3",
            "rh_LFJ2",
            "rh_LFJ1",
            "rh_THJ5",
            "rh_THJ4",
            "rh_THJ3",
            "rh_THJ2",
            "rh_THJ1",
        ]

    def _qpos_to_qacc(self, qpos):
        acc = np.zeros((20,))
        acc[:2] = qpos[:2]
        acc[2:7] = qpos[-5:]
        acc[7:9] = qpos[2:4]
        acc[9] = qpos[4] + qpos[5]
        acc[10:12] = qpos[6:8]
        acc[12] = qpos[8] + qpos[9]
        acc[13:15] = qpos[10:12]
        acc[15] = qpos[12] + qpos[13]
        acc[16:19] = qpos[14:17]
        acc[19] = qpos[17] + qpos[18]
        return acc
