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
    <!-- Approximate convex decompositions -->
    <mesh class="right_hand" file="forearm_1_c_0.stl"/>
    <mesh class="right_hand" file="forearm_1_c_1.stl"/>
    <mesh class="right_hand" file="forearm_1_c_2.stl"/>
    <mesh class="xflip" name="palm_c_0" file="palm_c_0_flip.stl"/>
    <mesh class="xflip" name="palm_c_1" file="palm_c_1_flip.stl"/>
    <mesh class="xflip" name="palm_c_2" file="palm_c_2_flip.stl"/>
    <mesh class="xflip" name="palm_c_3" file="palm_c_3_flip.stl"/>
    <mesh class="xflip" name="lf_metacarpal_c_0" file="lf_metacarpal_c_0_flip.stl"/>
    <mesh class="xflip" name="lf_metacarpal_c_1" file="lf_metacarpal_c_1_flip.stl"/>
  </asset>

  <worldbody>
    <body name="mocap" mocap="true" pos="{position}" quat="{quaternion}"/>
      <!-- Wrist -->
    <body name="rh_wrist" pos="{position}" quat="{quaternion}">
      <freejoint name="freejoint"/>
      <inertial mass="0.1" pos="0 0 0.029" quat="1 1 1 1" diaginertia="6.4e-05 4.38e-05 3.5e-05"/>
      <geom class="plastic_visual" mesh="wrist" material="metallic"/>
      <!-- Palm -->
      <body name="rh_palm" pos="0 0 0.034">
        <inertial mass="0.3" pos="0 0 0.035" quat="1 0 0 1" diaginertia="0.0005287 0.0003581 0.000191"/>
        <joint class="wrist_x" name="rh_WRJ1"/>
        <geom class="plastic_visual" mesh="palm"/>
        <geom class="plastic_collision" mesh="palm_c_0" type="mesh"/>
        <geom class="plastic_collision" mesh="palm_c_1" type="mesh"/>
        <geom class="plastic_collision" mesh="palm_c_2" type="mesh"/>
        <geom class="plastic_collision" mesh="palm_c_3" type="mesh"/>
        <!-- First finger -->
        <body name="rh_ffknuckle" pos="0.033 0 0.095">
          <inertial mass="0.008" pos="0 0 0" quat="1 1 -1 1" diaginertia="3.2e-07 2.6e-07 2.6e-07"/>
          <joint name="rh_FFJ4" class="knuckle"/>
          <geom pos="0 0 0.0005" class="plastic_visual" mesh="f_knuckle" material="metallic"/>
          <body name="rh_ffproximal">
            <inertial mass="0.03" pos="0 0 0.0225" quat="1 0 0 1" diaginertia="1e-05 9.8e-06 1.8e-06"/>
            <joint name="rh_FFJ3" class="proximal"/>
            <geom class="plastic_visual" mesh="f_proximal"/>
            <geom size="0.009 0.02" pos="0 0 0.025" type="capsule" class="plastic_collision"/>
            <site class="coupling_tendon" name="coupling_FFJ3_tendon_end" pos="0 -0.007 0.023"/>
            <body name="rh_ffmiddle" pos="0 0 0.045">
              <inertial mass="0.017" pos="0 0 0.0125" quat="1 0 0 1" diaginertia="2.7e-06 2.6e-06 8.7e-07"/>
              <joint name="rh_FFJ2" class="middle_distal"/>
              <geom class="plastic_visual" mesh="f_middle"/>
              <geom size="0.009 0.0125" pos="0 0 0.0125" type="capsule" class="plastic_collision"/>
              <geom class="coupling_tendon" name="coupling_FFJ2_pulley" size="0.01125"/>
              <site class="coupling_tendon" name="coupling_FFJ2_pulley_inside" pos="0 -0.011 0.011"/>
              <site class="coupling_tendon" name="coupling_FFJ2_between_pulleys" pos="0 -0.005 0.008"/>
              <body name="rh_ffdistal" pos="0 0 0.025">
                <inertial mass="0.013" pos="0 0 0.0130769" quat="1 0 0 1"
                  diaginertia="1.28092e-06 1.12092e-06 5.3e-07"/>
                <joint name="rh_FFJ1" class="middle_distal"/>
                <geom class="plastic_visual" mesh="f_distal_pst"/>
                <geom class="plastic_collision" type="mesh" mesh="f_distal_pst"/>
                <geom class="coupling_tendon" name="coupling_FFJ1_pulley" size="0.008929"/>
                <site class="coupling_tendon" name="coupling_FFJ1_pulley_outside" pos="0 0.010 -0.01"/>
                <site class="coupling_tendon" name="coupling_FFJ1_tendon_end" pos="0 0.007 0.015"/>
              </body>
            </body>
          </body>
        </body>
        <!-- Middle finger -->
        <body name="rh_mfknuckle" pos="0.011 0 0.099">
          <inertial mass="0.008" pos="0 0 0" quat="1 1 -1 1" diaginertia="3.2e-07 2.6e-07 2.6e-07"/>
          <joint name="rh_MFJ4" class="knuckle"/>
          <geom pos="0 0 0.0005" class="plastic_visual" mesh="f_knuckle" material="metallic"/>
          <body name="rh_mfproximal">
            <inertial mass="0.03" pos="0 0 0.0225" quat="1 0 0 1" diaginertia="1e-05 9.8e-06 1.8e-06"/>
            <joint name="rh_MFJ3" class="proximal"/>
            <geom class="plastic_visual" mesh="f_proximal"/>
            <geom size="0.009 0.02" pos="0 0 0.025" type="capsule" class="plastic_collision"/>
            <site class="coupling_tendon" name="coupling_MFJ3_tendon_end" pos="0 -0.007 0.023"/>
            <body name="rh_mfmiddle" pos="0 0 0.045">
              <inertial mass="0.017" pos="0 0 0.0125" quat="1 0 0 1" diaginertia="2.7e-06 2.6e-06 8.7e-07"/>
              <joint name="rh_MFJ2" class="middle_distal"/>
              <geom class="plastic_visual" mesh="f_middle"/>
              <geom size="0.009 0.0125" pos="0 0 0.0125" type="capsule" class="plastic_collision"/>
              <geom class="coupling_tendon" name="coupling_MFJ2_pulley" size="0.01125"/>
              <site class="coupling_tendon" name="coupling_MFJ2_pulley_inside" pos="0 -0.011 0.011"/>
              <site class="coupling_tendon" name="coupling_MFJ2_between_pulleys" pos="0 -0.005 0.008"/>
              <body name="rh_mfdistal" pos="0 0 0.025">
                <inertial mass="0.013" pos="0 0 0.0130769" quat="1 0 0 1"
                  diaginertia="1.28092e-06 1.12092e-06 5.3e-07"/>
                <joint name="rh_MFJ1" class="middle_distal"/>
                <geom class="plastic_visual" mesh="f_distal_pst"/>
                <geom class="plastic_collision" type="mesh" mesh="f_distal_pst"/>
                <geom class="coupling_tendon" name="coupling_MFJ1_pulley" size="0.008929"/>
                <site class="coupling_tendon" name="coupling_MFJ1_pulley_outside" pos="0 0.010 -0.01"/>
                <site class="coupling_tendon" name="coupling_MFJ1_tendon_end" pos="0 0.007 0.015"/>
              </body>
            </body>
          </body>
        </body>
        <!-- Ring finger -->
        <body name="rh_rfknuckle" pos="-0.011 0 0.095">
          <inertial mass="0.008" pos="0 0 0" quat="1 1 -1 1" diaginertia="3.2e-07 2.6e-07 2.6e-07"/>
          <joint name="rh_RFJ4" class="knuckle" axis="0 1 0"/>
          <geom pos="0 0 0.0005" class="plastic_visual" mesh="f_knuckle" material="metallic"/>
          <body name="rh_rfproximal">
            <inertial mass="0.03" pos="0 0 0.0225" quat="1 0 0 1" diaginertia="1e-05 9.8e-06 1.8e-06"/>
            <joint name="rh_RFJ3" class="proximal"/>
            <geom class="plastic_visual" mesh="f_proximal"/>
            <geom size="0.009 0.02" pos="0 0 0.025" type="capsule" class="plastic_collision"/>
            <site class="coupling_tendon" name="coupling_RFJ3_tendon_end" pos="0 -0.007 0.023"/>
            <body name="rh_rfmiddle" pos="0 0 0.045">
              <inertial mass="0.017" pos="0 0 0.0125" quat="1 0 0 1" diaginertia="2.7e-06 2.6e-06 8.7e-07"/>
              <joint name="rh_RFJ2" class="middle_distal"/>
              <geom class="plastic_visual" mesh="f_middle"/>
              <geom size="0.009 0.0125" pos="0 0 0.0125" type="capsule" class="plastic_collision"/>
              <geom class="coupling_tendon" name="coupling_RFJ2_pulley" size="0.01125"/>
              <site class="coupling_tendon" name="coupling_RFJ2_pulley_inside" pos="0 -0.011 0.011"/>
              <site class="coupling_tendon" name="coupling_RFJ2_between_pulleys" pos="0 -0.005 0.008"/>
              <body name="rh_rfdistal" pos="0 0 0.025">
                <inertial mass="0.013" pos="0 0 0.0130769" quat="1 0 0 1"
                  diaginertia="1.28092e-06 1.12092e-06 5.3e-07"/>
                <joint name="rh_RFJ1" class="middle_distal"/>
                <geom class="plastic_visual" mesh="f_distal_pst"/>
                <geom class="plastic_collision" type="mesh" mesh="f_distal_pst"/>
                <geom class="coupling_tendon" name="coupling_RFJ1_pulley" size="0.008929"/>
                <site class="coupling_tendon" name="coupling_RFJ1_pulley_outside" pos="0 0.010 -0.01"/>
                <site class="coupling_tendon" name="coupling_RFJ1_tendon_end" pos="0 0.007 0.015"/>
              </body>
            </body>
          </body>
        </body>
        <!-- Little finger -->
        <body name="rh_lfmetacarpal" pos="-0.033 0 0.02071">
          <inertial mass="0.03" pos="0 0 0.04" quat="1 0 0 1" diaginertia="1.638e-05 1.45e-05 4.272e-06"/>
          <joint name="rh_LFJ5" class="metacarpal"/>
          <geom class="plastic_visual" mesh="lf_metacarpal"/>
          <geom class="plastic_collision" mesh="lf_metacarpal_c_0" type="mesh"/>
          <geom class="plastic_collision" mesh="lf_metacarpal_c_1" type="mesh"/>
          <body name="rh_lfknuckle" pos="0 0 0.06579">
            <inertial mass="0.008" pos="0 0 0" quat="1 1 -1 1" diaginertia="3.2e-07 2.6e-07 2.6e-07"/>
            <joint name="rh_LFJ4" class="knuckle" axis="0 1 0"/>
            <geom pos="0 0 0.0005" class="plastic_visual" mesh="f_knuckle" material="metallic"/>
            <body name="rh_lfproximal">
              <inertial mass="0.03" pos="0 0 0.0225" quat="1 0 0 1" diaginertia="1e-05 9.8e-06 1.8e-06"/>
              <joint name="rh_LFJ3" class="proximal"/>
              <geom class="plastic_visual" mesh="f_proximal"/>
              <geom size="0.009 0.02" pos="0 0 0.025" type="capsule" class="plastic_collision"/>
              <site class="coupling_tendon" name="coupling_LFJ3_tendon_end" pos="0 -0.007 0.023"/>
              <body name="rh_lfmiddle" pos="0 0 0.045">
                <inertial mass="0.017" pos="0 0 0.0125" quat="1 0 0 1" diaginertia="2.7e-06 2.6e-06 8.7e-07"/>
                <joint name="rh_LFJ2" class="middle_distal"/>
                <geom class="plastic_visual" mesh="f_middle"/>
                <geom size="0.009 0.0125" pos="0 0 0.0125" type="capsule" class="plastic_collision"/>
                <geom class="coupling_tendon" name="coupling_LFJ2_pulley" size="0.01125"/>
                <site class="coupling_tendon" name="coupling_LFJ2_pulley_inside" pos="0 -0.011 0.011"/>
                <site class="coupling_tendon" name="coupling_LFJ2_between_pulleys" pos="0 -0.005 0.008"/>
                <body name="rh_lfdistal" pos="0 0 0.025">
                  <inertial mass="0.013" pos="0 0 0.0130769" quat="1 0 0 1"
                    diaginertia="1.28092e-06 1.12092e-06 5.3e-07"/>
                  <joint name="rh_LFJ1" class="middle_distal"/>
                  <geom class="plastic_visual" mesh="f_distal_pst"/>
                  <geom class="plastic_collision" type="mesh" mesh="f_distal_pst"/>
                  <geom class="coupling_tendon" name="coupling_LFJ1_pulley" size="0.008929"/>
                  <site class="coupling_tendon" name="coupling_LFJ1_pulley_outside" pos="0 0.010 -0.01"/>
                  <site class="coupling_tendon" name="coupling_LFJ1_tendon_end" pos="0 0.007 0.015"/>
                </body>
              </body>
            </body>
          </body>
        </body>
        <!-- Thumb -->
        <body name="rh_thbase" pos="0.034 -0.00858 0.029" quat="0.92388 0 0.382683 0">
          <inertial mass="0.01" pos="0 0 0" diaginertia="1.6e-07 1.6e-07 1.6e-07"/>
          <joint name="rh_THJ5" class="thbase"/>
          <geom class="plastic_collision" size="0.013"/>
          <body name="rh_thproximal">
            <inertial mass="0.04" pos="0 0 0.019" diaginertia="1.36e-05 1.36e-05 3.13e-06"/>
            <joint name="rh_THJ4" class="thproximal"/>
            <geom class="plastic_visual" mesh="th_proximal"/>
            <geom class="plastic_collision" size="0.0105 0.009" pos="0 0 0.02" type="capsule"/>
            <body name="rh_thhub" pos="0 0 0.038">
              <inertial mass="0.005" pos="0 0 0" diaginertia="1e-06 1e-06 3e-07"/>
              <joint name="rh_THJ3" class="thhub"/>
              <geom size="0.011" class="plastic_collision"/>
              <body name="rh_thmiddle">
                <inertial mass="0.02" pos="0 0 0.016" diaginertia="5.1e-06 5.1e-06 1.21e-06"/>
                <joint name="rh_THJ2" class="thmiddle"/>
                <geom class="plastic_visual" mesh="th_middle"/>
                <geom size="0.009 0.009" pos="0 0 0.012" type="capsule" class="plastic_collision"/>
                <geom size="0.01" pos="0 0 0.03" class="plastic_collision"/>
                <body name="rh_thdistal" pos="0 0 0.032" quat="1 0 0 -1">
                  <inertial mass="0.017" pos="0 0 0.0145588" quat="1 0 0 1"
                    diaginertia="2.37794e-06 2.27794e-06 1e-06"/>
                  <joint name="rh_THJ1" class="thdistal"/>
                  <geom class="plastic_visual" mesh="th_distal_pst"/>
                  <geom class="plastic_collision" type="mesh" mesh="th_distal_pst"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <contact>
    <exclude body1="rh_thproximal" body2="rh_thmiddle"/>
    <exclude body1="rh_lfmetacarpal" body2="rh_lfproximal"/>
    <exclude body1="rh_rfproximal" body2="rh_palm"/>
    <exclude body1="rh_mfproximal" body2="rh_palm"/>
    <exclude body1="rh_ffproximal" body2="rh_palm"/>
    <exclude body1="rh_thproximal" body2="rh_palm"/>
  </contact>

  <tendon>
    <fixed name="rh_FFT0" range="0 1.5707" group="4">
      <joint coef="1" joint="rh_FFJ2"/>
      <joint coef="-1" joint="rh_FFJ1"/>
    </fixed>
    <fixed name="rh_FFT1" range="0 3.14159" group="4">
      <joint coef="1" joint="rh_FFJ2"/>
      <joint coef="1" joint="rh_FFJ1"/>
    </fixed>
    <spatial name="rh_FFT2" range="0 0.080876" springlength="0.06" stiffness="3019.13" width="0.0005">
      <site site="coupling_FFJ1_tendon_end"/>
      <geom geom="coupling_FFJ1_pulley" sidesite="coupling_FFJ1_pulley_outside"/>
      <site site="coupling_FFJ2_between_pulleys"/>
      <geom geom="coupling_FFJ2_pulley" sidesite="coupling_FFJ2_pulley_inside"/>
      <site site="coupling_FFJ3_tendon_end"/>
    </spatial>
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
    <fixed name="rh_RFT0" range="0 1.5707" group="4">
      <joint coef="1" joint="rh_RFJ2"/>
      <joint coef="-1" joint="rh_RFJ1"/>
    </fixed>
    <fixed name="rh_RFT1" range="0 3.14159" group="4">
      <joint coef="1" joint="rh_RFJ2"/>
      <joint coef="1" joint="rh_RFJ1"/>
    </fixed>
    <spatial name="rh_RFT2" range="0 0.080876" springlength="0.056161" stiffness="3027.56" width="0.0005">
      <site site="coupling_RFJ1_tendon_end"/>
      <geom geom="coupling_RFJ1_pulley" sidesite="coupling_RFJ1_pulley_outside"/>
      <site site="coupling_RFJ2_between_pulleys"/>
      <geom geom="coupling_RFJ2_pulley" sidesite="coupling_RFJ2_pulley_inside"/>
      <site site="coupling_RFJ3_tendon_end"/>
    </spatial>
    <fixed name="rh_LFT0" range="0 1.5707" group="4">
      <joint coef="1" joint="rh_LFJ2"/>
      <joint coef="-1" joint="rh_LFJ1"/>
    </fixed>
    <fixed name="rh_LFT1" range="0 3.14159" group="4">
      <joint coef="1" joint="rh_LFJ2"/>
      <joint coef="1" joint="rh_LFJ1"/>
    </fixed>
    <spatial name="rh_LFT2" range="0 0.080876" springlength="0.059417" stiffness="3493.57" width="0.0005">
      <site site="coupling_LFJ1_tendon_end"/>
      <geom geom="coupling_LFJ1_pulley" sidesite="coupling_LFJ1_pulley_outside"/>
      <site site="coupling_LFJ2_between_pulleys"/>
      <geom geom="coupling_LFJ2_pulley" sidesite="coupling_LFJ2_pulley_inside"/>
      <site site="coupling_LFJ3_tendon_end"/>
    </spatial>
  </tendon>

  <actuator>
    <!-- Thumb -->
    <position name="rh_THJ5" joint="rh_THJ5" class="thbase"/>
    <position name="rh_THJ4" joint="rh_THJ4" class="thproximal"/>
    <position name="rh_THJ3" joint="rh_THJ3" class="thhub"/>
    <position name="rh_THJ2" joint="rh_THJ2" class="thmiddle"/>
    <position name="rh_THJ1" joint="rh_THJ1" class="thdistal"/>
    <!-- First finger -->
    <position name="rh_FFJ4" joint="rh_FFJ4" class="knuckle"/>
    <position name="rh_FFJ3" joint="rh_FFJ3" class="proximal"/>
    <position name="rh_FFJ0" tendon="rh_FFT1" class="middle_distal" forcerange="-0.465773 0.465773"/>
    <!-- Middle finger -->
    <position name="rh_MFJ4" joint="rh_MFJ4" class="knuckle"/>
    <position name="rh_MFJ3" joint="rh_MFJ3" class="proximal"/>
    <position name="rh_MFJ0" tendon="rh_MFT1" class="middle_distal" forcerange="-0.616938 0.616938"/>
    <!-- Ring finger -->
    <position name="rh_RFJ4" joint="rh_RFJ4" class="knuckle"/>
    <position name="rh_RFJ3" joint="rh_RFJ3" class="proximal"/>
    <position name="rh_RFJ0" tendon="rh_RFT1" class="middle_distal" forcerange="-0.464343 0.464343"/>
    <!-- Little finger -->
    <position name="rh_LFJ5" joint="rh_LFJ5" class="metacarpal"/>
    <position name="rh_LFJ4" joint="rh_LFJ4" class="knuckle"/>
    <position name="rh_LFJ3" joint="rh_LFJ3" class="proximal"/>
    <position name="rh_LFJ0" tendon="rh_LFT1" class="middle_distal" forcerange="-0.506173 0.506173"/>
  </actuator>
  <equality>
    <weld body1="mocap" body2="rh_wrist"/>
  </equality>

"""


class GripperShadowRightConstraint(MjShakableOpenCloseGripper, MjScannable):
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
        sim.data.mocap_pos = pose.pos
        sim.data.mocap_quat = pose.quat
        sim.data.ctrl[:] = self._qpos_to_qacc(
            np.array(
                [
                    -0.3464,
                    1.253,
                    0.7836,
                    -0.001106,
                    0.01103,
                    1.475,
                    0.6181,
                    0.0155,
                    -0.2083,
                    1.45,
                    0.75,
                    0.0,
                    0.13,
                    -0.4,
                    1.5,
                    0.95,
                    0.35,
                    0.07708,
                    1.21,
                    0.2023,
                    0.6614,
                    0.0102,
                ]
            )
        )  # type: ignore
        import time

        for i in range(300):
            mujoco.mj_step(sim.model, sim.data, 1)  # type: ignore
            # viewer.sync()
            # time.sleep(1.0 / 240.0)  # Cap at ~60 FPS

    def get_freejoint_idxs(self, sim: MjSimulation) -> List[int]:
        start_idx = sim.get_joint_idxs(["freejoint"])[0]
        return list(range(start_idx, start_idx + 7))

    def get_actuator_joint_names(
        self,
    ) -> List[str]:
        return [
            "rh_FFJ4",
            "rh_FFJ3",
            "rh_FFJ2",  # 2
            "rh_FFJ1",  # 3
            "rh_MFJ4",
            "rh_MFJ3",
            "rh_MFJ2",  # 6
            "rh_MFJ1",  # 7
            "rh_RFJ4",
            "rh_RFJ3",
            "rh_RFJ2",  # 10
            "rh_RFJ1",  # 11
            "rh_LFJ5",
            "rh_LFJ4",
            "rh_LFJ3",
            "rh_LFJ2",  # 15
            "rh_LFJ1",  # 16
            "rh_THJ5",
            "rh_THJ4",
            "rh_THJ3",
            "rh_THJ2",  # 20
            "rh_THJ1",  # 21
        ]

    def _qpos_to_qacc(self, qpos):
        acc = np.zeros((18,))
        acc[:5] = qpos[-5:]
        acc[5:7] = qpos[0:2]
        acc[7] = qpos[2] + qpos[3]
        acc[8:10] = qpos[4:6]
        acc[10] = qpos[6] + qpos[7]
        acc[11:13] = qpos[8:10]
        acc[13] = qpos[10] + qpos[11]
        acc[14:17] = qpos[12:15]
        acc[17] = qpos[15] + qpos[16]
        return acc
