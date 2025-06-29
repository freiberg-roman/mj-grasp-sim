import os
from typing import Any, Dict, List, Tuple

import mujoco
import numpy as np

from mgs.core.simualtion import MjSimulation
from mgs.util.const import ASSET_PATH
from mgs.util.geo.transforms import SE3Pose

from .base import MjScannable, MjShakableOpenCloseGripper

XML = """
  <!-- <compiler angle="radian"/> -->
  <asset>
    <material name="black" rgba="0.2 0.2 0.2 1" />

    <mesh name="palm" file="palm_right.obj"/>
    <mesh name="base" file="base.obj"/>
    <mesh name="proximal" file="proximal.obj"/>
    <mesh name="medial" file="medial.obj"/>
    <mesh name="distal" file="distal.obj"/>
    <mesh name="tip" file="tip.obj"/>
    <mesh name="thumb_base" file="thumb_base.obj"/>
    <mesh name="thumb_proximal" file="thumb_proximal.obj"/>
    <mesh name="thumb_distal" file="thumb_distal.obj"/>
    <mesh name="thumb_tip" file="thumb_tip.obj"/>
  </asset>

  <!-- defaults -->
  <default>
    <!-- constraint stiffness -->
    <geom solimp="0.999 0.999 0.001 0.0001 1" solref="0.0001 1" friction=".2"/>

    <!-- actuator defaults -->
    <position kp="3.0" kv="0.01" />
    <joint damping="0.03" frictionloss="0.001"/>

    <!-- geom class defaults -->
    <default class="visual">
      <geom group="1" type="mesh" contype="0" conaffinity="0" density="0" material="black" />
    </default>
    <default class="tip">
      <geom type="mesh" mesh="tip" friction="2.5"/>
    </default>
    <default class="thumb_tip">
      <geom type="mesh" mesh="thumb_tip" friction="2.5"/>
    </default>

    <!-- joint class defaults -->
    <default class="mcp">
      <joint pos="0 0 0" axis="0 0 -1"
        limited="true" range="-0.314 2.23" />
      <position ctrlrange="-0.314 2.23" />
    </default>
    <default class="rot">
      <joint pos="0 0 0" axis="0 0 -1"
        limited="true" range="-1.047 1.047" />
      <position ctrlrange="-1.047 1.047" />
    </default>
    <default class="pip">
      <joint pos="0 0 0" axis="0 0 -1"
        limited="true" range="-0.506 1.885" />
      <position ctrlrange="-0.506 1.885" />
    </default>
    <default class="dip">
      <joint pos="0 0 0" axis="0 0 -1"
        limited="true" range="-0.366 2.042" />
      <position ctrlrange="-0.366 2.042" />
    </default>
    <default class="thumb_cmc">
      <joint pos="0 0 0" axis="0 0 -1"
        limited="true" range="-0.349 2.094" />
      <position ctrlrange="-0.349 2.094" />
    </default>
    <default class="thumb_axl">
      <joint pos="0 0 0" axis="0 0 -1"
        limited="true" range="-0.47 2.443" />
      <position ctrlrange="-0.47 2.443" />
    </default>
    <default class="thumb_mcp">
      <joint pos="0 0 0" axis="0 0 -1"
        limited="true" range="-1.2 1.9" />
      <position ctrlrange="-1.2 1.9" />
    </default>
    <default class="thumb_ipl">
      <joint pos="0 0 0" axis="0 0 -1"
        limited="true" range="-1.34 1.88" />
      <position ctrlrange="-1.34 1.88" />
    </default>
  </default>

  <!-- collision filtering -->
  <contact>
    <!-- filter the palm from all other finger bodies except the tips -->
    <exclude body1="palm" body2="if_bs" />
    <exclude body1="palm" body2="mf_bs" />
    <exclude body1="palm" body2="rf_bs" />
    <exclude body1="palm" body2="th_mp" />

    <exclude body1="palm" body2="if_px" />
    <exclude body1="palm" body2="mf_px" />
    <exclude body1="palm" body2="rf_px" />
    <exclude body1="palm" body2="th_bs" />

    <exclude body1="palm" body2="if_md" />
    <exclude body1="palm" body2="mf_md" />
    <exclude body1="palm" body2="rf_md" />
    <exclude body1="palm" body2="th_px" />

    <!-- none of the base fingertip geoms can touch each other -->
    <exclude body1="if_bs" body2="mf_bs" />
    <exclude body1="if_bs" body2="rf_bs" />
    <exclude body1="mf_bs" body2="rf_bs" />

    <exclude body1="th_mp" body2="if_bs" />
    <exclude body1="th_mp" body2="mf_bs" />
    <exclude body1="th_mp" body2="rf_bs" />
  </contact>

  <!-- actuators -->
  <actuator>
    <!-- index -->
    <position name="if_mcp_act" joint="if_mcp" class="mcp" />
    <position name="if_rot_act" joint="if_rot" class="rot" />
    <position name="if_pip_act" joint="if_pip" class="pip" />
    <position name="if_dip_act" joint="if_dip" class="dip" />

    <!-- middle -->
    <position name="mf_mcp_act" joint="mf_mcp" class="mcp" />
    <position name="mf_rot_act" joint="mf_rot" class="rot" />
    <position name="mf_pip_act" joint="mf_pip" class="pip" />
    <position name="mf_dip_act" joint="mf_dip" class="dip" />

    <!-- ring -->
    <position name="rf_mcp_act" joint="rf_mcp" class="mcp" />
    <position name="rf_rot_act" joint="rf_rot" class="rot" />
    <position name="rf_pip_act" joint="rf_pip" class="pip" />
    <position name="rf_dip_act" joint="rf_dip" class="dip" />

    <!-- thumb -->
    <position name="th_cmc_act" joint="th_cmc" class="thumb_cmc" />
    <position name="th_axl_act" joint="th_axl" class="thumb_axl" />
    <position name="th_mcp_act" joint="th_mcp" class="thumb_mcp" />
    <position name="th_ipl_act" joint="th_ipl" class="thumb_ipl" />
  </actuator>

  <!-- sensors -->
  <sensor>
    <!-- index -->
    <jointpos name="if_mcp_sensor" joint="if_mcp" />
    <jointpos name="if_rot_sensor" joint="if_rot" />
    <jointpos name="if_pip_sensor" joint="if_pip" />
    <jointpos name="if_dip_sensor" joint="if_dip" />

    <!-- middle -->
    <jointpos name="mf_mcp_sensor" joint="mf_mcp" />
    <jointpos name="mf_rot_sensor" joint="mf_rot" />
    <jointpos name="mf_pip_sensor" joint="mf_pip" />
    <jointpos name="mf_dip_sensor" joint="mf_dip" />

    <!-- ring -->
    <jointpos name="rf_mcp_sensor" joint="rf_mcp" />
    <jointpos name="rf_rot_sensor" joint="rf_rot" />
    <jointpos name="rf_pip_sensor" joint="rf_pip" />
    <jointpos name="rf_dip_sensor" joint="rf_dip" />

    <!-- thumb -->
    <jointpos name="th_cmc_sensor" joint="th_cmc" />
    <jointpos name="th_axl_sensor" joint="th_axl" />
    <jointpos name="th_mcp_sensor" joint="th_mcp" />
    <jointpos name="th_ipl_sensor" joint="th_ipl" />
  </sensor>

  <equality>
    <weld body1="mocap" body2="palm"/>
  </equality>

  <worldbody>
    <body name="mocap" mocap="true" pos="{position}" quat="{quaternion}"/>
    <body name="palm" pos="{position}" quat="{quaternion}">
      <freejoint name="freejoint" />
      <inertial pos="-0.049542 -0.042914 -0.010227" quat="0.565586 0.427629 -0.574956 0.408254" mass="0.237" diaginertia="0.000407345 0.000304759 0.000180736"/>
      <geom name="palm_visual" pos="-0.02 0.02575 -0.0347" quat="1 0 0 0" class="visual" mesh="palm"/>
      <geom name="palm_collision_1" size="0.011 0.013 0.017" pos="-0.009 0.008 -0.011" type="box"/>
      <geom name="palm_collision_2" size="0.011 0.013 0.017" pos="-0.009 -0.037 -0.011" type="box"/>
      <geom name="palm_collision_3" size="0.011 0.013 0.017" pos="-0.009 -0.082 -0.011" type="box"/>
      <geom name="palm_collision_4" size="0.029 0.01 0.023" pos="-0.066 -0.078 -0.0115" quat="0.989016 0 0 -0.147806" type="box"/>
      <geom name="palm_collision_5" size="0.01 0.06 0.015" pos="-0.03 -0.035 -0.003" type="box"/>
      <geom name="palm_collision_6" size="0.005 0.06 0.01" pos="-0.032 -0.035 -0.024" quat="0.923956 0 0.382499 0" type="box"/>
      <geom name="palm_collision_7" size="0.012 0.058 0.023" pos="-0.048 -0.033 -0.0115" type="box"/>
      <geom name="palm_collision_8" size="0.022 0.026 0.023" pos="-0.078 -0.053 -0.0115" type="box"/>
      <geom name="palm_collision_9" size="0.002 0.018 0.017" pos="-0.098 -0.009 -0.006" type="box"/>
      <geom name="palm_collision_10" size="0.022 0.028 0.002" pos="-0.078 -0.003 0.01" type="box"/>

      <!-- index -->
      <body name="if_bs" pos="-0.007 0.023 -0.0187" quat="0.500003 0.5 0.5 -0.499997">
        <inertial pos="-0.022516 0.033882 0.016359" quat="0.388092 0.677951 -0.247713 0.573067" mass="0.044" diaginertia="1.74972e-05 1.61504e-05 7.21342e-06"/>
        <joint name="if_mcp" class="mcp"/>
        <geom name="if_bs_visual" pos="0.0084 0.0077 0.01465" quat="1 0 0 0" class="visual" mesh="base"/>
        <geom name="if_bs_collision_1" size="0.01 0.003 0.017" pos="0 0.018 0.0147" type="box"/>
        <geom name="if_bs_collision_2" size="0.014 0.02 0.01" pos="-0.027 0.042 0.015" type="box"/>
        <geom name="if_bs_collision_3" size="0.017 0.003 0.01" pos="-0.0262 0.02 0.0146" type="box"/>
        <geom name="if_bs_collision_4" size="0.01 0.012 0.004" pos="-0.0295 0.035 0.029" type="box"/>
        <geom name="if_bs_collision_5" size="0.007 0.01 0.002" pos="0 0.005 0.03" type="box"/>
        <geom name="if_bs_collision_6" size="0.007 0.01 0.002" pos="0 0.005 -0.001" type="box"/>

        <body name="if_px" pos="-0.0122 0.0381 0.0145" quat="0.500003 -0.5 -0.499997 0.5">
          <inertial pos="0.0075 -0.0002 -0.011" quat="0 0.707107 0 0.707107" mass="0.032" diaginertia="4.8853e-06 4.3733e-06 3.0933e-06"/>
          <joint name="if_rot" class="rot"/>
          <geom name="if_px_visual" pos="0.0096 0.0002 0.0007" quat="0.500003 -0.5 -0.5 -0.499997" class="visual" mesh="proximal"/>
          <geom name="if_px_collision" size="0.017 0.013 0.011" pos="0.0075 -0.0002 -0.011" type="box"/>

          <body name="if_md" pos="0.015 0.0143 -0.013" quat="0.500003 0.5 -0.5 0.499997">
            <inertial pos="0.0054215 -0.029148 0.015" quat="0.687228 0.687228 0.166487 0.166487" mass="0.037" diaginertia="8.28004e-06 8.1598e-06 5.39516e-06"/>
            <joint name="if_pip" class="pip"/>
            <geom name="if_md_visual" pos="0.0211 -0.0084 0.0097" quat="2.67949e-08 -1 0 0" class="visual" mesh="medial"/>
            <geom name="if_md_collision_1" size="0.008 0.003 0.013" pos="0 -0.02 0.015" type="box"/>
            <geom name="if_md_collision_2" size="0.01 0.002 0.017" pos="0 -0.016 0.015" type="box"/>
            <geom name="if_md_collision_3" size="0.007 0.01 0.002" pos="0 -0.0045 0.03" type="box"/>
            <geom name="if_md_collision_4" size="0.007 0.01 0.002" pos="0 -0.0045 0" type="box"/>
            <geom name="if_md_collision_5" size="0.017 0.011 0.013" pos="0.0075 -0.035 0.015" type="box"/>

            <body name="if_ds" pos="0 -0.0361 0.0002">
              <inertial pos="-0.0008794 -0.027019 0.014594" quat="0.702905 0.710643 -0.0212937 -0.0214203" mass="0.016" diaginertia="3.71863e-06 3.02396e-06 1.6518e-06"/>
              <joint name="if_dip" class="dip"/>
              <geom name="if_ds_visual" pos="0.0132 -0.0061 0.0144" quat="2.67949e-08 1 0 0" class="visual" mesh="distal"/>
              <geom name="if_ds_collision_1" size="0.01 0.003 0.015" pos="0 -0.017 0.015" type="box"/>
              <geom name="if_ds_collision_2" size="0.007 0.011 0.002" pos="0 -0.006 0.03" type="box"/>
              <geom name="if_ds_collision_3" size="0.007 0.011 0.002" pos="0 -0.006 -0.001" type="box"/>

              <geom name="if_tip" class="tip"/>
            </body>

          </body>
        </body>
      </body>  <!-- index -->

      <!-- middle -->
      <body name="mf_bs" pos="-0.0071 -0.0224 -0.0187" quat="0.500003 0.5 0.5 -0.499997">
        <inertial pos="-0.022516 0.033882 0.016359" quat="0.388092 0.677951 -0.247713 0.573067" mass="0.044" diaginertia="1.74972e-05 1.61504e-05 7.21342e-06"/>
        <joint name="mf_mcp" class="mcp"/>
        <geom name="mf_bs_visual" pos="0.0084 0.0077 0.01465" quat="1 0 0 0" class="visual" mesh="base"/>
        <geom name="mf_bs_collision_1" size="0.01 0.003 0.017" pos="0 0.018 0.0147" type="box"/>
        <geom name="mf_bs_collision_2" size="0.014 0.02 0.01" pos="-0.027 0.042 0.015" type="box"/>
        <geom name="mf_bs_collision_3" size="0.017 0.003 0.01" pos="-0.0262 0.02 0.0146" type="box"/>
        <geom name="mf_bs_collision_4" size="0.01 0.012 0.004" pos="-0.0295 0.035 0.029" type="box"/>
        <geom name="mf_bs_collision_5" size="0.007 0.01 0.002" pos="0 0.005 0.03" type="box"/>
        <geom name="mf_bs_collision_6" size="0.007 0.01 0.002" pos="0 0.005 -0.001" type="box"/>

        <body name="mf_px" pos="-0.0122 0.0381 0.0145" quat="0.500003 -0.5 -0.499997 0.5">
          <inertial pos="0.0075 -0.0002 -0.011" quat="0 0.707107 0 0.707107" mass="0.032" diaginertia="4.8853e-06 4.3733e-06 3.0933e-06"/>
          <joint name="mf_rot" class="rot"/>
          <geom name="mf_px_visual" pos="0.0096 0.0003 0.0007" quat="0.500003 -0.5 -0.5 -0.499997" class="visual" mesh="proximal"/>
          <geom name="mf_px_collision" size="0.017 0.013 0.011" pos="0.0075 -0.0002 -0.011" type="box"/>

          <body name="mf_md" pos="0.015 0.0143 -0.013" quat="0.500003 0.5 -0.5 0.499997">
            <inertial pos="0.0054215 -0.029148 0.015" quat="0.687228 0.687228 0.166487 0.166487" mass="0.037" diaginertia="8.28004e-06 8.1598e-06 5.39516e-06"/>
            <joint name="mf_pip" class="pip"/>
            <geom name="mf_md_visual" pos="0.0211 -0.0084 0.0097" quat="1.32679e-06 -1 0 0" class="visual" mesh="medial"/>
            <geom name="mf_md_collision_1" size="0.008 0.003 0.013" pos="0 -0.02 0.015" type="box"/>
            <geom name="mf_md_collision_2" size="0.01 0.002 0.017" pos="0 -0.016 0.015" type="box"/>
            <geom name="mf_md_collision_3" size="0.007 0.01 0.002" pos="0 -0.0045 0.03" type="box"/>
            <geom name="mf_md_collision_4" size="0.007 0.01 0.002" pos="0 -0.0045 0" type="box"/>
            <geom name="mf_md_collision_5" size="0.017 0.011 0.013" pos="0.0075 -0.035 0.015" type="box"/>

            <body name="mf_ds" pos="0 -0.0361 0.0002">
              <inertial pos="-0.0008794 -0.027019 0.014594" quat="0.702905 0.710643 -0.0212937 -0.0214203" mass="0.016" diaginertia="3.71863e-06 3.02396e-06 1.6518e-06"/>
              <joint name="mf_dip" class="dip"/>
              <geom name="mf_ds_visual" pos="0.0132 -0.0061 0.0145" quat="1.32679e-06 1 0 0" class="visual" mesh="distal"/>
              <geom name="mf_ds_collision_1" size="0.01 0.003 0.015" pos="0 -0.017 0.015" type="box"/>
              <geom name="mf_ds_collision_2" size="0.007 0.011 0.002" pos="0 -0.006 0.03" type="box"/>
              <geom name="mf_ds_collision_3" size="0.007 0.011 0.002" pos="0 -0.006 -0.001" type="box"/>

              <geom name="mf_tip" class="tip"/>
            </body>

          </body>
        </body>
      </body>  <!-- middle -->

      <!-- ring -->
      <body name="rf_bs" pos="-0.00709 -0.0678 -0.0187" quat="0.500003 0.5 0.5 -0.499997">
        <inertial pos="-0.022516 0.033882 0.016359" quat="0.388092 0.677951 -0.247713 0.573067" mass="0.044" diaginertia="1.74972e-05 1.61504e-05 7.21342e-06"/>
        <joint name="rf_mcp" class="mcp"/>
        <geom name="rf_bs_visual" pos="0.0084 0.0077 0.01465" quat="1 0 0 0" class="visual" mesh="base"/>
        <geom name="rf_bs_collision_1" size="0.01 0.003 0.017" pos="0 0.018 0.0147" type="box"/>
        <geom name="rf_bs_collision_2" size="0.014 0.02 0.01" pos="-0.027 0.042 0.015" type="box"/>
        <geom name="rf_bs_collision_3" size="0.017 0.003 0.01" pos="-0.0262 0.02 0.0146" type="box"/>
        <geom name="rf_bs_collision_4" size="0.01 0.012 0.004" pos="-0.0295 0.035 0.029" type="box"/>
        <geom name="rf_bs_collision_5" size="0.007 0.01 0.002" pos="0 0.005 0.03" type="box"/>
        <geom name="rf_bs_collision_6" size="0.007 0.01 0.002" pos="0 0.005 -0.001" type="box"/>

        <body name="rf_px" pos="-0.0122 0.0381 0.0145" quat="0.500003 -0.5 -0.499997 0.5">
          <inertial pos="0.0075 -0.0002 -0.011" quat="0 0.707107 0 0.707107" mass="0.032" diaginertia="4.8853e-06 4.3733e-06 3.0933e-06"/>
          <joint name="rf_rot" class="rot"/>
          <geom name="rf_px_visual" pos="0.0096 0.0003 0.0007" quat="0.500003 -0.5 -0.5 -0.499997" class="visual" mesh="proximal"/>
          <geom name="rf_px_collision" size="0.017 0.013 0.011" pos="0.0075 -0.0002 -0.011" type="box"/>

          <body name="rf_md" pos="0.015 0.0143 -0.013" quat="0.500003 0.5 -0.5 0.499997">
            <inertial pos="0.0054215 -0.029148 0.015" quat="0.687228 0.687228 0.166487 0.166487" mass="0.037" diaginertia="8.28004e-06 8.1598e-06 5.39516e-06"/>
            <joint name="rf_pip" class="pip"/>
            <geom name="rf_md_visual" pos="0.0211 -0.0084 0.0097" quat="1.32679e-06 -1 0 0" class="visual" mesh="medial"/>
            <geom name="rf_md_collision_1" size="0.008 0.003 0.013" pos="0 -0.02 0.015" type="box"/>
            <geom name="rf_md_collision_2" size="0.01 0.002 0.017" pos="0 -0.016 0.015" type="box"/>
            <geom name="rf_md_collision_3" size="0.007 0.01 0.002" pos="0 -0.0045 0.03" type="box"/>
            <geom name="rf_md_collision_4" size="0.007 0.01 0.002" pos="0 -0.0045 0" type="box"/>
            <geom name="rf_md_collision_5" size="0.017 0.011 0.013" pos="0.0075 -0.035 0.015" type="box"/>

            <body name="rf_ds" pos="0 -0.03609 0.0002">
              <inertial pos="-0.0008794 -0.027019 0.014594" quat="0.702905 0.710643 -0.0212937 -0.0214203" mass="0.016" diaginertia="3.71863e-06 3.02396e-06 1.6518e-06"/>
              <joint name="rf_dip" class="dip"/>
              <geom name="rf_ds_visual" pos="0.0132 -0.0061 0.0145" quat="1.32679e-06 1 0 0" class="visual" mesh="distal"/>
              <geom name="rf_ds_collision_1" size="0.01 0.003 0.015" pos="0 -0.017 0.015" type="box"/>
              <geom name="rf_ds_collision_2" size="0.007 0.011 0.002" pos="0 -0.006 0.03" type="box"/>
              <geom name="rf_ds_collision_3" size="0.007 0.011 0.002" pos="0 -0.006 -0.001" type="box"/>

              <geom name="rf_tip" class="tip"/>
            </body>

          </body>
        </body>
      </body>  <!-- ring -->

      <!-- thumb -->
      <body name="th_mp" pos="-0.0693 -0.0012 -0.0216" quat="0.707109 0 0.707105 0">
        <inertial pos="0.0075 -0.0002 -0.011" quat="0 0.707107 0 0.707107" mass="0.032" diaginertia="4.8853e-06 4.3733e-06 3.0933e-06"/>
        <joint name="th_cmc" class="thumb_cmc"/>
        <geom name="th_mp_visual" pos="-0.0053 0.0003 0.00078" quat="0.500003 -0.5 -0.5 -0.499997" class="visual" mesh="proximal"/>
        <geom name="th_mp_collision" size="0.017 0.013 0.011" pos="-0.0075 -0.0002 -0.011" type="box"/>

        <body name="th_bs" pos="0 0.0143 -0.013" quat="0.500003 0.5 -0.5 0.499997">
          <inertial pos="0 0 -0.0070806" quat="0.707107 0.707107 0 0" mass="0.003" diaginertia="6.1932e-07 5.351e-07 2.1516e-07"/>
          <joint name="th_axl" class="thumb_axl"/>
          <geom name="th_bs_visual" pos="0.01196 0 -0.0158" quat="0.707109 0.707105 0 0" class="visual" mesh="thumb_base"/>
          <geom name="th_bs_collision_1" size="0.009 0.0165 0.002" pos="0 0 -0.0015" type="box"/>
          <geom name="th_bs_collision_2" size="0.007 0.002 0.01" pos="0 -0.015 -0.013" type="box"/>
          <geom name="th_bs_collision_3" size="0.007 0.002 0.01" pos="0 0.015 -0.013" type="box"/>

          <body name="th_px" pos="0 0.0145 -0.017" quat="0.707109 -0.707105 0 0">
            <inertial pos="-0.0020593 0.015912 -0.013733" quat="0.698518 0.697382 -0.104933 0.121324" mass="0.038" diaginertia="9.87104e-06 9.32653e-06 4.36203e-06"/>
            <joint name="th_mcp" class="thumb_mcp"/>
            <geom name="th_px_visual" pos="0.0439 0.0579 -0.0086" quat="1 0 0 0" class="visual" mesh="thumb_proximal"/>
            <geom name="th_px_collision_1" size="0.01 0.02 0.012" pos="0 0.0105 -0.014" type="box"/>
            <geom name="th_px_collision_2" size="0.01 0.002 0.016" pos="0 0.031 -0.015" type="box"/>
            <geom name="th_px_collision_3" size="0.007 0.01 0.002" pos="0 0.042 0.001" type="box"/>
            <geom name="th_px_collision_4" size="0.007 0.01 0.002" pos="0 0.042 -0.029" type="box"/>
            <geom name="th_px_collision_5" size="0.005 0.012 0.009" pos="-0.0135 0.0175 -0.011656" type="box"/>

            <body name="th_ds" pos="0 0.0466 0.0002" quat="1.32679e-06 0 0 1">
              <inertial pos="0.00096191 -0.024203 -0.014419" quat="0.35287 0.311272 -0.632839 0.614904" mass="0.049" diaginertia="2.08591e-05 2.0402e-05 4.71335e-06"/>
              <joint name="th_ipl" class="thumb_ipl"/>
              <geom name="th_ds_visual" pos="0.0625 0.0784 0.0489" quat="1 0 0 0" class="visual" mesh="thumb_distal"/>
              <geom name="th_ds_collision_1" size="0.01 0.018 0.012" pos="0 -0.0085 -0.015" type="box"/>
              <geom name="th_ds_collision_2" size="0.01 0.002 0.015" pos="0 -0.029 -0.014" type="box"/>
              <geom name="th_ds_collision_3" size="0.004 0.012 0.009" pos="0.015 -0.0175 -0.0115" type="box"/>

              <geom name="th_tip" class="thumb_tip"/>
            </body>

          </body>
        </body>
      </body>  <!-- thumb -->

    </body>  <!-- palm -->
  </worldbody>
"""


class GripperLeap(MjShakableOpenCloseGripper, MjScannable):
    def __init__(self, pose: SE3Pose):
        super().__init__(pose, "palm")
        self.close_pose = np.array(
            [
                0.576,  # if_mcp_act
                0.0,  # if_rot_act
                1.43,  # if_pip_act
                0.453,  # if_dip_act
                0.856,  # mf_mcp_act
                0.0,  # mf_rot_act
                0.68,  # mf_pip_act
                0.826,  # mf_dip_act
                0.945,  # rf_mcp_act
                0.0,  # rf_rot_act
                1.3,  # rf_pip_act
                0.2,  # rf_dip_act
                1.81,  # th_cmc_act
                0.258,  # th_axl_act
                0.505,  # th_mcp_act
                0.351,  # th_ipl_act
            ]
        )

    def base_to_contact_transform(self) -> SE3Pose:
        # type: ignore
        return SE3Pose(
            np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]), type="wxyz"
        )

    def open_gripper(self, sim: MjSimulation):
        return
        # gripper_idxs = sim.get_joint_idxs(self.get_actuator_joint_names())
        # sim.set_qpos(np.copy(self.open_pose), gripper_idxs)  # type: ignore
        # sim.data.ctrl[:] = np.copy(self.open_pose)

    def close_gripper_at(self, sim: MjSimulation, pose: SE3Pose):
        self.set_pose(sim, pose)
        sim.data.ctrl[:] = np.copy(self.close_pose)
        mujoco.mj_step(sim.model, sim.data, 3000)

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
        base_path = os.path.join(ASSET_PATH, "leap")
        for file_name in os.listdir(base_path):
            path = os.path.join(base_path, file_name)
            with open(path, "rb") as f:
                ASSETS[file_name] = f.read()

        return (xml, ASSETS)

    def get_actuator_joint_names(self) -> List[str]:
        return [
            "if_mcp",
            "if_rot",
            "if_pip",
            "if_dip",
            "mf_mcp",
            "mf_rot",
            "mf_pip",
            "mf_dip",
            "rf_mcp",
            "rf_rot",
            "rf_pip",
            "rf_dip",
            "th_cmc",
            "th_axl",
            "th_mcp",
            "th_ipl",
        ]

    def get_freejoint_idxs(self, sim: MjSimulation) -> List[int]:
        start_idx = sim.get_joint_idxs(["freejoint"])[0]
        return list(range(start_idx, start_idx + 7))
