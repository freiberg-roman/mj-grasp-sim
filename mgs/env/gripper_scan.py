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

import mujoco
import mujoco.viewer
import numpy as np

from mgs.env.base import MjScanEnv
from mgs.gripper.base import MjScannable, MjScannableGripper
from mgs.util.camera import fibonacci_sphere, rnd_camera_pose_restricted
from mgs.util.geo.transforms import SE3Pose

XML = r"""
    <mujoco>
    <compiler angle="radian" autolimits="true" />
    <option integrator="implicitfast" timestep="0.001"/>
    <compiler discardvisual="false"/>
    <option noslip_iterations="1"> </option>
    <option><flag multiccd="enable"/> </option>
    <option cone="elliptic" gravity="0 0 -9.81" impratio="3" timestep="0.001" noslip_iterations="2" noslip_tolerance="1e-8" tolerance="1e-8"/>
    <option gravity="0 0 0" />
        {gripper}
        <option gravity="0 0 0" />
         <worldbody>
            {lights}
            <body name="center" pos="0.0 0.0 0.0" quat="1.0 0.0 0 0">
                <geom name="geom:center" size="0.000001" rgba="0 0 0 1.0"/>
            </body>
            <body name="body:camera" pos="0.0 0.0 .4" quat="1.0 0.0 0 0">
                <freejoint name="camera:joint"/>
                <geom name="geom:camera" size="0.01" />
                <camera name="camera" mode="targetbody" target="center"/>
            </body>
         </worldbody>
    </mujoco>
"""


class GripperScanEnv(MjScanEnv):
    def __init__(self, gripper: MjScannableGripper):
        self.gripper = gripper
        self.gripper_xml, self.gripper_assets = gripper.to_xml()

        rand_x, rand_y, rand_z = (
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-2.0, 2.0),
        )
        light_one = f"""<light name="light:one" pos="{rand_x} {rand_y} {rand_z}" attenuation="1.0 0.2 0.2" mode="targetbody" target="center"/>"""
        rand_x, rand_y, rand_z = (
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-2.0, 2.0),
        )
        light_two = f"""<light name="light:two" pos="{rand_x} {rand_y} {rand_z}" attenuation="1.0 0.2 0.2" mode="targetbody" target="center"/>"""
        rand_x, rand_y, rand_z = (
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-2.0, 2.0),
        )
        light_three = f"""<light name="light:three" pos="{rand_x} {rand_y} {rand_z}" attenuation="1.0 0.2 0.2" mode="targetbody" target="center"/>"""
        # num_lights = random.randint(1, 3)
        num_lights = 1
        light_xml = "".join([light_one, light_two, light_three][:num_lights])
        self.model_xml = XML.format(
            **{"gripper": self.gripper_xml, "body": gripper.base, "lights": light_xml}
        )

        self.model = mujoco.MjModel.from_xml_string(self.model_xml, self.gripper_assets)  # type: ignore
        self.data = mujoco.MjData(self.model)  # type: ignore
        super().__init__("camera", 480, 480)

        pose = SE3Pose(
            pos=np.array([0, 0, 0]), quat=np.array([1, 0, 0, 0]), type="wxyz"
        )
        b2c = self.gripper.base_to_contact_transform()
        pose_processed = pose @ b2c
        self.gripper.set_pose(self, pose_processed)

    def update_camera_settings(self, num_images, i):
        rnd_pos = fibonacci_sphere(total_num=num_images, i=i)
        jnt_adr_start = self.model.jnt("camera:joint").qposadr[0].item()
        self.data.qpos[jnt_adr_start : jnt_adr_start + 3] = rnd_pos
        mujoco.mj_forward(self.model, self.data)  # type: ignore
        options = (
            self.gripper.get_render_options()
            if isinstance(self.gripper, MjScannable)
            else None
        )
        self.renderer.update_scene(self.data, camera="camera", scene_option=options)
