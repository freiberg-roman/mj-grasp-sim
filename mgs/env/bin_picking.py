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

import random
from copy import deepcopy
from typing import List, TypedDict

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

from mgs.env.base import Loadable, MjScanEnv
from mgs.gripper.base import MjShakableOpenCloseGripper
from mgs.obj.base import CollisionMeshObject
from mgs.util.camera import rnd_camera_pose_restricted
from mgs.util.geo.convert import quat_xyzw_to_wxyz
from mgs.util.geo.transforms import SE3Pose


class BinPickingState(TypedDict):
    geom_conaffinity: np.ndarray
    geom_contype: np.ndarray
    geom_rgba: np.ndarray
    body_gravcomp: np.ndarray
    state: np.ndarray


XML = r"""
<mujoco>
    <compiler angle="radian" autolimits="true" />
    <size memory="32M" />
    <option integrator="implicitfast" timestep="0.001"/>
    <compiler discardvisual="false"/>
    <option noslip_iterations="3"> </option>
    <option><flag multiccd="enable"/> </option>
    <option cone="elliptic" gravity="0 0 -9.81" impratio="3" noslip_iterations="3" noslip_tolerance="1e-10" tolerance="1e-10"/>
    {gripper}
    <worldbody>
        {lights}
        <body name="body:table" pos="0.0 0 -0.02">
           <geom name="geom:table" pos="0 0 0" rgba="{table_color}" size="10 10 0.02" type="box" density="500" friction="1.0 0.1 0.1"/>
        </body>
        <body name="body:bin" pos="0.0 0 0.01">
           <geom name="geom:bin_groud" pos="0 0 0" rgba="{box_color}" size="{half-width} {half-length} {half-thickness}" type="box" density="500" friction="1.0 0.1 0.1"/>
           <geom name="geom:bin_front" pos="0 -{half-length-offset} {half-height-offset}" rgba="{box_color}" size="{half-width} {half-thickness} {half-height}" type="box" density="500" friction="1.0 0.1 0.1"/>
           <geom name="geom:bin_back" pos="0 {half-length-offset} {half-height-offset}" rgba="{box_color}" size="{half-width} {half-thickness} {half-height}" type="box" density="500" friction="1.0 0.1 0.1"/>
           <geom name="geom:bin_right" pos="{half-width-offset} 0 {half-height-offset}" rgba="{box_color}" size="{half-thickness} {half-length} {half-height}" type="box" density="500" friction="1.0 0.1 0.1"/>
           <geom name="geom:bin_left" pos="-{half-width-offset} 0 {half-height-offset}" rgba="{box_color}" size="{half-thickness} {half-length} {half-height}" type="box" density="500" friction="1.0 0.1 0.1"/>
        </body>
        <body name="body:camera" pos="0.0 0.0 -1.0" quat="1.0 0.0 0 0" gravcomp="1">
          <freejoint name="camera:joint"/>
          <geom name="geom:camera" size="0.01"/>
          <camera name="camera" mode="targetbody" target="base_origin"/>
        </body>
        <body name="base_origin" pos="0.0 0.0 -0.025" quat="1.0 0.0 0 0">
          <geom name="geom:base_origin" size="0.01" rgba="1.0 0 0 1"/>
        </body>
        <body name="body:wall_top" pos="0.0 1.0 0.1">
           <geom name="geom:wall_top" pos="0 0 0" rgba="1. 0. 0. 0." size="1.0 0.02 0.2" type="box" density="500"/>
        </body>
        <body name="body:wall_right" pos="1.0 0.0 0.1">
           <geom name="geom:wall_right" pos="0 0 0" rgba="1. 0. 0. 0." size="0.02 1.0 0.2" type="box" density="500"/>
        </body>
        <body name="body:wall_bottom" pos="0.0 -1.0 0.1">
           <geom name="geom:wall_bottom" pos="0 0 0" rgba="1. 0. 0. 0." size="1.0 0.02 0.2" type="box" density="500"/>
        </body>
        <body name="body:wall_left" pos="-1.0 0.0 0.1">
           <geom name="geom:wall_left" pos="0 0 0" rgba="1. 0. 0. 0." size="0.02 1.0 0.2" type="box" density="500"/>
        </body>
    </worldbody>
    {objects}
</mujoco>
"""


class BinPickingEnv(MjScanEnv, Loadable):
    def __init__(
        self,
        gripper: MjShakableOpenCloseGripper,
        objects: List[CollisionMeshObject],
        scene_randomization=True,
        box_size=None,
    ):
        self.gripper = gripper
        self.objects = objects
        self.gripper_xml, self.gripper_assets = gripper.to_xml()
        self.object_xml_assets = [obj.to_xml() for obj in objects]
        self.object_names = [obj.name for obj in objects]
        self.object_ids = [obj.object_id for obj in objects]

        self.objs_xml_concat = ""
        self.objs_assets = {}
        for obj_xml, obj_assets in self.object_xml_assets:
            self.objs_xml_concat += obj_xml
            self.objs_assets = {**self.objs_assets, **obj_assets}

        if scene_randomization:
            random_color = (
                " ".join([str(np.random.uniform(0, 1)) for _ in range(3)]) + " 1.0"
            )
            random_color_two = (
                " ".join([str(np.random.uniform(0, 1)) for _ in range(3)]) + " 1.0"
            )
        else:
            random_color = "0.5 0.5 0.5 1.0"
            random_color_two = "0.5 0.5 0.5 1.0"

        rand_x, rand_y = np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)
        light_one = f"""<light name="light:one" pos="{rand_x} {rand_y} 2.0" attenuation="1.0 0.2 0.2" mode="targetbody" target="base_origin"/>"""
        rand_x, rand_y = np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)
        light_two = f"""<light name="light:two" pos="{rand_x} {rand_y} 2.0" attenuation="1.0 0.2 0.2" mode="targetbody" target="base_origin"/>"""
        rand_x, rand_y = np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)
        light_three = f"""<light name="light:three" pos="{rand_x} {rand_y} 2.0" attenuation="1.0 0.2 0.2" mode="targetbody" target="base_origin"/>"""
        num_lights = random.randint(1, 3)
        light_xml = "".join([light_one, light_two, light_three][:num_lights])

        self.half_width = np.random.uniform(0.3, 0.6)
        self.half_length = np.random.uniform(0.2, 0.4)
        self.half_height = np.random.uniform(0.1, 0.2)
        self.half_thickness = np.random.uniform(0.005, 0.02)

        if box_size is not None:
            self.half_width = box_size["half_width"]
            self.half_length = box_size["half_length"]
            self.half_height = box_size["half_height"]
            self.half_thickness = box_size["half_thickness"]

        self.model_xml = XML.format(
            **{
                "gripper": self.gripper_xml,
                "objects": self.objs_xml_concat,
                "table_color": random_color,
                "box_color": random_color_two,
                "lights": light_xml,
                "half-width": str(self.half_width + self.half_thickness),
                "half-width-offset": str(self.half_width),
                "half-length": str(self.half_length + self.half_thickness),
                "half-length-offset": str(self.half_length),
                "half-height": str(self.half_height + self.half_thickness),
                "half-height-offset": str(self.half_height),
                "half-thickness": str(self.half_thickness),
            }
        )

        self.env_defintion = {
            "model_xml": self.model_xml,
            "assets": {**self.gripper_assets, **self.objs_assets},
        }
        self.model = mujoco.MjModel.from_xml_string(  # type: ignore
            self.model_xml, {**self.gripper_assets, **self.objs_assets}
        )
        self.data = mujoco.MjData(self.model)  # type: ignore
        mujoco.mj_forward(self.model, self.data)  # type: ignore

        self.next_x, self.next_y, self.counter = -5.5, -5.0, 0
        super().__init__("camera", 480, 480)

    def get_object(self, object_name: str):
        for obj in self.objects:
            if obj.name == object_name:
                return obj
        return None

    def remove_obj(self, obj):
        body_id = self.model.body(obj.name).id

        for i in range(self.model.ngeom):
            if self.model.geom_bodyid[i] == body_id:
                self.model.geom_conaffinity[i] = 0
                self.model.geom_contype[i] = 0
                self.model.geom_rgba[i] = [0, 0, 0, 0]
        self.model.body_gravcomp[body_id] = 1.0
        mujoco.mj_forward(self.model, self.data)  # type: ignore

    def settle(self):
        mujoco.mj_step(self.model, self.data, 10000)  # type: ignore

    def is_stable(self):
        stats = {}
        for obj in self.objects:
            stats[obj.name] = 0
        start_poses = []
        for _ in range(10):
            start_poses = []
            for obj in self.objects:
                obj_id = mujoco.mj_name2id(  # type:ignore
                    m=self.model,
                    name=(obj.name + ":joint"),
                    type=mujoco.mjtObj.mjOBJ_JOINT,  # type: ignore
                )
                obj_qpos_addr = self.model.jnt_qposadr[obj_id]
                start_pos = deepcopy(self.data.qpos[obj_qpos_addr : obj_qpos_addr + 3])
                start_poses.append(start_pos)

            mujoco.mj_step(self.model, self.data, 100)  # type: ignore

            for obj, start_pos in zip(self.objects, start_poses):
                obj_id = mujoco.mj_name2id(  # type: ignore
                    m=self.model,
                    name=(obj.name + ":joint"),
                    type=mujoco.mjtObj.mjOBJ_JOINT,  # type: ignore
                )
                obj_qpos_addr = self.model.jnt_qposadr[obj_id]
                end_pos = deepcopy(self.data.qpos[obj_qpos_addr : obj_qpos_addr + 3])
                delta = np.sum(np.abs(end_pos - start_pos))
                stats[obj.name] += delta

        max_delta = 0.0
        for delta in stats.values():
            if delta > max_delta:
                max_delta = delta

        return max_delta < 5e-3

    def gen_clutter(self):
        def random_pose():
            scipy_random_quat = Rotation.random().as_quat()  # type: ignore
            mujoco_random_quat = quat_xyzw_to_wxyz(scipy_random_quat)
            return SE3Pose(
                pos=np.array([0.0, 0.0, 0.8]), quat=mujoco_random_quat, type="wxyz"
            )

        drop_pose = random_pose()
        for obj_name in self.object_names:
            jnt_adr_start = (
                self.model.jnt("{}:joint".format(obj_name)).qposadr[0].item()
            )
            self.data.qpos[jnt_adr_start : jnt_adr_start + 7] = deepcopy(
                drop_pose.to_vec(layout="pq", type="wxyz")
            )
            self.data.qacc[:] = 0.0
            self.data.qvel[:] = 0.0
            for _ in range(900):
                np.clip(self.data.qacc, -50.0, 50.0, out=self.data.qacc)
                np.clip(self.data.qvel, -50.0, 50.0, out=self.data.qvel)
                mujoco.mj_step(self.model, self.data)  # type: ignore
        for _ in range(9000):
            np.clip(self.data.qacc, -1.0, 1.0, out=self.data.qacc)
            np.clip(self.data.qvel, -50.0, 50.0, out=self.data.qvel)
            mujoco.mj_step(self.model, self.data)  # type: ignore

    def update_camera_settings(self, num_images, i):
        rnd_pos, _ = rnd_camera_pose_restricted(radius=1.2, phi=np.pi * 0.125)
        jnt_adr_start = self.model.jnt("camera:joint").qposadr[0].item()
        self.data.qpos[jnt_adr_start : jnt_adr_start + 3] = rnd_pos
        mujoco.mj_forward(self.model, self.data)  # type: ignore
        self.renderer.update_scene(self.data, camera="camera")

    def check_gripper_collision(self):
        """
        As the geoms are ordered accordingly to the XML. We can simply
        check for contacts between obj geoms and gripper geoms by ids
        relative to the table (which is inbetween obj and gripper by construction)
        """
        table_id = self.model.geom("geom:table").id
        for contact_pairs in self.data.contact.geom:
            if (
                (contact_pairs[0] < table_id and contact_pairs[1] > table_id)
                or (contact_pairs[0] > table_id and contact_pairs[1] < table_id)
                or (contact_pairs[0] == table_id and contact_pairs[1] < table_id)
                or (contact_pairs[0] < table_id and contact_pairs[1] == table_id)
            ):
                return 1.0
        return 0.0

    def check_gripper_contact(self):
        table_id = self.model.geom("geom:table").id
        for contact_pairs in self.data.contact.geom:
            if (
                (contact_pairs[0] < table_id and contact_pairs[1] > table_id)
                or (contact_pairs[0] > table_id and contact_pairs[1] < table_id)
                and (
                    not (
                        (contact_pairs[0] == table_id and contact_pairs[1] < table_id)
                        or (
                            contact_pairs[0] < table_id and contact_pairs[1] == table_id
                        )
                    )
                )
            ):
                return 1.0
        return 0.0

    def grasp_stable_mask(self, grasps, env_state):
        is_grasp_stable = []

        b2c = self.gripper.base_to_contact_transform()
        grasps = grasps @ b2c
        for grasp in grasps:
            spec = mujoco.mjtState.mjSTATE_INTEGRATION  # type: ignore
            mujoco.mj_setState(self.model, self.data, env_state, spec)  # type: ignore
            self.gripper.open_gripper(self)
            self.gripper.set_pose(self, grasp)
            if self.check_gripper_collision():
                is_grasp_stable.append(False)
                continue

            self.gripper.close_gripper_at(self, grasp)
            current_pos, current_quat = np.copy(grasp.pos), np.copy(grasp.quat)
            z = np.copy(current_pos[2])
            for i in range(20000):
                self.data.mocap_pos[0, :] = current_pos
                self.data.mocap_quat[0, :] = current_quat
                self.data.mocap_pos[0, 2] = np.copy(z)
                z += 0.00003
                mujoco.mj_step(self.model, self.data)  # type: ignore
                if i in [x for x in range(3000, 18001, 500)]:
                    if not self.check_gripper_contact():
                        break

            if self.check_gripper_contact():
                is_grasp_stable.append(True)
            else:
                is_grasp_stable.append(False)

        stable_grasp_masks = np.array(is_grasp_stable)
        return stable_grasp_masks

    def get_obj_pose(self, object_name: str):
        mujoco.mj_forward(self.model, self.data)  # type: ignore
        jnt_adr_start = self.model.jnt("{}:joint".format(object_name)).qposadr[0].item()
        obj_position = np.copy(self.data.qpos[jnt_adr_start : jnt_adr_start + 3])
        obj_quat = np.copy(self.data.qpos[jnt_adr_start + 3 : jnt_adr_start + 7])
        return SE3Pose(obj_position, obj_quat, "wxyz")

    def grasp_collision_mask(self, grasps: SE3Pose) -> np.ndarray:
        current_state = self.get_state()
        collision_free_grasps = []
        for grasp in grasps:
            in_bound = (
                (grasp.pos[..., 0] < self.half_width + self.half_thickness)
                & (grasp.pos[..., 0] > -(self.half_width + self.half_thickness))
                & (grasp.pos[..., 1] < self.half_length + self.half_thickness)
                & (grasp.pos[..., 1] > -(self.half_length + self.half_thickness))
                & (grasp.pos[..., 2] < 1.0)
                & (grasp.pos[..., 2] > self.half_thickness)
            ).item()
            if not in_bound:
                collision_free_grasps.append(False)
                continue

            top_down_approach_vec = np.array([0, 0, -1.0], dtype=np.float32)
            rotated_vec = grasp.to_mat()[:3, :3] @ top_down_approach_vec
            angle = np.arccos(rotated_vec[2]) * 180.0 / np.pi
            if np.abs(angle) > 40:  # less than 40 degrees
                collision_free_grasps.append(False)
                continue

            self.set_state(current_state)
            self.gripper.open_gripper(self)
            b2c = self.gripper.base_to_contact_transform()
            self.gripper.set_pose(self, grasp @ b2c)
            has_collision = self.check_gripper_collision()
            collision_free_grasps.append(not has_collision)
        collision_free_masks = np.array(collision_free_grasps)
        return collision_free_masks

    def to_dict(self):
        state_dict: BinPickingState = {
            "geom_conaffinity": deepcopy(self.model.geom_conaffinity),
            "geom_contype": deepcopy(self.model.geom_contype),
            "geom_rgba": deepcopy(self.model.geom_rgba),
            "body_gravcomp": deepcopy(self.model.body_gravcomp),
            "state": self.get_state(),
        }
        dict = {
            "gripper": deepcopy(self.gripper),
            "objects": deepcopy(self.objects),
            "env_state": deepcopy(state_dict),
            "box_size": {
                "half_height": self.half_height,
                "half_width": self.half_width,
                "half_length": self.half_length,
                "half_thickness": self.half_thickness,
            },
        }
        return dict

    @classmethod
    def from_dict(cls, state_dict):
        gripper, obj_list, state, box_size = (
            state_dict["gripper"],
            state_dict["objects"],
            state_dict["env_state"],
            state_dict["box_size"],
        )
        env = BinPickingEnv(
            gripper, obj_list, scene_randomization=False, box_size=box_size
        )
        env.set_state(state["state"])

        env.model.geom_conaffinity[:] = state["geom_conaffinity"]
        env.model.geom_contype[:] = state["geom_contype"]
        env.model.geom_rgba[:] = state["geom_rgba"]
        env.model.body_gravcomp[:] = state["body_gravcomp"]

        return env
