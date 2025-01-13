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

from copy import deepcopy
from typing import List

import mujoco
import mujoco.viewer
import numpy as np

from mgs.core.simualtion import MjSimulation
from mgs.gripper.base import MjShakableOpenCloseGripper
from mgs.obj.base import CollisionMeshObject
from mgs.util.geo.transforms import SE3Pose

XML = r"""
<mujoco>
    <compiler angle="radian" autolimits="true" />
    <option integrator="implicitfast" timestep="0.001"/>
    <compiler discardvisual="false"/>
    <option noslip_iterations="1"> </option>
    <option><flag multiccd="enable"/> </option>
    <option cone="elliptic" impratio="3" timestep="0.001" noslip_iterations="2" noslip_tolerance="1e-8" tolerance="1e-8"/>
    <option gravity="0 0 0" />
    {gripper}
    <worldbody>
        <light name="light:top" pos="0 0 0.3"/>
        <light name="light:right" pos="0.3 0 0"/>
        <light name="light:left" pos="-0.3 0 0"/>
        <body name="body:ground" pos="0.0 0 -1.0">
           <geom name="geom:ground" pos="0 0 0" rgba="1.0 1.0 1.0 0.0" size="1.0 1.0 0.02" type="box" density="500"/>
        </body>
    </worldbody>
    {object}
</mujoco>
"""


class GravitylessObjectGrasping(MjSimulation):
    def __init__(self, gripper: MjShakableOpenCloseGripper, obj: CollisionMeshObject):
        self.gripper = gripper
        self.obj = obj
        self.gripper_xml, self.gripper_assets = gripper.to_xml()
        self.object_xml, self.object_assets = obj.to_xml()
        self.model_xml = XML.format(
            **{"gripper": self.gripper_xml, "object": self.object_xml}
        )

        self.model = mujoco.MjModel.from_xml_string(  # type: ignore
            self.model_xml, {**self.gripper_assets, **self.object_assets}
        )
        self.data = mujoco.MjData(self.model)  # type: ignore
        mujoco.mj_forward(self.model, self.data)  # type: ignore

    def idle_grasp(self, pose: SE3Pose):
        """Mainly used for graps allignment and testing. Not for actual grasping"""

        mujoco.mj_resetData(self.model, self.data)  # type: ignore
        b2c = self.gripper.base_to_contact_transform()
        pose_processed = pose @ b2c
        self.gripper.set_pose(self, pose_processed)
        self.gripper.open_gripper(self)
        mujoco.mj_forward(self.model, self.data)  # type: ignore
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while True:
                viewer.sync()
                mujoco.mj_step(self.model, self.data)  # type: ignore

    def grasp_stability_evaluation(self, poses: SE3Pose, stats=False) -> np.ndarray:
        """
        Function simulates all grasp poses and return a numpy array of 0s and
        1s for stable and unstable grasps, respectively.
        """
        # with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
        results: List[bool] = []
        positional_drift = []
        rotational_drift = []

        for pose in poses:
            b2c = self.gripper.base_to_contact_transform()
            pose_processed = pose @ b2c

            # proper non colliding grasp execution
            mujoco.mj_resetData(self.model, self.data)  # type: ignore
            obj_pose = self.get_object_transform(self.obj.name)
            mujoco.mj_forward(self.model, self.data)  # type: ignore
            self.gripper.open_gripper(self)
            # viewer.sync()
            self.gripper.set_pose(self, pose_processed)
            if self.check_contact():
                results.append(False)
                positional_drift.append(0.0)
                rotational_drift.append(0.0)
                continue
            self.gripper.close_gripper_at(self, pose_processed)
            if not self.check_contact():
                results.append(False)
                positional_drift.append(0.0)
                rotational_drift.append(0.0)
                continue

            # check for object movement
            obj_pose_after_grasps = self.get_object_transform(self.obj.name)
            positional_drift.append(
                np.sqrt(np.sum((obj_pose.pos - obj_pose_after_grasps.pos) ** 2))
            )
            rotational_drift.append(
                np.arccos(
                    2 * np.sum(obj_pose.quat * obj_pose_after_grasps.quat) ** 2 - 1
                )
                * 180
                / np.pi
            )
            self.gripper.move_back(self, pose_processed)
            # viewer.sync()
            if not self.check_contact():
                results.append(False)
                continue
            self.gripper.move_right(self, pose_processed)
            # viewer.sync()
            if not self.check_contact():
                results.append(False)
                continue
            self.gripper.move_left(self, pose_processed)
            # viewer.sync()
            if self.check_contact():
                results.append(True)
            else:
                results.append(False)

        if stats:
            return np.array(results), np.array(positional_drift), np.array(rotational_drift)  # type: ignore
        return (
            np.array(results)
            & (np.array(positional_drift) < 0.025)
            & (np.array(rotational_drift) < 25.0)
        )

    def find_contacts(self, grasp, viewer=False):
        if viewer:
            v_resource = mujoco.viewer.launch_passive(self.model, self.data)
            viewer = v_resource.__enter__()
        else:
            viewer = None

        grasp = deepcopy(grasp)
        mujoco.mj_resetData(self.model, self.data)  # type: ignore
        mujoco.mj_forward(self.model, self.data)  # type: ignore
        self.gripper.open_gripper(self)
        self.gripper.set_pose(self, grasp)
        if self.check_contact():
            return None, None
        self.gripper.close_gripper_at(self, grasp)
        if not self.check_contact():
            return None, None
        constacts, which_finger = self.get_panda_contact()
        if constacts is None:
            return None, None

        # since the object is likely to move during the grasp, we need to transform the contacts to object frame
        transform = self.get_object_transform(self.obj.name)

        homogenous_contacts = np.concatenate(
            [deepcopy(constacts), np.ones((*(constacts.shape[:-1]), 1))], axis=-1
        ).astype(np.float32)

        contacts = np.einsum(
            "ij,...j->...i", transform.inverse().to_mat(), homogenous_contacts
        )[..., :3]

        return contacts, which_finger

    def get_object_transform(self, object_name: str):
        jnt_adr_start = self.model.jnt("{}:joint".format(object_name)).qposadr[0].item()
        obj_position = np.copy(self.data.qpos[jnt_adr_start : jnt_adr_start + 3])
        obj_quat = np.copy(self.data.qpos[jnt_adr_start + 3 : jnt_adr_start + 7])
        return SE3Pose(
            obj_position.astype(np.float32), obj_quat.astype(np.float32), "wxyz"
        )

    def get_panda_contact(self):
        panda_geom_id = [
            self.model.geom("panda_col_{}".format(i)).id for i in range(1, 13)
        ]

        panda_geom_left = [
            self.model.geom("panda_col_{}".format(i)).id for i in range(1, 7)
        ]  #  by xml definition

        panda_geom_right = [
            self.model.geom("panda_col_{}".format(i)).id for i in range(7, 13)
        ]  #  by xml definition

        contact_positions = []
        which_finger = []
        for i, contact_pair in enumerate(self.data.contact.geom):
            # check for xor condition of panda geom and object geom contact
            if (
                (contact_pair[0] in panda_geom_id)
                and (contact_pair[1] not in panda_geom_id)
            ) or (
                (contact_pair[1] in panda_geom_id)
                and (contact_pair[0] not in panda_geom_id)
            ):
                # check if the contact is with the right or left gripper finger
                if (contact_pair[0] in panda_geom_left) or (
                    contact_pair[1] in panda_geom_left
                ):
                    contact_positions.append(self.data.contact.pos[i])
                    which_finger.append("l")
                elif (contact_pair[0] in panda_geom_right) or (
                    contact_pair[1] in panda_geom_right
                ):
                    contact_positions.append(self.data.contact.pos[i])
                    which_finger.append("r")

        if not contact_positions:
            return None, None
        return np.stack(contact_positions, axis=0), which_finger

    def check_contact(self):
        """
        As the geoms are ordered accordingly to the XML. We can simply
        check for contacts between obj geoms and gripper geoms by ids
        relative to the table (which is between obj and gripper by construction)
        """
        table_id = self.model.geom("geom:ground").id
        for contact_pairs in self.data.contact.geom:
            if (contact_pairs[0] < table_id and contact_pairs[1] > table_id) or (
                contact_pairs[0] > table_id and contact_pairs[1] < table_id
            ):
                return 1.0
        return 0.0
