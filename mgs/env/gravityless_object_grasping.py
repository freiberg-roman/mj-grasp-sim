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
from typing import List, Dict, Any, Tuple, Union  # Added Dict, Any, Tuple, Union

import mujoco
import mujoco.viewer
import numpy as np

from mgs.core.simualtion import MjSimulation

# Make sure the gripper has the set_gripper_width method
from mgs.gripper.base import MjShakableOpenCloseGripper

# Import specific grippers ONLY if type checking requires it after set_gripper_width add
# from mgs.gripper.panda import GripperPanda
# from mgs.gripper.vx300 import GripperVX300
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
        # (Implementation remains the same)
        mujoco.mj_resetData(self.model, self.data)
        b2c = self.gripper.base_to_contact_transform()
        pose_processed = pose @ b2c
        self.gripper.set_pose(self, pose_processed)
        self.gripper.open_gripper(self)
        mujoco.mj_forward(self.model, self.data)
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while True:
                viewer.sync()
                mujoco.mj_step(self.model, self.data)

    def grasp_stability_evaluation(
        self,
        poses: SE3Pose,
        aux_info: Dict[str, Any],  # Expects 'grasp_widths'
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Simulates grasp poses: sets initial width, moves to pose, closes fully,
        and evaluates stability.

        Args:
            poses: An SE3Pose object containing N grasp poses.
            aux_info: A dictionary containing auxiliary grasp information,
                      must include 'grasp_widths' (np.ndarray of shape [N]).
            stats: If True, returns detailed stats (results, pos_drift, rot_drift).
                   If False, returns boolean array indicating stable grasps meeting drift thresholds.
            render: If True, launch passive viewer for debugging each grasp evaluation.
            render_wait_time: Sleep duration within render loop (if render=True).

        Returns:
            If stats is False: np.ndarray (boolean mask of stable grasps).
            If stats is True: Tuple[np.ndarray, np.ndarray, np.ndarray] (results, pos_drift, rot_drift).
        """
        results: List[bool] = []
        positional_drift = []
        rotational_drift = []

        # --- Input Checks ---
        if "grasp_widths" not in aux_info:
            raise ValueError("Auxiliary info dictionary must contain 'grasp_widths'.")
        grasp_widths = aux_info["grasp_widths"]
        for i, pose in enumerate(poses):
            print(f"Evaluating grasp {i + 1}/{len(poses)}...")
            width = grasp_widths[i]

            b2c = self.gripper.base_to_contact_transform()
            pose_processed = pose @ b2c
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            self.gripper.set_gripper_width(self, width)
            self.gripper.set_pose(self, pose_processed)
            mujoco.mj_forward(self.model, self.data)

            # 4. Check for collision *at preset width*
            if self.check_contact():
                results.append(False)
                positional_drift.append(np.nan)
                rotational_drift.append(np.nan)
                continue
            obj_pose_before_close = self.get_object_transform(self.obj.name)
            self.gripper.close_gripper_at(self, pose_processed)

            if not self.check_contact():
                results.append(False)
                positional_drift.append(np.nan)
                rotational_drift.append(np.nan)
                continue

            obj_pose_after_close = self.get_object_transform(self.obj.name)
            pos_drift = np.linalg.norm(
                obj_pose_before_close.pos - obj_pose_after_close.pos
            )
            dot_product = np.clip(
                np.sum(obj_pose_before_close.quat * obj_pose_after_close.quat),
                -1.0,
                1.0,
            )
            rot_drift_rad = np.arccos(2 * dot_product**2 - 1)
            rot_drift_deg = np.degrees(rot_drift_rad)
            positional_drift.append(pos_drift)
            rotational_drift.append(rot_drift_deg)
            shake_passed = True
            self.gripper.move_back(
                self, pose_processed
            )
            if not self.check_contact():
                shake_passed = False
            if shake_passed:
                self.gripper.move_right(
                    self, pose_processed
                )
                if not self.check_contact():
                    shake_passed = False
            if shake_passed:
                self.gripper.move_left(
                    self, pose_processed
                )
                if not self.check_contact():
                    shake_passed = False

            results.append(shake_passed)
            if shake_passed:
                print("  Passed.")

        # --- Process Results ---
        results_arr = np.array(results)
        pos_drift_arr = np.array(positional_drift)
        rot_drift_arr = np.array(rotational_drift)

        pos_drift_arr_filled = np.nan_to_num(pos_drift_arr, nan=np.inf)
        rot_drift_arr_filled = np.nan_to_num(rot_drift_arr, nan=np.inf)
        stable_mask = (
            results_arr
            & (pos_drift_arr_filled < 0.025)
            & (rot_drift_arr_filled < 25.0)  # Using 25 degrees threshold
        )
        return stable_mask

    def get_object_transform(self, object_name: str):
        # (Implementation remains the same)
        jnt_adr_start = self.model.jnt("{}:joint".format(object_name)).qposadr[0].item()
        obj_position = np.copy(self.data.qpos[jnt_adr_start : jnt_adr_start + 3])
        obj_quat = np.copy(self.data.qpos[jnt_adr_start + 3 : jnt_adr_start + 7])
        return SE3Pose(
            obj_position.astype(np.float32), obj_quat.astype(np.float32), "wxyz"
        )

    def check_contact(self):
        # (Implementation remains the same)
        table_id = self.model.geom("geom:ground").id
        for i in range(self.data.ncon):  # Iterate through active contacts
            contact = self.data.contact[i]
            geom1_is_obj = contact.geom1 < table_id
            geom2_is_obj = contact.geom2 < table_id
            geom1_is_gripper = contact.geom1 > table_id
            geom2_is_gripper = contact.geom2 > table_id

            if (geom1_is_obj and geom2_is_gripper) or (
                geom1_is_gripper and geom2_is_obj
            ):
                return True  # Contact between object and gripper found
        return False
