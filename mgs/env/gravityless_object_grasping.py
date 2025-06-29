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

    def idle_grasp(self, pose: SE3Pose, joints: np.ndarray):
        # (Implementation remains the same)
        mujoco.mj_resetData(self.model, self.data)
        b2c = self.gripper.base_to_contact_transform()
        pose_processed = pose @ b2c
        gripper_joint_idxs = self.get_joint_idxs(
            self.gripper.get_actuator_joint_names()
        )
        self.set_qpos(joints, gripper_joint_idxs)
        self.gripper.set_pose(self, pose_processed)

        mujoco.mj_forward(self.model, self.data)
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while True:
                viewer.sync()
                mujoco.mj_step(self.model, self.data)

    def grasp_collision_mask(
        self,
        poses: SE3Pose,
        joints: np.ndarray,
    ) -> np.ndarray:
        if len(poses) != len(joints):
            raise ValueError(
                f"Number of poses ({len(poses)}) must match number of joint configurations ({len(joints)})."
            )
        if joints.shape[1] != len(self.gripper.get_actuator_joint_names()):
            raise ValueError(
                f"Joints array has incorrect dimension ({joints.shape[1]}), expected {len(self.gripper.get_actuator_joint_names())}."
            )

        collision_free_mask: List[bool] = []
        num_grasps = len(poses)
        gripper_joint_idxs = self.get_joint_idxs(
            self.gripper.get_actuator_joint_names()
        )  # Cache indices

        initial_state = self.get_state()

        for i in range(num_grasps):
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)

            b2c = self.gripper.base_to_contact_transform()
            pose_processed = poses[i] @ b2c  # Apply base-to-contact transform
            self.set_qpos(joints[i], gripper_joint_idxs)
            self.gripper.set_pose(self, pose_processed)
            mujoco.mj_forward(self.model, self.data)
            has_contact = self.check_contact()
            collision_free_mask.append(not has_contact)

        self.set_state(initial_state)
        return np.array(collision_free_mask)

    def grasp_stability_evaluation_from_joints(
        self,
        poses: SE3Pose,
        joints: np.ndarray,
        nstep_lift: int = 3000,  # Steps for lifting simulation
        lift_dist: float = 0.1,  # Distance to lift
        shake_steps: int = 500,  # Steps per shake direction
        shake_dist: float = 0.02,  # Distance to shake side-to-side
        enough_stable=None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(poses) != len(joints):
            raise ValueError(
                f"Number of poses ({len(poses)}) must match number of joint configurations ({len(joints)})."
            )
        results: List[bool] = []
        positional_drift: List[float] = []
        rotational_drift: List[float] = []
        num_grasps = len(poses)
        gripper_joint_idxs = self.get_joint_idxs(
            self.gripper.get_actuator_joint_names()
        )

        count_stable = 0
        for i in range(num_grasps):
            if enough_stable is not None:
                if count_stable >= enough_stable:
                    positional_drift.append(np.nan)
                    rotational_drift.append(np.nan)
                    results.append(False)
                    continue

            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)

            b2c = self.gripper.base_to_contact_transform()
            pose_processed = poses[i] @ b2c
            self.set_qpos(joints[i], gripper_joint_idxs)
            self.gripper.set_pose(self, pose_processed)
            mujoco.mj_forward(self.model, self.data)  # Update geom positions
            obj_pose_before_close = self.get_object_transform(self.obj.name)
            self.gripper.close_gripper_at(self, pose_processed)
            if not self.check_contact_with_object():
                # print("  Failed: No contact after closing.") # Optional debug print
                results.append(False)
                positional_drift.append(np.nan)
                rotational_drift.append(np.nan)
                continue

            # 5. Calculate drift
            obj_pose_after_close = self.get_object_transform(self.obj.name)
            pos_drift = np.linalg.norm(
                obj_pose_before_close.pos - obj_pose_after_close.pos
            )
            # Ensure quaternions are aligned for angle calculation
            dot_product = np.clip(
                np.sum(obj_pose_before_close.quat * obj_pose_after_close.quat),
                -1.0,
                1.0,
            )
            # Correct angle calculation: theta = acos(2*dot^2 - 1)
            angle_rad = np.arccos(2 * dot_product**2 - 1)
            # Handle potential numerical inaccuracies leading to NaN
            if np.isnan(angle_rad):
                angle_rad = (
                    0.0 if np.isclose(dot_product**2, 0.5) else np.pi
                )  # If dot is +/- sqrt(0.5), angle is 90 deg, if dot is +/- 1, angle is 0, otherwise something went wrong -> pi
                print(
                    f"Warning: NaN encountered in rotation drift calculation for grasp {i}. Dot product: {dot_product}. Setting angle to {np.degrees(angle_rad)} deg."
                )

            rot_drift_deg = np.degrees(angle_rad)

            positional_drift.append(pos_drift)
            rotational_drift.append(rot_drift_deg)

            # 6. Perform lift and shake tests
            shake_passed = True
            # --- Lift ---
            start_pos_lift = np.copy(self.data.mocap_pos[0, :])
            lift_target_z = start_pos_lift[2] + lift_dist
            for t in range(nstep_lift):
                # Simple linear interpolation for lifting
                current_z = start_pos_lift[2] + (lift_target_z - start_pos_lift[2]) * (
                    t / nstep_lift
                )
                # Directly manipulate mocap
                self.data.mocap_pos[0, 2] = current_z
                mujoco.mj_step(self.model, self.data)
                # Check contact periodically during lift
                if t > 0 and t % 100 == 0 and not self.check_contact_with_object():
                    shake_passed = False
                    # print(f"  Failed: Lost contact during lift at step {t}.") # Optional debug
                    break
            if not shake_passed:
                results.append(False)
                continue
            if not self.check_contact_with_object():  # Final check after lift
                # print("  Failed: Lost contact after lift.") # Optional debug
                results.append(False)
                continue

            # --- Shake ---
            current_mocap_pose = SE3Pose(
                np.copy(self.data.mocap_pos[0, :]),
                np.copy(self.data.mocap_quat[0, :]),
                "wxyz",
            )

            # Move back (relative to current pose)
            rot_mat = current_mocap_pose.to_mat()[:3, :3]
            back_direction = rot_mat @ np.array([0, 0, -1.0])
            target_pos_back = current_mocap_pose.pos + back_direction * shake_dist
            start_pos_shake = np.copy(self.data.mocap_pos[0, :])
            for t in range(shake_steps):
                self.data.mocap_pos[0, :] = start_pos_shake + (
                    target_pos_back - start_pos_shake
                ) * (t / shake_steps)
                mujoco.mj_step(self.model, self.data)
            if not self.check_contact_with_object():
                shake_passed = False
                # print("  Failed: Lost contact after move back.") # Optional debug

            # Move right (relative to current pose)
            if shake_passed:
                right_direction = rot_mat @ np.array([0, 1.0, 0])
                target_pos_right = target_pos_back + right_direction * shake_dist
                start_pos_shake = np.copy(self.data.mocap_pos[0, :])
                for t in range(shake_steps):
                    self.data.mocap_pos[0, :] = start_pos_shake + (
                        target_pos_right - start_pos_shake
                    ) * (t / shake_steps)
                    mujoco.mj_step(self.model, self.data)
                if not self.check_contact_with_object():
                    shake_passed = False
                    # print("  Failed: Lost contact after move right.") # Optional debug

            # Move left (relative to current pose)
            if shake_passed:
                left_direction = rot_mat @ np.array([0, -1.0, 0])
                # Start from the right-most position and move double the distance left
                target_pos_left = start_pos_shake + left_direction * (2 * shake_dist)
                # start_pos_shake = np.copy(self.data.mocap_pos[0, :]) # Already at right pos
                for t in range(shake_steps * 2):  # Longer move across
                    self.data.mocap_pos[0, :] = start_pos_shake + (
                        target_pos_left - start_pos_shake
                    ) * (t / (shake_steps * 2))
                    mujoco.mj_step(self.model, self.data)
                if not self.check_contact_with_object():
                    shake_passed = False
                    # print("  Failed: Lost contact after move left.") # Optional debug

            results.append(shake_passed)
            count_stable += int(shake_passed)

        # --- Process and Return Results ---
        results_arr = np.array(results)
        # Ensure float for NaN
        pos_drift_arr = np.array(positional_drift, dtype=float)
        # Ensure float for NaN
        rot_drift_arr = np.array(rotational_drift, dtype=float)

        # Apply stability thresholds here if desired, or return raw drifts
        # Example thresholding:
        # pos_drift_ok = np.nan_to_num(pos_drift_arr, nan=np.inf) < 0.025
        # rot_drift_ok = np.nan_to_num(rot_drift_arr, nan=np.inf) < 25.0
        # final_stable_mask = results_arr & pos_drift_ok & rot_drift_ok
        # return final_stable_mask, pos_drift_arr, rot_drift_arr

        return results_arr  # , pos_drift_arr, rot_drift_arr

    def get_object_transform(self, object_name: str):
        # (Implementation remains the same)
        jnt_adr_start = self.model.jnt("{}:joint".format(object_name)).qposadr[0].item()
        obj_position = np.copy(self.data.qpos[jnt_adr_start : jnt_adr_start + 3])
        obj_quat = np.copy(self.data.qpos[jnt_adr_start + 3 : jnt_adr_start + 7])
        return SE3Pose(
            obj_position.astype(np.float32), obj_quat.astype(np.float32), "wxyz"
        )

    def check_contact(self):
        return self.data.ncon != 0

    def check_contact_with_object(self):
        """
        As the geoms are ordered accordingly to the XML. We can simply
        check for contacts between obj geoms and gripper geoms by ids
        relative to the table (which is inbetween obj and gripper by construction)
        """
        table_id = self.model.geom("geom:ground").id
        for contact_pairs in self.data.contact.geom:
            if (contact_pairs[0] < table_id and contact_pairs[1] > table_id) or (
                contact_pairs[0] > table_id and contact_pairs[1] < table_id
            ):
                return True
        return False
