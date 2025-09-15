from typing import List

import mujoco
import numpy as np

from mgs.core.simualtion import MjSimulation
from mgs.gripper.shadow import GripperShadowRight
from mgs.util.geo.transforms import SE3Pose


class StaticGripperShadowRight(GripperShadowRight):

    OPEN_POSE = [
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
    CLOSE_POSE = [
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
    OFFSET = np.array([0.1, -0.02, -0.08])

    def base_to_contact_transform(self) -> SE3Pose:
        theta_z = np.pi / 2.0
        theta_y = -np.pi / 2.0
        rot_z = SE3Pose(np.array([0, 0, 0]), np.array([np.cos(theta_z / 2.0), 0.0, 0.0, np.sin(theta_z / 2.0)]), type="wxyz")  # type: ignore
        rot_y = SE3Pose(np.array([0, 0, 0]), np.array([np.cos(theta_y / 2.0), 0.0, np.sin(theta_y / 2.0), 0.0]), type="wxyz")  # type: ignore
        return SE3Pose(self.OFFSET, (rot_y @ rot_z).quat, type="wxyz")

    def open_gripper(self, sim: MjSimulation):
        gripper_idxs = sim.get_joint_idxs(self.get_actuator_joint_names())
        open_pose = np.array(self.OPEN_POSE)
        sim.set_qpos(open_pose, gripper_idxs)  # type: ignore
        sim.data.ctrl[:] = self._qpos_to_qacc(np.copy(open_pose))  # type: ignore

    def close_gripper_at(self, sim: MjSimulation, pose: SE3Pose):
        sim.data.mocap_pos = pose.pos
        sim.data.mocap_quat = pose.quat
        sim.data.ctrl[:] = self._qpos_to_qacc(np.array(self.CLOSE_POSE))  # type: ignore
        mujoco.mj_step(sim.model, sim.data, 3000)  # type: ignore

    def width_to_joints(self, width: np.ndarray):
        joints = np.zeros((width.shape[0], 22))
        joints[:, :] = np.array([self.OPEN_POSE])  # broadcast static open pose
        return joints
