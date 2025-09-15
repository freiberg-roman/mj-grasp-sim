import mujoco
import numpy as np

from mgs.core.simualtion import MjSimulation
from mgs.gripper.allegro import GripperAllegro
from mgs.util.geo.transforms import SE3Pose


class StaticGripperAllegro(GripperAllegro):
    OPEN_POSE = [
        -0.08,
        0.715,
        0.710,
        0.95,
        0,
        0.8,
        0.71,
        0.67,
        0.08,
        0.715,
        0.710,
        0.95,
        1.4,
        0.55,
        -0.19,
        1.45,
    ]
    CLOSE_POSE = [
        -0.08,
        0.95,
        1,
        0.95,
        0,
        0.95,
        1.2,
        0.85,
        0.08,
        0.95,
        1.2,
        0.9,
        1.4,
        0.55,
        0.29,
        1.45,
    ]

    def base_to_contact_transform(self) -> SE3Pose:
        theta = -np.pi / 2.0
        rot_offset = SE3Pose(np.array([0, 0, 0]), np.array([np.cos(theta / 2.0), 0.0, np.sin(theta / 2.0), 0.0]), type="wxyz")  # type: ignore
        offset = rot_offset @ SE3Pose(
            np.array([-0.08, 0.0, 0.01]), np.array([1.0, 0, 0, 0]), type="wxyz"
        )
        return SE3Pose(offset.pos, np.array([np.cos(theta / 2.0), 0.0, np.sin(theta / 2.0), 0.0]), type="wxyz")  # type: ignore

    def open_gripper(self, sim: MjSimulation):
        gripper_idxs = sim.get_joint_idxs(self.get_actuator_joint_names())
        sim.set_qpos(np.copy(self.open_pose), gripper_idxs)  # type: ignore
        sim.data.ctrl[:] = np.copy(self.open_pose)

    def close_gripper_at(self, sim: MjSimulation, pose: SE3Pose):
        self.set_pose(sim, pose)
        sim.data.ctrl[:] = np.copy(self.close_pose)
        mujoco.mj_step(sim.model, sim.data, 3000)  # type: ignore

    def width_to_joints(self, width: np.ndarray):
        joints = np.zeros((width.shape[0], 16))
        joints[:, :] = np.array([self.OPEN_POSE])  # broadcast static open pose
        return joints
