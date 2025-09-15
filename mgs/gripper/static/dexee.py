import mujoco
import numpy as np

from mgs.core.simualtion import MjSimulation
from mgs.gripper.dexee import GripperDexee
from mgs.util.geo.transforms import SE3Pose


class StaticGripperDexee(GripperDexee):
    OPEN_POSE = [0, -1.3963, 0, 0, 0, -1.3963, 0, 0, 0, -1.3963, 0, 0]

    def base_to_contact_transform(self) -> SE3Pose:
        pos = np.array([0.0, 0.0, -0.31])
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        return SE3Pose(pos, quat, type="wxyz")

    def open_gripper(self, sim: MjSimulation):
        idxs = sim.get_joint_idxs(self.get_actuator_joint_names())

        qpos = np.array(self.OPEN_POSE)
        sim.set_qpos(np.copy(qpos), idxs)
        sim.data.ctrl[:] = np.copy(qpos)

    def close_gripper_at(self, sim: MjSimulation, pose: SE3Pose):
        self.set_pose(sim, pose)
        qpos = np.array(
            [0, -0.0325, 0, 0.00143, 0.0655, -0.0369, 0, 0, -0.0654, -0.0337, 0, 0]
        )
        sim.data.ctrl[:] = np.copy(qpos)
        mujoco.mj_step(sim.model, sim.data, 500)  # type: ignore

    def width_to_joints(self, width: np.ndarray):
        joints = np.zeros((width.shape[0], 12))
        joints[:, :] = np.array(self.OPEN_POSE)  # broadcast static open pose
        return joints
