import numpy as np

from mgs.gripper.panda import GripperPanda
from mgs.util.geo.transforms import SE3Pose


class StaticGripperPanda(GripperPanda):
    MIN_WIDTH_TARGET = 0.0  # Target closed width before clamping
    MAX_WIDTH = 0.08  # Max open width (8cm)
    MIN_WIDTH_CLAMP = 0.003  # Minimum physical width clamp (3mm)

    # Joint limits from XML
    Q1_RANGE = [0.0, 0.04]
    Q2_RANGE = [-0.04, 0.0]

    def base_to_contact_transform(self) -> SE3Pose:
        pos = np.array([0, 0, -0.102])
        quat = np.array([0.707106781, 0.0, 0.0, 0.707106781])
        return SE3Pose(pos, quat, type="wxyz")

    def width_to_joints(self, width: np.ndarray):
        joints = np.zeros(shape=(width.shape[0], 2))
        joints[:, 0] = 0.04  # return always an open gripper
        return joints
