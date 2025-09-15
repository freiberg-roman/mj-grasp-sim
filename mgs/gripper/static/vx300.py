import numpy as np

from mgs.gripper.vx300 import GripperVX300
from mgs.util.geo.transforms import SE3Pose


class StaticGripperVX300(GripperVX300):
    """
    Represents the Interbotix WidowX-series gripper (like VX300s) in MuJoCo.
    Uses independent position control for each finger.
    """

    # Define min/max width constants based on joint ranges
    # Left finger (q1): [0.021, 0.057] -> Outward movement is positive
    # Right finger (q2): [-0.057, -0.021] -> Outward movement is negative (more negative)
    # Width = q1 - q2 (distance along the sliding axis, assuming symmetric placement)
    # Max width: 0.057 - (-0.057) = 0.114 meters (11.4 cm)
    # Min width: 0.021 - (-0.021) = 0.042 meters (4.2 cm)
    MIN_WIDTH = 0.042
    MAX_WIDTH = 0.114
    # Clamp slightly above zero for stability if needed, though MIN_WIDTH is already large
    MIN_WIDTH_CLAMP = 0.003

    # Joint limits from XML
    Q1_RANGE = [0.021, 0.057]  # left_finger
    Q2_RANGE = [-0.057, -0.021]  # right_finger

    def base_to_contact_transform(self) -> SE3Pose:
        rot_around_y = SE3Pose(
            np.array([0, 0, 0]),
            np.array([0.707106781, 0, -0.707106781, 0]),
            type="wxyz",
        )
        rot_around_z = SE3Pose(
            np.array([0, 0, 0]),
            np.array([0.707106781, 0, 0.0, 0.707106781]),
            type="wxyz",
        )

        rot = rot_around_z @ rot_around_y
        rot.pos = np.array([0, 0, -0.12])

        return rot

    def width_to_joints(self, width: np.ndarray):
        joints = np.zeros((width.shape[0], 2))
        joints[:, 0] = 0.057  # static open pose
        joints[:, 1] = -0.057
        return joints
