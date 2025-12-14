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

import math

import hydra
import numpy as np
from omegaconf import DictConfig

from mgs.env.gravityless_object_grasping import GravitylessObjectGrasping
from mgs.gripper.selector import get_gripper
from mgs.obj.marker import Marker
from mgs.util.geo.transforms import SE3Pose


def setup(gripper):
    marker_pose = SE3Pose(
        pos=np.array([0, 0, 0]), quat=np.array([1, 0, 0, 0]), type="wxyz"
    )
    marker = Marker(pose=marker_pose, name="contact_marker")
    scene = GravitylessObjectGrasping(gripper, marker)  # type: ignore
    qpos = np.array([0, -1.3963, 0, 0, 0, -1.3963, 0, 0, 0, -1.3963, 0, 0])
    # qpos = np.array([0.057, -0.057])
    # qpos = np.array(
    #     [
    #         -0.3497,
    #         0.1074,
    #         0.1424,
    #         -0.008296,
    #         -0.05406,
    #         -0.07509,
    #         0.4658,
    #         0.01033,
    #         -0.1062,
    #         0.2738,
    #         0.001251,
    #         0.02586,
    #         0.1551,
    #         -0.3593,
    #         0.1445,
    #         -0.00691,
    #         -0.001588,
    #         -0.1413,
    #         0.8383,
    #         0.199,
    #         0.4945,
    #         0.1291,
    #     ]
    # )
    # qpos = np.array(
    #     [
    #         0.22960272,
    #         0.6664361,
    #         0.90993171,
    #         0.44160989,
    #         0.30307757,
    #         0.41678267,
    #         1.08125465,
    #         0.90171303,
    #         0.23372727,
    #         0.62730184,
    #         0.88023867,
    #         0.53751856,
    #         1.0182761,
    #         0.05897662,
    #         0.76506179,
    #         0.84908875,
    #     ]
    # )
    sqrt_half = 1 / math.sqrt(2.0)
    scene.idle_grasp(
        pose=SE3Pose(
            pos=np.array([0, 0, 0]),
            quat=np.array([1, 0, 0, 0]),
            type="wxyz",
        ),
        joints=qpos,
    )
    print(rots, positions)


@hydra.main(config_path="config", config_name="show_gripper_contact")
def main(cfg: DictConfig):
    gripper = get_gripper(cfg.gripper)
    setup(gripper)


if __name__ == "__main__":
    main()
