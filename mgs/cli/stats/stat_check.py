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

import os
from typing import Tuple

import pandas as pd
from omegaconf import DictConfig

from mgs.util.const import GIT_PATH

"""File exampel
name,number_successful_grasps,total_time,pos_drift_under_005,pos_drift_under_010,pos_drift_under_015,pos_drift_under_025,rot_drift_under_010,rot_drift_under_012,rot_drift_under_015,rot_drift_under_025
SLACK_CRUISER,294,172.80670261383057,120,163,202,240,214,229,245,278
Cole_Hardware_Electric_Pot_Assortment_55,296,208.28376984596247,124,196,231,268,236,250,264,285
Romantic_Blush_Tieks_Metallic_Italian_Leather_Ballet_Flats,560,364.0808382034302,228,347,433,510,446,473,492,535
Phillips_Stool_Softener_Liquid_Gels_30_liquid_gels,673,337.0605733394623,309,470,548,620,526,548,573,620
Office_Depot_HP_61Tricolor_Ink_Cartridge,478,246.99677968025208,209,326,388,439,377,394,409,440
...
"""

STAT_TABLE = {
    "GoogleGripper": "google-default_stat.csv",
    "PandaGripper": "panda-default_stat.csv",
    "RethinkGripper": "rethink-default_stat.csv",
    "Robotiq2f85Gripper": "robotiq_2f_85-default_stat.csv",
    "AllegroGripper": "allegro-default_stat.csv",
    "DexeeGripper": "dexee-default_stat.csv",
    "ShadowHand": {
        "two_finger_pinch": "shadow-two_finger_pinch_stat.csv",
        "three_finger_pinch": "shadow-three_finger_pinch_stat.csv",
        "grasp_hard": "shadow-grasp_hard_stat.csv",
    },
    "VXGripper": "vx300-default_stat.csv",
}


def is_graspable(
    gripper_cfg: DictConfig, object_id: str, eta=20000
) -> Tuple[bool, float]:
    """Determine whether an object will generate sufficient grasps.

    This function computes the estimated time of arrival for a given gripper-object combination using prior statistics.
    Since grippers have undergone multiple changes during development, the statistics might be inaccurate.
    More conservative gripper settings may severely influence these numbers. Generally, more conservative settings
    lead to better and more stable grasps, but at the cost of additional computation.
    """
    if gripper_cfg.name == "ShadowHand":
        stat_file = STAT_TABLE[gripper_cfg.name][gripper_cfg.grasp_type]
    else:
        stat_file = STAT_TABLE[gripper_cfg.name]
    stat_path = os.path.join(GIT_PATH, "mgs", "cli", "stats", stat_file)

    # Load panda table
    df = pd.read_csv(stat_path)

    if (
        object_id not in df["name"].values
        or df[df["name"] == object_id]["rot_pos_setting_4"].values[0] == 0
    ):
        return False
    estimated_eta = (
        df[df["name"] == object_id]["total_time"]
        * 1000.0
        / df[df["name"] == object_id]["rot_pos_setting_4"]
    )
    assert estimated_eta.values.shape[0] == 1
    return estimated_eta.values[0] < eta
