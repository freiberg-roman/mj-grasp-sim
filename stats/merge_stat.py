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

import pandas as pd

"""Example file
name,number_successful_grasps,total_time,pos_drift_under_005,pos_drift_under_010,pos_drift_under_015,pos_drift_under_025,rot_drift_under_010,rot_drift_under_012,rot_drift_under_015,rot_drift_under_025
053_mini_soccer_ball,611,368.83021116256714,449,511,544,601,575,593,605,610
...
"""


def main():
    current_dir = os.path.dirname(__file__)
    all_directories_in_current_dir = os.listdir(current_dir)

    # filter out files
    all_directories_in_current_dir = [
        directory
        for directory in all_directories_in_current_dir
        if os.path.isdir(os.path.join(current_dir, directory))
    ]

    for directory in all_directories_in_current_dir:
        all_files_in_directory = os.listdir(os.path.join(current_dir, directory))

        all_csvs = []
        for file in all_files_in_directory:
            file_path = os.path.join(current_dir, directory, file)
            all_csvs.append(pd.read_csv(file_path))

        assert len([item["name"][0] for item in all_csvs]) == len(
            set([item["name"][0] for item in all_csvs])
        )

        # merge all the csvs
        save_path = os.path.join(current_dir, directory + "_stat.csv")
        pd.concat(all_csvs).to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
