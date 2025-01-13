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
import shutil


def main():
    with open(os.path.join(os.path.dirname(__file__), "gso_to_delete.txt"), "r") as f:
        lines = f.readlines()

    # check if all lines have a corresponding directory
    for l in lines:
        l = l.strip()
        path_to_check = os.path.join(os.path.dirname(__file__), l)
        if not os.path.exists(path_to_check):
            raise FileNotFoundError(f"Directory {path_to_check} does not exist")

        # delete all non empty directory
        shutil.rmtree(path_to_check)


if __name__ == "__main__":
    main()
