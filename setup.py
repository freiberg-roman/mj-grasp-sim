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

from pathlib import Path

import setuptools
from setuptools import setup


def read_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line and not line.startswith("-")]


with open("README.md", "r") as f:
    long_description = f.read()


init_str = Path("mgs/__init__.py").read_text()
version = init_str.split("__version__ = ")[1].rstrip().split('"')[1]


setup(
    name="mj-grasp-sim",
    version=version,
    author="Roman Freiberg",
    description="MuJoCo based grasping simulation for multiple gripper types",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="grasping, robotics, simulation",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.11",
    install_requires=read_requirements(),
)
