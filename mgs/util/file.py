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

import secrets
import time


def generate_unique_hash(length=16):
    random_string = secrets.token_hex(length)
    return random_string


def generate_unique_filename(extension="h5", length=32, addon=""):
    random_string = secrets.token_hex(length // 2)
    timestamp = int(time.time() * 1000)
    unique_filename = f"{random_string}_{timestamp}{addon}.{extension}"
    return unique_filename
