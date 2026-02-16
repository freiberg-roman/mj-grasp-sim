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

FROM python:3.11-slim

# Accept build args
ARG http_proxy
ARG https_proxy
ARG ftp_proxy
ARG no_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG FTP_PROXY
ARG NO_PROXY

ENV http_proxy=$http_proxy \
    https_proxy=$https_proxy \
    ftp_proxy=$ftp_proxy \
    no_proxy=$no_proxy \
    HTTP_PROXY=$HTTP_PROXY \
    HTTPS_PROXY=$HTTPS_PROXY \
    FTP_PROXY=$FTP_PROXY \
    NO_PROXY=$NO_PROXY

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y \
    locales \
    libglew-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8 \
    && mkdir -p /out /stats /in

ENV LANG=en_US.utf8
ENV MUJOCO_GL=egl
ENV MGS_OUTPUT_DIR=/out
ENV MGS_INPUT_DIR=/in

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy the rest of the project
COPY . .
RUN uv sync --frozen --no-dev

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

ENTRYPOINT ["python", "-m"]
