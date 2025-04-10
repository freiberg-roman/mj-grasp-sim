# --- File: cli/inspect.py ---

# Copyright (c) 2025 Robert Bosch GmbH
# Author: Your Name/AI Assistant based on user request
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

"""
Inspects MuJoCo objects listed in names_common.txt using a custom viewer loop.

This script loads the first N objects specified in the names_common.txt file,
one by one, and displays them in a MuJoCo viewer created manually with glfw,
showing both visual and collision geometries (group 3). The user can close the
viewer window to proceed to the next object. Includes basic mouse controls.
"""

import os
import sys
import time
from pathlib import Path

# Attempt to set up paths relative to the script location
try:
    _project_root = Path(__file__).resolve().parent.parent.parent
    _mgs_path = _project_root / "mgs"
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    if str(_mgs_path) not in sys.path:
         sys.path.insert(0, str(_mgs_path))
    GIT_PATH = str(_project_root)
    ASSET_PATH = os.path.join(GIT_PATH, "asset")
    if not os.path.isdir(ASSET_PATH):
         print(f"Warning: Default ASSET_PATH '{ASSET_PATH}' not found.", file=sys.stderr)
         try:
             from mgs.util.const import ASSET_PATH as const_asset_path
             ASSET_PATH = const_asset_path
             print(f"Using ASSET_PATH from mgs.util.const: {ASSET_PATH}")
         except ImportError:
             print("Could not determine ASSET_PATH automatically.", file=sys.stderr)

except IndexError:
    print("Warning: Could not automatically determine project structure.", file=sys.stderr)
    GIT_PATH = os.getcwd() # Fallback
    ASSET_PATH = os.path.join(GIT_PATH, "asset") # Fallback

# Now import project modules
try:
    import mujoco
    # import mujoco.viewer # We are replacing launch_passive
    import glfw # Need glfw for manual window
    import numpy as np
    from omegaconf import OmegaConf

    from mgs.obj.selector import get_object
    from mgs.gripper.selector import get_gripper
    from mgs.env.gravityless_object_grasping import GravitylessObjectGrasping
    from mgs.util.geo.transforms import SE3Pose
    from mgs.util.file import generate_unique_hash # Implicitly used
    from mgs.obj.ycb import ObjectYCB # Needed for get_object checks
    from mgs.obj.gso import ObjectGSO # Needed for get_object checks

except ImportError as e:
    print(f"Error importing necessary modules: {e}", file=sys.stderr)
    print("Please ensure the script is run from the project root directory", file=sys.stderr)
    print("or the 'mgs' package is in your PYTHONPATH.", file=sys.stderr)
    print("You might also need to install glfw: pip install glfw", file=sys.stderr)
    sys.exit(1)


# --- Configuration ---
NAMES_FILE_PATH = os.path.join(GIT_PATH, "names_common_clean.txt")
NUM_OBJECTS_TO_INSPECT = 236
GRIPPER_CONFIG = OmegaConf.create({"name": "PandaGripper"})
GRIPPER_DEFAULT_POSE = SE3Pose(np.array([.4, .0, .0]), np.array([1.0, 0.0, 0.0, 0.0]), type="wxyz")
COLLISION_GEOM_GROUP = 3 # MuJoCo default group for collisions
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 900

# --- GLFW Viewer Globals (for callbacks) ---
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

# --- GLFW Callback Functions (adapted from mujoco-python simulate.py) ---
def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    global button_left, button_middle, button_right
    button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
    glfw.get_cursor_pos(window) # Update cursor position

def mouse_move(window, xpos, ypos):
    global lastx, lasty, button_left, button_middle, button_right
    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos
    if not (button_left or button_middle or button_right):
        return

    # Get window size
    width, height = glfw.get_window_size(window)
    PRESS_LEFT_SHIFT = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS)
    PRESS_LEFT_CONTROL = (glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS)

    # Determine action based on button and modifiers
    action = None
    if button_right:
        action = mujoco.mjtMouse.mjMOUSE_MOVE_H if PRESS_LEFT_SHIFT else mujoco.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if PRESS_LEFT_SHIFT else mujoco.mjtMouse.mjMOUSE_ROTATE_V
    elif button_middle:
        action = mujoco.mjtMouse.mjMOUSE_ZOOM

    if action:
        mujoco.mjv_moveCamera(model, action, dx / width, dy / height, scene, cam)


def scroll(window, xoffset, yoffset):
    action = mujoco.mjtMouse.mjMOUSE_ZOOM
    mujoco.mjv_moveCamera(model, action, 0.0, -0.05 * yoffset, scene, cam)

# --- Helper Function for Rendering ---
def render_passive_with_options(local_model, local_data, local_opt, window_title="Object Inspector"):
    """Creates a glfw window and runs a passive rendering loop."""
    global model, data, cam, scene # Make model/data accessible to callbacks

    model = local_model
    data = local_data

    if not glfw.init():
        print("Could not initialize GLFW", file=sys.stderr)
        return

    window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, window_title, None, None)
    if not window:
        glfw.terminate()
        print("Could not create GLFW window", file=sys.stderr)
        return

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # Set callbacks
    glfw.set_key_callback(window, keyboard)
    glfw.set_cursor_pos_callback(window, mouse_move)
    glfw.set_mouse_button_callback(window, mouse_button)
    glfw.set_scroll_callback(window, scroll)

    # Initialize MuJoCo visualization objects
    cam = mujoco.MjvCamera()
    scene = mujoco.MjvScene(model, maxgeom=10000) # Increase maxgeom if needed
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

    # Set initial camera viewpoint (optional, adjust as needed)
    cam.azimuth = 90
    cam.elevation = -30
    cam.distance = 1.0
    cam.lookat[:] = [0.0, 0.0, 0.0] # Look at origin

    # Get initial cursor position
    global lastx, lasty
    lastx, lasty = glfw.get_cursor_pos(window)

    print("  Viewer launched. Use mouse to navigate. Close window to continue.")

    while not glfw.window_should_close(window):
        time_prev = data.time

        # Advance simulation only if needed (for dynamic elements, not here)
        # while data.time - time_prev < 1.0 / 60.0: # Example: 60Hz target
        #     mujoco.mj_step(model, data)

        # Get framebuffer size
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

        # Update scene and render using the provided options (local_opt)
        mujoco.mjv_updateScene(model, data, local_opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, context)

        # Swap buffers and poll events
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()
    print("  Viewer closed.")


# --- Main Execution Logic ---
def main():
    """Loads and displays objects for inspection."""

    # --- 1. Read Object IDs ---
    if not os.path.exists(NAMES_FILE_PATH):
        print(f"Error: Cannot find object list file at '{NAMES_FILE_PATH}'", file=sys.stderr)
        sys.exit(1)

    try:
        with open(NAMES_FILE_PATH, 'r') as f:
            object_ids = [line.strip() for line in f if line.strip()]
    except IOError as e:
        print(f"Error reading file '{NAMES_FILE_PATH}': {e}", file=sys.stderr)
        sys.exit(1)

    if not object_ids:
        print(f"Error: No object IDs found in '{NAMES_FILE_PATH}'.", file=sys.stderr)
        sys.exit(1)

    object_ids_to_inspect = object_ids[:NUM_OBJECTS_TO_INSPECT]
    num_to_show = len(object_ids_to_inspect)

    print(f"Found {len(object_ids)} objects in '{os.path.basename(NAMES_FILE_PATH)}'.")
    print(f"Attempting to inspect the first {num_to_show} objects.")
    print(f"Collision geometries (group {COLLISION_GEOM_GROUP}) will be shown.")
    print("-" * 40)

    # --- 2. Setup Gripper ---
    try:
        gripper = get_gripper(GRIPPER_CONFIG, default_pose=GRIPPER_DEFAULT_POSE)
    except Exception as e:
        print(f"Error initializing gripper '{GRIPPER_CONFIG.name}': {e}", file=sys.stderr)
        sys.exit(1)

    # --- 3. Loop and Display Objects ---
    for i, obj_id_str in enumerate(object_ids_to_inspect):
        print(f"[{i+1}/{num_to_show}] Displaying object: {obj_id_str}")
        env = None
        try:
            # Get the object instance
            obj = get_object(obj_id_str)

            # Create the simulation environment
            env = GravitylessObjectGrasping(gripper, obj)

            # ----> Create and Modify MjvOption <----
            options = mujoco.MjvOption()
            options.geomgroup[COLLISION_GEOM_GROUP] = 1 # Enable collision geoms
            # -----------------------------------------

            # Call the helper function to render passively
            render_passive_with_options(env.model, env.data, options, window_title=f"Object: {obj_id_str}")

        except FileNotFoundError as e:
            print(f"  Warning: Could not load assets for object '{obj_id_str}'. Skipping. Error: {e}", file=sys.stderr)
        except ValueError as e:
             print(f"  Warning: Could not identify or load '{obj_id_str}'. Skipping. Error: {e}", file=sys.stderr)
        except Exception as e:
            print(f"  Error displaying object '{obj_id_str}'. Skipping. Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        finally:
             # Clean up simulation resources if they were created
             if env:
                 # MuJoCo Python bindings might handle this, but explicit deletion can help
                 try:
                     if hasattr(env, 'data'): del env.data
                     if hasattr(env, 'model'): del env.model
                     del env
                 except Exception as cleanup_e:
                     print(f"  Note: Error during env cleanup: {cleanup_e}", file=sys.stderr)

        print("-" * 40)

    print("Finished inspecting objects.")

if __name__ == "__main__":
    if not os.path.isdir(ASSET_PATH):
         print(f"Critical Error: ASSET_PATH '{ASSET_PATH}' is not a valid directory.", file=sys.stderr)
         print("Object loading will fail. Please check your setup.", file=sys.stderr)
         sys.exit(1)
    else:
         print(f"Using ASSET_PATH: {ASSET_PATH}")

    main()