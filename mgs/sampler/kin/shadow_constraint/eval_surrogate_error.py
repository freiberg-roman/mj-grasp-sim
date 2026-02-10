"""
Evaluate the maximum absolute error of each finger's piecewise-linear
surrogate against the MuJoCo ground truth (steady-state qpos).

For every finger we:
  1. Build the isolated single-finger MuJoCo model.
  2. Sweep the actuation domain [0, pi] on a dense grid (NUM_EVAL points).
  3. Record the GT qpos (J2, J1) from MuJoCo.
  4. Evaluate the surrogate on the same grid.
  5. Report max / mean absolute error per joint per finger.
"""

import numpy as np
import tqdm

from mgs.env.gravityless_object_grasping import GravitylessObjectGrasping
from mgs.gripper.shadow_segments.shadow_ff import ShadowRightFF
from mgs.gripper.shadow_segments.shadow_lf import ShadowRightLF
from mgs.gripper.shadow_segments.shadow_mf import ShadowRightMF
from mgs.gripper.shadow_segments.shadow_rf import ShadowRightRF
from mgs.obj.marker import Marker
from mgs.sampler.kin.shadow_constraint.ff_surrogate import (
    load_surrogate_npz as load_ff,
)
from mgs.sampler.kin.shadow_constraint.lf_surrogate import (
    load_surrogate_npz as load_lf,
)
from mgs.sampler.kin.shadow_constraint.mf_surrogate import (
    load_surrogate_npz as load_mf,
)
from mgs.sampler.kin.shadow_constraint.rf_surrogate import (
    load_surrogate_npz as load_rf,
)
from mgs.util.geo.transforms import SE3Pose

NUM_EVAL = 2048


def _get_gt_qpos(gripper_cls, num_eval: int):
    """Sweep actuation [0, pi] and return (act_grid, gt_qpos) from MuJoCo."""
    marker_pose = SE3Pose(
        pos=np.array([0, 0, 0]), quat=np.array([1, 0, 0, 0]), type="wxyz"
    )
    marker = Marker(pose=marker_pose, name="contact_marker")
    gripper = gripper_cls(marker_pose)
    env = GravitylessObjectGrasping(gripper, marker)
    joint_idxs = env.get_joint_idxs(env.gripper.get_actuator_joint_names())

    act_grid = np.linspace(0, np.pi, num_eval, endpoint=True)
    gt = np.empty((num_eval, len(joint_idxs)))
    for i in tqdm.tqdm(range(num_eval), desc=gripper_cls.__name__, leave=False):
        gt[i, :] = env.acc_to_qpos(act_grid[i : i + 1])[joint_idxs]
    return act_grid, gt


def _eval_surrogate(surrogate, act_grid):
    """Evaluate surrogate on the grid, return (num_eval, 2) array."""
    out = np.array(surrogate(np.asarray(act_grid)))
    return out


def main():
    fingers = [
        ("FF", ShadowRightFF, load_ff),
        ("MF", ShadowRightMF, load_mf),
        ("RF", ShadowRightRF, load_rf),
        ("LF", ShadowRightLF, load_lf),
    ]

    print(f"Evaluating surrogate error on {NUM_EVAL} points per finger\n")
    print(
        f"{'Finger':<6} {'Joint':<5} {'Max |err| (rad)':<18} {'Mean |err| (rad)':<18}"
    )
    print("-" * 50)

    for name, gripper_cls, load_fn in fingers:
        act_grid, gt = _get_gt_qpos(gripper_cls, NUM_EVAL)
        surrogate = load_fn()
        pred = _eval_surrogate(surrogate, act_grid)

        for j, joint_label in enumerate(["J2", "J1"]):
            err = np.abs(gt[:, j] - pred[:, j])
            print(f"{name:<6} {joint_label:<5} {err.max():<18.6f} {err.mean():<18.6f}")

    print()


if __name__ == "__main__":
    main()
