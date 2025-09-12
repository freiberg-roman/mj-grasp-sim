import os
import time

import hydra
import numpy as np
from omegaconf import DictConfig

from mgs.env.gravityless_object_grasping import GravitylessObjectGrasping
from mgs.gripper.panda import GripperPanda
from mgs.gripper.selector import get_gripper
from mgs.gripper.vx300 import GripperVX300
from mgs.obj.selector import get_object
from mgs.sampler.antipodal import AntipodalGraspGenerator
from mgs.sampler.contact import ContactBasedDiff
from mgs.sampler.kin.shadow import ShadowKinematicsModel
from mgs.util.const import ASSET_PATH
from mgs.util.file import generate_unique_hash  # uses secrets.token_hex under the hood
from mgs.util.geo.transforms import SE3Pose


def _count_grasps_in_dir(dirpath: str) -> int:
    """Robustly sum #poses across all .npz files in dir."""
    total = 0
    for fn in os.listdir(dirpath):
        if not fn.endswith(".npz"):
            continue
        fpath = os.path.join(dirpath, fn)
        try:
            with np.load(fpath) as z:
                total += int(len(z["poses"]))
        except Exception:
            # ignore any unreadable/partial files (shouldn't happen with atomic rename)
            continue
    return total


def _atomic_save_npz(final_path: str, **arrays):
    """
    Atomic write: save to a temp file in the same directory, then os.replace.
    This avoids readers ever seeing a partially-written file.
    """
    d = os.path.dirname(final_path)
    tmp_name = f".tmp-{os.getpid()}-{generate_unique_hash(8)}.npz"
    tmp_path = os.path.join(d, tmp_name)
    # ensure same filesystem/dir for atomic replace
    np.savez(tmp_path, **arrays)
    os.replace(tmp_path, final_path)  # atomic on POSIX when same filesystem


@hydra.main(
    version_base="1.3.2", config_path="config", config_name="gen_gripper_object_grasps"
)
def main(cfg: DictConfig):
    # --- select object ---
    object_id_file = os.path.join(ASSET_PATH, "mj-objects", "fast_eta_objects.txt")
    with open(object_id_file, "r") as file:
        all_object_ids = file.read().splitlines()
    object_id = all_object_ids[int(cfg.object_id)]

    # --- components ---
    obj = get_object(object_id)
    assert cfg.gripper.grasp_sampler in [
        "Antipodal",
        "ContactGradient",
    ], "Unsupported sampler"
    sampler = (
        AntipodalGraspGenerator(obj)
        if cfg.gripper.grasp_sampler == "Antipodal"
        else ContactBasedDiff(obj)
    )
    gripper = get_gripper(cfg.gripper)

    print(
        f"Generating grasp candidates using gripper: {cfg.gripper.name}"
        f"\nfor object {object_id} using {cfg.gripper.grasp_sampler} sampler"
    )

    # --- output dir ---
    output_root = os.getenv("MGS_OUTPUT_DIR") or os.path.join(os.getcwd(), "outputs")
    output_dir = os.path.join(output_root, cfg.gripper.name, object_id)
    os.makedirs(output_dir, exist_ok=True)

    # how many are already saved (from all processes)?
    num_total_grasps = _count_grasps_in_dir(output_dir)
    grasps_to_generate = cfg.target_grasps - num_total_grasps
    if grasps_to_generate <= 0:
        print("Nothing to do — target already met.")
        return

    print(f"Grasps to generate (target minus existing): {grasps_to_generate}")

    env = GravitylessObjectGrasping(gripper, obj)

    # rolling buffers
    buf_poses, buf_joints = [], []

    # per-process stats
    total_attempts = 0
    total_stable = 0
    attempts_since_last_save = 0
    stable_since_last_save = 0

    def maybe_flush(final_flush: bool = False):
        """
        Save a chunk with a random-hash filename. After saving, re-scan the dir to
        see if the global target is already met by any process.
        """
        nonlocal buf_poses, buf_joints
        nonlocal attempts_since_last_save, stable_since_last_save
        nonlocal num_total_grasps, total_stable

        if not buf_poses:
            return False

        poses_all = np.concatenate(buf_poses, axis=0)
        joints_all = np.concatenate(buf_joints, axis=0)

        # decide how many to save now
        if final_flush:
            want = len(poses_all)  # don't waste any grasps on exit
        else:
            want = min(
                int(cfg.collect_grasps_till_save),
                len(poses_all),
            )

        if want <= 0:
            return False

        save_poses = poses_all[:want]
        save_joints = joints_all[:want]
        leftovers_poses = poses_all[want:]
        leftovers_joints = joints_all[want:]
        buf_poses = [leftovers_poses] if len(leftovers_poses) else []
        buf_joints = [leftovers_joints] if len(leftovers_joints) else []

        # report success rates (per-process)
        file_sr = (
            float(stable_since_last_save) / float(attempts_since_last_save)
            if attempts_since_last_save > 0
            else float("nan")
        )
        cum_sr = (
            (float(total_stable) / float(total_attempts))
            if total_attempts > 0
            else float("nan")
        )
        # random-hash filename to avoid collisions between processes
        fname = f"{int(time.time())}-{os.getpid()}-{generate_unique_hash(8)}.npz"
        fpath = os.path.join(output_dir, fname)
        _atomic_save_npz(
            fpath,
            poses=save_poses,
            joints=save_joints,
            success_rate_during_gen=np.asarray(file_sr),
        )
        print(
            f"[SAVE] {fname} → {len(save_poses)} grasps | "
            f"since-last-save SR={file_sr*100:.1f}% | cumulative SR={cum_sr*100:.1f}%"
        )
        attempts_since_last_save = 0
        stable_since_last_save = 0

        # re-scan the dir (all processes) to decide whether to continue
        num_total_grasps = _count_grasps_in_dir(output_dir)
        if num_total_grasps >= cfg.target_grasps:
            # we still keep any local leftovers by doing a final flush on exit
            return True  # signal: global target reached
        return False

    round_idx = 0
    while round_idx < int(cfg.max_rounds):
        # check global progress before starting a new round
        num_total_grasps = _count_grasps_in_dir(output_dir)
        if num_total_grasps >= cfg.target_grasps:
            break

        round_idx += 1

        # ---- collect collision-free grasps until eval threshold ----
        collected_poses, collected_joints = [], []

        while sum(len(p) for p in collected_poses) < int(cfg.collect_grasps_till_eval):

            if cfg.gripper.grasp_sampler == "Antipodal":
                poses_mat, aux_info = sampler.generate_grasps(
                    num=int(cfg.sample_grasps)  # type: ignore
                )
                padding = 0.01
                joints = gripper.width_to_joints(aux_info["width"] + padding)
            elif cfg.gripper.grasp_sampler == "ContactGradient":
                all_kins = {
                    "ShadowHand": ShadowKinematicsModel(),
                    "Allegro": None,
                }
                kin_model = all_kins[cfg.gripper.name]
                poses_mat, aux_info = sampler.generate_grasps(  # type: ignore
                    num=int(cfg.sample_grasps), gripper=kin_model  # type: ignore
                )
                joints = aux_info["joints"]
            else:
                raise ValueError("Not known grasp sampler")
            if len(poses_mat) == 0:
                continue

            poses_se3 = SE3Pose.from_mat(poses_mat)
            collision_mask = env.grasp_collision_mask(
                poses_se3, joints, with_padding=0.002
            )
            if np.any(collision_mask):
                collected_poses.append(poses_se3.to_mat()[collision_mask])
                collected_joints.append(joints[collision_mask])

        if not collected_poses:
            continue

        cf_poses_mat = np.concatenate(collected_poses, axis=0)
        cf_joints = np.concatenate(collected_joints, axis=0)
        cf_poses = SE3Pose.from_mat(cf_poses_mat)

        # ---- stability evaluation ----
        stable_mask = env.grasp_stability_evaluation_from_joints(
            cf_poses, cf_joints, impulse_force=float(cfg.force)
        )
        attempts = len(cf_poses)
        stables = int(np.count_nonzero(stable_mask))

        total_attempts += attempts
        attempts_since_last_save += attempts
        total_stable += stables
        stable_since_last_save += stables

        if stables:
            buf_poses.append(cf_poses.to_mat()[stable_mask])
            buf_joints.append(cf_joints[stable_mask])

        # save if chunk ready
        if sum(len(p) for p in buf_poses) >= int(cfg.collect_grasps_till_save):
            stop = maybe_flush(final_flush=False)
            if stop:
                break

    # final flush of leftovers so no grasps are wasted
    if sum(len(p) for p in buf_poses):
        _ = maybe_flush(final_flush=True)

    print("Done!")


if __name__ == "__main__":
    main()
