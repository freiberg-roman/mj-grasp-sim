import os

import hydra
import numpy as np
from omegaconf import DictConfig

from mgs.env.gravityless_object_grasping import GravitylessObjectGrasping
from mgs.gripper.panda import GripperPanda
from mgs.obj.selector import get_object
from mgs.sampler.antipodal import AntipodalGraspGenerator
from mgs.util.const import ASSET_PATH
from mgs.util.geo.transforms import SE3Pose


@hydra.main(
    version_base="1.3.2", config_path="config", config_name="gen_gripper_object_grasps"
)
def main(cfg: DictConfig):
    # --- load object id list ---
    object_id_file = os.path.join(ASSET_PATH, "mj-objects", "fast_eta_objects.txt")
    with open(object_id_file, "r") as file:
        all_object_ids = file.read().splitlines()
    object_id = all_object_ids[int(cfg.object_id)]

    # --- components ---
    obj = get_object(object_id)
    sampler = (
        AntipodalGraspGenerator(obj)
        if cfg.gripper.grasp_sampler == "Antipodal"
        else None
    )
    assert sampler is not None, "Unsupported sampler"
    gripper = GripperPanda(
        SE3Pose.from_vec(
            np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), type="wxyz", layout="pq"
        )
    )

    print(
        f"Generating grasp candidates using gripper: {cfg.gripper.name}"
        f"\nfor object {object_id} using {cfg.gripper.grasp_sampler} sampler"
    )

    # --- output dir & existing files ---
    output_root = os.getenv("MGS_OUTPUT_DIR") or os.path.join(os.getcwd(), "outputs")
    output_dir = os.path.join(output_root, cfg.gripper.name, object_id)
    os.makedirs(output_dir, exist_ok=True)

    existing = sorted(
        [f for f in os.listdir(output_dir) if f.endswith(".npz") and f[:-4].isdigit()]
    )
    next_file_index = len(existing)

    # how many are already saved?
    num_total_grasps = 0
    for f in existing:
        p = np.load(os.path.join(output_dir, f))
        try:
            num_total_grasps += len(p["poses"])
        finally:
            p.close()

    grasps_to_generate = cfg.target_grasps - num_total_grasps
    if grasps_to_generate <= 0:
        print("Nothing to do — target already met.")
        return

    print(f"Grasps to generate (target minus existing): {grasps_to_generate}")

    env = GravitylessObjectGrasping(gripper, obj)

    # rolling buffers (to be saved in chunks)
    buf_poses = []
    buf_joints = []

    # stats
    total_attempts = 0
    total_stable = 0
    attempts_since_last_save = 0
    stable_since_last_save = 0

    round_idx = 0
    while (total_stable < grasps_to_generate) and (round_idx < cfg.max_rounds):
        round_idx += 1

        # ---- collect collision-free grasps until we hit the eval threshold ----
        collected_poses = []
        collected_joints = []

        while sum(len(p) for p in collected_poses) < int(cfg.collect_grasps_till_eval):
            # sample
            poses_mat, aux_info = sampler.generate_grasps(num=int(cfg.sample_grasps))
            if len(poses_mat) == 0:
                continue

            # joints from width (+ small padding)
            padding = 0.01
            joints = gripper.width_to_joints(aux_info["width"] + padding)

            # collision filter (with local pose padding for robustness)
            poses_se3 = SE3Pose.from_mat(poses_mat)
            collision_mask = env.grasp_collision_mask(
                poses_se3, joints, with_padding=0.002
            )
            if np.any(collision_mask):
                collected_poses.append(poses_se3.to_mat()[collision_mask])
                collected_joints.append(joints[collision_mask])

            # keep looping until threshold reached

        if not collected_poses:
            continue

        # concatenate for stability evaluation
        cf_poses_mat = np.concatenate(collected_poses, axis=0)
        cf_joints = np.concatenate(collected_joints, axis=0)
        cf_poses = SE3Pose.from_mat(cf_poses_mat)

        # ---- stability evaluation (impulse-only, as implemented in your env) ----
        stable_mask = env.grasp_stability_evaluation_from_joints(cf_poses, cf_joints)
        attempts = len(cf_poses)
        stables = int(np.count_nonzero(stable_mask))

        # update stats
        total_attempts += attempts
        attempts_since_last_save += attempts
        total_stable += stables
        stable_since_last_save += stables

        # push stable into buffers
        if stables:
            buf_poses.append(cf_poses.to_mat()[stable_mask])
            buf_joints.append(cf_joints[stable_mask])

        # ---- flush to disk if we have enough for a file, or if we’re near the target ----
        def _flush_one_file(final_flush: bool = False):
            nonlocal buf_poses, buf_joints, next_file_index
            nonlocal attempts_since_last_save, stable_since_last_save, total_stable

            if not buf_poses:
                return False

            # flatten buffers
            poses_all = np.concatenate(buf_poses, axis=0)
            joints_all = np.concatenate(buf_joints, axis=0)

            remaining_needed = grasps_to_generate - (
                num_total_grasps + total_stable - len(poses_all)
            )
            # choose how many to save this time
            want = int(cfg.collect_grasps_till_save)
            if final_flush:
                want = max(
                    1, min(len(poses_all), grasps_to_generate - num_total_grasps)
                )
            else:
                want = min(want, len(poses_all), grasps_to_generate - num_total_grasps)

            if want <= 0:
                return False

            save_poses = poses_all[:want]
            save_joints = joints_all[:want]

            # rewrite buffers with leftovers
            leftover_poses = poses_all[want:]
            leftover_joints = joints_all[want:]
            buf_poses = [leftover_poses] if len(leftover_poses) else []
            buf_joints = [leftover_joints] if len(leftover_joints) else []

            filename = os.path.join(output_dir, f"{next_file_index:04d}.npz")
            np.savez(
                filename, poses=save_poses, joints=save_joints
            )  # saves named arrays. :contentReference[oaicite:1]{index=1}
            next_file_index += 1

            # per-file success rate = stable_since_last_save / attempts_since_last_save
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

            print(
                f"[SAVE] {os.path.basename(filename)} → {len(save_poses)} grasps | "
                f"since-last-save SR={file_sr*100:.1f}% "
                f"| cumulative SR={cum_sr*100:.1f}% "
                f"| total_saved={num_total_grasps + (total_stable - len(poses_all)) + len(save_poses)}"
            )

            # reset “since last save” counters
            attempts_since_last_save = 0
            stable_since_last_save = 0
            return True

        # try to save one file this round if we’ve reached the chunk size or nearing target
        if sum(len(p) for p in buf_poses) >= int(cfg.collect_grasps_till_save):
            _flush_one_file(final_flush=False)
        elif (
            num_total_grasps + total_stable + sum(len(p) for p in buf_poses)
        ) >= cfg.target_grasps:
            _flush_one_file(final_flush=True)

    # final flush for leftovers (if any)
    if sum(len(p) for p in buf_poses):
        # we’re done generating; save what’s left (even if < chunk size)
        # compute a last per-file SR from what has accumulated since last save
        def _final_attempts_sr():
            return (
                float(stable_since_last_save) / float(attempts_since_last_save)
                if attempts_since_last_save > 0
                else float("nan")
            )

        poses_all = np.concatenate(buf_poses, axis=0)
        joints_all = np.concatenate(buf_joints, axis=0)
        want = min(len(poses_all), cfg.target_grasps - num_total_grasps)
        filename = os.path.join(output_dir, f"{next_file_index:04d}.npz")
        np.savez(
            filename, poses=poses_all[:want], joints=joints_all[:want]
        )  # .npz bundle. :contentReference[oaicite:2]{index=2}
        next_file_index += 1
        cum_sr = (
            (float(total_stable) / float(total_attempts))
            if total_attempts > 0
            else float("nan")
        )
        print(
            f"[SAVE] {os.path.basename(filename)} → {want} grasps | "
            f"since-last-save SR={_final_attempts_sr()*100:.1f}% | "
            f"cumulative SR={cum_sr*100:.1f}%"
        )

    print("Done!")


if __name__ == "__main__":
    main()
