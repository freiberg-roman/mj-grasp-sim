import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from mgs.env.gravityless_object_grasping import GravitylessObjectGrasping
from mgs.gripper.shadow_segments.shadow_ff import ShadowRightFF
from mgs.gripper.shadow_segments.shadow_lf import ShadowRightLF
from mgs.gripper.shadow_segments.shadow_mf import ShadowRightMF
from mgs.gripper.shadow_segments.shadow_rf import ShadowRightRF
from mgs.obj.marker import Marker
from mgs.util.geo.transforms import SE3Pose

SAMPLES = 256
JOINT = "lf"  # others ff, mf, rf, lf


def save_surrogate_npz(path: str, xp: np.ndarray, fp: np.ndarray):
    """
    xp: (N,) float
    fp: (N, D) float  (D = number of joints)
    """
    xp = np.asarray(xp)
    fp = np.asarray(fp)
    assert xp.ndim == 1
    assert fp.ndim == 2 and fp.shape[0] == xp.shape[0]
    np.savez_compressed(path, xp=xp, fp=fp)


def fit():
    # Boilerplate setup
    marker_pose = SE3Pose(
        pos=np.array([0, 0, 0]), quat=np.array([1, 0, 0, 0]), type="wxyz"
    )
    marker = Marker(pose=marker_pose, name="contact_marker")
    if JOINT == "ff":
        gripper = ShadowRightFF(marker_pose)
    elif JOINT == "mf":
        gripper = ShadowRightMF(marker_pose)
    elif JOINT == "rf":
        gripper = ShadowRightRF(marker_pose)
    elif JOINT == "lf":
        gripper = ShadowRightLF(marker_pose)

    env = GravitylessObjectGrasping(gripper, marker)
    gripper_joint_idxs = env.get_joint_idxs(env.gripper.get_actuator_joint_names())

    domain_space = np.linspace(
        0,
        3.1415,
        SAMPLES + 1,
        endpoint=True,
    )
    q_pos = np.empty(shape=(SAMPLES + 1, len(gripper_joint_idxs)))
    for i in tqdm.tqdm(range(SAMPLES + 1)):
        x = domain_space[i][None]
        q_pos[i, :] = env.acc_to_qpos(x)[gripper_joint_idxs]

    # store function model
    save_surrogate_npz(JOINT, domain_space, q_pos)

    domain_space_j = jnp.asarray(domain_space)
    q_pos_j = jnp.asarray(q_pos)
    interpolations = []
    for i in range(len(gripper_joint_idxs)):
        interpolations.append(
            lambda x, i=i: jnp.interp(x, domain_space_j, q_pos_j[:, i])
        )

    # Dense evaluation grid for visualization (much denser than knots)
    x_dense = jnp.linspace(domain_space_j[0], domain_space_j[-1], 20000)

    n_joints = len(gripper_joint_idxs)

    # 1) Plot sampled points + interpolated curve for each joint
    for i in range(n_joints):
        y_dense = interpolations[i](x_dense)

        # bring to numpy for matplotlib
        xk = np.asarray(domain_space)
        yk = np.asarray(q_pos[:, i])
        xd = np.asarray(x_dense)
        yd = np.asarray(y_dense)

        plt.figure()
        plt.plot(xd, yd)  # interpolated curve
        plt.plot(xk, yk, marker=".", linestyle="")  # sampled knots
        plt.xlabel("actuation (rad)")
        plt.ylabel(f"qpos[joint_idx={gripper_joint_idxs[i]}]")
        plt.title(f"Joint {i}: knots vs jnp.interp")
        plt.grid(True)

    # 2) Residual on knots (should be ~0, useful for catching dtype / ordering issues)
    max_abs_residual = []
    for i in range(n_joints):
        y_on_knots = interpolations[i](domain_space_j)
        res = y_on_knots - q_pos_j[:, i]
        max_abs_residual.append(float(jnp.max(jnp.abs(res))))

    print("Max |interp(x_k) - y_k| per joint:", max_abs_residual)

    x_dense = jnp.linspace(domain_space_j[0], domain_space_j[-1], 20000)

    for i in range(n_joints):
        # scalar-valued function for grad
        def f_scalar(x, i=i):
            # x is scalar
            return jnp.interp(x, domain_space_j, q_pos_j[:, i])

        df_dx = jax.grad(f_scalar)

        # Evaluate derivative on dense grid (vectorize)
        dydx_dense = jax.vmap(df_dx)(x_dense)

        # Convert to numpy for plotting
        xd = np.asarray(x_dense)
        dydx = np.asarray(dydx_dense)

        plt.figure()
        plt.plot(xd, dydx)
        plt.xlabel("actuation (rad)")
        plt.ylabel(f"d qpos / d act (joint {i})")
        plt.title(f"Joint {i}: gradient via jax.grad (PWL -> step function)")
        plt.grid(True)

        # Optional: overlay knot locations as vertical markers (lightweight)
        # This helps confirm whether the "kink" aligns with a knot or is a broader region.
        # Comment out if too busy.
        for xk in domain_space[:: max(1, SAMPLES // 16)]:  # thin markers
            plt.axvline(xk, linewidth=0.3, alpha=0.2)

    plt.show()


if __name__ == "__main__":
    fit()
