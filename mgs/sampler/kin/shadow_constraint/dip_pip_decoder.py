import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from mgs.env.gravityless_object_grasping import GravitylessObjectGrasping
from mgs.gripper.shadow_constraint import GripperShadowRightConstraint
from mgs.obj.marker import Marker
from mgs.util.geo.transforms import SE3Pose


class DiPPiPDecoder(nnx.Module):
    def __init__(self, hid_layers=[16] * 3, *, rngs):
        self.layers = []
        prev = 1
        for dim in hid_layers:
            self.layers.append(nnx.Linear(prev, dim, rngs=rngs))
            prev = dim

        self.layers.append(nnx.Linear(prev, 2, rngs=rngs))

    def __call__(self, ff_j0_acc: jnp.ndarray) -> jnp.ndarray:
        assert ff_j0_acc.shape[-1] == 1

        x = self.layers[0](ff_j0_acc)
        for l in self.layers[1:]:
            x = nnx.elu(x)
            x = l(x)
        return x


class AllDiPPiPDecoder(nnx.Module):
    def __init__(self, num_modules=4, *, rngs):
        self.num_modules = num_modules
        self.mods = []
        for _ in range(num_modules):
            self.mods.append(DiPPiPDecoder(rngs=rngs))

    def __call__(self, acc):
        assert acc.shape[-2] == self.num_modules and acc.shape[-1] == 1
        out = []
        for i, mod in enumerate(self.mods):
            out.append(mod(acc[..., i, :]))
        return jnp.stack(out, axis=-2)


# ids which are responsible for J1 / J2 joints
J0_ACC_ID = [7, 10, 13, 17]
J1J2_QPOS_ID = [2, 3, 6, 7, 10, 11, 15, 16]

BATCH = 64
NUM_ITERATIONS = 100 * 1000


@jax.jit
def step(graph, state, inputs, targets):
    model, opt = nnx.merge(graph, state)

    def loss_fn(model):
        preds = model(inputs)
        loss = jnp.mean((targets - preds) ** 2)
        return loss

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    opt.update(grads)

    return loss, nnx.state((model, opt))


def train():
    # Boilerplate setup
    gripper = GripperShadowRightConstraint(
        SE3Pose(
            pos=np.array([0.0, 0.0, 0.0]),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            type="wxyz",
        )
    )
    marker_pose = SE3Pose(
        pos=np.array([0, 0, 0]), quat=np.array([1, 0, 0, 0]), type="wxyz"
    )
    marker = Marker(pose=marker_pose, name="contact_marker")
    env = GravitylessObjectGrasping(gripper, marker)
    gripper_joint_idxs = env.get_joint_idxs(env.gripper.get_actuator_joint_names())

    # Networks
    rngs = nnx.Rngs(0)
    net = AllDiPPiPDecoder(rngs=rngs)
    tx = optax.adamw(learning_rate=0.001)
    optim = nnx.Optimizer(net, tx)
    graph, state = nnx.split((net, optim))

    for iter in range(NUM_ITERATIONS):
        j0_ids = np.array(J0_ACC_ID, dtype=np.int32)
        j1j2_ids = np.array(J1J2_QPOS_ID, dtype=np.int32)

        j0s = np.zeros(shape=(BATCH, 4, 1))
        j1j2s = np.zeros(shape=(BATCH, 4, 2))
        acc = np.zeros(shape=(BATCH, 18))
        q_pos = np.zeros(shape=(BATCH, 22))

        start = time.time()
        for i in range(BATCH):
            # collect data
            # sampling
            acc_for_j0 = np.random.uniform(low=0.0, high=np.pi / 2.0, size=(4,))
            acc[i, j0_ids] = acc_for_j0
            j0s[i, :, :] = acc_for_j0[..., None]
            q_pos[i, :] = env.acc_to_qpos(acc[i])[gripper_joint_idxs]
            j1j2s[i, :, :] = q_pos[i, j1j2_ids].reshape(4, 2)  # FF, MF, RF, LF
        end = time.time()
        print(f"Time per data collection: {end - start}")

        j0s = jnp.asarray(j0s)
        j1j2s = jnp.asarray(j1j2s)

        start = time.time()
        loss, state = step(graph, state, j0s, j1j2s)
        end = time.time()
        print(f"Time for update {end - start}")
        print(f"Iteration: {iter}, loss: {loss}")


if __name__ == "__main__":
    train()
