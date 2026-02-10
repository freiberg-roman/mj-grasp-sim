from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class rf_acc_to_qpos:
    xp: jax.Array  # (N,)
    fp: jax.Array  # (N, 2)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        x: scalar or any shape (...,)
        returns: (..., 2)
        """
        # vmap over joint axis of fp (axis=1): each slice is (N,)
        x = jnp.clip(x, 0.0, 3.1415)
        return jax.vmap(
            lambda fp1: jnp.interp(x, self.xp, fp1), in_axes=1, out_axes=-1
        )(self.fp)

    # PyTree plumbing (so jit/grad sees xp/fp as leaves)
    def tree_flatten(self):
        return (self.xp, self.fp), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        xp, fp = children
        return cls(xp=xp, fp=fp)


def load_surrogate_npz() -> rf_acc_to_qpos:
    import os

    path = os.path.join(os.path.dirname(__file__), "rf.npz")
    d = np.load(path)
    xp = jnp.asarray(d["xp"], dtype=jnp.float32)
    fp = jnp.asarray(d["fp"], dtype=jnp.float32)
    return rf_acc_to_qpos(xp=xp, fp=fp)


if __name__ == "__main__":
    f = load_surrogate_npz()
    J = jax.jacrev(f)(jnp.array(1.0))  # shape (2,) if x is scalar
    print(J)
