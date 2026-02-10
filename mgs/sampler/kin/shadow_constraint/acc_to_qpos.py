from dataclasses import dataclass

import jax
import jax.numpy as jnp

from mgs.sampler.kin.shadow_constraint.ff_surrogate import ff_acc_to_qpos
from mgs.sampler.kin.shadow_constraint.ff_surrogate import load_surrogate_npz as load_ff
from mgs.sampler.kin.shadow_constraint.lf_surrogate import lf_acc_to_qpos
from mgs.sampler.kin.shadow_constraint.lf_surrogate import load_surrogate_npz as load_lf
from mgs.sampler.kin.shadow_constraint.mf_surrogate import load_surrogate_npz as load_mf
from mgs.sampler.kin.shadow_constraint.mf_surrogate import mf_acc_to_qpos
from mgs.sampler.kin.shadow_constraint.rf_surrogate import load_surrogate_npz as load_rf
from mgs.sampler.kin.shadow_constraint.rf_surrogate import rf_acc_to_qpos


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class acc_to_qpos:
    ff: ff_acc_to_qpos
    mf: mf_acc_to_qpos
    rf: rf_acc_to_qpos
    lf: lf_acc_to_qpos

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        # Expected 18 dim acc. input
        # TH 5
        # TH 4
        # TH 3
        # TH 2
        # TH 1
        # FF 4
        # FF 3
        # FF 0
        # MF 4
        # MF 3
        # MF 0
        # RF 4
        # RF 3
        # RF 0
        # LF 5
        # LF 4
        # LF 3
        # LF 0

        # Converted to 22 joint dims
        # FF 4
        # FF 3
        # FF 2
        # FF 1
        # MF 4
        # MF 3
        # MF 2
        # MF 1
        # RF 4
        # RF 3
        # RF 2
        # RF 1
        # LF 5
        # LF 4
        # LF 3
        # LF 2
        # LF 1
        # TH 5
        # TH 4
        # TH 3
        # TH 2
        # TH 1
        """
        x = jnp.asarray(x)
        x_ff, x_mf, x_rf, x_lf = x[..., 7], x[..., 10], x[..., 13], x[..., 17]
        y_ff = self.ff(x_ff)  # (...,2)
        y_mf = self.mf(x_mf)
        y_rf = self.rf(x_rf)
        y_lf = self.lf(x_lf)
        return jnp.concatenate(
            [
                x[..., 5:7],
                y_ff,
                x[..., 8:10],
                y_mf,
                x[..., 11:13],
                y_rf,
                x[..., 14:17],
                y_lf,
                x[..., 0:5],
            ],
            axis=-1,
        )

    def tree_flatten(self):
        return (self.ff, self.mf, self.rf, self.lf), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        ff, mf, rf, lf = children
        return cls(ff=ff, mf=mf, rf=rf, lf=lf)


def load_shadow_acc_to_qpos() -> acc_to_qpos:
    ff = load_ff()
    mf = load_mf()
    rf = load_rf()
    lf = load_lf()
    return acc_to_qpos(ff=ff, mf=mf, rf=rf, lf=lf)


if __name__ == "__main__":
    f = load_shadow_acc_to_qpos()
    x0 = jnp.array([[1.0] * 18])
    y0 = f(x0)  # (1,22,)
    J = jax.jacrev(f)(x0)  # (1 ,22, 1, 18)
    print(y0, J)
