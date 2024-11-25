import jax
from typing import Callable
import jax.numpy as jnp
from sklearn.utils import gen_batches


def vmap_batch(
    vmapped_fn: Callable[..., jax.Array], start: int, max_vmap: int
) -> Callable[..., jax.Array]:
    def chunked_fn(*args):
        batch_len = len(args[start])
        batches = list(gen_batches(batch_len, max_vmap))
        res = [
            vmapped_fn(*args[:start], *[arg[single_slice] for arg in args[start:]])
            for single_slice in batches
        ]
        # jnp.concatenate needs to act on arrays with the same shape, so pad the last array if necessary
        if batch_len / max_vmap % 1 != 0.0:
            diff = max_vmap - len(res[-1])
            res[-1] = jnp.pad(
                res[-1], [(0, diff), *[(0, 0)] * (len(res[-1].shape) - 1)]
            )
            return jnp.concatenate(res)[:-diff]
        else:
            return jnp.concatenate(res)

    return chunked_fn