import jax
from typing import Callable, Tuple
import jax.numpy as jnp
from sklearn.utils import gen_batches
from jax.sharding import Mesh, PartitionSpec, NamedSharding


jax.config.update("jax_default_matmul_precision", "highest")


def vmap_batch(
    vmapped_fn: Callable[..., jax.Array], start: int, max_batch_size: int
) -> Callable[..., jax.Array]:
    def chunked_fn(*args):
        batch_len = len(args[start])
        batches = list(gen_batches(batch_len, max_batch_size))
        res = [
            vmapped_fn(*args[:start], *[arg[single_slice] for arg in args[start:]])
            for single_slice in batches
        ]
        if batch_len / max_batch_size % 1 != 0.0:
            diff = max_batch_size - len(res[-1])
            res[-1] = jnp.pad(
                res[-1], [(0, diff), *[(0, 0)] * (len(res[-1].shape) - 1)]
            )
            return jnp.concatenate(res)[:-diff]
        else:
            return jnp.concatenate(res)

    return chunked_fn


def mesh_sharding(pspec: PartitionSpec, mesh_grid: Mesh | None = None) -> NamedSharding:
    if mesh_grid is None:
        axis_shapes = best_axis_shapes(len(jax.local_devices()))
        axis_names = ("a", "b")
        mesh_grid = jax.make_mesh(axis_shapes, axis_names)
    return NamedSharding(mesh_grid, pspec)


def best_axis_shapes(num_devices: int) -> Tuple[int, int]:
    for i in range(int(jnp.sqrt(num_devices)), 0, -1):
        if num_devices % i == 0:
            return (i, num_devices // i)
    raise ValueError(f"No valid mesh split found for {num_devices}.")
