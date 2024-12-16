import jax
from typing import Callable, Tuple
import jax.numpy as jnp
from sklearn.utils import gen_batches
from jax.sharding import Mesh, PartitionSpec, NamedSharding

"""
When running on GPU/TPU this line is needed, such that gram matrix
stays symmetric and has ones across diagonal.
Otherwise the effect of 
finite precision is so worse that the tests crash and 
code doesn't work on GPU/TPU.
"""
jax.config.update("jax_default_matmul_precision", "highest")


def vmap_batch(
    vmapped_fn: Callable[..., jax.Array], start: int, max_batch_size: int
) -> Callable[..., jax.Array]:
    """
    Wraps a "vmapped" function (see https://jax.readthedocs.io/en/latest/automatic-vectorization.html)
    to process inputs in chunks of a specified maximum batch size.

    This utility function is designed to handle large batched computations by splitting the input
    into smaller chunks, applying the vmapped function on each chunk, and then concatenating
    the results. If the input batch size is not evenly divisible (python integer
    division should have no rest then) by the maximum batch size,
    the final batch is padded such that it works.

    Parameters
    ----------
    vmapped_fn : Callable[..., jax.Array]
        The function to be applied to each batch. It must be compatible with JAX's `vmap`
        (https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html#jax.vmap)
        mechanism and accept the appropriate array shapes.
    start : int
        The index in `args` where the batched inputs start. Arguments before this index
        are assumed to be non-batched, while arguments at and after this index are
        batched inputs to be chunked.
    max_batch_size : int
        The maximum size of each batch. Input data is split into chunks of this size
        before processing.

    Returns
    -------
    Callable[..., jax.Array]
        A wrapped version of `vmapped_fn` that processes inputs in chunks, handles padding
        if necessary, and returns the concatenated results.

    Notes
    -----
    - This function assumes that all batched inputs have the same leading dimension
      (batch size).
    - Padding is only applied to the final batch if its size is smaller than `max_batch_size`.
      The padded entries are excluded from the concatenated results.

    Examples given:
    --------
    >>> import jax.numpy as jnp
    >>> import jax
    >>> def fn(x):
    ...     return x ** 2
    >>> vmapped_fn = jax.vmap(fn)
    >>> chunked_fn = vmap_batch(vmapped_fn, start=0, max_batch_size=3)
    >>> inputs = jnp.arange(10)
    >>> chunked_fn(inputs)
    DeviceArray([ 0,  1,  4,  9, 16, 25, 36, 49, 64, 81], dtype=int32)
    """

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
    """
    Creates a mesh grid for GPU/TPU-sharding with multiple
    (GPU/TPU-)devices.

    Parameters
    ----------
    pspec : PartitionSpec
        Tuple describing how to partition an array across a mesh of devices.
    mesh_grid : Mesh | None, optional
        Tuple (n_devices, 1), where n_devices means the number of
        devices available, by default None

    Returns
    -------
    NamedSharding
        From JAX library:
        >>> NamedSharding is a pair of a Mesh of devices and
        PartitionSpec which describes how to shard an array across that
        mesh.
    """
    if mesh_grid is None:
        axis_shapes = best_axis_shapes(len(jax.local_devices()))
        axis_names = ("a", "b")
        mesh_grid = jax.make_mesh(axis_shapes, axis_names)
    return NamedSharding(mesh_grid, pspec)


def best_axis_shapes(num_devices: int) -> Tuple[int, int]:
    """
    I didn't want to hardcode the number of devices when
    I run on let's say multiple GPUs/TPUs and then later switch
    to testing on my single CPU. So this method will individually
    switch this by checking which device I'm on and creating the
    correct mesh for computations.

    Parameters
    ----------
    num_devices : int
        The number of computational devices available.

    Returns
    -------
    Tuple[int, int]
        A tuple of (n_devices, 1) where n_devices specifies the
        number of computational devices available.

    Raises
    ------
    ValueError
        When there's an invalid split.
    """
    for i in range(int(jnp.sqrt(num_devices)), 0, -1):
        if num_devices % i == 0:
            return (i, num_devices // i)
    raise ValueError(f"No valid mesh split found for {num_devices}.")
