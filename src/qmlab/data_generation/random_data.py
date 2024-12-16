from typing import Any, Callable, List, Tuple
from functools import reduce
import itertools as it

import jax
import jax.numpy as jnp
import numpy as np
from qiskit_algorithms.utils import algorithm_globals
from numpy.typing import NDArray

# once again this is needed on GPU/TPU
jax.config.update("jax_default_matmul_precision", "highest")


def generate_random_data(
    feature_dimension: int,
    training_examples_per_class: int,
    test_examples_per_class: int,
    delta: float = 0.3,
    random_state: int = 42,
    interval: Tuple[float, float] = (0.0, 2 * np.pi),
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Generate random dataset for binary classification based
    on the paper of Havlicek et al. (https://arxiv.org/pdf/1804.11326).

    This function creates artificial data for training and testing based on a
    quantum-inspired feature map and random unitary transformations. The generated
    dataset includes features and labels for two classes, `+1` and `-1`.

    This data is known (see source) to be hard to classify classically
    and a quantum-based model can achieve a higher accuracy.

    Parameters
    ----------
    feature_dimension : int
        The number of features for each sample. Determines the dimensionality
        of the data embedding.
    training_examples_per_class : int
        The number of training examples to generate per class.
    test_examples_per_class : int
        The number of test examples to generate per class.
    delta : float, optional
        Controls the decision boundary for the dataset (gap). Default is 0.3.
        In the paper they denote it with $\Delta$.
    random_state : int, optional
        Seed for the random number generator to ensure reproducibility. Default is 42.
    interval : Tuple[float, float], optional
        Range of values for the feature grid. Default is $[0.0, 2 \cdot \pi)$ just
        as Havlicek et al. propose.

    Returns
    -------
    Tuple[NDArray, NDArray, NDArray, NDArray]
        A python tuple containing:
        - X_train : ndarray
            Training feature data of shape (2 * training_examples_per_class, feature_dimension).
        - y_train : ndarray
            Training labels of shape (2 * training_examples_per_class,).
        - X_test : ndarray
            Testing feature data of shape (2 * test_examples_per_class, feature_dimension).
        - y_test : ndarray
            Testing labels of shape (2 * test_examples_per_class,)
            (it's always 2 since we have binary data [-1, +1])

    Raises
    ------
    ValueError
        If `interval` is not a tuple of two floating-point values.

    Notes
    -----
    This function uses a quantum-inspired feature map derived from a tensor-product
    structure with rotation gates. The feature map is parameterized by `feature_dimension`,
    and the classification problem involves evaluating the expectation values of
    a random Hermitian operator after applying a sequence of unitary transformations.

    Examples
    --------
    Generate a dataset with a 2-dimensional feature space and 50 training and
    20 test examples per class (fix the seed/random state for reproducable results):

    >>> X_train, y_train, X_test, y_test = generate_random_data(
    ...     feature_dimension=2,
    ...     training_examples_per_class=50,
    ...     test_examples_per_class=20,
    ...     delta=0.5,
    ...     random_state=42,
    ... )
    >>> X_train.shape
    (100, 2)
    >>> y_train
    NDArray([1, 1, -1, ..., -1, 1])
    """
    key = jax.random.key(seed=random_state)
    algorithm_globals.random_seed = random_state
    class_labels = [r"+1", r"-1"]

    if not isinstance(interval, tuple) or len(interval) != 2:
        raise ValueError(
            "Parameter `interval` must be a tuple containing two floating point values!"
        )

    num_points = 100 if feature_dimension == 2 else 20
    xvals = jnp.linspace(start=interval[0], stop=interval[1], num=num_points)
    I_2 = jnp.eye(2)
    H_2 = jnp.array([[1, 1], [1, -1]]) / jnp.sqrt(2)
    H_n = reduce(jnp.kron, [H_2] * feature_dimension)
    psi_init = jnp.ones(2**feature_dimension) / jnp.sqrt(2**feature_dimension)
    single_z = jnp.diag(jnp.array([1, -1]))
    z_rotations = jnp.stack(
        [
            reduce(
                jnp.kron,
                [I_2] * i + [single_z] + [I_2] * (feature_dimension - i - 1),
            )
            for i in range(feature_dimension)
        ]
    )

    par_op = reduce(jnp.kron, [single_z] * feature_dimension)
    assert _is_hermitian(par_op) and _is_unitary(par_op) is True

    """Generate real random values and create a hermitian operator via the trick
    that is explained in the docstring of the function `_hermitian_random`!
    """
    random_hermitian = _hermitian_random(size=2**feature_dimension, key=key)
    assert _is_hermitian(random_hermitian) is True

    eigvals, eigvecs = jnp.linalg.eigh(random_hermitian)
    idcs = jnp.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idcs]
    assert _is_unitary(eigvecs) is True
    random_unitary = eigvecs.conj().T @ par_op @ eigvecs
    assert _is_unitary(random_unitary) is True

    def compute_expval(x: jax.Array) -> jax.Array:
        x = jnp.asarray(x)
        fm = jnp.sum(x[:, None, None] * z_rotations, axis=0)
        fm += sum(
            (
                (jnp.pi - x[i]) * (jnp.pi - x[j]) * z_rotations[i] @ z_rotations[j]
                for i, j in it.combinations(range(feature_dimension), 2)
            )
        )
        unitary_fm = jnp.diag(jnp.exp(1j * jnp.diagonal(fm)))
        psi = unitary_fm @ H_n @ unitary_fm @ psi_init
        exp_val = jnp.real(psi.conj().T @ random_unitary @ psi)
        return exp_val

    sample_grid = _generate_sample_grid(
        num_points, feature_dimension, xvals, compute_expval, delta
    )

    x_sample, y_sample = _sample_data(
        sample_grid,
        xvals,
        training_examples_per_class + test_examples_per_class,
        feature_dimension,
    )

    training_input = {
        key: (x_sample[y_sample == k, :])[:training_examples_per_class]
        for k, key in enumerate(class_labels)
    }
    test_input = {
        key: (x_sample[y_sample == k, :])[
            training_examples_per_class : training_examples_per_class
            + test_examples_per_class
        ]
        for k, key in enumerate(class_labels)
    }

    X_train, y_train = _features_and_labels_transform(training_input, class_labels)
    X_test, y_test = _features_and_labels_transform(test_input, class_labels)

    return (X_train, y_train, X_test, y_test)


def _generate_sample_grid(
    num_points: int,
    feature_dimension: int,
    xvals: jax.Array,
    compute_expval: Callable[..., jax.Array],
    delta: float,
) -> jax.Array:
    """
    Generates a labeled grid of points based on the computed expectation values.

    Args:
        num_points (int): Number of points along each dimension of the grid.
        feature_dimension (int): Dimensionality of the feature space.
        xvals (jax.Array): A 1D array of values to use as coordinates for the grid points.
        compute_expval (Callable[..., jax.Array]): A function to compute the expectation value
            for a given input, which is vectorized internally for efficiency.
        delta (float): Threshold for labeling grid points. If the absolute value of the
            computed expectation exceeds this threshold, the grid point is labeled with the
            sign of the expectation value. Otherwise, it is labeled as 0.

    Returns:
        jax.Array: A labeled grid of shape `(num_points, ... , num_points)` with a size corresponding
        to `feature_dimension`. Each point in the grid is labeled based on the expectation value:
        1 for positive, -1 for negative, and 0 if the value is within `-delta` to `delta`.

    Notes:
        - The grid points are generated by creating a
        meshgrid from `xvals` repeated `feature_dimension` times.
        - The expectation value function is vectorized using
            JAX's automatic vectorization.
        - The resulting grid is reshaped into a tensor matching
        the desired grid shape.

    Example given:
        >>> num_points = 3
        >>> feature_dimension = 2
        >>> xvals = jnp.array([-1.0, 0.0, 1.0])
        >>> def compute_expval(x): return x.sum(axis=-1)  # Example function
        >>> delta = 0.5
        >>> labeled_grid = _generate_sample_grid(num_points, feature_dimension, xvals, compute_expval, delta)
        >>> print(labeled_grid)
        [[-1  0  1]
         [-1  0  1]
         [-1  0  1]]
    """
    mesh_axes = jnp.meshgrid(*[xvals] * feature_dimension, indexing="ij")
    grid_points = jnp.stack([axis.flatten() for axis in mesh_axes], axis=-1)
    compute_expval_vmap = jax.vmap(compute_expval)
    exp_vals = compute_expval_vmap(grid_points)
    labels = jnp.where(jnp.abs(exp_vals) > delta, jnp.sign(exp_vals), 0)
    grid_shape = [num_points] * feature_dimension
    labeled_grid = labels.reshape(*grid_shape)
    return labeled_grid


def _hermitian_random(size: int, key: jax.Array) -> jax.Array:
    """
    Use trick from the second chapter of the book by Nielsen & Chuang
    (https://en.wikipedia.org/wiki/Quantum_Computation_and_Quantum_Information).

    Every operator $ CC^\dagger $
    with

    $$ C = A + i B $$

    is hermitian since

    $$ CC^\dagger = (A - i B) (A + i B) = A^2 + B^2. $$

    and then using 2.25 (every positive operator is hermitian)
    we get that $ CC^\dagger $ is hermitian.

    Parameters
    ----------
    size : int
        Dimension of A and B, is equal to 2^n, where n denotes the
        number of qubits of the quantum system.
    key : jax.Array
        A JAX-key used for the generation of reproducable
        random numbers. The same key produces the exact
        same values.

    Returns
    -------
    jax.Array
        A random hermitian operator.
    """
    A, B = _rand(size, key)
    C = A + 1j * B
    return C.conj().T @ C


def _sample_data(
    sample_total: jax.Array,
    xvals: jax.Array,
    total_num_examples: int,
    feature_dimension: int,
) -> Tuple[jax.Array, jax.Array]:
    """
    Samples data points from a given set of samples
    based on their associated labels.

    Args:
        sample_total (jax.Array): A 1D array containing the labels
        of the total set of possible samples.
        The labels are assumed to be either 1 or -1, and these are
        then used to determine which samples to select.
        xvals (jax.Array): A 1D array of feature values from which
        to sample. Each element in `xvals`
        corresponds to one coordinate in the feature space.
        total_num_examples (int): The total number of examples to
        sample for each label (positive and negative).
        feature_dimension (int): The number of dimensions in each
        sample, which determines the shape of the samples.

    Returns:
        Tuple[jax.Array, jax.Array]: A tuple containing:
            - `samples` (jax.Array): A 2D array of shape `(2 * total_num_examples, feature_dimension)`,
            representing the sampled data points.
            - `labels` (jax.Array): A 1D array of shape `(2 * total_num_examples,)`, where the first
            `total_num_examples` elements are labeled `0` (representing positive samples) and the
            next `total_num_examples` elements are labeled `1` (representing negative samples).

    Notes:
        - This function generates random samples by choosing points from `sample_total` based on their
        labels, and then selecting corresponding feature values from `xvals` to form the samples.
        - Positive samples are labeled `1`, and negative samples are labeled `-1`.
        - The sampling process continues until the desired number of examples for each label is met.

    Example:
        >>> sample_total = jnp.array([1, -1, 1, -1, 1, -1])  # Example label array
        >>> xvals = jnp.array([0.1, 0.2, 0.3, 0.4])  # Example feature values
        >>> total_num_examples = 2
        >>> feature_dimension = 2
        >>> samples, labels = _sample_data(sample_total, xvals, total_num_examples, feature_dimension)
        >>> print(samples)
        [[0.1, 0.2],
        [0.3, 0.4],
        [0.1, 0.2],
        [0.3, 0.4]]
        >>> print(labels)
        [0, 0, 1, 1]
    """
    count = sample_total.shape[0]
    sample_pos: List[Any] = []
    sample_neg: List[Any] = []
    for i, sample_list in enumerate([sample_pos, sample_neg]):
        label = 1 if i == 0 else -1
        while len(sample_list) < total_num_examples:
            draws = tuple(
                algorithm_globals.random.choice(count) for i in range(feature_dimension)
            )
            if sample_total[draws] == label:
                sample_list.append([xvals[d] for d in draws])

    labels = jnp.array([0] * total_num_examples + [1] * total_num_examples)
    samples = jnp.array([sample_pos, sample_neg])
    samples = jnp.reshape(samples, (2 * total_num_examples, feature_dimension))
    return samples, labels


def _features_and_labels_transform(
    dataset: dict, class_labels: list
) -> Tuple[NDArray, NDArray]:
    features = np.concatenate(list(dataset.values()))
    raw_labels = np.concatenate(
        [np.full((v.shape[0],), k) for k, v in enumerate(dataset.values())]
    )
    label_mapping = {class_labels[0]: -1, class_labels[1]: 1}
    labels = np.array(
        [label_mapping[class_labels[int(label)]] for label in raw_labels],
    )
    return features, labels


def _rand(
    size: int, key: jax.Array, low: float = 0.0, high: float = 1.0
) -> Tuple[jax.Array, jax.Array]:
    """Creates a tuple of two random arrays that
    are created by samping a gaussian normal distribution.

    Parameters
    ----------
    size : int
        The dimension of the operators, will be equal to 2^n,
        where n denotes the number of qubits in the quantum
        system.
    key : jax.Array
        A JAX-key used for the generation of reproducable
        random numbers. The same key produces the exact
        same values.
    low : float, optional
        the lowest random value that will be drawn from the
        normal distribution, by default 0.0
    high : float, optional
        the highest random value that will be drawn from the
        normal distribution, by default 1.0

    Returns
    -------
    Tuple[jax.Array, jax.Array]
        Two arrays containing samples of a gaussian normal distribution.
    """
    new_key, subkey1, subkey2 = jax.random.split(key, num=3)
    del key
    A = jax.random.normal(subkey1, (size, size)) * (high - low) + low
    B = jax.random.normal(subkey2, (size, size)) * (high - low) + low
    return A, B


def _is_hermitian(H: jax.Array) -> bool:
    """Checks if the given operator is hermitian, i.e.
    whether H^\dagger = H holds up to a small deviation.
    See also section 1 of the thesis for supplementary
    information on hermitian operators.

    Parameters
    ----------
    H : jax.Array
        linear Operator, of shape (2^n, 2^n), where n
        denotes the number of qubits in the quantum system.

    Returns
    -------
    bool
        - True if H is hermitian,
        - False if H isn't hermitian.
    """
    H_dagger = jnp.conj(H.T)
    return jnp.allclose(H, H_dagger, atol=1e-6, rtol=1e-6).item()


def _is_unitary(U: jax.Array) -> bool:
    """Checks the two unitary relations, i.e. if
    the inverse of U is equal to its hermitian conjugate.
    See also section 1 of the thesis for more information
    on unitary operators.

    Parameters
    ----------
    U : jax.Array
        Operator of shape (2^n, 2^n), where n denotes the
        number of qubits in the Hilbert space.

    Returns
    -------
    bool
        True if U is unitary,
        False if U isn't unitary.
    """
    U_dagger = jnp.conj(U.T)
    relation_1 = U_dagger @ U
    relation_2 = U @ U_dagger
    return (
        jnp.allclose(jnp.eye(U.shape[0]), relation_1, atol=1e-3, rtol=1e-3).item()
        and jnp.allclose(jnp.eye(U.shape[0]), relation_2, atol=1e-3, rtol=1e-3).item()
    )


if __name__ == "__main__":
    print(_is_hermitian(jnp.eye(2)))  # True
