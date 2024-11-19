from typing import Callable, Tuple
from functools import reduce
import itertools as it

import jax
import jax.numpy as jnp
from qiskit_algorithms.utils import algorithm_globals


def generate_random_data(
    feature_dimension: int,
    training_examples_per_class: int,
    test_examples_per_class: int,
    delta: float = 0.3,
    random_state: int = 42,
    interval: Tuple[float, float] = (0.0, 2 * jnp.pi),
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    key = jax.random.key(seed=random_state)
    algorithm_globals.random_seed = random_state
    class_labels = [r"+1", r"-1"]

    if not isinstance(interval, tuple) or len(interval) != 2:
        raise ValueError(
            "`interval` must be a tuple containing two floating point values!"
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

    # generate real random vals and create hermitian by the trick
    # explained in the function _hermitian_random docstring!
    random_hermitian = _hermitian_random(size=2**feature_dimension, key=key)
    assert _is_hermitian(random_hermitian) is True

    eigvals, eigvecs = jnp.linalg.eigh(random_hermitian)
    idcs = jnp.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idcs]
    assert _is_unitary(eigvecs) is True
    random_unitary = eigvecs.conj().T @ par_op @ eigvecs
    assert _is_unitary(random_unitary) is True

    def compute_expval(x: jnp.ndarray) -> jnp.ndarray:
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
    xvals: jnp.ndarray,
    compute_expval: Callable[..., jnp.ndarray],
    delta: float,
) -> jnp.ndarray:
    mesh_axes = jnp.meshgrid(*[xvals] * feature_dimension, indexing="ij")
    grid_points = jnp.stack([axis.flatten() for axis in mesh_axes], axis=-1)
    compute_expval_vmap = jax.vmap(compute_expval)
    exp_vals = compute_expval_vmap(grid_points)
    labels = jnp.where(jnp.abs(exp_vals) > delta, jnp.sign(exp_vals), 0)
    grid_shape = [num_points] * feature_dimension
    labeled_grid = labels.reshape(*grid_shape)
    return labeled_grid


def _hermitian_random(size: int, key: jnp.ndarray) -> jnp.ndarray:
    A, B = _rand(size, key)
    C = A + 1j * B
    return C.conj().T @ C


def _sample_data(sample_total, xvals, total_num_examples, feature_dimension):
    count = sample_total.shape[0]
    sample_pos, sample_neg = [], []
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
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    features = jnp.concatenate(list(dataset.values()))
    raw_labels = jnp.concatenate(
        [jnp.full((v.shape[0],), k) for k, v in enumerate(dataset.values())]
    )
    label_mapping = {class_labels[0]: -1, class_labels[1]: 1}
    labels = jnp.array(
        [label_mapping[class_labels[int(label)]] for label in raw_labels],
    )
    return features, labels


def _rand(
    size: int, key: jnp.ndarray, low: float = 0.0, high: float = 1.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    new_key, subkey1, subkey2 = jax.random.split(key, num=3)
    del key
    A = jax.random.normal(subkey1, (size, size)) * (high - low) + low
    B = jax.random.normal(subkey2, (size, size)) * (high - low) + low
    return A, B


def _is_hermitian(H: jnp.ndarray) -> bool:
    H_dagger = jnp.conj(H.T)
    return jnp.allclose(H, H_dagger, atol=1e-6, rtol=1e-6).item()


def _is_unitary(U: jnp.ndarray) -> bool:
    U_dagger = jnp.conj(U.T)
    relation_1 = U_dagger @ U
    relation_2 = U @ U_dagger
    return (
        jnp.allclose(jnp.eye(U.shape[0]), relation_1, atol=1e-6, rtol=1e-6).item()
        and jnp.allclose(jnp.eye(U.shape[0]), relation_2, atol=1e-6, rtol=1e-6).item()
    )


if __name__ == "__main__":
    print(_is_hermitian(jnp.eye(2)))
