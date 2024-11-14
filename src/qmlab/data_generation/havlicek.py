import numpy as np
from typing import Dict, List, Tuple
from functools import reduce
import itertools as it


def havlicek_data(
    feature_dimension: int,
    training_examples_per_class: int,
    test_examples_per_class: int,
    delta: float = 0.3,
    random_state: int = 12345,
    interval: Tuple[float, float] = (0.0, 2 * np.pi),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    np.random.seed(random_state)

    class_labels = [r"+1", r"-1"]

    if feature_dimension not in [2, 3]:
        raise ValueError(
            f"Supported values of `feature_dimension` are 2 and 3 - you provided {feature_dimension}."
        )

    if not isinstance(interval, tuple):
        raise ValueError(
            f"You must specify `interval` as a Tuple - you provided {type(interval)}!"
        )
    else:
        assert len(interval) == 2, "you must provide exactly 2 values in `interval`!"
        for vals in interval:
            if not isinstance(vals, float):
                raise ValueError(
                    f"values for generation must be floats but you provided {type(vals)}"
                )

    num_points = 100 if feature_dimension == 2 else 20
    xvals = np.linspace(
        start=interval[0], stop=interval[1], num=num_points, endpoint=False
    )

    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    H_wall = reduce(np.kron, [H] * feature_dimension)
    single_z = np.diag([1, -1])

    # start in |+^n>, i.e. uniform superposition
    psi_init = np.ones(2**feature_dimension) / np.sqrt(2**feature_dimension)

    z_rotations = np.array(
        [
            reduce(
                np.kron,
                [np.eye(2)] * i
                + [single_z]
                + [np.eye(2)] * (feature_dimension - i - 1),
            )
            for i in range(feature_dimension)
        ]
    )

    state_labels = ["".join(b) for b in it.product("01", repeat=feature_dimension)]
    parity_diag = [b.count("1") % 2 for b in state_labels]
    par_op = np.diag((-1) ** np.array(parity_diag))
    print(par_op)

    # par_op = np.kron(single_z, single_z)

    # P |b> = (-1)^(parity(b)) |b>
    assert _is_hermitian(par_op) and _is_unitary(par_op) is True

    random_hermitian = _hermitian_random(size=2**feature_dimension)
    assert _is_hermitian(random_hermitian) is True
    eigvals, eigvecs = np.linalg.eig(random_hermitian)
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx]
    assert _is_unitary(eigvecs) is True
    rand_unitary = eigvecs.conj().T @ par_op @ eigvecs
    assert _is_unitary(rand_unitary) is True

    samples = []
    for x in it.product(*[xvals] * feature_dimension):
        fm = sum(x[i] * z_rotations[i] for i in range(feature_dimension))
        fm += sum(
            (
                (np.pi - x[i]) * (np.pi - x[j]) * z_rotations[i] @ z_rotations[j]
                for i, j in it.combinations(range(feature_dimension), 2)
            )
        )
        unitary_fm = np.diag(np.exp(1j * np.diag(fm)))
        psi = unitary_fm @ H_wall @ unitary_fm @ psi_init
        exp_val = np.real(psi.conj().T @ rand_unitary @ psi)
        samples.append(np.sign(exp_val) if np.abs(exp_val) > delta else 0)

    sample_grid = np.array(samples).reshape(tuple([num_points] * feature_dimension))

    x_sample, y_sample = _sample_havlicek_data(
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
            training_examples_per_class : (
                training_examples_per_class + test_examples_per_class
            )
        ]
        for k, key in enumerate(class_labels)
    }

    X_train, y_train = _features_and_labels_transform(training_input, class_labels)
    X_test, y_test = _features_and_labels_transform(test_input, class_labels)

    return (X_train, y_train, X_test, y_test)


def _hermitian_random(size: int) -> np.ndarray:
    # Use the trick from Nielsen & Chuang to create arbitrary hermitian:
    # C = A + iB, C^\dagger = A - iB
    # C = C^\dagger C = A^\dagger A + B^\dagger B + i (A^\dagger B - B^\dagger A)
    # => C^\dagger = C !
    C = _rand(size, size) + 1j * _rand(size, size)
    return C.conj().T @ C


def _sample_havlicek_data(
    sample_grid, X_vals, num_examples_per_class, feature_dimension
):
    count = sample_grid.shape[0]
    plus_class, minus_class = [], []
    for label, sample_list in [(1, plus_class), (-1, minus_class)]:
        while len(sample_list) < num_examples_per_class:
            idx = tuple(np.random.choice(count) for _ in range(feature_dimension))
            if sample_grid[idx] == label:
                sample_list.append([X_vals[i] for i in idx])
    samples = np.vstack([plus_class, minus_class])
    labels = np.array([0] * num_examples_per_class + [1] * num_examples_per_class)
    return samples, labels


def _prepare_datasets(samples, labels, num_train, num_test):
    x_train, x_test = samples[:num_train], samples[num_train : num_train + num_test]
    y_train, y_test = labels[:num_train], labels[num_train : num_train + num_test]
    return x_train, y_train, x_test, y_test


def _features_and_labels_transform(
    dataset: Dict[str, np.ndarray], class_labels: List[str]
) -> Tuple[np.ndarray, np.ndarray]:

    features = np.concatenate(list(dataset.values()))

    raw_labels = []
    for category in dataset.keys():
        num_samples = dataset[category].shape[0]
        raw_labels += [category] * num_samples

    if not raw_labels:
        # no labels, empty dataset
        labels = np.zeros((0, len(class_labels)))
        return features, labels

    # Map class labels to -1 and +1
    label_mapping = {class_labels[0]: -1, class_labels[1]: +1}
    labels = np.array([label_mapping[label] for label in raw_labels])

    return features, labels


def _rand(*shape, low=0.0, high=1.0, dtype="float32"):
    """Generate random numbers uniform between low and high"""
    array = np.random.rand(*shape) * (high - low) + low
    return np.array(array, dtype=dtype)


def _is_hermitian(h: np.ndarray) -> bool:
    h_dagger = np.conjugate(h.T)
    return np.allclose(h, h_dagger, atol=1e-8, rtol=1e-8)


def _is_unitary(u: np.ndarray) -> bool:
    u_dagger = np.conjugate(u.T)
    unitary_rel_1 = u_dagger @ u
    unitary_rel_2 = u @ u_dagger
    return np.allclose(
        np.eye(u.shape[0]), unitary_rel_1, atol=1e-7, rtol=1e-7
    ) and np.allclose(np.eye(u.shape[0]), unitary_rel_2, atol=1e-7, rtol=1e-7)
