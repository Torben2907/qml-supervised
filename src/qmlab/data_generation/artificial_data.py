import numpy as np
from typing import List, Tuple
from functools import reduce
import itertools as it
import torch


def generate_random_data(
    feature_dimension: int,
    training_examples_per_class: int,
    test_examples_per_class: int,
    delta: float = 0.3,
    random_state: int = 42,
    interval: Tuple[float, float] = (0.0, 2 * np.pi),
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    torch.manual_seed(random_state)
    class_labels = [r"+1", r"-1"]

    if not isinstance(interval, tuple) or len(interval) != 2:
        raise ValueError(
            "`interval` must be a tuple containing two floating point values!"
        )

    num_points = 100 if feature_dimension == 2 else 20
    xvals = torch.linspace(
        start=interval[0],
        end=interval[1],
        steps=num_points,
        device=device,
        dtype=torch.float32,
    )

    H = torch.tensor(
        [[1, 1], [1, -1]], dtype=torch.complex64, device=device
    ) / torch.sqrt(torch.tensor([2.0], dtype=torch.complex64, device=device))
    H_wall = reduce(torch.kron, [H] * feature_dimension).to(device)
    psi_init = torch.ones(
        2**feature_dimension, dtype=torch.complex64, device=device
    ) / torch.sqrt(torch.tensor(2**feature_dimension, device=device))
    single_z = torch.diag(torch.tensor([1, -1], device=device))
    z_rotations = torch.stack(
        [
            reduce(
                torch.kron,
                [torch.eye(2, device=device)] * i
                + [single_z]
                + [torch.eye(2, device=device)] * (feature_dimension - i - 1),
            )
            for i in range(feature_dimension)
        ]
    )

    par_op = reduce(torch.kron, [single_z] * feature_dimension)
    assert _is_hermitian(par_op) and _is_unitary(par_op) is True

    # generate real random vals and create hermitian by the trick
    # explained in the function _hermitian_random docstring!
    random_hermitian = _hermitian_random(
        size=2**feature_dimension, dtype=torch.float32, device=device
    )
    assert _is_hermitian(random_hermitian) is True
    eigvals, eigvecs = torch.linalg.eigh(random_hermitian)
    assert _is_unitary(eigvecs) is True
    random_unitary = eigvecs.conj().t() @ par_op.to(dtype=eigvecs.dtype) @ eigvecs
    assert _is_unitary(random_unitary) is True

    samples: List[float] = []
    for x in it.product(*[xvals] * feature_dimension):
        fm = sum(x[i] * z_rotations[i] for i in range(feature_dimension))
        fm += sum(
            (
                (np.pi - x[i]) * (np.pi - x[j]) * z_rotations[i] @ z_rotations[j]
                for i, j in it.combinations(range(feature_dimension), 2)
            )
        )
        unitary_fm = torch.diag(torch.exp(1j * torch.diagonal(fm)))
        psi = unitary_fm @ H_wall @ unitary_fm @ psi_init
        exp_val = torch.real(psi.conj().t() @ random_unitary @ psi)
        samples.append(torch.sign(exp_val).item() if torch.abs(exp_val) > delta else 0)

    sample_grid = torch.tensor(samples).reshape(tuple([num_points] * feature_dimension))

    x_sample, y_sample = _sample_train_test_data(
        sample_grid,
        xvals,
        training_examples_per_class + test_examples_per_class,
        feature_dimension,
        device,
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

    X_train, y_train = _features_and_labels_transform(
        training_input, class_labels, device
    )
    X_test, y_test = _features_and_labels_transform(test_input, class_labels, device)

    return (
        X_train.cpu().numpy(),
        y_train.cpu().numpy(),
        X_test.cpu().numpy(),
        y_test.cpu().numpy(),
    )


def _hermitian_random(size: int, dtype: torch.dtype, device: str) -> torch.Tensor:
    r"""Creates an arbitrary hermitian using the fact:
    :math:`C = A + iB, C^\dagger = A - iB`
    :math:`C = C^\dagger C = A^\dagger A + B^\dagger B + i (A^\dagger B - B^\dagger A)`
    :math:`\implies C^\dagger = C `

    Parameters
    ----------
    size : int
        Size of the output matrix. In context this will be 2 to the power of the number
        of features.
    dtype : _type_, optional
        The datatype used for the generation of the
    device : str, optional
        _description_, by default "mps"

    Returns
    -------
    torch.Tensor
        _description_
    """
    A = _rand(size, size, dtype=dtype, device=device).to(dtype=torch.complex64)
    B = _rand(size, size, dtype=dtype, device=device).to(dtype=torch.complex64)
    C = A + 1j * B
    return C.conj().T @ C


def _sample_train_test_data(
    sample_grid: torch.Tensor, X_vals, num_examples_per_class, feature_dimension, device
):
    count = sample_grid.shape[0]
    plus_class: List[torch.Tensor] = []
    minus_class: List[torch.Tensor] = []
    for label, sample_list in [(1, plus_class), (-1, minus_class)]:
        while len(sample_list) < num_examples_per_class:
            idx = tuple(torch.randint(0, count, (feature_dimension,), device=device))
            if sample_grid[idx] == label:
                sample_list.append(
                    torch.tensor([X_vals[i] for i in idx], device=device)
                )
    samples = torch.vstack([torch.stack(plus_class), torch.stack(minus_class)])
    labels = torch.tensor(
        [0] * num_examples_per_class + [1] * num_examples_per_class, device=device
    )
    return samples, labels


def _features_and_labels_transform(
    dataset: dict, class_labels: list, device: str = "mps"
) -> Tuple[torch.Tensor, torch.Tensor]:
    features = torch.cat(list(dataset.values()), dim=0)
    raw_labels = torch.cat(
        [
            torch.full((v.shape[0],), k, device=device)
            for k, v in enumerate(dataset.values())
        ]
    )
    label_mapping = {class_labels[0]: -1, class_labels[1]: 1}
    labels = torch.tensor(
        [label_mapping[class_labels[int(label)]] for label in raw_labels.cpu()],
        device=device,
    )
    return features, labels


def _rand(
    *shape,
    low: float = 0.0,
    high: float = 1.0,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    """Generate random numbers uniformly between the values `low` and `high`.

    Parameters
    ----------
    dtype : torch.dtype
        the datatype of the random values that are generated
    low : float, optional
        the lowest possible value that can be generated, by default 0.0
    high : float, optional
        the highest possible value that can be generated, by default 1.0
    device : str, optional
        the device used to perform the computations.
    Returns
    -------
    torch.Tensor
        the random numbers in form of a tensor which has the desired shape
        as the first arguments specify.
    """
    return torch.rand(*shape, dtype=dtype) * (high - low) + low


def _is_hermitian(H: torch.Tensor) -> bool:
    r"""Check whether given Tensor is hermitian, i.e.
        $H = H^\dagger$

    Parameters
    ----------
    H : torch.Tensor
        Operator in the complex Hilbert space.

    Returns
    -------
    bool
        `True` if H is hermitian, `False` otherwise.
    """
    H_dagger = torch.conj(H.t())
    return torch.allclose(H, H_dagger, atol=1e-6, rtol=1e-6)


def _is_unitary(U: torch.Tensor) -> bool:
    r"""Check whether given operator U fulfills the unitary relations:
    ```math
    U^\dagger U = U U^\dagger = \mathbb{1}
    ```
    where :math:`1` is the identity matrix of dimension
    :math:`2 ** feature_dimension`.

    Parameters
    ----------
    U : torch.Tensor
       Operator in the complex Hilbert space :

    Returns
    -------
    bool
        _description_
    """
    U_dagger = torch.conj(U.t())
    unitary_rel_1 = U_dagger @ U
    unitary_rel_2 = U @ U_dagger
    return torch.allclose(
        torch.eye(U.shape[0], dtype=U.dtype),
        unitary_rel_1,
        atol=1e-6,
        rtol=1e-6,
    ) and torch.allclose(
        torch.eye(U.shape[0], dtype=U.dtype),
        unitary_rel_2,
        atol=1e-6,
        rtol=1e-6,
    )


if __name__ == "__main__":
    print(_is_hermitian(torch.eye(2)))
