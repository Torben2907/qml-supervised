import numpy as np
from functools import partial
from typing import Union, Optional, Sequence
import copy
import matplotlib.pyplot as plt

# Qiskit imports
from qiskit.utils import algorithm_globals
from qiskit.utils import QuantumInstance
from qiskit.primitives import Sampler
from qiskit_algorithms.optimizers import SPSA, Optimizer
from .trainable_kernel import TrainableQuantumKernel
from .kernel_loss import KernelLoss
from qiskit_algorithms.variational_algorithm import VariationalResult
from qiskit import QuantumCircuit

# Custom loss functions
from .kernel_loss import SVCLoss, KTALoss


class QuantumKernelTrainerResult(VariationalResult):
    """Quantum Kernel Trainer Result."""

    def __init__(self) -> None:
        super().__init__()
        self._quantum_kernel: TrainableQuantumKernel | None = None

    @property
    def quantum_kernel(self) -> Optional[TrainableQuantumKernel]:
        """Return the optimized quantum kernel object."""
        return self._quantum_kernel

    @quantum_kernel.setter
    def quantum_kernel(self, quantum_kernel: TrainableQuantumKernel) -> None:
        self._quantum_kernel = quantum_kernel


def QuantumKernelTraining(
    qfm: QuantumCircuit,
    X_train: np.ndarray,
    y_train: np.ndarray,
    init_params: Sequence[float],
    optimizer: Optimizer = None,
    loss: KernelLoss | str | None = None,
    backend: QuantumInstance = None,
    plot: bool = False,
    random_state: int | None = None,
) -> QuantumKernelTrainerResult:

    algorithm_globals.random_seed = random_state

    if loss is None:
        loss = "svc_loss"

    callback = QKTCallback()
    if optimizer is None:
        optimizer = SPSA(
            maxiter=100,
            learning_rate=None,
            perturbation=None,
            callback=callback.callback,
            termination_checker=TerminationChecker(0.001, N=10),
        )
    else:
        optimizer.callback = callback.callback

    if backend is None:
        backend = Sampler()

    quantum_kernel = TrainableQuantumKernel(
        feature_map=qfm, training_params=qfm.training_params, quantum_instance=backend
    )

    qkt_results = QuantumKernelTrainer(
        quantum_kernel=quantum_kernel,
        loss=loss,
        optimizer=optimizer,
        initial_point=init_params,
    ).fit(X_train, y_train)

    if plot:
        # optimization summary
        print("Loss function value: ", qkt_results.optimal_value)
        print("Optimal parameters:", qkt_results.optimal_point)
        # Visualize the optimization workflow
        plot_data = callback.get_callback_data()
        plt.rcParams["font.size"] = 20
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.plot(
            [i + 1 for i in range(len(plot_data[0]))],
            np.array(plot_data[2]),
            c="k",
            marker="o",
        )
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss")
        ax.title.set_text("Optimization")
        plt.show()

    return qkt_results


class QuantumKernelTrainer:
    def __init__(
        self,
        quantum_kernel: TrainableQuantumKernel,
        loss: Optional[Union[str, KernelLoss]] = None,
        optimizer: Optional[Optimizer] = None,
        initial_point: Optional[Sequence[float]] = None,
    ):
        # Class fields
        self._quantum_kernel = quantum_kernel
        self._initial_point = initial_point
        self._optimizer = optimizer or SPSA()

        # Loss setter
        self._set_loss(loss)

    @property
    def quantum_kernel(self) -> TrainableQuantumKernel:
        """Return the quantum kernel object."""
        return self._quantum_kernel

    @quantum_kernel.setter
    def quantum_kernel(self, quantum_kernel: TrainableQuantumKernel) -> None:
        """Set the quantum kernel."""
        self._quantum_kernel = quantum_kernel

    @property
    def loss(self) -> KernelLoss | str:
        """Return the loss object."""
        return self._loss

    @loss.setter
    def loss(self, loss: Optional[Union[str, KernelLoss]]) -> None:
        self._set_loss(loss)

    @property
    def optimizer(self) -> Optimizer:
        """Return an optimizer to be used in training."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer) -> None:
        """Set the optimizer."""
        self._optimizer = optimizer

    @property
    def initial_point(self) -> Optional[Sequence[float]]:
        """Return initial point"""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: Optional[Sequence[float]]) -> None:
        """Set the initial point"""
        self._initial_point = initial_point

    def fit(
        self,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> QuantumKernelTrainerResult:

        # Number of parameters to tune
        # num_params = len(self._quantum_kernel.user_parameters) # depricated
        num_params = len(self._quantum_kernel._feature_parameters)
        if num_params == 0:
            msg = "Quantum kernel cannot be fit because there are no user parameters specified."
            raise ValueError(msg)

        # Bind inputs to objective function
        output_kernel = copy.deepcopy(self._quantum_kernel)

        # Randomly initialize the initial point if one was not passed
        if self._initial_point is None:
            self._initial_point = algorithm_globals.random.random(num_params)

        # Perform kernel optimization
        loss_function = partial(
            self._loss.evaluate_kernel,
            quantum_kernel=self._quantum_kernel,
            data=data,
            labels=labels,
        )

        # Check if the optimizer support bounds
        if self._optimizer.is_bounds_required:
            bounds = np.array(self._optimizer._options["bounds"])
        else:
            bounds = None

        # Run the optimizer
        opt_results = self._optimizer.minimize(
            fun=loss_function,
            x0=self._initial_point,
            bounds=bounds,
        )

        # Return kernel training results
        result = QuantumKernelTrainerResult()
        result.optimizer_evals = opt_results.nfev
        result.optimal_value = opt_results.fun
        result.optimal_point = opt_results.x
        result.optimal_parameters = dict(
            # zip(output_kernel.user_parameters, opt_results.x)
            zip(output_kernel._feature_parameters, opt_results.x)
        )

        # Return the QuantumKernel in optimized state
        output_kernel.assign_training_parameters(result.optimal_parameters)
        result.quantum_kernel = output_kernel

        return result

    def _set_loss(self, loss: str | KernelLoss | None) -> None:
        """Internal setter."""
        if loss is None:
            loss = SVCLoss()
        elif isinstance(loss, str):
            loss = self._str_to_loss(loss)

        self._loss = loss

    def _str_to_loss(self, loss_str: str) -> KernelLoss:
        """Function which maps strings to default KernelLoss objects."""
        if loss_str == "svc_loss":
            return SVCLoss()
        elif loss_str == "kta_loss":
            return KTALoss()
        else:
            raise ValueError(f"Unknown loss {loss_str}!")


class QKTCallback:
    """Callback wrapper class for retrieving the data from optimization run."""

    def __init__(self) -> None:
        self._data: list[list] = [[] for i in range(5)]

    def callback(self, x0, x1=None, x2=None, x3=None, x4=None):
        """
        Args:
            x0: number of function evaluations
            x1: the parameters
            x2: the function value
            x3: the stepsize
            x4: whether the step was accepted
        """
        self._data[0].append(x0)
        self._data[1].append(x1)
        self._data[2].append(x2)
        self._data[3].append(x3)
        self._data[4].append(x4)

    def get_callback_data(self):
        return self._data

    def clear_callback_data(self):
        self._data = [[] for i in range(5)]

    def plot(self):
        """Visualize the optimization results"""
        plot_data = self.get_callback_data()
        plt.rcParams["font.size"] = 20
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.plot(
            [i + 1 for i in range(len(plot_data[0]))],
            np.array(plot_data[2]),
            c="k",
            marker="o",
        )
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss")
        ax.title.set_text("Optimization")
        plt.show()


class TerminationChecker:
    """Custom termination checker. Added check for preliminary convergence."""

    def __init__(self, tol: float, N: int = 5):
        self.tol = tol
        self.N = N
        self.values: list[int] = []

    def __call__(self, nfev, parameters, value, stepsize, accepted) -> bool:
        self.values.append(value)
        if len(self.values) > self.N:
            last_values = self.values[-self.N :]
            std = np.std(last_values)
            if std < self.tol:
                return True
        return False
