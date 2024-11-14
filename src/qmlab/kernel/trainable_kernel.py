from typing import Mapping, Sequence
from .fidelity_quantum_kernel import FidelityQuantumKernel
from qiskit.circuit import ParameterVector, Parameter, ParameterExpression
from qiskit import QuantumCircuit
from qiskit_algorithms.state_fidelities import BaseStateFidelity
import numpy as np


class TrainableQuantumKernel(FidelityQuantumKernel):
    def __init__(
        self,
        *,
        feature_map: QuantumCircuit = None,
        fidelity: BaseStateFidelity | None = None,
        training_params: ParameterVector | Sequence[ParameterVector] = None,
        **kwargs,
    ):
        super().__init__(feature_map=feature_map, fidelity=fidelity)

        if not training_params:
            training_params = []

        self._training_params = training_params
        self._num_training_params = len(self._training_params)

        self._parameter_dict = {p: None for p in training_params}

        self._feature_parameters: Sequence[Parameter] = []

    def assign_training_parameters(
        self,
        parameter_values: (
            Mapping[Parameter, ParameterExpression] | Sequence[ParameterExpression]
        ),
    ) -> None:
        """
        Fix the training parameters to numerical values.
        """
        if not isinstance(parameter_values, dict):
            if len(parameter_values) != self._num_training_params:
                raise ValueError(
                    f"The number of given parameters is wrong: {len(parameter_values)}, "
                    f"expected {self._num_training_params}."
                )
            self._parameter_dict.update(
                {
                    parameter: parameter_values[i]
                    for i, parameter in enumerate(self._training_params)
                }
            )
        else:
            for key in parameter_values:
                if key not in self._training_params:
                    raise ValueError(
                        f"Parameter {key} is not a trainable parameter of the feature map and "
                        f"thus cannot be bound. Make sure {key} is provided in the the trainable "
                        "parameters when initializing the kernel."
                    )
                self._parameter_dict[key] = parameter_values[key]

    @property
    def parameter_values(self) -> np.ndarray:
        """
        Returns numerical values assigned to the training parameters as a numpy array.
        """
        return np.asarray(
            [self._parameter_dict[param] for param in self._training_params]
        )

    @property
    def training_parameters(self) -> ParameterVector | Sequence[Parameter]:
        """
        Returns the vector of training parameters.
        """
        return self._training_params

    @property
    def num_training_parameters(self) -> int:
        """
        Returns the number of training parameters.
        """
        return len(self._training_params)

    def _parameter_array(self, x_vec: np.ndarray) -> np.ndarray:
        """
        Combines the feature values and the trainable parameters into one array.
        """
        self._check_trainable_parameters()
        full_array = np.zeros(
            (x_vec.shape[0], self._feature_dimension + self._num_training_params)
        )
        for i, x in enumerate(x_vec):
            self._parameter_dict.update(
                {
                    feature_param: x[j]
                    for j, feature_param in enumerate(self._feature_parameters)
                }
            )
            full_array[i, :] = list(self._parameter_dict.values())
        return full_array

    def _check_trainable_parameters(self) -> None:
        for param in self._training_params:
            if self._parameter_dict[param] is None:
                raise ValueError(
                    f"Trainable parameter {param} has not been bound. Make sure to bind all"
                    "trainable parameters to numerical values using `.assign_training_parameters()`"
                    "before calling `.evaluate()`."
                )
