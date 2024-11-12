from copy import copy
from typing import Sequence
from qiskit_algorithms.state_fidelities import BaseStateFidelity
from qiskit_algorithms.state_fidelities import StateFidelityResult
from qiskit.primitives import Sampler
from qiskit.providers import Options
from qiskit import QuantumCircuit
from ..exceptions.exceptions import AlgorithmError


class SwapTest(BaseStateFidelity):
    def __init__(self, sampler: Sampler = None) -> None:
        # simulate classically if no backend is setup:
        if not sampler:
            sampler = Sampler()
        self._sampler = sampler
        super().__init__()

    def _run(
        self,
        circuit_1: QuantumCircuit,
        circuit_2: QuantumCircuit,
        values_1: Sequence[float] | Sequence[Sequence[float]],
        values_2: Sequence[float] | Sequence[Sequence[float]],
    ) -> StateFidelityResult:

        swap_test = self.create_fidelity_circuit(circuit_1, circuit_2)

        if len(swap_test) == 0:
            raise ValueError(
                "At least one pair of circuits must be defined to compute the state overlap"
            )

        parameter_values = self._construct_value_list(
            circuit_1, circuit_2, values_1, values_2
        )

        job = self._sampler.run(circuits=swap_test, parameter_values=parameter_values)
        try:
            result = job.result()
        except Exception as _:
            raise AlgorithmError()

        quasi_dist = result.quasi_dists[0]
        prob_only_zeros = quasi_dist.get(0, 0)
        fidelity = 2 * prob_only_zeros - 1
        return StateFidelityResult(fidelities=fidelity, raw_fidelities=None)

    def create_fidelity_circuit(
        self, circuit_1: QuantumCircuit, circuit_2: QuantumCircuit
    ) -> QuantumCircuit:

        self.num_qubits = circuit_1.num_qubits + circuit_2.num_qubits + 1

        # first parameter qubits, second parameter classical bits
        circuit = QuantumCircuit(self.num_qubits, 1)

        ancilla_idx = 0
        psi_idcs = range(1, self.num_qubits + 1)
        phi_idcs = range(self.num_qubits + 1, 2 * self.num_qubits + 1)

        circuit.h(ancilla_idx)

        circuit.compose(circuit_1, qubits=psi_idcs, inplace=True)
        circuit.compose(circuit_2, qubits=phi_idcs, inplace=True)

        for psi_q, phi_q in zip(psi_idcs, phi_idcs):
            circuit.cswap(ancilla_idx, psi_q, phi_q)

        circuit.h(ancilla_idx)

        circuit.measure(ancilla_idx, 0)

        return circuit

    @property
    def options(self) -> Options:
        """Return the union of estimator options setting and fidelity default options,
        where, if the same field is set in both, the fidelity's default options override
        the primitive's default setting.

        Returns:
            The fidelity default + estimator options.
        """
        return self._get_local_options(self._default_options.__dict__)

    def update_default_options(self, **options):
        """Update the fidelity's default options setting.

        Args:
            **options: The fields to update the default options.
        """

        self._default_options.update_options(**options)

    def _get_local_options(self, options: Options) -> Options:
        """Return the union of the primitive's default setting,
        the fidelity default options, and the options in the ``run`` method.
        The order of priority is: options in ``run`` method > fidelity's
                default options > primitive's default setting.

        Args:
            options: The fields to update the options

        Returns:
            The fidelity default + estimator + run options.
        """
        opts = copy(self._sampler.options)
        opts.update_options(**options)
        return opts

    @staticmethod
    def _get_global_fidelity(probability_distribution: dict[int, float]) -> float:
        """Process the probability distribution of a measurement to determine the
        global fidelity.

        Args:
            probability_distribution: Obtained from the measurement result

        Returns:
            The global fidelity.
        """
        return probability_distribution.get(0, 0)

    @staticmethod
    def _get_local_fidelity(
        probability_distribution: dict[int, float], num_qubits: int
    ) -> float:
        """Process the probability distribution of a measurement to determine the
        local fidelity by averaging over single-qubit projectors.

        Args:
            probability_distribution: Obtained from the measurement result

        Returns:
            The local fidelity.
        """
        fidelity = 0.0
        for qubit in range(num_qubits):
            for bitstring, prob in probability_distribution.items():
                # Check whether the bit representing the current qubit is 0
                if not bitstring >> qubit & 1:
                    fidelity += prob / num_qubits
        return fidelity
