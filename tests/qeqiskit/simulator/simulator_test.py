import pytest
import numpy as np
import os
from openfermion.ops import QubitOperator
import qiskit.providers.aer.noise as AerNoise

from zquantum.core.circuit import Circuit, Gate, Circuit, Qubit
from zquantum.core.interfaces.backend_test import (
    QuantumSimulatorTests,
    QuantumSimulatorGatesTest,
)
from zquantum.core.measurement import ExpectationValues
from qeqiskit.simulator import QiskitSimulator
from qeqiskit.noise import get_qiskit_noise_model


@pytest.fixture(
    params=[
        {
            "device_name": "qasm_simulator",
            "n_samples": 1,
            "api_token": os.getenv("ZAPATA_IBMQ_API_TOKEN"),
        },
    ]
)
def backend(request):
    return QiskitSimulator(**request.param)


@pytest.fixture(
    params=[
        {
            "device_name": "statevector_simulator",
        },
    ]
)
def wf_simulator(request):
    return QiskitSimulator(**request.param)


@pytest.fixture(
    params=[
        {
            "device_name": "qasm_simulator",
        },
        {
            "device_name": "statevector_simulator",
        },
    ]
)
def sampling_simulator(request):
    return QiskitSimulator(**request.param)


@pytest.fixture(
    params=[
        {"device_name": "qasm_simulator", "n_samples": 1000, "optimization_level": 0},
    ]
)
def noisy_simulator(request):
    ibmq_api_token = os.getenv("ZAPATA_IBMQ_API_TOKEN")
    noise_model, connectivity = get_qiskit_noise_model(
        "ibmqx2", api_token=ibmq_api_token
    )

    return QiskitSimulator(
        **request.param, noise_model=noise_model, device_connectivity=connectivity
    )


class TestQiskitSimulator(QuantumSimulatorTests):
    def test_run_circuitset_and_measure(self, sampling_simulator):
        # Given
        qubits = [Qubit(i) for i in range(3)]
        X = Gate("X", qubits=[Qubit(0)])
        CNOT = Gate("CNOT", qubits=[Qubit(1), Qubit(2)])
        circuit = Circuit()
        circuit.qubits = qubits
        circuit.gates = [X, CNOT]

        # When
        sampling_simulator.n_samples = 100
        measurements_set = sampling_simulator.run_circuitset_and_measure([circuit])
        # Then
        assert len(measurements_set) == 1
        for measurements in measurements_set:
            assert len(measurements.bitstrings) == 100
            assert all(bitstring == (1, 0, 0) for bitstring in measurements.bitstrings)

        # When
        sampling_simulator.n_samples = 100
        measurements_set = sampling_simulator.run_circuitset_and_measure(
            [circuit] * 100
        )
        # Then
        assert len(measurements_set) == 100
        for measurements in measurements_set:
            assert len(measurements.bitstrings) == 100
            assert all(bitstring == (1, 0, 0) for bitstring in measurements.bitstrings)

    def test_setup_basic_simulators(self):
        simulator = QiskitSimulator("qasm_simulator")
        assert isinstance(simulator, QiskitSimulator)
        assert simulator.device_name == "qasm_simulator"
        assert simulator.n_samples is None
        assert simulator.noise_model is None
        assert simulator.device_connectivity is None
        assert simulator.basis_gates is None

        simulator = QiskitSimulator("statevector_simulator")
        assert isinstance(simulator, QiskitSimulator)
        assert simulator.device_name == "statevector_simulator"
        assert simulator.n_samples is None
        assert simulator.noise_model is None
        assert simulator.device_connectivity is None
        assert simulator.basis_gates is None

    def test_simulator_that_does_not_exist(self):
        # Given/When/Then
        with pytest.raises(RuntimeError):
            QiskitSimulator("DEVICE DOES NOT EXIST")

    def test_expectation_value_with_noisy_simulator(self, noisy_simulator):
        # Given
        # Initialize in |1> state
        X = Gate("X", qubits=[Qubit(0)])
        circuit = Circuit()
        circuit.qubits = [Qubit(0)]
        circuit.gates = [X]

        # Flip qubit an even number of times to remain in the |1> state, but allow decoherence to take effect
        circuit.gates += [X] * 10
        qubit_operator = QubitOperator("Z0")
        noisy_simulator.n_samples = 8192
        # When
        expectation_values_10_gates = noisy_simulator.get_expectation_values(
            circuit, qubit_operator
        )
        # Then
        assert isinstance(expectation_values_10_gates, ExpectationValues)
        assert len(expectation_values_10_gates.values) == 1
        assert expectation_values_10_gates.values[0] > -1
        assert expectation_values_10_gates.values[0] < 0.0
        assert isinstance(noisy_simulator, QiskitSimulator)
        assert noisy_simulator.device_name == "qasm_simulator"
        assert noisy_simulator.n_samples == 8192
        assert isinstance(noisy_simulator.noise_model, AerNoise.NoiseModel)
        assert noisy_simulator.device_connectivity is not None
        assert noisy_simulator.basis_gates is not None

        # Given
        # Initialize in |1> state
        circuit.gates = [X]
        # Flip qubit an even number of times to remain in the |1> state, but allow decoherence to take effect
        circuit.gates += [X] * 50
        qubit_operator = QubitOperator("Z0")
        noisy_simulator.n_samples = 8192
        # When
        expectation_values_50_gates = noisy_simulator.get_expectation_values(
            circuit, qubit_operator
        )
        # Then
        assert isinstance(expectation_values_50_gates, ExpectationValues)
        assert len(expectation_values_50_gates.values) == 1
        assert expectation_values_50_gates.values[0] > -1
        assert expectation_values_50_gates.values[0] < 0.0
        assert (
            expectation_values_50_gates.values[0]
            > expectation_values_10_gates.values[0]
        )
        assert isinstance(noisy_simulator, QiskitSimulator)
        assert noisy_simulator.device_name == "qasm_simulator"
        assert noisy_simulator.n_samples == 8192
        assert isinstance(noisy_simulator.noise_model, AerNoise.NoiseModel)
        assert noisy_simulator.device_connectivity is not None
        assert noisy_simulator.basis_gates is not None

    def test_optimization_level_of_transpiler(self):
        # Given
        noise_model, connectivity = get_qiskit_noise_model(
            "ibmqx2", api_token=os.getenv("ZAPATA_IBMQ_API_TOKEN")
        )
        simulator = QiskitSimulator(
            "qasm_simulator",
            n_samples=8192,
            noise_model=noise_model,
            device_connectivity=connectivity,
            optimization_level=0,
        )
        qubit_operator = QubitOperator("Z0")
        # Initialize in |1> state
        X = Gate("X", qubits=[Qubit(0)])
        circuit = Circuit()
        circuit.qubits = [Qubit(0)]
        circuit.gates = [X]
        # Flip qubit an even number of times to remain in the |1> state, but allow decoherence to take effect
        circuit.gates += [X] * 50

        # When
        expectation_values_no_compilation = simulator.get_expectation_values(
            circuit, qubit_operator
        )

        simulator.optimization_level = 3
        expectation_values_full_compilation = simulator.get_expectation_values(
            circuit, qubit_operator
        )

        # Then
        assert (
            expectation_values_full_compilation.values[0]
            < expectation_values_no_compilation.values[0]
        )


class TestQiskitSimulatorGates(QuantumSimulatorGatesTest):
    pass
