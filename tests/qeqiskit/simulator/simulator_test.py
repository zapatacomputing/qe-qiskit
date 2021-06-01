import pytest
import os
from openfermion.ops import QubitOperator
import qiskit.providers.aer.noise as AerNoise

from zquantum.core.wip.circuits import Circuit, X, CNOT
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
        circuit = Circuit([X(0), CNOT(1, 2)])

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
        circuit = Circuit([X(0)])

        # Flip qubit an even number of times to remain in the |1> state, but allow decoherence to take effect
        circuit += Circuit([X(0) for i in range(10)])
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
        circuit = Circuit([X(0)])
        # Flip qubit an even number of times to remain in the |1> state, but allow decoherence to take effect
        circuit += Circuit([X(0) for i in range(50)])
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
        circuit = Circuit([X(0)])
        # Flip qubit an even number of times to remain in the |1> state, but allow decoherence to take effect
        circuit += Circuit([X(0) for i in range(50)])

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

    def test_run_circuit_and_measure_seed(self):
        # Given
        circuit = Circuit([X(0), CNOT(1, 2)])
        simulator1 = QiskitSimulator("qasm_simulator", seed=643, n_samples=100)
        simulator2 = QiskitSimulator("qasm_simulator", seed=643, n_samples=100)

        # When
        measurements1 = simulator1.run_circuit_and_measure(circuit)
        measurements2 = simulator2.run_circuit_and_measure(circuit)

        # Then
        for (meas1, meas2) in zip(measurements1.bitstrings, measurements2.bitstrings):
            assert meas1 == meas2

    def test_get_wavefunction_seed(self):
        # Given
        circuit = Circuit([X(0), CNOT(1, 2)])
        simulator1 = QiskitSimulator("statevector_simulator", seed=643)
        simulator2 = QiskitSimulator("statevector_simulator", seed=643)

        # When
        wavefunction1 = simulator1.get_wavefunction(circuit)
        wavefunction2 = simulator2.get_wavefunction(circuit)

        # Then
        for (ampl1, ampl2) in zip(wavefunction1.amplitudes, wavefunction2.amplitudes):
            assert ampl1 == ampl2


class TestQiskitSimulatorGates(QuantumSimulatorGatesTest):
    gates_to_exclude = ["RH", "XY"]
    pass
