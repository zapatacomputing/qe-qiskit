import unittest
import numpy as np
import os
from pyquil import Program
from pyquil.gates import H, CNOT, RX, CZ, X
from openfermion.ops import QubitOperator
import qiskit.providers.aer.noise as AerNoise

from zquantum.core.circuit import Circuit
from zquantum.core.interfaces.backend_test import QuantumSimulatorTests
from zquantum.core.measurement import ExpectationValues
from ..simulator import QiskitSimulator
from ..noise import get_qiskit_noise_model


class TestQiskitSimulator(unittest.TestCase, QuantumSimulatorTests):
    def setUp(self):
        self.ibmq_api_token = os.getenv("ZAPATA_IBMQ_API_TOKEN")
        self.wf_simulator = QiskitSimulator("statevector_simulator")
        self.sampling_simulators = [QiskitSimulator("qasm_simulator")]
        noise_model, connectivity = get_qiskit_noise_model(
            "ibmqx2", api_token=self.ibmq_api_token
        )
        self.noisy_simulators = [
            QiskitSimulator(
                "qasm_simulator",
                n_samples=1000,
                noise_model=noise_model,
                device_connectivity=connectivity,
                optimization_level=0,
            ),
        ]
        self.all_simulators = (
            [self.wf_simulator] + self.sampling_simulators + self.noisy_simulators
        )

        # Inherited tests
        # NOTE: With our current version of qiskit, the statevector_simulator has a bug which won't return
        #       the correct number of measurements
        self.backends = self.sampling_simulators
        self.wf_simulators = [self.wf_simulator]

    def test_run_circuitset_and_measure(self):
        for simulator in self.sampling_simulators:
            # Given
            circuit = Circuit(Program(X(0), CNOT(1, 2)))
            # When
            simulator.n_samples = 100
            measurements_set = simulator.run_circuitset_and_measure([circuit])
            # Then
            self.assertEqual(len(measurements_set), 1)
            for measurements in measurements_set:
                self.assertEqual(len(measurements.bitstrings), 100)
                for measurement in measurements.bitstrings:
                    self.assertEqual(measurement, (1, 0, 0))

            # Given
            circuit = Circuit(Program(X(0), CNOT(1, 2)))
            # When
            simulator.n_samples = 100
            measurements_set = simulator.run_circuitset_and_measure([circuit] * 100)
            # Then
            self.assertEqual(len(measurements_set), 100)
            for measurements in measurements_set:
                self.assertEqual(len(measurements.bitstrings), 100)
                for measurement in measurements.bitstrings:
                    self.assertEqual(measurement, (1, 0, 0))

    def test_setup_basic_simulators(self):
        simulator = QiskitSimulator("qasm_simulator")
        self.assertIsInstance(simulator, QiskitSimulator)
        self.assertEqual(simulator.device_name, "qasm_simulator")
        self.assertEqual(simulator.n_samples, None)
        self.assertEqual(simulator.noise_model, None)
        self.assertEqual(simulator.device_connectivity, None)
        self.assertEqual(simulator.basis_gates, None)

        simulator = QiskitSimulator("statevector_simulator")
        self.assertIsInstance(simulator, QiskitSimulator)
        self.assertEqual(simulator.device_name, "statevector_simulator")
        self.assertEqual(simulator.n_samples, None)
        self.assertEqual(simulator.noise_model, None)
        self.assertEqual(simulator.device_connectivity, None)
        self.assertEqual(simulator.basis_gates, None)

    def test_simulator_that_does_not_exist(self):
        # Given/When/Then
        self.assertRaises(
            RuntimeError, lambda: QiskitSimulator("DEVICE DOES NOT EXIST")
        )

    def test_expectation_value_with_noisy_simulator(self):
        for simulator in self.noisy_simulators:
            # Given
            # Initialize in |1> state
            circuit = Circuit(Program(X(0)))
            # Flip qubit an even number of times to remain in the |1> state, but allow decoherence to take effect
            circuit += Circuit(Program([X(0) for _ in range(10)]))
            qubit_operator = QubitOperator("Z0")
            simulator.n_samples = 8192
            # When
            expectation_values_10_gates = simulator.get_expectation_values(
                circuit, qubit_operator
            )
            # Then
            self.assertIsInstance(expectation_values_10_gates, ExpectationValues)
            self.assertEqual(len(expectation_values_10_gates.values), 1)
            self.assertGreater(expectation_values_10_gates.values[0], -1)
            self.assertLess(expectation_values_10_gates.values[0], 0.0)
            self.assertIsInstance(simulator, QiskitSimulator)
            self.assertEqual(simulator.device_name, "qasm_simulator")
            self.assertEqual(simulator.n_samples, 8192)
            self.assertIsInstance(simulator.noise_model, AerNoise.NoiseModel)
            self.assertNotEqual(simulator.device_connectivity, None)
            self.assertNotEqual(simulator.basis_gates, None)

            # Given
            # Initialize in |1> state
            circuit = Circuit(Program(X(0)))
            # Flip qubit an even number of times to remain in the |1> state, but allow decoherence to take effect
            circuit += Circuit(Program([X(0) for _ in range(50)]))
            qubit_operator = QubitOperator("Z0")
            simulator.n_samples = 8192
            # When
            expectation_values_50_gates = simulator.get_expectation_values(
                circuit, qubit_operator
            )
            # Then
            self.assertIsInstance(expectation_values_50_gates, ExpectationValues)
            self.assertEqual(len(expectation_values_50_gates.values), 1)
            self.assertGreater(expectation_values_50_gates.values[0], -1)
            self.assertLess(expectation_values_50_gates.values[0], 0.0)
            self.assertGreater(
                expectation_values_50_gates.values[0],
                expectation_values_10_gates.values[0],
            )
            self.assertIsInstance(simulator, QiskitSimulator)
            self.assertEqual(simulator.device_name, "qasm_simulator")
            self.assertEqual(simulator.n_samples, 8192)
            self.assertIsInstance(simulator.noise_model, AerNoise.NoiseModel)
            self.assertNotEqual(simulator.device_connectivity, None)
            self.assertNotEqual(simulator.basis_gates, None)

    def test_optimization_level_of_transpiler(self):
        # Given
        noise_model, connectivity = get_qiskit_noise_model(
            "ibmqx2", api_token=self.ibmq_api_token
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
        circuit = Circuit(Program(X(0)))
        # Flip qubit an even number of times to remain in the |1> state, but allow decoherence to take effect
        circuit += Circuit(Program([X(0) for _ in range(50)]))

        # When
        expectation_values_no_compilation = simulator.get_expectation_values(
            circuit, qubit_operator
        )

        simulator.optimization_level = 3
        expectation_values_full_compilation = simulator.get_expectation_values(
            circuit, qubit_operator
        )

        # Then
        self.assertLess(
            expectation_values_full_compilation.values[0],
            expectation_values_no_compilation.values[0],
        )
