import pytest
import os
from qiskit import IBMQ
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit import execute

from zquantum.core.circuit import Circuit, Gate, Circuit, Qubit
from zquantum.core.interfaces.backend_test import QuantumBackendTests
from .backend import QiskitBackend

import math


@pytest.fixture(
    params=[
        {
            "device_name": "ibmq_qasm_simulator",
            "n_samples": 1,
            "api_token": os.getenv("ZAPATA_IBMQ_API_TOKEN"),
        },
    ]
)
def backend(request):
    return QiskitBackend(**request.param)


class TestQiskitBackend(QuantumBackendTests):
    def x_cnot_circuit(self):
        qubits = [Qubit(i) for i in range(3)]
        X = Gate("X", qubits=[Qubit(0)])
        CNOT = Gate("CNOT", qubits=[Qubit(1), Qubit(2)])
        circuit = Circuit()
        circuit.qubits = qubits
        circuit.gates = [X, CNOT]
        return circuit

    def test_transform_circuitset_to_ibmq_experiments(self, backend):
        circuit = self.x_cnot_circuit()
        circuitset = (circuit,) * 2
        backend.n_samples = backend.max_shots + 1

        (
            experiments,
            n_samples_for_experiments,
            multiplicities,
        ) = backend.transform_circuitset_to_ibmq_experiments(circuitset)
        assert multiplicities == [2, 2]
        assert n_samples_for_experiments == [
            backend.n_samples - 1,
            1,
            backend.n_samples - 1,
            1,
        ]
        assert len(set([circuit.name for circuit in experiments])) == 4

    def test_batch_experiments(self, backend):
        circuit = self.x_cnot_circuit()
        n_circuits = backend.batch_size + 1
        experiments = (circuit.to_qiskit(),) * n_circuits
        n_samples_for_ibmq_circuits = (10,) * n_circuits
        batches, n_samples_for_batches = backend.batch_experiments(
            experiments, n_samples_for_ibmq_circuits
        )
        assert len(batches) == 2
        assert len(batches[0]) == backend.batch_size
        assert len(batches[1]) == 1
        assert n_samples_for_batches == [10, 10]

    def test_aggregate_measurements(self, backend):
        circuit = self.x_cnot_circuit().to_qiskit()
        circuit.barrier(range(3))
        circuit.measure(range(3), range(3))
        batches = [
            [circuit.copy("circuit1"), circuit.copy("circuit2")],
            [circuit.copy("circuit3"), circuit.copy("circuit4")],
        ]
        multiplicities = [3, 1]
        jobs = [
            execute(
                batch,
                backend.device,
                shots=10,
            )
            for batch in batches
        ]

        circuit = self.x_cnot_circuit()
        measurements_set = backend.aggregregate_measurements(
            jobs,
            batches,
            multiplicities,
        )

        assert (
            measurements_set[0].bitstrings
            == [
                (1, 0, 0),
            ]
            * 30
        )
        assert (
            measurements_set[1].bitstrings
            == [
                (1, 0, 0),
            ]
            * 10
        )
        assert len(measurements_set) == 2

    def test_run_circuitset_and_measure(self, backend):
        # Given
        num_circuits = 10
        circuit = self.x_cnot_circuit()
        n_samples = 100
        # When
        backend.n_samples = n_samples
        measurements_set = backend.run_circuitset_and_measure([circuit] * num_circuits)
        # Then
        assert len(measurements_set) == num_circuits
        for measurements in measurements_set:
            assert len(measurements.bitstrings) == n_samples

            # Then (since SPAM error could result in unexpected bitstrings, we make sure the most common bitstring is
            #   the one we expect)
            counts = measurements.get_counts()
            assert max(counts, key=counts.get) == "100"

    def test_run_circuitset_and_measure_split_circuits_and_jobs(self, backend):
        # Given
        num_circuits = 200
        circuit = self.x_cnot_circuit()
        n_samples = 8193

        # Verify that we are actually going to need to split circuits
        assert n_samples > backend.max_shots

        # Verify that we are actually going to need multiple batches
        assert (
            num_circuits * math.ceil(n_samples / backend.max_shots) > backend.batch_size
        )

        # When
        backend.n_samples = n_samples
        measurements_set = backend.run_circuitset_and_measure([circuit] * num_circuits)
        # Then
        assert len(measurements_set) == num_circuits
        for measurements in measurements_set:
            assert len(measurements.bitstrings) == n_samples or len(
                measurements.bitstrings
            ) == backend.max_shots * math.ceil(n_samples / backend.max_shots)

            # Then (since SPAM error could result in unexpected bitstrings, we make sure the most common bitstring is
            #   the one we expect)
            counts = measurements.get_counts()
            assert max(counts, key=counts.get) == "100"

    def test_readout_correction_works_run_circuit_and_measure(self):
        # Given
        ibmq_api_token = os.getenv("ZAPATA_IBMQ_API_TOKEN")
        backend = QiskitBackend(
            device_name="ibmq_qasm_simulator",
            n_samples=1000,
            api_token=ibmq_api_token,
            readout_correction=True,
        )
        circuit = self.x_cnot_circuit()

        # When
        backend.run_circuit_and_measure(circuit)

        # Then
        assert backend.readout_correction
        assert backend.readout_correction_filter is not None

    def test_readout_correction_works_run_circuitset_and_measure(self):
        # Given
        ibmq_api_token = os.getenv("ZAPATA_IBMQ_API_TOKEN")
        backend = QiskitBackend(
            device_name="ibmq_qasm_simulator",
            n_samples=1000,
            api_token=ibmq_api_token,
            readout_correction=True,
        )
        circuit = self.x_cnot_circuit()

        # When
        backend.run_circuitset_and_measure([circuit] * 10)

        # Then
        assert backend.readout_correction
        assert backend.readout_correction_filter is not None

    def test_device_that_does_not_exist(self):
        # Given/When/Then
        with pytest.raises(QiskitBackendNotFoundError):
            QiskitBackend("DEVICE DOES NOT EXIST")
