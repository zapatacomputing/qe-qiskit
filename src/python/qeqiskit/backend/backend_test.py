import pytest
import os
from pyquil import Program
from pyquil.gates import X, CNOT
from qiskit import IBMQ
from qiskit.providers.exceptions import QiskitBackendNotFoundError

from zquantum.core.circuit import Circuit
from zquantum.core.interfaces.backend_test import QuantumBackendTests
from .backend import QiskitBackend


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
    def test_run_circuitset_and_measure(self, backend):
        # Given
        num_circuits = 10
        circuit = Circuit(Program(X(0), CNOT(1, 2)))
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

    def test_readout_correction_works_run_circuit_and_measure(self):
        # Given
        ibmq_api_token = os.getenv("ZAPATA_IBMQ_API_TOKEN")
        backend = QiskitBackend(
            device_name="ibmq_qasm_simulator",
            n_samples=1000,
            api_token=ibmq_api_token,
            readout_correction=True,
        )
        circuit = Circuit(Program(X(0), CNOT(1, 2)))

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
        circuit = Circuit(Program(X(0), CNOT(1, 2)))

        # When
        backend.run_circuitset_and_measure([circuit] * 10)

        # Then
        assert backend.readout_correction
        assert backend.readout_correction_filter is not None

    def test_device_that_does_not_exist(self):
        # Given/When/Then
        with pytest.raises(QiskitBackendNotFoundError):
            QiskitBackend("DEVICE DOES NOT EXIST")
