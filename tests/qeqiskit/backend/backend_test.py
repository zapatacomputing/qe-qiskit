import math
import os
import pickle

import pytest
from qeqiskit.backend import QiskitBackend
from qeqiskit.conversions import export_to_qiskit
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from zquantum.core.circuits import CNOT, Circuit, X
from zquantum.core.interfaces.backend_test import QuantumBackendTests
from zquantum.core.measurement import Measurements


@pytest.fixture(
    params=[
        {
            "device_name": "ibmq_qasm_simulator",
            "api_token": os.getenv("ZAPATA_IBMQ_API_TOKEN"),
            "retry_delay_seconds": 1,
        },
    ]
)
def backend(request):
    return QiskitBackend(**request.param)


@pytest.fixture(
    params=[
        {
            "device_name": "ibmq_qasm_simulator",
            "api_token": os.getenv("ZAPATA_IBMQ_API_TOKEN"),
            "readout_correction": True,
            "n_samples_for_readout_calibration": 1,
            "retry_delay_seconds": 1,
        },
    ]
)
def backend_with_readout_correction(request):
    return QiskitBackend(**request.param)


class TestQiskitBackend(QuantumBackendTests):
    def x_cnot_circuit(self):
        return Circuit([X(0), CNOT(1, 2)])

    def x_circuit(self):
        return Circuit([X(0)])

    def test_transform_circuitset_to_ibmq_experiments(self, backend):
        circuit = self.x_cnot_circuit()
        circuitset = (circuit,) * 2
        n_samples = [backend.max_shots + 1] * 2

        (
            experiments,
            n_samples_for_experiments,
            multiplicities,
        ) = backend.transform_circuitset_to_ibmq_experiments(circuitset, n_samples)
        assert multiplicities == [2, 2]
        assert n_samples_for_experiments == [
            backend.max_shots,
            1,
            backend.max_shots,
            1,
        ]
        assert len(set([circuit.name for circuit in experiments])) == 4

    def test_batch_experiments(self, backend):
        circuit = self.x_cnot_circuit()
        n_circuits = backend.batch_size + 1
        experiments = (export_to_qiskit(circuit),) * n_circuits
        n_samples_for_ibmq_circuits = (10,) * n_circuits
        batches, n_samples_for_batches = backend.batch_experiments(
            experiments, n_samples_for_ibmq_circuits
        )
        assert len(batches) == 2
        assert len(batches[0]) == backend.batch_size
        assert len(batches[1]) == 1
        assert n_samples_for_batches == [10, 10]

    def test_aggregate_measurements(self, backend):
        """
        Get pickle file for output of following code

        circuit = export_to_qiskit(self.x_cnot_circuit())
        circuit.barrier(range(3))
        circuit.add_register(qiskit.ClassicalRegister(3))
        circuit.measure(range(3), range(3))
        batches = [
            [circuit.copy("circuit1"), circuit.copy("circuit2")],
            [circuit.copy("circuit3"), circuit.copy("circuit4")],
        ]
        multiplicities = [3, 1]
        jobs = [
            backend.execute_with_retries(
                batch,
                10,
            )
            for batch in batches
        ]
        """

        FileHandler = open(
            "tests/qeqiskit/backend/test_aggregate_measurements.pickle", "rb"
        )
        jobs = pickle.load(FileHandler)
        batches = pickle.load(FileHandler)
        multiplicities = pickle.load(FileHandler)

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
        circuit = self.x_circuit()
        n_samples = 100
        # When
        measurements_set = backend.run_circuitset_and_measure(
            [circuit] * num_circuits, [n_samples] * num_circuits
        )
        # Then
        assert len(measurements_set) == num_circuits
        for measurements in measurements_set:
            assert len(measurements.bitstrings) == n_samples

            # Then (since SPAM error could result in unexpected bitstrings, we make sure
            # the most common bitstring is the one we expect)
            counts = measurements.get_counts()
            assert max(counts, key=counts.get) == "1"

    def test_execute_with_retries(self, backend):
        # This test has a race condition where the IBMQ server might finish
        # executing the first job before the last one is submitted. The test
        # will still pass in the case, but will not actually perform a retry.
        # We can address in the future by using a mock provider.

        # Given
        circuit = export_to_qiskit(self.x_circuit())
        n_samples = 10
        num_jobs = backend.device.job_limit().maximum_jobs + 1

        # When
        jobs = [
            backend.execute_with_retries([circuit], n_samples) for _ in range(num_jobs)
        ]

        # Then

        # The correct number of jobs were submitted
        assert len(jobs) == num_jobs

        # Each job has a unique ID
        assert len(set([job.job_id() for job in jobs])) == num_jobs

    @pytest.mark.skip(reason="Test has long runtime and just fails anyway.")
    def test_execute_with_retries_timeout(self, backend):
        # This test has a race condition where the IBMQ server might finish
        # executing the first job before the last one is submitted, causing the
        # test to fail. We can address this in the future using a mock provider.

        # Given
        circuit = export_to_qiskit(self.x_cnot_circuit())
        n_samples = 10
        backend.retry_timeout_seconds = 0
        num_jobs = backend.device.job_limit().maximum_jobs + 1

        # Then
        with pytest.raises(RuntimeError):

            # When
            for _ in range(num_jobs):
                backend.execute_with_retries([circuit], n_samples)

    def test_run_circuitset_and_measure_split_circuits_and_jobs(self, backend):
        # Given
        num_circuits = 2  # Minimum number of circuits to require batching
        circuit = self.x_cnot_circuit()
        n_samples = backend.max_shots + 1
        backend.batch_size = 2

        # Verify that we are actually going to need multiple batches
        assert (
            num_circuits * math.ceil(n_samples / backend.max_shots) > backend.batch_size
        )

        # When
        measurements_set = backend.run_circuitset_and_measure(
            [circuit] * num_circuits, [n_samples] * num_circuits
        )
        # Then
        assert len(measurements_set) == num_circuits
        for measurements in measurements_set:
            assert len(measurements.bitstrings) == n_samples or len(
                measurements.bitstrings
            ) == backend.max_shots * math.ceil(n_samples / backend.max_shots)

            # Then (since SPAM error could result in unexpected bitstrings, we make sure
            # the most common bitstring is the one we expect)
            counts = measurements.get_counts()
            assert max(counts, key=counts.get) == "100"

    def test_readout_correction_works_run_circuit_and_measure(
        self, backend_with_readout_correction
    ):
        # Given
        circuit = self.x_cnot_circuit()

        # When
        backend_with_readout_correction.run_circuit_and_measure(circuit, n_samples=1)

        # Then
        assert backend_with_readout_correction.readout_correction
        assert backend_with_readout_correction.readout_correction_filter is not None

    def test_readout_correction_works_run_circuitset_and_measure(
        self, backend_with_readout_correction
    ):
        # Given
        circuit = self.x_cnot_circuit()
        n_samples = 10

        # When
        backend_with_readout_correction.run_circuitset_and_measure(
            [circuit] * 2, [n_samples] * 2
        )

        # Then
        assert backend_with_readout_correction.readout_correction
        assert backend_with_readout_correction.readout_correction_filter is not None

    def test_device_that_does_not_exist(self):
        # Given/When/Then
        with pytest.raises(QiskitBackendNotFoundError):
            QiskitBackend("DEVICE DOES NOT EXIST")

    def test_run_circuitset_and_measure_n_samples(self, backend):
        # We override the base test because the qiskit integration may return
        # more samples than requested due to the fact that each circuit in a
        # batch must have the same number of measurements.

        # Given
        backend.number_of_circuits_run = 0
        backend.number_of_jobs_run = 0

        first_circuit = Circuit(
            [
                X(0),
                X(0),
                X(1),
                X(1),
                X(2),
            ]
        )

        second_circuit = Circuit(
            [
                X(0),
                X(1),
                X(2),
            ]
        )

        n_samples = [2, 3]

        # When
        measurements_set = backend.run_circuitset_and_measure(
            [first_circuit, second_circuit], n_samples
        )

        # Then (since SPAM error could result in unexpected bitstrings, we make sure the
        # most common bitstring is the one we expect)
        counts = measurements_set[0].get_counts()
        assert max(counts, key=counts.get) == "001"
        counts = measurements_set[1].get_counts()
        assert max(counts, key=counts.get) == "111"

        assert len(measurements_set[0].bitstrings) >= n_samples[0]
        assert len(measurements_set[1].bitstrings) >= n_samples[1]

        assert backend.number_of_circuits_run == 2

    @pytest.mark.parametrize("n_samples", [1, 2, 10])
    def test_run_circuit_and_measure_correct_num_measurements_attribute(
        self, backend, n_samples
    ):
        # Overriding to reduce number of samples required

        # Given
        backend.number_of_circuits_run = 0
        backend.number_of_jobs_run = 0
        circuit = self.x_cnot_circuit()

        # When
        measurements = backend.run_circuit_and_measure(circuit, n_samples)

        # Then
        assert isinstance(measurements, Measurements)
        assert len(measurements.bitstrings) == n_samples
        assert backend.number_of_circuits_run == 1
        assert backend.number_of_jobs_run == 1

    def test_run_circuit_and_measure_correct_indexing(self, backend):
        # Overriding to reduce number of samples required

        # Given
        backend.number_of_circuits_run = 0
        backend.number_of_jobs_run = 0
        circuit = self.x_cnot_circuit()
        n_samples = 2  # qiskit only runs simulators, so we can use low n_samples
        measurements = backend.run_circuit_and_measure(circuit, n_samples)

        counts = measurements.get_counts()
        assert max(counts, key=counts.get) == "100"
        assert backend.number_of_circuits_run == 1
        assert backend.number_of_jobs_run == 1
