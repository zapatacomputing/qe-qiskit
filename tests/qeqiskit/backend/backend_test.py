################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import math
import os
from copy import deepcopy

import pytest
import qiskit
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
        {
            "device_name": "ibmq_qasm_simulator",
            "api_token": os.getenv("ZAPATA_IBMQ_API_TOKEN"),
            "readout_correction": True,
            "n_samples_for_readout_calibration": 1,
            "retry_delay_seconds": 1,
            "noise_inversion_method": "pseudo_inverse",
        },
    ],
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

        multiplicities = [3, 1]

        circuit = export_to_qiskit(self.x_cnot_circuit())
        circuit.barrier(range(3))
        circuit.add_register(qiskit.ClassicalRegister(3))
        circuit.measure(range(3), range(3))
        batches = [
            [circuit.copy("circuit1"), circuit.copy("circuit2")],
            [circuit.copy("circuit3"), circuit.copy("circuit4")],
        ]

        jobs = [
            backend.execute_with_retries(
                batch,
                10,
            )
            for batch in batches
        ]

        measurements_set = backend.aggregate_measurements(
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

    def test_execute_with_retries_timeout(self, backend):
        # This test has a race condition where the IBMQ server might finish
        # executing the first job before the last one is submitted, causing the
        # test to fail. We can address this in the future using a mock provider.

        # Given
        circuit = export_to_qiskit(self.x_cnot_circuit())
        n_samples = 10
        backend.retry_timeout_seconds = 0
        # need large number here as + 1 was not enough
        num_jobs = backend.device.job_limit().maximum_jobs + int(10e20)

        # Then
        with pytest.raises(RuntimeError):

            # When
            for _ in range(num_jobs):
                backend.execute_with_retries([circuit], n_samples)

    @pytest.mark.skip(reason="test will always succeed.")
    def test_run_circuitset_and_measure_readout_correction_retries(
        self, backend_with_readout_correction
    ):
        # This test has a race condition where the IBMQ server might finish
        # executing the first job before the last one is submitted. The test
        # will still pass in the case, but will not actually perform a retry.
        # We can address in the future by using a mock provider.

        # Given
        circuit = self.x_cnot_circuit()
        n_samples = 10
        num_circuits = (
            backend_with_readout_correction.batch_size
            * backend_with_readout_correction.device.job_limit().maximum_jobs
            + 1
        )

        # When
        measurements_set = backend_with_readout_correction.run_circuitset_and_measure(
            [circuit] * num_circuits, [n_samples] * num_circuits
        )

        # Then
        assert len(measurements_set) == num_circuits

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
        assert backend_with_readout_correction.readout_correction_filters is not None

    def test_readout_correction_for_distributed_circuit(
        self, backend_with_readout_correction
    ):
        # Given
        num_circuits = 10
        circuit = self.x_circuit() + X(5)
        n_samples = 100

        # When
        measurements_set = backend_with_readout_correction.run_circuitset_and_measure(
            [circuit] * num_circuits, [n_samples] * num_circuits
        )

        # Then
        assert backend_with_readout_correction.readout_correction
        assert (
            backend_with_readout_correction.readout_correction_filters.get(str([0, 5]))
            is not None
        )
        assert len(measurements_set) == num_circuits
        for measurements in measurements_set:
            assert len(measurements.bitstrings) == n_samples
            counts = measurements.get_counts()
            assert max(counts, key=counts.get) == "11"

    @pytest.mark.parametrize(
        "counts, active_qubits",
        [
            ({"100000000000000000001": 10}, [0, 20]),
            ({"100000000000000000100": 10}, [0, 18, 20]),
            ({"001000000000000000001": 10}, [2, 20]),
        ],
    )
    def test_subset_readout_correction(
        self, counts, active_qubits, backend_with_readout_correction
    ):
        # Given
        copied_counts = deepcopy(counts)

        # When
        mitigated_counts = backend_with_readout_correction._apply_readout_correction(
            copied_counts, active_qubits
        )

        # Then
        assert backend_with_readout_correction.readout_correction
        assert backend_with_readout_correction.readout_correction_filters.get(
            str(active_qubits)
        )
        assert copied_counts == pytest.approx(mitigated_counts, 10e-5)

    def test_subset_readout_correction_with_unspecified_active_qubits(
        self, backend_with_readout_correction
    ):
        # Given
        counts = {"11": 10}

        # When
        mitigated_counts = backend_with_readout_correction._apply_readout_correction(
            counts
        )

        # Then
        assert backend_with_readout_correction.readout_correction
        assert backend_with_readout_correction.readout_correction_filters.get(
            str([0, 1])
        )
        assert counts == pytest.approx(mitigated_counts, 10e-5)

    def test_must_define_n_samples_for_readout_calibration_for_readout_correction(
        self, backend_with_readout_correction
    ):
        # Given
        counts, active_qubits = ({"11": 10}, None)
        backend_with_readout_correction.n_samples_for_readout_calibration = None

        # When/Then
        with pytest.raises(TypeError):
            backend_with_readout_correction._apply_readout_correction(
                counts, active_qubits
            )

    def test_subset_readout_correction_for_multiple_subsets(
        self, backend_with_readout_correction
    ):
        # Given
        counts_1, active_qubits_1 = ({"100000000000000000001": 10}, [0, 20])
        counts_2, active_qubits_2 = ({"001000000000000000001": 10}, [2, 20])

        # When
        mitigated_counts_1 = backend_with_readout_correction._apply_readout_correction(
            counts_1, active_qubits_1
        )
        mitigated_counts_2 = backend_with_readout_correction._apply_readout_correction(
            counts_2, active_qubits_2
        )

        # Then
        assert backend_with_readout_correction.readout_correction
        assert backend_with_readout_correction.readout_correction_filters.get(
            str(active_qubits_1)
        )
        assert backend_with_readout_correction.readout_correction_filters.get(
            str(active_qubits_2)
        )
        assert counts_1 == pytest.approx(mitigated_counts_1, 10e-5)
        assert counts_2 == pytest.approx(mitigated_counts_2, 10e-5)

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
