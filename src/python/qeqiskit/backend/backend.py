import math
import time
from typing import List, Optional, Tuple

from qeqiskit.conversions import export_to_qiskit
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, execute
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter, complete_meas_cal
from qiskit.providers.ibmq import IBMQ
from qiskit.providers.ibmq.exceptions import IBMQAccountError, IBMQBackendJobLimitError
from qiskit.providers.ibmq.job import IBMQJob
from qiskit.result import Counts
from zquantum.core.circuits import Circuit
from zquantum.core.interfaces.backend import QuantumBackend
from zquantum.core.measurement import Measurements


class QiskitBackend(QuantumBackend):
    def __init__(
        self,
        device_name: str,
        hub: Optional[str] = "ibm-q",
        group: Optional[str] = "open",
        project: Optional[str] = "main",
        api_token: Optional[str] = None,
        readout_correction: Optional[bool] = False,
        optimization_level: Optional[int] = 0,
        retry_delay_seconds: Optional[int] = 60,
        retry_timeout_seconds: Optional[int] = 86400,
        n_samples_for_readout_calibration: Optional[int] = None,
        **kwargs,
    ):
        """Get a qiskit QPU that adheres to the
        zquantum.core.interfaces.backend.QuantumBackend

        qiskit currently offers 2 types of qasm simulators:
        1. qasm_simulator - a local simulator that is depreciated.
        2. IBMQ_qasm_simulator - a remote simulator.
        All implementation of qasm_simulator have been removed since it's depreciation
        but IBMQ_qasm_simulator is still tested by this module.

        Args:
            device_name: the name of the device
            hub: IBMQ hub
            group: IBMQ group
            project: IBMQ project
            api_token: IBMQ Api Token
            readout_correction: flag of whether or not to use basic readout correction
            optimization_level: optimization level for the default qiskit transpiler (0,
                1, 2, or 3).
            retry_delay_seconds: Number of seconds to wait to resubmit a job when
                backend job limit is reached.
            retry_timeout_seconds: Number of seconds to wait
        """
        super().__init__()
        self.device_name = device_name

        if api_token is not None:
            try:
                IBMQ.enable_account(api_token)
            except IBMQAccountError as e:
                if e.message != (
                    "An IBM Quantum Experience account is already in use for the session."  # noqa: E501
                ):
                    raise RuntimeError(e)

        provider = IBMQ.get_provider(hub=hub, group=group, project=project)
        self.device = provider.get_backend(name=self.device_name)
        self.max_shots = self.device.configuration().max_shots
        self.batch_size = self.device.configuration().max_experiments
        self.supports_batching = True
        self.readout_correction = readout_correction
        self.readout_correction_filter = None
        self.optimization_level = optimization_level
        self.basis_gates = kwargs.get(
            "basis_gates", self.device.configuration().basis_gates
        )
        self.retry_delay_seconds = retry_delay_seconds
        self.retry_timeout_seconds = retry_timeout_seconds
        self.n_samples_for_readout_calibration = n_samples_for_readout_calibration

    def run_circuit_and_measure(self, circuit: Circuit, n_samples: int) -> Measurements:
        """Run a circuit and measure a certain number of bitstrings. Note: the
        number of bitstrings measured is derived from self.n_samples

        Args:
            circuit: the circuit to prepare the state
            n_samples: The number of samples to collect.
        """
        assert isinstance(n_samples, int) and n_samples > 0
        return self.run_circuitset_and_measure([circuit], [n_samples])[0]

    def transform_circuitset_to_ibmq_experiments(
        self,
        circuitset: List[Circuit],
        n_samples: List[int],
    ) -> Tuple[List[QuantumCircuit], List[int], List[int]]:
        """Convert circuits to qiskit and duplicate those whose measurement
        count exceeds the maximum allowed by the backend.

        Args:
            circuitset: The circuits to be executed.
            n_samples: A list of the number of samples to be collected for each
                circuit.

        Returns:
            Tuple containing:
            - The expanded list of circuits, converted to qiskit and each
              assigned a unique name.
            - An array indicating how many duplicates there are for each of the
              original circuits.
        """
        ibmq_circuitset = []
        n_samples_for_ibmq_circuits = []
        multiplicities = []

        for n_samples_for_circuit, circuit in zip(n_samples, circuitset):
            ibmq_circuit = export_to_qiskit(circuit)
            full_qubit_indices = list(range(circuit.n_qubits))
            ibmq_circuit.barrier(full_qubit_indices)
            ibmq_circuit.add_register(ClassicalRegister(size=circuit.n_qubits))
            ibmq_circuit.measure(full_qubit_indices, full_qubit_indices)

            multiplicities.append(math.ceil(n_samples_for_circuit / self.max_shots))

            for i in range(multiplicities[-1]):
                ibmq_circuitset.append(ibmq_circuit.copy(f"{ibmq_circuit.name}_{i}"))

            for i in range(math.floor(n_samples_for_circuit / self.max_shots)):
                n_samples_for_ibmq_circuits.append(self.max_shots)

            if n_samples_for_circuit % self.max_shots != 0:
                n_samples_for_ibmq_circuits.append(
                    n_samples_for_circuit % self.max_shots
                )
        return ibmq_circuitset, n_samples_for_ibmq_circuits, multiplicities

    def batch_experiments(
        self,
        experiments: List[QuantumCircuit],
        n_samples_for_ibmq_circuits: List[int],
    ) -> Tuple[List[List[QuantumCircuit]], List[int]]:
        """Batch a set of experiments (circuits to be executed) into groups
        whose size is no greater than the maximum allowed by the backend.

        Args:
            experiments: The circuits to be executed.
            n_samples_for_ibmq_circuits: The number of samples desired for each
                experiment.

        Returns:
            A tuple containing:
            - A list of batches, where each batch is a list of experiments.
            - An array containing the number of measurements that must be
              performed for each batch so that each experiment receives at least
              as many samples as specified by n_samples_for_ibmq_circuits.
        """

        batches: List = []
        n_samples_for_batches = []
        while len(batches) * self.batch_size < len(experiments):
            batches.append(
                [
                    experiments[i]
                    for i in range(
                        len(batches) * self.batch_size,
                        min(
                            len(batches) * self.batch_size + self.batch_size,
                            len(experiments),
                        ),
                    )
                ]
            )

            n_samples_for_batches.append(
                max(
                    [
                        n_samples_for_ibmq_circuits[i]
                        for i in range(
                            len(batches) * self.batch_size - self.batch_size,
                            min(
                                len(batches) * self.batch_size,
                                len(experiments),
                            ),
                        )
                    ]
                )
            )

        return batches, n_samples_for_batches

    def aggregregate_measurements(
        self,
        jobs: List[IBMQJob],
        batches: List[List[QuantumCircuit]],
        multiplicities: List[int],
    ) -> List[Measurements]:
        """Combine samples from a circuit set that has been expanded and batched
        to obtain a set of measurements for each of the original circuits. Also
        applies readout correction after combining.

        Args:
            jobs: The submitted IBMQ jobs.
            batches: The batches of experiments submitted.
            multiplicities: The number of copies of each of the original
                circuits.

        Returns:
            A list of list of measurements, where each list of measurements
            corresponds to one of the circuits of the original (unexpanded)
            circuit set.
        """
        ibmq_circuit_counts_set = []
        for job, batch in zip(jobs, batches):
            for experiment in batch:
                ibmq_circuit_counts_set.append(job.result().get_counts(experiment))

        measurements_set = []
        ibmq_circuit_index = 0
        for multiplicity in multiplicities:
            combined_counts = Counts({})
            for i in range(multiplicity):
                for bitstring, counts in ibmq_circuit_counts_set[
                    ibmq_circuit_index
                ].items():
                    combined_counts[bitstring] = (
                        combined_counts.get(bitstring, 0) + counts
                    )
                ibmq_circuit_index += 1

            if self.readout_correction:
                combined_counts = self._apply_readout_correction(combined_counts)

            # qiskit counts object maps bitstrings in reversed order to ints, so we must
            # flip the bitstrings
            reversed_counts = {}
            for bitstring in combined_counts.keys():
                reversed_counts[bitstring[::-1]] = int(combined_counts[bitstring])

            measurements = Measurements.from_counts(reversed_counts)
            measurements_set.append(measurements)

        return measurements_set

    def run_circuitset_and_measure(
        self,
        circuits: List[Circuit],
        n_samples: List[int],
    ) -> List[Measurements]:
        """Run a set of circuits and measure a certain number of bitstrings.
        Note: the number of bitstrings measured is derived from self.n_samples

        Args:
            circuitset: the circuits to run
            n_samples: The number of shots to perform on each circuit. If
                None, then self.n_samples shots are performed for each circuit.

        Returns:
            A list of Measurements objects containing the observed bitstrings.
        """

        (
            experiments,
            n_samples_for_experiments,
            multiplicities,
        ) = self.transform_circuitset_to_ibmq_experiments(circuits, n_samples)
        batches, n_samples_for_batches = self.batch_experiments(
            experiments, n_samples_for_experiments
        )

        jobs = [
            self.execute_with_retries(batch, n_samples)
            for n_samples, batch in zip(n_samples_for_batches, batches)
        ]

        self.number_of_circuits_run += len(circuits)
        self.number_of_jobs_run += len(batches)

        return self.aggregregate_measurements(jobs, batches, multiplicities)

    def execute_with_retries(
        self, batch: List[QuantumCircuit], n_samples: int
    ) -> IBMQJob:
        """Execute a job, resubmitting if the the backend job limit has been
        reached.

        The number of seconds between retries is specified by
        self.retry_delay_seconds. If self.retry_timeout_seconds is defined, then
        an exception will be raised if the submission does not succeed in the
        specified number of seconds.

        Args:
            batch: The batch of qiskit ircuits to be executed.
            n_samples: The number of shots to perform on each circuit.

        Returns:
            The qiskit representation of the submitted job.
        """

        start_time = time.time()
        while True:
            try:
                job = execute(
                    batch,
                    self.device,
                    shots=n_samples,
                    basis_gates=self.basis_gates,
                    optimization_level=self.optimization_level,
                    backend_properties=self.device.properties(),
                )
                return job
            except IBMQBackendJobLimitError:
                if self.retry_timeout_seconds is not None:
                    elapsed_time_seconds = time.time() - start_time
                    if elapsed_time_seconds > self.retry_timeout_seconds:
                        raise RuntimeError(
                            f"Failed to submit job in {elapsed_time_seconds}s due to "
                            "backend job limit."
                        )
                print(f"Job limit reached. Retrying in {self.retry_delay_seconds}s.")
                time.sleep(self.retry_delay_seconds)  # type: ignore

    def _apply_readout_correction(self, counts, qubit_list=None):
        if self.readout_correction_filter is None:

            for key in counts.keys():
                num_qubits = len(key)
                break

            if qubit_list is None or qubit_list == {}:
                qubit_list = [i for i in range(num_qubits)]

            qr = QuantumRegister(num_qubits)
            meas_cals, state_labels = complete_meas_cal(qubit_list=qubit_list, qr=qr)

            # Execute the calibration circuits
            job = self.execute_with_retries(
                meas_cals, self.n_samples_for_readout_calibration
            )
            cal_results = job.result()

            # Make a calibration matrix
            meas_fitter = CompleteMeasFitter(cal_results, state_labels)
            # Create a measurement filter from the calibration matrix
            self.readout_correction_filter = meas_fitter.filter

        mitigated_counts = self.readout_correction_filter.apply(counts)
        return mitigated_counts
