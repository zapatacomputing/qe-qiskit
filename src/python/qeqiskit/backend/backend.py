from qiskit import execute, QuantumRegister
from qiskit.providers.ibmq import IBMQ
from qiskit.ignis.mitigation.measurement import (
    complete_meas_cal,
    CompleteMeasFitter,
)
from qiskit.providers.ibmq.exceptions import IBMQAccountError
from openfermion.ops import IsingOperator
from zquantum.core.openfermion import change_operator_type
from zquantum.core.interfaces.backend import QuantumBackend
from zquantum.core.measurement import (
    expectation_values_to_real,
    Measurements,
)


class QiskitBackend(QuantumBackend):
    def __init__(
        self,
        device_name,
        n_samples=None,
        hub="ibm-q",
        group="open",
        project="main",
        api_token=None,
        batch_size=75,
        readout_correction=False,
        optimization_level=0,
        **kwargs
    ):
        """Get a qiskit QPU that adheres to the
        zquantum.core.interfaces.backend.QuantumBackend

        Args:
            device_name (string): the name of the device
            n_samples (int): the number of samples to use when running the device
            hub (string): IBMQ hub
            group (string): IBMQ group
            project (string): IBMQ project
            api_token (string): IBMQ Api Token
            readout_correction (bool): indication of whether or not to use basic readout correction
            optimization_level (int): optimization level for the default qiskit transpiler (0, 1, 2, or 3)

        Returns:
            qeqiskit.backend.QiskitBackend
        """
        self.number_of_circuits_run = 0
        self.number_of_jobs_run = 0
        self.device_name = device_name
        self.n_samples = n_samples
        self.batch_size = batch_size

        if api_token is not None:
            try:
                IBMQ.enable_account(api_token)
            except IBMQAccountError as e:
                if (
                    e.message
                    != "An IBM Quantum Experience account is already in use for the session."
                ):
                    raise RuntimeError(e)

        provider = IBMQ.get_provider(hub=hub, group=group, project=project)
        self.device = provider.get_backend(name=self.device_name)

        self.readout_correction = readout_correction
        self.readout_correction_filter = None
        self.optimization_level = optimization_level

    def run_circuit_and_measure(self, circuit, **kwargs):
        """Run a circuit and measure a certain number of bitstrings. Note: the
        number of bitstrings measured is derived from self.n_samples

        Args:
            circuit (zquantum.core.circuit.Circuit): the circuit to prepare the state

        Returns:
            a list of bitstrings (a list of tuples)
        """
        num_qubits = len(circuit.qubits)

        ibmq_circuit = circuit.to_qiskit()
        ibmq_circuit.barrier(range(num_qubits))
        ibmq_circuit.measure(range(num_qubits), range(num_qubits))
        self.number_of_circuits_run += 1
        self.number_of_jobs_run += 1

        # Run job on device and get counts
        raw_counts = (
            execute(
                ibmq_circuit,
                self.device,
                shots=self.n_samples,
                optimization_level=self.optimization_level,
            )
            .result()
            .get_counts()
        )

        if self.readout_correction:
            raw_counts = self.apply_readout_correction(raw_counts, kwargs)

        # qiskit counts object maps bitstrings in reversed order to ints, so we must flip the bitstrings
        reversed_counts = {}
        for bitstring in raw_counts.keys():
            reversed_counts[bitstring[::-1]] = int(raw_counts[bitstring])
        measurements = Measurements.from_counts(reversed_counts)

        return measurements

    def run_circuitset_and_measure(self, circuitset, **kwargs):
        """Run a set of circuits and measure a certain number of bitstrings.
        Note: the number of bitstrings measured is derived from self.n_samples

        Args:
            circuitset (List[zquantum.core.circuit.Circuit]): the circuits to run

        Returns:
            a list of lists of bitstrings (a list of lists of tuples)
        """
        ibmq_circuitset = []
        self.number_of_circuits_run += len(circuit_set)
        for circuit in circuitset:
            num_qubits = len(circuit.qubits)

            ibmq_circuit = circuit.to_qiskit()
            ibmq_circuit.barrier(range(num_qubits))
            ibmq_circuit.measure(range(num_qubits), range(num_qubits))

            ibmq_circuitset.append(ibmq_circuit)

        # Run job on device and get counts
        experiments = []
        while len(experiments) * self.batch_size < len(circuitset):
            experiments.append(
                [
                    ibmq_circuitset[i]
                    for i in range(
                        len(experiments) * self.batch_size,
                        min(
                            len(experiments) * self.batch_size + self.batch_size,
                            len(circuitset),
                        ),
                    )
                ]
            )

        jobs = [
            execute(
                experiment,
                self.device,
                shots=self.n_samples,
                optimization_level=self.optimization_level,
            )
            for experiment in experiments
        ]

        measurements_set = []
        for i, ibmq_circuit in enumerate(ibmq_circuitset):
            job = jobs[int(i / self.batch_size)]
            circuit_counts = job.result().get_counts(ibmq_circuit)
            self.number_of_jobs_run += 1

            if self.readout_correction:
                circuit_counts = self.apply_readout_correction(circuit_counts, kwargs)

            # qiskit counts object maps bitstrings in reversed order to ints, so we must flip the bitstrings
            reversed_counts = {}
            for bitstring in circuit_counts.keys():
                reversed_counts[bitstring[::-1]] = int(circuit_counts[bitstring])

            measurements = Measurements.from_counts(reversed_counts)
            measurements_set.append(measurements)

        return measurements_set

    def get_expectation_values(self, circuit, operator, **kwargs):
        """Run a circuit and measure the expectation values with respect to a
        given operator. Note: the number of bitstrings measured is derived
        from self.n_samples - if self.n_samples = None, then this will use
        self.get_exact_expectation_values

        Args:
            circuit (zquantum.core.circuit.Circuit): the circuit to prepare the state
            operator (openfermion.ops.IsingOperator or openfermion.ops.QubitOperator): the operator to measure
        Returns:
            zquantum.core.measurement.ExpectationValues: the expectation values of each term in the operator
        """
        operator = change_operator_type(operator, IsingOperator)
        measurements = self.run_circuit_and_measure(circuit)
        expectation_values = measurements.get_expectation_values(operator)

        expectation_values = expectation_values_to_real(expectation_values)
        return expectation_values

    def get_expectation_values_for_circuitset(self, circuitset, operator, **kwargs):
        """Run a set of circuits and measure the expectation values with respect to a
        given operator.

        Args:
            circuitset (list of zquantum.core.circuit.Circuit objects): the circuits to prepare the states
            operator (openfermion.ops.IsingOperator or openfermion.ops.QubitOperator): the operator to measure
        Returns:
            list of zquantum.core.measurement.ExpectationValues objects: a list of the expectation values of each
                term in the operator with respect to the various state preparation circuits
        """
        operator = change_operator_type(operator, IsingOperator)
        measurements_set = self.run_circuitset_and_measure(circuitset)

        expectation_values_set = []
        for measurements in measurements_set:
            expectation_values = measurements.get_expectation_values(operator)
            expectation_values = expectation_values_to_real(expectation_values)
            expectation_values_set.append(expectation_values)

        return expectation_values_set

    def apply_readout_correction(self, counts, qubit_list=None, **kwargs):
        if self.readout_correction_filter is None:

            for key in counts.keys():
                num_qubits = len(key)
                break

            if qubit_list is None or qubit_list == {}:
                qubit_list = [i for i in range(num_qubits)]

            qr = QuantumRegister(num_qubits)
            meas_cals, state_labels = complete_meas_cal(qubit_list=qubit_list, qr=qr)

            # Execute the calibration circuits
            job = execute(meas_cals, self.device, shots=self.n_samples)
            cal_results = job.result()

            # Make a calibration matrix
            meas_fitter = CompleteMeasFitter(cal_results, state_labels)
            # Create a measurement filter from the calibration matrix
            self.readout_correction_filter = meas_fitter.filter

        mitigated_counts = self.readout_correction_filter.apply(counts)
        return mitigated_counts
