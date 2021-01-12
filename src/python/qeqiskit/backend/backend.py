from qiskit import IBMQ, execute, QuantumRegister
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
    supports_batching = True

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
        super().__init__(n_samples)
        self.device_name = device_name
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
        super().run_circuit_and_measure(circuit)
        num_qubits = len(circuit.qubits)

        ibmq_circuit = circuit.to_qiskit()
        ibmq_circuit.barrier(range(num_qubits))
        ibmq_circuit.measure(range(num_qubits), range(num_qubits))

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
        super().run_circuitset_and_measure(circuitset)
        ibmq_circuitset = []
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

            if self.readout_correction:
                circuit_counts = self.apply_readout_correction(circuit_counts, kwargs)

            # qiskit counts object maps bitstrings in reversed order to ints, so we must flip the bitstrings
            reversed_counts = {}
            for bitstring in circuit_counts.keys():
                reversed_counts[bitstring[::-1]] = int(circuit_counts[bitstring])

            measurements = Measurements.from_counts(reversed_counts)
            measurements_set.append(measurements)

        return measurements_set

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
