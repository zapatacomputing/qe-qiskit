import numpy as np
from qiskit import Aer, execute
from qiskit.providers.ibmq import IBMQ
from qiskit.providers.ibmq.exceptions import IBMQAccountError
from qiskit.transpiler import CouplingMap
from pyquil.wavefunction import Wavefunction
from openfermion.ops import IsingOperator

from zquantum.core.openfermion import expectation, change_operator_type
from zquantum.core.interfaces.backend import QuantumSimulator
from zquantum.core.measurement import (
    expectation_values_to_real,
    ExpectationValues,
    Measurements,
)


class QiskitSimulator(QuantumSimulator):
    supports_batching = True

    def __init__(
        self,
        device_name,
        n_samples=None,
        noise_model=None,
        device_connectivity=None,
        basis_gates=None,
        api_token=None,
        optimization_level=0,
        **kwargs,
    ):
        """Get a qiskit device (simulator or QPU) that adheres to the
        zquantum.core.interfaces.backend.QuantumSimulator

        Args:
            device_name (string): the name of the device
            n_samples (int): the number of samples to use when running the device
            noise_model (qiskit.providers.aer.noise.NoiseModel): an optional
                noise model to pass in for noisy simulations
            device_connectivity (zquantum.core.circuit.CircuitConnectivity): an optional input of an object representing
                the connectivity of the device that will be used in simulations
            basis_gates (list): an optional input of the list of basis gates
                used in simulations
            api_token (string): IBMQ Api Token
            optimization_level (int): optimization level for the default qiskit transpiler (0, 1, 2, or 3)

        Returns:
            qeqiskit.backend.QiskitSimulator
        """
        super().__init__(n_samples=n_samples)
        self.device_name = device_name
        self.noise_model = noise_model
        self.device_connectivity = device_connectivity

        if basis_gates is None and self.noise_model is not None:
            self.basis_gates = self.noise_model.basis_gates
        else:
            self.basis_gates = basis_gates

        if api_token is not None:
            try:
                IBMQ.enable_account(api_token)
            except IBMQAccountError as e:
                if (
                    e.message
                    != "An IBM Quantum Experience account is already in use for the session."
                ):
                    raise RuntimeError(e)

        self.optimization_level = optimization_level
        self.get_device(**kwargs)

    def get_device(self, noisy=False, **kwargs):
        """Get the ibm device used for executing circuits

        Args:
            noisy (bool): a boolean indicating if the user wants to use noisy
                simulations
        Returns:
            The ibm device that can use the ibm execute api
        """
        # If not doing noisy simulation...
        if len(Aer.backends(self.device_name)) > 0:
            self.device = Aer.get_backend(self.device_name)
        else:
            raise RuntimeError(
                "Could not find simulator with name: {}".format(self.device_name)
            )

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

        coupling_map = None
        if self.device_connectivity is not None:
            coupling_map = CouplingMap(self.device_connectivity.connectivity)

        # Run job on device and get counts
        raw_counts = (
            execute(
                ibmq_circuit,
                self.device,
                shots=self.n_samples,
                noise_model=self.noise_model,
                coupling_map=coupling_map,
                basis_gates=self.basis_gates,
                optimization_level=self.optimization_level,
            )
            .result()
            .get_counts()
        )

        # qiskit counts object maps bitstrings in reversed order to ints, so we must flip the bitstrings
        reversed_counts = {}
        for bitstring in raw_counts.keys():
            reversed_counts[bitstring[::-1]] = raw_counts[bitstring]

        return Measurements.from_counts(reversed_counts)

    def run_circuitset_and_measure(self, circuitset, **kwargs):
        """Run a set of circuits and measure a certain number of bitstrings.
        Note: the number of bitstrings measured is derived from self.n_samples

        Args:
            circuit (zquantum.core.circuit.Circuit): the circuit to prepare the state

        Returns:
            a list of lists of bitstrings (a list of lists of tuples)
        """
        self.number_of_circuits_run += len(circuitset)
        ibmq_circuitset = []
        for circuit in circuitset:
            num_qubits = len(circuit.qubits)

            ibmq_circuit = circuit.to_qiskit()
            ibmq_circuit.barrier(range(num_qubits))
            ibmq_circuit.measure(range(num_qubits), range(num_qubits))

            ibmq_circuitset.append(ibmq_circuit)

        coupling_map = None
        if self.device_connectivity is not None:
            coupling_map = CouplingMap(self.device_connectivity.connectivity)

        # Run job on device and get counts
        job = execute(
            ibmq_circuitset,
            self.device,
            shots=self.n_samples,
            noise_model=self.noise_model,
            coupling_map=coupling_map,
            basis_gates=self.basis_gates,
            optimization_level=self.optimization_level,
        )
        measurements_set = []
        for i, ibmq_circuit in enumerate(ibmq_circuitset):
            self.number_of_jobs_run += 1
            circuit_counts = job.result().get_counts(ibmq_circuit)

            # qiskit counts object maps bitstrings in reversed order to ints, so we must flip the bitstrings
            reversed_counts = {}
            for bitstring in circuit_counts.keys():
                reversed_counts[bitstring[::-1]] = circuit_counts[bitstring]

            measurements = Measurements.from_counts(reversed_counts)
            measurements_set.append(measurements)

        return measurements_set

    def get_expectation_values(self, circuit, qubit_operator, **kwargs):
        """Run a circuit and measure the expectation values with respect to a
        given operator. Note: the number of bitstrings measured is derived
        from self.n_samples - if self.n_samples = None, then this will use
        self.get_exact_expectation_values

        Args:
            circuit (zquantum.core.circuit.Circuit): the circuit to prepare the state
            qubit_operator (openfermion.ops.QubitOperator): the operator to measure
        Returns:
            zquantum.core.measurement.ExpectationValues: the expectation values
                of each term in the operator
        """
        if self.n_samples == None:
            return self.get_exact_expectation_values(circuit, qubit_operator, **kwargs)
        else:
            operator = change_operator_type(qubit_operator, IsingOperator)
            measurements = self.run_circuit_and_measure(circuit)
            expectation_values = measurements.get_expectation_values(operator)

            expectation_values = expectation_values_to_real(expectation_values)
            return expectation_values

    def get_exact_expectation_values(self, circuit, qubit_operator, **kwargs):
        """Run a circuit to prepare a wavefunction and measure the exact
        expectation values with respect to a given operator.

        Args:
            circuit (zquantum.core.circuit.Circuit): the circuit to prepare the state
            qubit_operator (openfermion.ops.QubitOperator): the operator to measure
        Returns:
            zquantum.core.measurement.ExpectationValues: the expectation values
                of each term in the operator
        """
        wavefunction = self.get_wavefunction(circuit)

        # Pyquil does not support PauliSums with no terms.
        if len(qubit_operator.terms) == 0:
            return ExpectationValues(np.zeros((0,)))

        values = []

        for op in qubit_operator:
            values.append(expectation(op, wavefunction))
        return expectation_values_to_real(ExpectationValues(np.asarray(values)))

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

    def get_wavefunction(self, circuit):
        """Run a circuit and get the wavefunction of the resulting statevector.

        Args:
            circuit (zquantum.core.circuit.Circuit): the circuit to prepare the state
        Returns:
            pyquil.wavefunction.Wavefunction
        """
        super().get_wavefunction(circuit)
        ibmq_circuit = circuit.to_qiskit()

        coupling_map = None
        if self.device_connectivity is not None:
            coupling_map = CouplingMap(self.device_connectivity.connectivity)

        # Execute job to get wavefunction
        job = execute(
            ibmq_circuit,
            self.device,
            noise_model=self.noise_model,
            coupling_map=coupling_map,
            basis_gates=self.basis_gates,
        )
        wavefunction = job.result().get_statevector(ibmq_circuit, decimals=20)
        return Wavefunction(wavefunction)
