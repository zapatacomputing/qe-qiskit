import sys

from qeqiskit.conversions import export_to_qiskit

from qiskit import Aer, ClassicalRegister, execute
from qiskit.providers.ibmq import IBMQ
from qiskit.providers.ibmq.exceptions import IBMQAccountError
from qiskit.transpiler import CouplingMap

from zquantum.core.circuits import Circuit
from zquantum.core.interfaces.backend import QuantumSimulator
from zquantum.core.measurement import Measurements, sample_from_wavefunction
from zquantum.core.wavefunction import Wavefunction


class QiskitSimulator(QuantumSimulator):
    supports_batching = False
    batch_size = sys.maxsize

    def __init__(
        self,
        device_name,
        noise_model=None,
        device_connectivity=None,
        basis_gates=None,
        api_token=None,
        optimization_level=0,
        seed=None,
        **kwargs,
    ):
        """Get a qiskit device (simulator or QPU) that adheres to the
        zquantum.core.interfaces.backend.QuantumSimulator

        Args:
            device_name (string): the name of the device
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
        super().__init__()
        self.device_name = device_name
        self.noise_model = noise_model
        self.device_connectivity = device_connectivity
        self.seed = seed

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

    def run_circuit_and_measure(self, circuit: Circuit, n_samples: int) -> Measurements:
        """Run a circuit and measure a certain number of bitstrings. Note: the
        number of bitstrings measured is derived from self.n_samples

        Args:
            circuit: the circuit to prepare the state
            n_samples: The number of samples to collect.
        """

        super().run_circuit_and_measure(circuit, n_samples)
        num_qubits = circuit.n_qubits

        ibmq_circuit = export_to_qiskit(circuit)
        ibmq_circuit.barrier(range(num_qubits))
        ibmq_circuit.add_register(ClassicalRegister(size=circuit.n_qubits))
        ibmq_circuit.measure(range(num_qubits), range(num_qubits))

        coupling_map = None
        if self.device_connectivity is not None:
            coupling_map = CouplingMap(self.device_connectivity.connectivity)

        if self.device_name == "statevector_simulator":
            wavefunction = self.get_wavefunction(circuit)
            return Measurements(sample_from_wavefunction(wavefunction, n_samples))
        else:
            # Run job on device and get counts
            raw_counts = (
                execute(
                    ibmq_circuit,
                    self.device,
                    shots=n_samples,
                    noise_model=self.noise_model,
                    coupling_map=coupling_map,
                    basis_gates=self.basis_gates,
                    optimization_level=self.optimization_level,
                    seed_simulator=self.seed,
                    seed_transpiler=self.seed,
                )
                .result()
                .get_counts()
            )

        # qiskit counts object maps bitstrings in reversed order to ints, so we must flip the bitstrings
        reversed_counts = {}
        for bitstring in raw_counts.keys():
            reversed_counts[bitstring[::-1]] = raw_counts[bitstring]

        return Measurements.from_counts(reversed_counts)

    def get_wavefunction(self, circuit):
        """Run a circuit and get the wavefunction of the resulting statevector.

        Args:
            circuit (zquantum.core.circuit.Circuit): the circuit to prepare the state
        Returns:
            pyquil.wavefunction.Wavefunction
        """
        super().get_wavefunction(circuit)
        ibmq_circuit = export_to_qiskit(circuit)

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
            seed_simulator=self.seed,
            seed_transpiler=self.seed,
        )
        wavefunction = job.result().get_statevector(ibmq_circuit, decimals=20)
        return Wavefunction(wavefunction)
