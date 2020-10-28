import numpy as np
import qiskit.providers.aer.noise as AerNoise
from qiskit import IBMQ
from qiskit.providers.ibmq.exceptions import IBMQAccountError
from zquantum.core.circuit import CircuitConnectivity
from qiskit.providers.aer.noise import (amplitude_damping_error, 
                                        phase_damping_error, 
                                        phase_amplitude_damping_error, 
                                        pauli_error)
from qiskit.providers.aer.noise import NoiseModel
from qiskit.quantum_info import Kraus


def get_qiskit_noise_model(
    device_name, hub="ibm-q", group="open", project="main", api_token=None
):
    """ Get a qiskit noise model to use noisy simulations with a qiskit simulator

    Args:
        device_name (string): The name of the device trying to be emulated
        hub (string): The ibmq hub (see qiskit documentation)
        group (string): The ibmq group (see qiskit documentation)
        project (string): The ibmq project (see qiskit documentation)
        api_token (string): The ibmq api token (see qiskit documentation)
    Returns:
        qiskit.providers.aer.noise.NoiseModel
        zquantum.core.circuit.CircuitConnectivity: the qubit connectivity of the device
    """
    if api_token is not None and api_token is not "None":
        try:
            IBMQ.enable_account(api_token)
        except IBMQAccountError as e:
            if (
                e.message
                != "An IBM Quantum Experience account is already in use for the session."
            ):
                raise RuntimeError(e)

    # Get qiskit noise model from qiskit
    provider = IBMQ.get_provider(hub=hub, group=group, project=project)
    noisy_device = provider.get_backend(device_name)

    noise_model = AerNoise.NoiseModel.from_backend(noisy_device)
    coupling_map = noisy_device.configuration().coupling_map

    return noise_model, CircuitConnectivity(coupling_map)

def create_amplitude_damping_noise(T_1, t_step=10e-9):
    """ Creates an amplitude damping noise model
    
    Args:
        T_1 (float) : Relaxation time (seconds)
        t_step (float) : Discretized time step over which the relaxation occurs over (seconds)
    
    Returns:
        qiskit.providers.aer.noise.NoiseModel
    """

    gamma = (1 - pow(np.e, - 1/T_1*t_step))
    error = amplitude_damping_error(gamma)
    gate_error = error.tensor(error)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error, ['id', 'u3'])
    noise_model.add_all_qubit_quantum_error(gate_error, ['cx'])
    return noise_model

def create_phase_damping_noise(T_2, t_step=10e-9):
    """ Creates a dephasing noise model
    
    Args:
        T_2 (float) : dephasing time (seconds)
        t_step (float) : Discretized time step over which the relaxation occurs over (seconds)
    
    Returns:
        qiskit.providers.aer.noise.NoiseModel
    """
    gamma = (1 - pow(np.e, - 1/T_2*t_step))
    error = phase_damping_error(gamma)
    gate_error = error.tensor(error)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error, ['id', 'u3'])
    noise_model.add_all_qubit_quantum_error(gate_error, ['cx'])
    return noise_model

def create_phase_and_amplitude_damping_error(T_1, T_2, t_step=10e-9):
    """ Creates a noise model that does both phase and amplitude damping
    
    Args:
        T_1 (float) : Relaxation time (seconds)
        T_2 (float) : dephasing time  (seonds)
        t_step (float) : Discretized time step over which the relaxation occurs over (seconds)
    
    Returns:
        qiskit.providers.aer.noise.NoiseModel
    """

    param_amp = (1 - pow(np.e, - 1/T_1*t_step))
    param_phase = (1 - pow(np.e, - 1/T_2*t_step))
    error = phase_amplitude_damping_error(param_amp, param_phase)
    gate_error = error.tensor(error)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error, ['id', 'u3'])
    noise_model.add_all_qubit_quantum_error(gate_error, ['cx'])
    return noise_model

def create_pta_channel(T_1, T_2, t_step=10e-9):
    """ Creates a noise model that does both phase and amplitude damping but in the
        Pauli Twirling Approximation discussed the following reference 
        https://arxiv.org/pdf/1305.2021.pdf

    
    Args:
        T_1 (float) : Relaxation time (seconds)
        T_2 (float) : dephasing time (seconds)
        t_step (float) : Discretized time step over which the relaxation occurs over (seconds)
    
    Returns:
        qiskit.providers.aer.noise.NoiseModel
    """

    if T_1 == T_2:
        t_phi = 2*T_1
    elif 2*T_1 == T_2:
        raise RuntimeError(" T_2 == 2*T_1 only in a pure amplitude damping case ")
    else:
        t_phi = T_2 - 2*T_1
    
    p_x = 0.25*(1- pow(np.e, - t_step/T_1))
    p_y = 0.25*(1- pow(np.e, - t_step/T_1))
    
    exp_1 = pow(np.e, -t_step/(2*T_1))
    exp_2 = pow(np.e, -t_step/t_phi)
    p_z = (0.5 - p_x - 0.5*exp_1*exp_2)
    p_i = 1 - p_x - p_y - p_z
    errors = [('X', p_x), ('Y', p_y), ('Z', p_z), ('I', p_i)]
    pta_error = pauli_error(errors)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(pta_error, ['id', 'u3'])
    gate_error = pta_error.tensor(pta_error)
    noise_model.add_all_qubit_quantum_error(gate_error, ['cx'])
    return noise_model


def get_kraus_matrices_from_ibm_noise_model(noise_model):
    """Gets the kraus operators from a pre defined noise model

    Args:
        noise_model (qiskit.providers.aer.noise.NoiseModel): Noise model for circuit
    
    Return
        dict_of_kraus_operators(dict): A dictionary labelled by keys which are the basis gates and values are the list of kraus operators

    """

    retrieved_quantum_error_dict = noise_model._default_quantum_errors
    dict_of_kraus_operators = { gate: Kraus(retrieved_quantum_error_dict[gate]).data for gate in retrieved_quantum_error_dict }
    return dict_of_kraus_operators 