################################################################################
# © Copyright 2020-2021 Zapata Computing Inc.
################################################################################
from qeqiskit.noise import (
    create_amplitude_damping_noise as _create_amplitude_damping_noise,
)
from qeqiskit.noise import (
    create_phase_and_amplitude_damping_error as _create_pa_damping_error,
)
from qeqiskit.noise import create_phase_damping_noise as _create_phase_damping_noise
from qeqiskit.noise import create_pta_channel as _create_pta_channel
from qeqiskit.noise import get_qiskit_noise_model as _get_qiskit_noise_model
from qeqiskit.utils import save_qiskit_noise_model
from zquantum.core.circuits.layouts import save_circuit_connectivity


def get_qiskit_noise_model(
    device_name, hub="ibm-q", group="open", project="main", api_token=None
):

    if api_token == "None":
        api_token = None

    noise_model, device_connectivity = _get_qiskit_noise_model(
        device_name,
        hub=hub,
        group=group,
        project=project,
        api_token=api_token,
    )

    save_qiskit_noise_model(noise_model, "noise-model.json")
    save_circuit_connectivity(device_connectivity, "device-connectivity.json")


def create_amplitude_damping_noise(T_1, **kwargs):
    noise_model = _create_amplitude_damping_noise(T_1, **kwargs)
    save_qiskit_noise_model(noise_model, "noise-model.json")


def create_phase_damping_noise(T_2, **kwargs):
    noise_model = _create_phase_damping_noise(T_2, **kwargs)
    save_qiskit_noise_model(noise_model, "noise-model.json")


def create_phase_and_amplitude_damping_error(T_1, T_2, **kwargs):
    noise_model = _create_pa_damping_error(T_1, T_2, **kwargs)
    save_qiskit_noise_model(noise_model, "noise-model.json")


def create_pta_channel(T_1, T_2, **kwargs):
    noise_model = _create_pta_channel(T_1, T_2, **kwargs)
    save_qiskit_noise_model(noise_model, "noise-model.json")
