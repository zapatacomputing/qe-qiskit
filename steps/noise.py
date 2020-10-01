from zquantum.core.circuit import save_circuit_connectivity
from qeqiskit.utils import save_qiskit_noise_model
from qeqiskit.noise import get_qiskit_noise_model as _get_qiskit_noise_model


def get_qiskit_noise_model(
    device_name, hub="ibm-q", group="open", project="main", api_token=None
):

    if api_token is "None":
        api_token = None

    noise_model, device_connectivity = _get_qiskit_noise_model(
        device_name,
        hub=hub,
        group=group,
        project=project,
        api_token=api_token,
    )

    save_qiskit_noise_model(noise_model, "noise_model.json")
    save_circuit_connectivity(device_connectivity, "device_connectivity.json")