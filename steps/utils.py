from typing import List, Optional, Union

from qeqiskit.conversions import export_to_qiskit, import_from_qiskit
from qiskit.providers.ibmq import IBMQ
from qiskit.providers.ibmq.exceptions import IBMQAccountError

from zquantum.core.circuits import (
    Circuit,
    load_circuit,
    load_circuitset,
    save_circuit,
    save_circuitset,
)

from qiskit.compiler import transpile


def transpile_circuit_for_ibm_device(
    circuit_set: Union[str, List[Circuit]],
    device_name: str,
    hub: str = "ibm-q",
    group: str = "open",
    project: str = "main",
    api_token: str = None,
    **kwargs,
):

    if isinstance(circuit_set, str):
        circuit_set = load_circuitset(circuit_set)

    qiskit_circuit_set = [export_to_qiskit(c) for c in circuit_set]

    if api_token is not None:
        try:
            IBMQ.enable_account(api_token)
        except IBMQAccountError as e:
            if e.message != (
                "An IBM Quantum Experience account is already in use for the session."  # noqa: E501
            ):
                raise RuntimeError(e)

    provider = IBMQ.get_provider(hub=hub, group=group, project=project)
    device = provider.get_backend(device_name)

    # transpile the circuits
    transpiled_qiskit_circuit_set = [
        transpile(c, backend=device, **kwargs) for c in qiskit_circuit_set
    ]
    orquestra_circuit_set = [
        import_from_qiskit(c) for c in transpiled_qiskit_circuit_set
    ]

    # save circuit set
    save_circuitset(orquestra_circuit_set, "circuit-set.json")
