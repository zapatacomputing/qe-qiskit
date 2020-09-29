import unittest
import os
import qiskit.providers.aer.noise as AerNoise
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from zquantum.core.circuit import CircuitConnectivity

from .basic import get_qiskit_noise_model


class TestBasic(unittest.TestCase):
    def setUp(self):
        self.ibmq_api_token = os.getenv("ZAPATA_IBMQ_API_TOKEN")
        self.all_devices = ["ibmqx2"]

    def test_get_qiskit_noise_model(self):
        # Given
        for device in self.all_devices:
            # When
            noise_model, coupling_map = get_qiskit_noise_model(
                device, api_token=self.ibmq_api_token
            )

            # Then
            self.assertIsInstance(noise_model, AerNoise.NoiseModel)
            self.assertIsInstance(coupling_map, CircuitConnectivity)

    def test_get_qiskit_noise_model_no_device(self):
        # Given
        not_real_devices = ["THIS IS NOT A REAL DEVICE", "qasm_simulator"]

        for device in not_real_devices:
            # When/then
            self.assertRaises(
                QiskitBackendNotFoundError,
                lambda: get_qiskit_noise_model(device, api_token=self.ibmq_api_token),
            )
