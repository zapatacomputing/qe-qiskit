import unittest
import os
import qiskit.providers.aer.noise as AerNoise
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from zquantum.core.circuits.layouts import CircuitConnectivity

from qeqiskit.noise.basic import (
    get_qiskit_noise_model,
    create_amplitude_damping_noise,
    create_phase_damping_noise,
    create_phase_and_amplitude_damping_error,
    create_pta_channel,
    get_kraus_matrices_from_ibm_noise_model,
)


class TestBasic(unittest.TestCase):
    def setUp(self):
        self.ibmq_api_token = os.getenv("ZAPATA_IBMQ_API_TOKEN")
        self.all_devices = ["ibmqx2"]
        self.T_1 = 10e-7
        self.T_2 = 30e-7
        self.t_step = 10e-9
        self.t_1_t_2_models = [
            create_phase_and_amplitude_damping_error,
            create_pta_channel,
        ]

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

    def test_t_1_t_2_noise_models(self):
        for noise in self.t_1_t_2_models:
            self.assertIsInstance(
                noise(self.T_1, self.T_2, self.t_step), AerNoise.NoiseModel
            )

    def test_amplitude_damping_model(self):
        self.assertIsInstance(
            create_amplitude_damping_noise(self.T_1, self.t_step), AerNoise.NoiseModel
        )

    def test_phase_damping_noise(self):
        self.assertIsInstance(
            create_phase_damping_noise(self.T_2, self.t_step), AerNoise.NoiseModel
        )

    def test_getting_kraus_matrices_from_noise_model(self):
        noise_model = create_amplitude_damping_noise(self.T_1, self.t_step)
        kraus_dict = get_kraus_matrices_from_ibm_noise_model(noise_model)

        # Test to see if basis gates are in
        self.assertEqual("id" in kraus_dict, True)
        self.assertEqual("u3" in kraus_dict, True)
        self.assertEqual("cx" in kraus_dict, True)

        # Test to see if the number of kraus operators is right for a basis gate
        self.assertEqual(len(kraus_dict["id"]), 2)
        self.assertEqual(len(kraus_dict["u3"]), 2)
        self.assertEqual(len(kraus_dict["cx"]), 4)
