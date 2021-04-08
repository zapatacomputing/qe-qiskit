import unittest
import numpy as np
import os
import subprocess
import json
import qiskit.providers.aer.noise as AerNoise

from zquantum.core.utils import load_noise_model, save_noise_model
from qeqiskit.utils import save_qiskit_noise_model, load_qiskit_noise_model, save_kraus_operators
from qeqiskit.noise.basic import create_amplitude_damping_noise,  get_kraus_matrices_from_ibm_noise_model

class TestQiskitUtils(unittest.TestCase):
    def setUp(self):
        self.T_1 = 10e-7
        self.t_step = 10e-9

    def test_save_qiskit_noise_model(self):
        # Given
        noise_model = AerNoise.NoiseModel()
        quantum_error = AerNoise.depolarizing_error(0.0, 1)
        coherent_error = np.asarray(
            [
                np.asarray(
                    [0.87758256 - 0.47942554j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]
                ),
                np.asarray(
                    [0.0 + 0.0j, 0.87758256 + 0.47942554j, 0.0 + 0.0j, 0.0 + 0.0j]
                ),
                np.asarray(
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.87758256 + 0.47942554j, 0.0 + 0.0j]
                ),
                np.asarray(
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.87758256 - 0.47942554j]
                ),
            ]
        )
        noise_model.add_quantum_error(
            AerNoise.coherent_unitary_error(coherent_error), ["cx"], [0, 1]
        )
        filename = "noise_model.json"

        # When
        save_qiskit_noise_model(noise_model, filename)

        # Then
        with open("noise_model.json", "r") as f:
            data = json.loads(f.read())
        self.assertEqual(data["module_name"], "qeqiskit.utils")
        self.assertEqual(data["function_name"], "load_qiskit_noise_model")
        self.assertIsInstance(data["data"], dict)

        # Cleanup
        subprocess.run(["rm", "noise_model.json"])

    def test_noise_model_io_using_core_functions(self):
        # Given
        noise_model = AerNoise.NoiseModel()
        quantum_error = AerNoise.depolarizing_error(0.0, 1)
        coherent_error = np.asarray(
            [
                np.asarray(
                    [0.87758256 - 0.47942554j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]
                ),
                np.asarray(
                    [0.0 + 0.0j, 0.87758256 + 0.47942554j, 0.0 + 0.0j, 0.0 + 0.0j]
                ),
                np.asarray(
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.87758256 + 0.47942554j, 0.0 + 0.0j]
                ),
                np.asarray(
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.87758256 - 0.47942554j]
                ),
            ]
        )
        noise_model.add_quantum_error(
            AerNoise.coherent_unitary_error(coherent_error), ["cx"], [0, 1]
        )
        noise_model_data = noise_model.to_dict(serializable=True)
        module_name = "qeqiskit.utils"
        function_name = "load_qiskit_noise_model"
        filename = "noise_model.json"

        # When
        save_noise_model(noise_model_data, module_name, function_name, filename)
        new_noise_model = load_noise_model(filename)

        # Then
        self.assertEqual(
            noise_model.to_dict(serializable=True),
            new_noise_model.to_dict(serializable=True),
        )

        # Cleanup
        subprocess.run(["rm", "noise_model.json"])

    def test_save_kraus_operators(self):
        noise_model = create_amplitude_damping_noise(self.T_1, self.t_step)
        kraus_dict = get_kraus_matrices_from_ibm_noise_model(noise_model)
        save_kraus_operators(kraus_dict, 'kraus_operators.json')

        # Cleanup
        os.remove("kraus_operators.json")



     

