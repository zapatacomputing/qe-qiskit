import json
import os
import subprocess

import numpy as np
import pytest
import qiskit.providers.aer.noise as AerNoise
from qeqiskit.noise.basic import (
    create_amplitude_damping_noise,
    get_kraus_matrices_from_ibm_noise_model,
)
from qeqiskit.utils import (
    load_qiskit_noise_model,
    save_kraus_operators,
    save_qiskit_noise_model,
)
from zquantum.core.utils import load_noise_model, save_noise_model


class TestQiskitUtils:
    def test_save_qiskit_noise_model(self):
        pytest.xfail(
            """This test requires fixing.
            Depending on qiskit version it might fail or give false positives."""
        )

        # Given
        noise_model = AerNoise.NoiseModel()
        coherent_error = np.asarray(
            [np.exp(-1j * 0.5), 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, np.exp(1j * 0.5), 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, np.exp(1j * 0.5), 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, np.exp(-1j * 0.5)],
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

        new_noise_model = load_qiskit_noise_model(data)
        assert data["module_name"] == "qeqiskit.utils"
        assert data["function_name"] == "load_qiskit_noise_model"
        assert isinstance(data["data"], dict)
        assert noise_model.to_dict(serializable=True) == new_noise_model.to_dict(
            serializable=True
        )
        # Cleanup
        subprocess.run(["rm", "noise_model.json"])

    def test_noise_model_io_using_core_functions(self):

        pytest.xfail(
            """This test requires fixing.
            Depending on qiskit version it might fail or give false positives."""
        )

        # Given
        noise_model = AerNoise.NoiseModel()
        coherent_error = np.asarray(
            [np.exp(-1j * 0.5), 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, np.exp(1j * 0.5), 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, np.exp(1j * 0.5), 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, np.exp(-1j * 0.5)],
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
        assert noise_model.to_dict(serializable=True) == new_noise_model.to_dict(
            serializable=True
        )

        # Cleanup
        subprocess.run(["rm", "noise_model.json"])

    def test_save_kraus_operators(self):
        T_1 = 10e-7
        t_step = 10e-9
        noise_model = create_amplitude_damping_noise(T_1, t_step)
        kraus_dict = get_kraus_matrices_from_ibm_noise_model(noise_model)
        save_kraus_operators(kraus_dict, "kraus_operators.json")

        # Cleanup
        os.remove("kraus_operators.json")
