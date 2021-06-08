from zquantum.core.history.recorder import recorder
from qeqiskit.optimizer import QiskitOptimizer
from zquantum.core.interfaces.optimizer_test import OptimizerTests
from zquantum.core.interfaces.optimizer_test import sum_x_squared
import numpy as np
import pytest


@pytest.fixture(
    params=[
        {"method": "ADAM"},
        {
            "method": "SPSA",
            "options": {
                "maxiter": int(1e5),
                "learning_rate": 1e-3,
                "perturbation": 1e-5,
            },
        },
        {"method": "AMSGRAD", "options": {"maxiter": 2e5, "tol": 1e-9, "lr": 1e-4}},
    ]
)
def optimizer(request):
    return QiskitOptimizer(**request.param)


@pytest.fixture(params=[True, False])
def keep_history(request):
    return request.param


class TestQiskitOptimizerTests(OptimizerTests):
    def test_optimizer_succeeds_on_cost_function_without_gradient(self, optimizer):
        cost_function = sum_x_squared

        results = optimizer.minimize(cost_function, initial_params=np.array([1, -1]))
        assert results.opt_value == pytest.approx(0, abs=1e-5)
        assert results.opt_params == pytest.approx(np.zeros(2), abs=1e-4)

        assert "nfev" in results
        assert "nit" in results
        assert "opt_value" in results
        assert "opt_params" in results
        assert "history" in results
