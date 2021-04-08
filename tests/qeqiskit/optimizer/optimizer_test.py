import unittest
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
                "c0": 1e-3,
                "c1": 1e-4,
                "c2": 1e-3,
                "c3": 1e-4,
            },
        },
        {"method": "AMSGRAD", "options": {"maxiter": 2e5, "tol": 1e-9, "lr": 1e-4}},
    ]
)
def optimizer(request):
    return QiskitOptimizer(**request.param)


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

    def test_optimizer_records_history_if_keep_value_history_is_added_as_option(
        self, optimizer
    ):
        optimizer.keep_value_history = True

        # To check that history is recorded correctly, we wrap cost_function
        # with a recorder. Optimizer should wrap it a second time and
        # therefore we can compare two histories to see if they agree.
        cost_function = recorder(sum_x_squared)

        result = optimizer.minimize(cost_function, np.array([-1, 1]))

        assert result.history == cost_function.history

    def test_optimizier_does_not_record_history_if_keep_value_history_is_set_to_false(
        self, optimizer
    ):
        optimizer.keep_value_history = False

        result = optimizer.minimize(sum_x_squared, np.array([-2, 0.5]))

        assert result.history == []

    def test_optimizer_does_not_record_history_if_keep_value_history_by_default(
        self, optimizer
    ):

        result = optimizer.minimize(sum_x_squared, np.array([-2, 0.5]))

        assert result.history == []
