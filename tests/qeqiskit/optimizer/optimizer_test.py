import numpy as np
import pytest
from qeqiskit.optimizer import QiskitOptimizer
from zquantum.core.gradients import finite_differences_gradient
from zquantum.core.history.recorder import recorder
from zquantum.core.interfaces.functions import FunctionWithGradient
from zquantum.core.interfaces.optimizer_test import (
    MANDATORY_OPTIMIZATION_RESULT_FIELDS,
    OptimizerTests,
)


@pytest.fixture(
    params=[
        {"method": "ADAM"},
        {
            "method": "SPSA",
            "optimizer_kwargs": {
                "maxiter": int(1e5),
                "learning_rate": 1e-3,
                "perturbation": 1e-5,
            },
        },
        {
            "method": "AMSGRAD",
            "optimizer_kwargs": {"maxiter": 2e5, "tol": 1e-9, "lr": 1e-4},
        },
    ]
)
def optimizer(request):
    return QiskitOptimizer(**request.param)


# A fast, inaccurate optimizer used when resulting values aren't being tested
@pytest.fixture(
    params=[
        {
            "method": "ADAM",
            "optimizer_kwargs": {
                "maxiter": int(5),
            },
        },
        {
            "method": "SPSA",
            "optimizer_kwargs": {"maxiter": int(5)},
        },
        {
            "method": "AMSGRAD",
            "optimizer_kwargs": {"maxiter": 5},
        },
    ]
)
def low_iter_optimizer(request):
    return QiskitOptimizer(**request.param)


@pytest.fixture(
    params=[
        {
            "method": "NFT",
            "optimizer_kwargs": {"maxiter": 2e5},
        },
    ]
)
def sinusoidal_optimizer(request):
    return QiskitOptimizer(**request.param)


@pytest.fixture(
    params=[
        {
            "method": "NFT",
            "optimizer_kwargs": {"maxiter": 5},
        },
    ]
)
def low_iter_sinusoidal_optimizer(request):
    return QiskitOptimizer(**request.param)


@pytest.fixture(params=[True, False])
def keep_history(request):
    return request.param


class TestQiskitOptimizerTests(OptimizerTests):
    def test_optimizer_succeeds_on_cost_function_without_gradient(
        self, optimizer, sum_x_squared
    ):
        cost_function = sum_x_squared

        results = optimizer.minimize(cost_function, initial_params=np.array([1, -1]))
        assert results.opt_value == pytest.approx(0, abs=1e-5)
        assert results.opt_params == pytest.approx(np.zeros(2), abs=1e-4)

        assert "nfev" in results
        assert "nit" in results
        assert "opt_value" in results
        assert "opt_params" in results
        assert "history" in results

    def test_optimizer_records_history_if_keep_history_is_true(
        self, low_iter_optimizer, sum_x_squared
    ):
        super().test_optimizer_records_history_if_keep_history_is_true(
            low_iter_optimizer, sum_x_squared
        )

    def test_optimizer_does_not_record_history_if_keep_history_is_set_to_false(
        self, low_iter_optimizer, sum_x_squared
    ):
        super().test_optimizer_does_not_record_history_if_keep_history_is_set_to_false(
            low_iter_optimizer, sum_x_squared
        )

    def test_optimizer_does_not_record_history_by_default(
        self, low_iter_optimizer, sum_x_squared
    ):
        super().test_optimizer_does_not_record_history_by_default(
            low_iter_optimizer, sum_x_squared
        )


class TestQiskitSinusoidalOptimizerTests(OptimizerTests):
    """Some optimizers assume that the cost function can be expressed as a sum of
    products of sines and cosines (Eq. 12 of arXiv:1903.12166), a functional form which
    reflects the cost function of many variational quantum algorithms. Such optimizers
    may fail to find the minimum of cost functions that are not of this form.

    This class removes the tests from the base class that check whether the optimizer
    finds the minimum of such non-sinusoidal cost functions, and replaces them with a
    test on a sinuisoidal cost function."""

    @pytest.fixture
    def sum_cos_x(self):
        """Create a sinuisoidal function whose minimum is at the origin with a value of
        zero."""
        return lambda x: -np.sum(np.cos(x)) + x.shape[0]

    @pytest.mark.skip(reason="Optimizer only supports sinusoidal cost functions")
    def test_optimizer_succeeds_with_optimizing_rosenbrock_function(self):
        pass

    @pytest.mark.skip(reason="Optimizer only supports sinusoidal cost functions")
    def test_optimizer_succeeds_with_optimizing_sum_of_squares_function(self):
        pass

    @pytest.mark.skip(reason="Optimizer only supports sinusoidal cost functions")
    def test_optimizer_succeeds_on_cost_function_without_gradient(self):
        pass

    def test_optimizer_succeeds_on_sum_of_cosines_function(
        self, sinusoidal_optimizer, sum_cos_x, keep_history
    ):

        cost_function = FunctionWithGradient(
            sum_cos_x, finite_differences_gradient(sum_cos_x)
        )

        results = sinusoidal_optimizer.minimize(
            cost_function, initial_params=np.array([1, -1]), keep_history=keep_history
        )

        assert results.opt_value == pytest.approx(0, abs=1e-5)
        assert results.opt_params == pytest.approx(np.zeros(2), abs=1e-4)

        assert all(field in results for field in MANDATORY_OPTIMIZATION_RESULT_FIELDS)

        assert "history" in results or not keep_history
        assert "gradient_history" in results or not keep_history

    def test_optimizer_records_history_if_keep_history_is_true(
        self, low_iter_sinusoidal_optimizer, sum_x_squared
    ):
        super().test_optimizer_records_history_if_keep_history_is_true(
            low_iter_sinusoidal_optimizer, sum_x_squared
        )

    def test_optimizer_does_not_record_history_if_keep_history_is_set_to_false(
        self, low_iter_sinusoidal_optimizer, sum_x_squared
    ):
        super().test_optimizer_does_not_record_history_if_keep_history_is_set_to_false(
            low_iter_sinusoidal_optimizer, sum_x_squared
        )

    def test_optimizer_does_not_record_history_by_default(
        self, low_iter_sinusoidal_optimizer, sum_x_squared
    ):
        super().test_optimizer_does_not_record_history_by_default(
            low_iter_sinusoidal_optimizer, sum_x_squared
        )
