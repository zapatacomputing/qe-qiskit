import unittest
from .optimizer import QiskitOptimizer
from zquantum.core.interfaces.optimizer_test import OptimizerTests
import numpy as np
from zquantum.core.interfaces.optimizer_test import sum_x_squared


class QiskitOptimizerTests(unittest.TestCase, OptimizerTests):
    def setUp(self):
        self.optimizers = [
            QiskitOptimizer(method="ADAM"),
            QiskitOptimizer(
                method="SPSA",
                options={
                    "max_trials": int(1e5),
                    "c0": 1e-3,
                    "c1": 1e-4,
                    "c2": 1e-3,
                    "c3": 1e-4,
                },
            ),
            QiskitOptimizer(
                method="AMSGRAD", options={"maxiter": 2e5, "tol": 1e-9, "lr": 1e-4}
            ),
        ]

    def test_optimizer_succeeds_on_cost_function_without_gradient(self):
        for optimizer in self.optimizers:
            cost_function = sum_x_squared

            results = optimizer.minimize(
                cost_function, initial_params=np.array([1, -1])
            )
            self.assertAlmostEqual(results.opt_value, 0, places=5)
            self.assertAlmostEqual(results.opt_params[0], 0, places=4)
            self.assertAlmostEqual(results.opt_params[1], 0, places=4)

            self.assertIn("nfev", results.keys())
            self.assertIn("nit", results.keys())
            self.assertIn("opt_value", results.keys())
            self.assertIn("opt_params", results.keys())
            self.assertIn("history", results.keys())
