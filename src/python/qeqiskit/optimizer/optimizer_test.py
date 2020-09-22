import unittest
from .optimizer import QiskitOptimizer
from zquantum.core.interfaces.optimizer_test import OptimizerTests


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
