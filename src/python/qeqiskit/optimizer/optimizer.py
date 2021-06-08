import numpy as np
from zquantum.core.history.recorder import recorder as _recorder
from zquantum.core.interfaces.optimizer import (
    Optimizer,
    optimization_result,
    construct_history_info,
)
from zquantum.core.interfaces.functions import CallableWithGradient
from zquantum.core.typing import RecorderFactory
from typing import Optional, Dict
from qiskit.algorithms.optimizers import SPSA, ADAM


class QiskitOptimizer(Optimizer):
    def __init__(
        self,
        method: str,
        options: Optional[Dict] = None,
        recorder: RecorderFactory = _recorder,
    ):
        """
        Args:
            method: specifies optimizer to be used. Currently supports "ADAM", "AMSGRAD" and "SPSA".
            options: dictionary with additional options for the optimizer.
            recorder: recorder object which defines how to store the optimization history.

        Supported values for the options dictionary:
        Options:
            **kwargs: options specific for particular qiskit optimizers.

        """
        super().__init__(recorder=recorder)
        self.method = method
        if options is None:
            self.options = {}
        else:
            self.options = options

    def _minimize(
        self,
        cost_function: CallableWithGradient,
        initial_params: np.ndarray = None,
        keep_history: bool = False,
    ):
        """
        Minimizes given cost function using optimizers from Qiskit Aqua.

        Args:
            cost_function: python method which takes numpy.ndarray as input
            initial_params(np.ndarray): initial parameters to be used for optimization

        Returns:
            optimization_results(scipy.optimize.OptimizeResults): results of the optimization.
        """
        history = []

        if self.method == "SPSA":
            optimizer = SPSA(**self.options)
        elif self.method == "ADAM" or self.method == "AMSGRAD":
            if self.method == "AMSGRAD":
                self.options["amsgrad"] = True
            optimizer = ADAM(**self.options)

        number_of_variables = len(initial_params)

        gradient_function = None
        if hasattr(cost_function, "gradient") and callable(
            getattr(cost_function, "gradient")
        ):
            gradient_function = cost_function.gradient

        solution, value, nfev = optimizer.optimize(
            num_vars=number_of_variables,
            objective_function=cost_function,
            initial_point=initial_params,
            gradient_function=gradient_function,
        )

        if self.method == "ADAM" or self.method == "AMSGRAD":
            nit = optimizer._t
        else:
            nit = optimizer.maxiter

        return optimization_result(
            opt_value=value,
            opt_params=solution,
            nit=nit,
            nfev=nfev,
            **construct_history_info(cost_function, keep_history)
        )
