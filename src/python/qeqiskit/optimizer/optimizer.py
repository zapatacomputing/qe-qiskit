################################################################################
# © Copyright 2020-2022 Zapata Computing Inc.
################################################################################
from typing import Callable, Dict, Optional, Union

import numpy as np
from qiskit.algorithms.optimizers import ADAM, NFT, SPSA
from scipy.optimize import OptimizeResult
from zquantum.core.history.recorder import recorder as _recorder
from zquantum.core.interfaces.functions import CallableWithGradient
from zquantum.core.interfaces.optimizer import (
    Optimizer,
    construct_history_info,
    optimization_result,
)
from zquantum.core.typing import RecorderFactory


class QiskitOptimizer(Optimizer):
    def __init__(
        self,
        method: str,
        optimizer_kwargs: Optional[Dict] = None,
        recorder: RecorderFactory = _recorder,
    ):
        """
        Args:
            method: specifies optimizer to be used.
                Currently supports "ADAM", "AMSGRAD" and "SPSA".
            optimizer_kwargs: dictionary with additional optimizer_kwargs
                for the optimizer.
            recorder: recorder object which defines how to store
                the optimization history.

        """
        super().__init__(recorder=recorder)
        self.method = method
        if optimizer_kwargs is None:
            self.optimizer_kwargs = {}
        else:
            self.optimizer_kwargs = optimizer_kwargs

        if self.method == "SPSA":
            self.optimizer = SPSA(**self.optimizer_kwargs)
        elif self.method == "ADAM" or self.method == "AMSGRAD":
            if self.method == "AMSGRAD":
                self.optimizer_kwargs["amsgrad"] = True
            self.optimizer = ADAM(**self.optimizer_kwargs)
        elif self.method == "NFT":
            self.optimizer = NFT(**self.optimizer_kwargs)

    def _minimize(
        self,
        cost_function: Union[CallableWithGradient, Callable],
        initial_params: np.ndarray,
        keep_history: bool = False,
    ) -> OptimizeResult:
        """
        Minimizes given cost function using optimizers from Qiskit Aqua.

        Args:
            cost_function: python method which takes numpy.ndarray as input
            initial_params: initial parameters to be used for optimization

        Returns:
            optimization_results: results of the optimization.
        """

        number_of_variables = len(initial_params)

        gradient_function = None
        if isinstance(cost_function, CallableWithGradient):
            gradient_function = cost_function.gradient

        solution, value, nfev = self.optimizer.optimize(
            num_vars=number_of_variables,
            objective_function=cost_function,
            initial_point=initial_params,
            gradient_function=gradient_function,
        )

        if self.method == "ADAM" or self.method == "AMSGRAD":
            nit = self.optimizer._t
        elif self.method == "NFT":
            nit = self.optimizer._options["maxiter"]
        else:
            nit = self.optimizer.maxiter

        return optimization_result(
            opt_value=value,
            opt_params=solution,
            nit=nit,
            nfev=nfev,
            **construct_history_info(cost_function, keep_history)
        )
