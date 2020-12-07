from zquantum.core.history.recorder import recorder
from zquantum.core.interfaces.optimizer import Optimizer, optimization_result
from qiskit.aqua.components.optimizers import SPSA, ADAM
from scipy.optimize import OptimizeResult


class _CostFunctionWrapper:
    def __init__(self, cost_function):
        self.cost_function = cost_function
        self.number_of_calls = 0

    def __call__(self, params):
        self.number_of_calls += 1
        return self.cost_function(params)


class QiskitOptimizer(Optimizer):
    def __init__(self, method, options={}):
        """
        Args:
            method(str): specifies optimizer to be used. Currently supports "ADAM", "AMSGRAD" and "SPSA".
            options(dict): dictionary with additional options for the optimizer.

        Supported values for the options dictionary:
        Options:
            keep_value_history(bool): boolean flag indicating whether the history of evaluations should be stored or not.
            **kwargs: options specific for particular scipy optimizers.

        """

        self.method = method
        self.options = options
        self.keep_value_history = self.options.pop("keep_value_history", False)

    def minimize(self, cost_function, initial_params=None):
        """
        Minimizes given cost function using optimizers from Qiskit Aqua.

        Args:
            cost_function(): python method which takes numpy.ndarray as input
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

        if self.keep_value_history:
            cost_function_wrapper = recorder(cost_function)
        else:
            cost_function_wrapper = _CostFunctionWrapper(cost_function)

        gradient_function = None
        if hasattr(cost_function, "gradient") and callable(
            getattr(cost_function, "gradient")
        ):
            gradient_function = cost_function.gradient

        solution, value, nit = optimizer.optimize(
            num_vars=number_of_variables,
            objective_function=cost_function_wrapper,
            initial_point=initial_params,
            gradient_function=gradient_function,
        )

        if self.keep_value_history:
            nfev = len(cost_function_wrapper.history)
            history = cost_function_wrapper.history
        else:
            nfev = cost_function_wrapper.number_of_calls
            history = []

        return optimization_result(
            opt_value=value,
            opt_params=solution,
            nit=nit,
            history=history,
            nfev=nfev,
        )
