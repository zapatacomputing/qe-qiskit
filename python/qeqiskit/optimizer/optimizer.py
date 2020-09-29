from zquantum.core.interfaces.optimizer import Optimizer
from qiskit.aqua.components.optimizers import SPSA, ADAM
from scipy.optimize import OptimizeResult


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
        if "keep_value_history" not in self.options.keys():
            self.keep_value_history = False
        else:
            self.keep_value_history = self.options["keep_value_history"]
            del self.options["keep_value_history"]
            Warning(
                "Orquestra does not support keeping history of the evaluations yet."
            )

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

        def wrapped_(params):
            history.append({"params": params})
            if self.keep_value_history:
                value = cost_function.evaluate(params).value
                history[-1]["value"] = value
                print(f"Iteration {len(history)}: {value}", flush=True)
            else:
                print(f"iteration {len(history)}")
            print(f"{params}", flush=True)

        number_of_variables = len(initial_params)
        cost_function_wrapper = lambda params: cost_function.evaluate(params).value
        solution, value, nit = optimizer.optimize(
            num_vars=number_of_variables,
            objective_function=cost_function_wrapper,
            initial_point=initial_params,
            gradient_function=cost_function.get_gradient,
        )
        optimization_results = {}
        optimization_results["opt_value"] = value
        optimization_results["opt_params"] = solution
        optimization_results["history"] = {}
        optimization_results["nfev"] = len(cost_function.evaluations_history)
        optimization_results["nit"] = nit

        return OptimizeResult(optimization_results)
