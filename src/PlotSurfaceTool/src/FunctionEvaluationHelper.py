import numpy as np

class FunctionEvaluationHelper:
    @staticmethod
    def evaluate_function(fixed_func, x):
        y = np.zeros(len(x))  # Create an array to store the y-values
        for i in range(0, len(x)):
            result_base = fixed_func(x[i])
            if isinstance(result_base, tuple):
                  y[i], _ = result_base
            else:
                y[i] = result_base
        return y

    @staticmethod
    def compute_delta(fixed_func, x, hFinDiff, hardCodedh = None):
        if hardCodedh is None:
            h = hFinDiff
        else:
            h = hardCodedh   
        delta = np.zeros(len(x))
        for i in range(0, len(x)):
            result_base = fixed_func(x[i])
            if isinstance(result_base, tuple):
                  _, delta[i] = result_base
            else:
                delta[i] = np.divide(fixed_func(x[i] + h) - result_base, h)
        return delta

    @staticmethod
    def compute_gamma(fixed_func, x, hFinDiff, hardCodedh = None):
        if hardCodedh is None:
            h = hFinDiff
        else:
            h = hardCodedh   
        gamma = np.zeros(len(x))
        for i in range(0, len(x)):
            result_base = fixed_func(x[i] - h)
            if isinstance(result_base, tuple):
                _, delta_minus = result_base
                _, delta_plus = fixed_func(x[i] + h)
                gamma[i] = (delta_plus - delta_minus) / (2 * h)
            else:
                gamma[i] = np.divide(fixed_func(x[i] + h)  - 2*  fixed_func(x[i])  + result_base , h * h)
        return gamma
