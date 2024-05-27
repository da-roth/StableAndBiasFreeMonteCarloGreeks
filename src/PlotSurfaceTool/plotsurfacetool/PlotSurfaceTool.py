import matplotlib.pyplot as plt
from .evaluation import FunctionEvaluationHelper

class PlotSurfaceTool:
    @staticmethod
    def Run(fixed_func, PlotSettings):
        if PlotSettings.OutputStatistic == OutputStatistic.PresentValue:
            print(PlotSettings.OutputStatistic)
            y = FunctionEvaluationHelper.evaluate_function(fixed_func, PlotSettings.S0Array)
            PlotSurfaceTool.plot_valuation_1d(PlotSettings.S0Array, y, 'Present value')
        elif PlotSettings.OutputStatistic== OutputStatistic.Delta:
            y = FunctionEvaluationHelper.evaluate_function(fixed_func, PlotSettings.S0Array)
            delta = FunctionEvaluationHelper.compute_delta(fixed_func, PlotSettings.S0Array, PlotSettings.FiniteDifferencesStepWidth)
            fig, axs = plt.subplots(1, 2, figsize=(10, 3))
            axs[0].plot(PlotSettings.S0Array, y)
            axs[0].set_xlabel('x')
            axs[0].set_ylabel('Value')
            axs[0].set_title('Present value')
            axs[1].plot(PlotSettings.S0Array, delta)
            axs[1].set_xlabel('x')
            axs[1].set_ylabel('Delta')
            axs[1].set_title('Delta')
            plt.tight_layout()
            plt.show()
        elif PlotSettings.OutputStatistic == OutputStatistic.Gamma:
            y = FunctionEvaluationHelper.evaluate_function(fixed_func, PlotSettings.S0Array)
            delta = FunctionEvaluationHelper.compute_delta(fixed_func, PlotSettings.S0Array, PlotSettings.FiniteDifferencesStepWidth)
            gamma = FunctionEvaluationHelper.compute_gamma(fixed_func, PlotSettings.S0Array, PlotSettings.FiniteDifferencesStepWidth)
            fig, axs = plt.subplots(1, 3, figsize=(10, 3))
            axs[0].plot(PlotSettings.S0Array, y)
            axs[0].set_xlabel('x')
            axs[0].set_ylabel('Value')
            axs[0].set_title('Present value')
            axs[1].plot(PlotSettings.S0Array, delta)
            axs[1].set_xlabel('x')
            axs[1].set_ylabel('Delta')
            axs[1].set_title('Delta')
            axs[2].plot(PlotSettings.S0Array, gamma)
            axs[2].set_xlabel('x')
            axs[2].set_ylabel('Gamma')
            axs[2].set_title('Gamma')
            plt.tight_layout()
            plt.show()

    @staticmethod
    def plot_valuation_1d(x, y, title):
        plt.plot(PlotSettings.S0Array, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        plt.show()
