

class PlotSettings:
    def __init__(self):
        self._plot_setting = None
        self._h_finite_differences = None
        self._s0_array = None

    @property
    def OutputStatistic(self):
        return self._plot_setting

    def set_OutputStatistic(self, value):
        self._plot_setting = value

    @property
    def S0Array(self):
        return self._s0_array

    def set_S0Array(self, value):
        self._s0_array = value

    @property
    def FiniteDifferencesStepWidth(self):
        return self._h_finite_differences

    def set_FiniteDifferencesStepWidth(self, value):
        self._h_finite_differences = value
