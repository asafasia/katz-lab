from abc import ABC, abstractmethod
from qm import QuantumMachinesManager, SimulationConfig

# from katz_lab.utils import DC
from katz_lab.utils.configuration import (
    qubit_flux_bias_channel,
    flux_bias,
    qm_host,
    load_config,
)

from katz_lab.utils.options import Options
from katz_lab.utils.configuration import *


class BaseExperiment(ABC):
    def __init__(
        self,
        qubit: str,
        options: Options,
        config: dict = None,
        qmm: QuantumMachinesManager = None,
    ):

        self.qubit = qubit
        self.qubit_params = args[qubit]["qubit"]

        if qmm is None:
            # qmm = QuantumMachinesManager(host=qm_host)
            qmm = QuantumMachinesManager()

        if config is None:
            config = load_config(options)

        self.qmm = qmm
        self.config = config

        self.options = options

    @abstractmethod
    def define_program(self):
        pass

    @abstractmethod
    def execute_program(self):
        pass

    @abstractmethod
    def analyze_results(self):
        pass

    @abstractmethod
    def plot_results(self):
        pass

    @abstractmethod
    def save_results(self):
        pass

    def run(self):
        self.define_program()

        if self.options.simulate:
            import matplotlib.pyplot as plt

            simulation_config = SimulationConfig(
                duration=10_000
            )  # In clock cycles = 4ns
            job = self.qmm.simulate(self.config, self.program, simulation_config)
            job.get_simulated_samples().con1.plot()

            plt.show()
        else:

            if self.options.dc_set_voltage:
                # DC.set_voltage(qubit_flux_bias_channel, flux_bias)
                pass

            self.execute_program()

            if self.options.dc_set_voltage:
                # DC.set_voltage(qubit_flux_bias_channel, 0)
                pass

            self.analyze_results()
            if self.options.plot:
                self.plot_results()
            if self.options.save:
                self.save_results()
