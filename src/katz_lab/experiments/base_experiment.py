from abc import ABC, abstractmethod
from qm import QuantumMachinesManager, SimulationConfig

from katz_lab.utils import DC
from katz_lab.utils.configuration import (
    qubit_flux_bias_channel,
    flux_bias,
    qm_host,
    load_config,
)

from katz_lab.utils.options import Options


class BaseExperiment(ABC):
    def __init__(
        self,
        qubit: str,
        options: Options,
        config: dict = None,
        qmm: QuantumMachinesManager = None,
    ):

        self.qubit = qubit
        self.
        if qmm is None:
            qmm = QuantumMachinesManager(host=qm_host)

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

        DC.set_voltage(qubit_flux_bias_channel, flux_bias)  # Set the flux bias voltage

        self.execute_program()

        DC.set_voltage(qubit_flux_bias_channel, 0)  # Set the flux bias voltage

        self.analyze_results()
        if self.options.plot:
            self.plot_results()
        if self.options.save:
            self.save_results()
