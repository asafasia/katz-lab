from dataclasses import dataclass
from abc import ABC, abstractmethod
from qm import QuantumMachinesManager, SimulationConfig

from utils import DC
from utils.configuration import qubit_flux_bias_channel


@dataclass
class Options:
    n_avg: int = 1000
    state_discrimination: bool = True
    plot: bool = True
    simulate: bool = False
    save: bool = False


class BaseExperiment(ABC):
    def __init__(
        self, options: Options, config: dict, qmm: QuantumMachinesManager = None
    ):
        self.options = options
        self.config = config
        self.qmm = qmm

    @abstractmethod
    def define_program(self):
        raise NotImplementedError

    @abstractmethod
    def execute_program(self):
        raise NotImplementedError

    @abstractmethod
    def analyze_results(self):
        raise NotImplementedError

    @abstractmethod
    def plot_results(self):
        raise NotImplementedError

    def save_results(self):
        raise NotImplementedError

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
