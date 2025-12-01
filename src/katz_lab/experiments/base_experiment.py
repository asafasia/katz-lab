from dataclasses import dataclass
from abc import ABC, abstractmethod
from qm import QuantumMachinesManager, SimulationConfig


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
        self.execute_program()
        self.analyze_results()
        if self.options.plot:
            self.plot_results()
        if self.options.save:
            self.save_results()
