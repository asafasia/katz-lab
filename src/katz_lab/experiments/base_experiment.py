from dataclasses import dataclass
from abc import ABC, abstractmethod
from qm import QuantumMachinesManager, SimulationConfig

# try:
#     from katz_lab.utils import DC
#     from katz_lab.utils.configuration import qubit_flux_bias_channel, flux_bias

# except ImportError:
#     DC = None

from qualang_tools.results import progress_counter, fetching_tool
from katz_lab.utils.configuration import qm_host, load_config
from dataclasses import dataclass, field


@dataclass
class Options:
    n_avg: int = 100
    state_discrimination: bool = True
    plot: bool = True
    simulate: bool = False
    save: bool = False
    dc_set_voltage: bool = False
    states_to_measure: list = field(default_factory=lambda: ["gef"])
    update_args: bool = False
    active_reset: bool = False
    active_reset_n: int = 2


class BaseExperiment(ABC):
    def __init__(
        self,
        qubit: str,
        options: Options,
        config: dict = None,
        qmm: QuantumMachinesManager = None,
    ):

        self.qubit = qubit
        if qmm is None:
            qmm = QuantumMachinesManager(host=qm_host)

        if config is None:
            config = load_config()

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

        # DC.set_voltage(qubit_flux_bias_channel, flux_bias)  # Set the flux bias voltage

        self.execute_program()

        # DC.set_voltage(qubit_flux_bias_channel, 0)  # Set the flux bias voltage

        self.analyze_results()
        if self.options.plot:
            self.plot_results()
        if self.options.save:
            self.save_results()
