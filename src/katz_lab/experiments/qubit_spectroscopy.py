from scipy.optimize import curve_fit
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from qm.qua import *
from qm import QuantumMachinesManager, SimulationConfig

from katz_lab.experiments.base_experiment import BaseExperiment
from katz_lab.utils.options import Options

from katz_lab.utils.configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array

from katz_lab.utils.macros import *


@dataclass
class OptionsQubitSpectroscopy(Options):
    two_photon: bool = False


class QubitSpectroscopy(BaseExperiment):
    def __init__(
        self,
        qubit: str,
        frequencies: np.ndarray,
        options: OptionsQubitSpectroscopy = None,
        config: dict = None,
        qmm: QuantumMachinesManager = None,
    ):
        if options is None:
            options = OptionsQubitSpectroscopy()
        super().__init__(qubit=qubit, options=options, config=config, qmm=qmm)

        self.frequencies = qubit_LO - frequencies

    def define_program(self):
        with program() as qubit_spec:
            n = declare(int)  # QUA variable for the averaging loop
            df = declare(int)  # QUA variable for the qubit frequency
            n_st = declare_stream()  # Stream for the averaging iteration 'n'
            state_st = declare_stream()  # Stream for the qubit state
            I_st = declare_stream()  # Stream for the 'I' quadrature
            Q_st = declare_stream()  # Stream for the 'Q' quadrature

            with for_(n, 0, n < self.options.n_avg, n + 1):
                with for_(*from_array(df, self.frequencies)):
                    qubit_initialization(self.options.active_reset)
                    update_frequency("qubit", df)
                    play(
                        "saturation",
                        "qubit",
                        duration=self.qubit_params["saturation_pulse"]["length"] // 4,
                    )
                    wait(100, "qubit")
                    align("qubit", "resonator")
                    state, I, Q = readout_macro_two_state()
                    save(state, state_st)
                    save(I, I_st)
                    save(Q, Q_st)
                save(n, n_st)

            with stream_processing():
                I_st.buffer(len(self.frequencies)).average().save("I")
                Q_st.buffer(len(self.frequencies)).average().save("Q")
                state_st.buffer(len(self.frequencies)).average().save("state")
                n_st.save("iteration")

        self.program = qubit_spec

    def execute_program(self):
        qm = self.qmm.open_qm(self.config)
        job = qm.execute(self.program)
        results = fetching_tool(
            job, data_list=["state", "I", "Q", "iteration"], mode="live"
        )
        while results.is_processing():
            state, I, Q, iteration = results.fetch_all()
            progress_counter(
                iteration, self.options.n_avg, start_time=results.get_start_time()
            )

        results = dict()
        results["state"] = state
        results["I"] = I
        results["Q"] = Q

        self.results = results

        return results

    def analyze_results(self):
        pass

    def plot_results(self):

        plt.figure()
        if self.options.state_discrimination:
            plt.plot(self.frequencies / 1e6, self.results["state"])
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("State")
        else:

            plt.subplot(211)
            plt.plot(self.frequencies / 1e6, self.results["I"])
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("I")
            plt.subplot(212)
            plt.plot(self.frequencies / 1e6, self.results["Q"])
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Q")
        plt.tight_layout()
        plt.show()

    def update_max_freq(self, from_fit=True):
        pass

    def save_results(self):
        pass


if __name__ == "__main__":
    qubit = "q10"

    options = OptionsQubitSpectroscopy()
    options.state_discrimination = False
    options.n_avg = 3
    options.active_reset = False
    options.simulate = False

    span = 100e6
    N = 11
    frequencies = np.arange(-span / 2, span / 2, span // N)

    experiment = QubitSpectroscopy(
        frequencies=frequencies,
        qubit=qubit,
        options=options,
    )
    experiment.run()
