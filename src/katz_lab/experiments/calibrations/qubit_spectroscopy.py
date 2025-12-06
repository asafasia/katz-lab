from scipy.optimize import curve_fit
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from qm.qua import *
from qm import QuantumMachinesManager, SimulationConfig

from katz_lab.experiments.core.base_experiment import BaseExperiment
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
        relative_amplitude: float = 1,
        options: OptionsQubitSpectroscopy = None,
        config: dict = None,
        qmm: QuantumMachinesManager = None,
    ):
        if options is None:
            options = OptionsQubitSpectroscopy()
        super().__init__(qubit=qubit, options=options, config=config, qmm=qmm)

        qubit_IF = qubit_LO - qubit_freq
        self.frequencies_IF = qubit_IF + frequencies
        self.frequencies_RF = qubit_LO - self.frequencies_IF
        self.relative_amplitude = relative_amplitude

    def define_program(self):
        with program() as qubit_spec:
            n = declare(int)  # QUA variable for the averaging loop
            df = declare(int)  # QUA variable for the qubit frequency
            n_st = declare_stream()  # Stream for the averaging iteration 'n'
            state_st = declare_stream()  # Stream for the qubit state
            I_st = declare_stream()  # Stream for the 'I' quadrature
            Q_st = declare_stream()  # Stream for the 'Q' quadrature

            with for_(n, 0, n < self.options.n_avg, n + 1):
                with for_(*from_array(df, self.frequencies_IF)):
                    qubit_initialization(self.options.active_reset)
                    update_frequency("qubit", df)
                    play(
                        "saturation" * amp(self.relative_amplitude),
                        "qubit",
                        duration=self.qubit_params["saturation_pulse"]["length"] // 4,
                    )
                    wait(100, "qubit")
                    align("qubit", "resonator")
                    state, I, Q = readout_macro()
                    save(state, state_st)
                    save(I, I_st)
                    save(Q, Q_st)
                save(n, n_st)

            with stream_processing():
                I_st.buffer(len(self.frequencies_IF)).average().save("I")
                Q_st.buffer(len(self.frequencies_IF)).average().save("Q")
                state_st.buffer(len(self.frequencies_IF)).average().save("state")
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

        data = dict()
        data["states"] = state
        data["I"] = I
        data["Q"] = Q
        data["freqs"] = self.frequencies_RF

        self.data = data

        return data

    def analyze_results(self):

        freqs = self.data["freqs"]
        states = self.data["states"]
        I = self.data["I"]
        Q = self.data["Q"]

        # find max freq
        max_freq = self.frequencies_RF[np.argmax(states)]

        self.data["max_freq"] = max_freq

        print(f"Max freq: {max_freq}")

    def plot_results(self):

        freqs = self.frequencies_RF

        plt.figure()
        if self.options.state_discrimination:
            plt.plot(freqs / 1e6, self.data["states"])
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("State")
        else:

            plt.subplot(211)
            plt.plot(freqs / 1e6, self.data["I"])
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("I")
            plt.subplot(212)
            plt.plot(freqs / 1e6, self.data["Q"])
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Q")

        plt.axvline(x=self.data["max_freq"] / 1e6, color="r", linestyle="--")
        plt.tight_layout()

        plt.show()

    def update_params(self):
        answer = input(f"Update {self.qubit} frequency? (y/n) [y] ").strip().lower()
        if answer in ("", "y"):
            # user pressed Enter or typed 'y'
            args[f"{self.qubit}/qubit/qubit_ge_freq"] = int(self.data["max_freq"])
        else:
            print("Not updating frequency")

    def save_results(self):
        pass


if __name__ == "__main__":
    qubit = "q10"

    options = OptionsQubitSpectroscopy()
    options.state_discrimination = True
    options.n_avg = 100
    options.simulate = False
    options.update_args = True

    span = 1e6
    N = 101
    frequencies = np.linspace(-span / 2, span / 2, N, dtype=int)

    relative_amplitude = 0.001

    experiment = QubitSpectroscopy(
        frequencies=frequencies,
        qubit=qubit,
        options=options,
        relative_amplitude=relative_amplitude,
    )

    experiment.run()
