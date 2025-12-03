# Refactored Power Rabi Experiment (Updated Script)

import numpy as np
from scipy.optimize import curve_fit
from qm.qua import *
import matplotlib.pyplot as plt

from katz_lab.utils.macros import readout_macro, qubit_initialization


from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array

from katz_lab.experiments.base_experiment import BaseExperiment
from katz_lab.utils.options import Options
from katz_lab.utils.configuration import *


class OptionsPowerRabi(Options):
    num_pis: int = 4


class PowerRabiExperiment(BaseExperiment):
    def __init__(
        self,
        qubit: str,
        options,
        amplitudes,
    ):
        super().__init__(qubit, options)

        self.amplitudes = amplitudes
        self.state_discrimination = False
        self.rabi_amp = 0.02

    def define_program(self):
        with program() as power_rabi:
            n = declare(int)
            a = declare(fixed)
            I_st = declare_stream()
            Q_st = declare_stream()
            n_st = declare_stream()
            state_st = declare_stream()

            with for_(n, 0, n < self.options.n_avg, n + 1):
                with for_(*from_array(a, self.amplitudes)):
                    qubit_initialization(self.options.active_reset)

                    for _ in range(self.options.num_pis):
                        play("x180" * amp(a), "qubit")

                    align("qubit", "resonator")
                    state, I, Q = readout_macro()

                    save(I, I_st)
                    save(Q, Q_st)
                    save(state, state_st)

                save(n, n_st)

            with stream_processing():
                I_st.buffer(len(self.amplitudes)).average().save("I")
                Q_st.buffer(len(self.amplitudes)).average().save("Q")
                state_st.buffer(len(self.amplitudes)).average().save("state")
                n_st.save("iteration")

        self.program = power_rabi

    def execute_program(self):

        qm = self.qmm.open_qm(self.config)
        self.job = qm.execute(self.program)
        results = fetching_tool(
            self.job, data_list=["I", "Q", "state", "iteration"], mode="live"
        )

        while results.is_processing():
            I, Q, state, iteration = results.fetch_all()
            # I, Q = u.demod2volts(I, readout_len), u.demod2volts(Q, readout_len)
            # S = u.demod2volts(I + 1j * Q, readout_len)

            progress_counter(
                iteration, self.options.n_avg, start_time=results.get_start_time()
            )

        self.results = results

    def analyze_results(self):
        I, Q, state, iteration = self.results.fetch_all()

        self.data["amplitudes"] = self.amplitudes
        self.data["I"] = I
        self.data["Q"] = Q
        self.data["state"] = state

        if self.state_discrimination:
            self.y = state
        else:
            Ic = np.max(I) - np.min(I)
            Qc = np.max(Q) - np.min(Q)
            self.y = I if Ic > Qc else Q
            self.quad_name = "I" if Ic > Qc else "Q"

        def cos_fit(x, a, b, c, d):
            return a * np.cos(2 * np.pi * 1 / b * x + c) + d

        try:
            self.fit_args = curve_fit(
                cos_fit,
                self.x,
                self.y,
                p0=[
                    max(self.y) / 2 - min(self.y) / 2,
                    self.rabi_amp * 2,
                    np.pi,
                    np.mean(self.y),
                ],
                maxfev=100000,
                xtol=1e-8,
                ftol=1e-8,
            )[0]
        except:
            self.fit_args = None

    def plot_results(self):

        amplitudes = self.data["amplitudes"]
        I = self.data["I"]
        Q = self.data["Q"]
        state = self.data["state"]

        plt.plot(amplitudes * self.rabi_amp * self.options.num_pis * 1e3, state, ".")

        # if self.fit_args is not None:

        #     def cos_fit(x, a, b, c, d):
        #         return a * np.cos(2 * np.pi * 1 / b * x + c) + d

        #     plt.plot(
        #         self.x * 1e3,
        #         cos_fit(self.x, *self.fit_args),
        #         "r-",
        #         label=f"fit rabi amp = {self.fit_args[1] / 2:.5f} V",
        #     )

        plt.title("Power Rabi g->e transition")
        plt.xlabel("Rabi amplitude (mV)")
        ylabel = (
            "state"
            if self.options.state_discrimination
            else f"{self.quad_name} quadrature (V)"
        )
        plt.ylabel(ylabel)
        # plt.legend()
        plt.show()

    def save_results(self):
        # meta_data = {
        #     "user": user,
        #     "n_avg": self.n_avg,
        #     "args": args,
        #     "tags": [self.qubit],
        # }

        # measured_data = dict(states=self.state, I=self.I, Q=self.Q)
        # sweep_parameters = dict(rabi_amp=self.x)
        # units = dict(rabi_amp="V", I="V", Q="V")

        # exp_result = dict(
        #     measured_data=measured_data,
        #     sweep_parameters=sweep_parameters,
        #     units=units,
        #     meta_data=meta_data,
        # )

        # lu.create_logfile("power-rabi", **exp_result, loop_type="1d")

        # if self.update_args and self.fit_args is not None:
        #     modify_json(
        #         self.qubit, "qubit", f"{pulse_type}180_amp", self.fit_args[1] / 2
        #     )
        #     modify_json(
        #         self.qubit, "qubit", f"{pulse_type}90_amp", self.fit_args[1] / 4
        #     )
        pass


if __name__ == "__main__":
    qubit = "q10"
    options = OptionsPowerRabi()
    options.n_avg = 100
    options.n_a = 100
    options.num_pis = 4

    amps = np.linspace(0, 1, 100)
    experiment = PowerRabiExperiment(
        qubit=qubit,
        options=options,
        amplitudes=amps,
    )
    experiment.run()
