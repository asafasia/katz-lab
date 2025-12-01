from scipy.optimize import curve_fit
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


# from experiment_utils.MHz_to_Volt import amp_Volt_to_MHz
# from experiment_utils.macros import readout_macro, qubit_initialization, readout_macro_mahalabonis
# import configuration
# from experiment_utils.saver import Saver
# reload(configuration)
from qm.qua import *
from qm import QuantumMachinesManager, SimulationConfig

from katz_lab.experiments.base_experiment import BaseExperiment
from katz_lab.utils.options import Options

from katz_lab.utils.configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array


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

        self.frequencies = frequencies

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
                    # qubit_initialization(
                    #     self.active_reset, three_state=self.AR_three_state
                    # )
                    update_frequency("qubit", df)
                    play(
                        "saturation",
                        "qubit",
                        duration=self.pulse_length // 4,
                    )
                    wait(100, "qubit")
                    align("qubit", "resonator")
                    state, I, Q = readout_macro_mahalabonis()
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
        # qm = self.qmm.open_qm(self.config)
        # job = qm.execute(self.program)
        # results = fetching_tool(
        #     job, data_list=["state", "I", "Q", "iteration"], mode="live"
        # )
        # while results.is_processing():
        #     state, I, Q, iteration = results.fetch_all()

        #     progress_counter(iteration, self.n_avg, start_time=results.get_start_time())

        # results = dict()
        # results["state"] = state
        # results["I"] = I
        # results["Q"] = Q

        # return results
        pass

    def analyze_results(self):
        pass

    def plot_results(self, with_fit=False):
        # if self.state_discrimination:
        #     Y = self.state
        # else:
        #     # I_contrast = (np.max(np.max(self.I)) - np.min(np.min(self.I))) / np.abs(np.mean(np.mean(self.I)))
        #     # Q_contrast = (np.max(np.max(self.Q)) - np.min(np.min(self.Q))) / np.abs(np.mean(np.mean(self.Q)))

        #     I_contrast = (np.max(np.max(self.I)) - np.min(np.min(self.I)))
        #     Q_contrast = (np.max(np.max(self.Q)) - np.min(np.min(self.Q)))
        #     Y = self.I if I_contrast > Q_contrast else self.Q
        #     quad_name = "I" if I_contrast > Q_contrast else "Q"

        # t1 = qubit_args['T1']
        # t2 = qubit_args['T2']
        # t2_limit = 1 / t2 / np.pi * 1e3

        # if self.span < 0.8 * u.MHz:
        #     plt.axvline(0, color='k', linestyle='--')
        #     plt.axvline(t2_limit / 2, color='b', linestyle='-', label=f'T2 limit = {t2_limit * 1e3:.1f}  kHz')
        #     plt.axvline(-t2_limit / 2, color='b', linestyle='-')

        # plt.figure()
        # plt.plot(self.detunings / 1e6, Y)

        # try:
        #     def lorentzian(x, a, b, c, d):
        #         return a / (1 + ((x - b) / c) ** 2) + d

        #     if self.two_photon:
        #         freq_guess = qubit_anharmonicity / 2
        #     else:
        #         freq_guess = 0
        #     start_d = np.mean(Y)
        #     start_a = Y[np.argmax(np.abs(Y - np.mean(Y)))] - start_d
        #     p0 = [start_a, freq_guess, self.span / 5, start_d]
        #     # bounds = [[0,1],[-50e6,50e6],[-50e6,50e6],[0,1]]
        #     bounds = [[0, -50e6, -50e6, 0], [1, 50e6, 50e6, 1]]
        #     # popt = curve_fit(lorentzian, self.detunings, Y, p0=p0, bounds=bounds)[0]
        #     popt = curve_fit(lorentzian, self.detunings, Y, p0=p0)[0]
        #     a = popt[0]
        #     b = popt[1]
        #     c = popt[2]
        #     d = popt[3]
        #     a = self.pulse_amp * saturation_amp
        #     self.a_MHz = amp_Volt_to_MHz(a)
        #     plt.title(
        #         f'Qubit spectroscopy {self.qubit}\n amp = {self.pulse_amp * saturation_amp * 1e3:.2f} mV ({self.a_MHz:.3f} MHz), ')
        #     if with_fit:
        #         plt.plot(self.detunings / 1e6, lorentzian(self.detunings, *popt), label='fit')
        #     max_detuning = b

        #     fwhm = 2 * c / 1e3

        #     print('FWHM = ', fwhm, 'kHz')

        #     print(d)
        #     if with_fit:
        #         plt.axhline(d, color='k')
        #         plt.axvline(b / 1e6 + fwhm / 2 / 1e3, color='r', linestyle='--', label=f'fit fwhm = {fwhm:.2f} kHz')
        #         plt.axvline(b / 1e6 - fwhm / 2 / 1e3, color='r', linestyle='--')
        #         plt.axvline(b / 1e6, color='g', linestyle=':', label=f'fit center = {b / 1e6:.2f} MHz')
        #         plt.xlim([self.detunings[0] / 1e6, self.detunings[-1] / 1e6])
        #         plt.xlabel("Detuning (MHz)")
        #         plt.ylabel("State" if self.state_discrimination else quad_name)
        #         # plt.legend()
        #     # plt.show()

        #     print('freq fit = ', b + qubit_freq, 'Hz')
        #     self.qubit_fit_freq = b + qubit_freq
        # except Exception as e:
        #     print("fit failed")
        #     print(e)
        #     max_detuning = self.detunings[np.argmax(self.state)]
        #     print("Max detuning = ", max_detuning / 1e6, "MHz")

        # self.qubit_max_freq = qubit_freq + max_detuning
        # print('qubit freq = ', qubit_freq)
        # print('max detuning = ', max_detuning)
        # print('max freq = ', self.qubit_max_freq)
        # # plt.axvline(qubit_freq/1e6, color='k', linestyle='--', label = "current")
        # plt.axvline(max_detuning / 1e6, color='blue', linestyle='--', label='max')
        # print('T2 limit = ', t2_limit * 1e3, 'kHz')
        # plt.xlabel("Detuning (MHz)")
        # plt.ylabel("State" if self.state_discrimination else quad_name)
        # plt.legend()
        # # plt.ylim([0, 1])
        # plt.show()
        pass

    def update_max_freq(self, from_fit=True):
        # # max_freq = self.frequencies[np.argmax(self.state)]
        # update_freq = self.qubit_fit_freq if from_fit else self.qubit_max_freq
        # if not self.two_photon:
        #     key_to_change = "qubit_freq_sweet_spot" if sweet_spot_flag else "qubit_freq_zero_bias"
        #     modify_json(self.qubit, 'qubit', key_to_change, int(update_freq))
        # else:
        #     alpha = (qubit_freq - update_freq) * 2
        #     print('new alpha:', alpha / 1e6, 'MHz')
        #     modify_json(self.qubit, 'qubit', 'qubit_anharmonicity', -int(alpha / 10) * 10)

        pass

    def save_results(self):
        pass


if __name__ == "__main__":
    qubit = "q10"
    options = OptionsQubitSpectroscopy()

    qubit_LO = 5.5e9
    center = 0
    span = 100e6
    N = 100
    frequencies = qubit_LO - np.arange(center - span / 2, center + span / 2, span // N)

    experiment = QubitSpectroscopy(
        frequencies=frequencies,
        qubit=qubit,
        options=options,
    )
    experiment.run()
