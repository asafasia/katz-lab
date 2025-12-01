"""
        RESONATOR SPECTROSCOPY
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to extract the
'I' and 'Q' quadratures across varying readout intermediate frequencies.
The data is then post-processed to determine the resonator resonance frequency.
This frequency can be used to update the readout intermediate frequency in the configuration under "resonator_IF".

Prerequisites:
    - Ensure calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibrate the IQ mixer connected to the readout line (whether it's an external mixer or an Octave port).
    - Define the readout pulse amplitude and duration in the configuration.
    - Specify the expected resonator depletion time in the configuration.

Before proceeding to the next node:
    - Update the readout frequency, labeled as "resonator_IF", in the configuration.
"""

import numpy as np
from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig

import DC
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
from scipy import signal

from configuration import *
from utils.options import Options


###################
# The QUA program #
###################


class OptionsResonatorSpectroscopy(Options):
    long_pulse: bool = False
    discriminate_ef: bool = False  # long_pulse has to be False


class ResonatorSpectroscopy:
    def __init__(
        self, frequencies: np.ndarray, options: OptionsResonatorSpectroscopy, qubit: str
    ):
        self.frequencies = frequencies
        self.options = options
        self.qubit = qubit
        self.config = config

    def define_program(self):
        with program() as resonator_spec:
            n = declare(int)  # QUA variable for the averaging loop
            f = declare(int)  # QUA variable for the readout frequency
            I1 = declare(fixed)  # QUA variable for the measured 'I' quadrature
            Q1 = declare(fixed)  # QUA variable for the measured 'Q' quadrature\
            I2 = declare(fixed)  # QUA variable for the measured 'I' quadrature
            Q2 = declare(fixed)  # QUA variable for the measured 'Q' quadrature
            I_st1 = declare_stream()  # Stream for the 'I' quadrature
            Q_st1 = declare_stream()  # Stream for the 'Q' quadrature
            I_st2 = declare_stream()  # Stream for the 'I' quadrature
            Q_st2 = declare_stream()  # Stream for the 'Q' quadrature
            n_st = declare_stream()  # Stream for the averaging iteration 'n'

            with for_(
                n, 0, n < self.options.n_avg, n + 1
            ):  # QUA for_ loop for averaging
                with for_(
                    *from_array(f, self.frequencies)
                ):  # QUA for_ loop for sweeping the frequency
                    update_frequency("resonator", f)
                    if self.options.discriminate_ef and not self.options.long_pulse:
                        play("x180", "qubit")
                        wait(100, "qubit")
                        align("qubit", "resonator")
                    measure(
                        "readout",
                        "resonator",
                        None,
                        dual_demod.full("cos", "out1", "sin", "out2", I1),
                        dual_demod.full("minus_sin", "out1", "cos", "out2", Q1),
                    )
                    wait(thermalization_time // 4, "resonator")
                    if self.options.long_pulse:
                        play("saturation", "qubit")
                        # play("saturation", "qubit2")

                    else:
                        if self.options.discriminate_ef:
                            play("x180", "qubit")
                            play("x180_ef", "qubit_ef")
                        else:
                            play("x180", "qubit")
                    wait(100, "qubit")
                    align("qubit", "resonator")
                    measure(
                        "readout",
                        "resonator",
                        None,
                        dual_demod.full("cos", "out1", "sin", "out2", I2),
                        dual_demod.full("minus_sin", "out1", "cos", "out2", Q2),
                    )
                    save(I1, I_st1)
                    save(Q1, Q_st1)
                    save(I2, I_st2)
                    save(Q2, Q_st2)
                    wait(thermalization_time // 4, "resonator")

                save(n, n_st)

            self.program = resonator_spec

            with stream_processing():
                I_st1.buffer(len(self.frequencies)).buffer(self.options.n_avg).save(
                    "I1"
                )
                Q_st1.buffer(len(self.frequencies)).buffer(self.options.n_avg).save(
                    "Q1"
                )
                I_st2.buffer(len(self.frequencies)).buffer(self.options.n_avg).save(
                    "I2"
                )
                Q_st2.buffer(len(self.frequencies)).buffer(self.options.n_avg).save(
                    "Q2"
                )
                n_st.save("iteration")

    def run(self):
        self.define_program()

        config = self.config
        qm = qmm.open_qm(config)
        job = qm.execute(self.program)
        self.results = fetching_tool(
            job, data_list=["I1", "Q1", "I2", "Q2", "iteration"], mode="live"
        )

    def plot(self):
        self.results.plot("I1", "Q1", "I2", "Q2", "iteration")


if __name__ == "__main__":

    qubit = "q10"

    span = 4 * u.MHz
    f_min = resonator_freq - span / 2
    f_max = resonator_freq + span / 2
    df = 20 * u.kHz

    frequencies = resonator_LO - np.arange(f_min, f_max + 0.1, df)

    options = OptionsResonatorSpectroscopy()

    options.n_avg = 100

    qmm = QuantumMachinesManager(host=qm_host)

    experiment = ResonatorSpectroscopy(frequencies, options, qubit)

    experiment.run()
    

    experiment.results
