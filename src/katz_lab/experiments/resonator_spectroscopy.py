import numpy as np
from qm.qua import *
from qm import QuantumMachinesManager
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array

from utils.configuration import *
from experiments.base_experiment import Options
import matplotlib.pyplot as plt

from experiments.base_experiment import BaseExperiment


class OptionsResonatorSpectroscopy(Options):
    long_pulse: bool = False
    discriminate_ef: bool = False


class ResonatorSpectroscopy(BaseExperiment):
    def __init__(
        self,
        qubit: str,
        frequencies: np.ndarray,
        options: OptionsResonatorSpectroscopy,
        config: dict,
    ):
        self.frequencies = frequencies
        self.options = options
        self.qubit = qubit
        self.config = config

        super().__init__(options=options, config=config, qmm=None)

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

        while self.results.is_processing():
            I1, Q1, I2, Q2, iteration = self.results.fetch_all()

            S1 = u.demod2volts(I1 + 1j * Q1, readout_len)
            S2 = u.demod2volts(I2 + 1j * Q2, readout_len)
            R1 = np.abs(S1)  # Amplitude
            R2 = np.abs(S2)  # Amplitude

            R1_var = np.var(R1, axis=0)
            R2_var = np.var(R2, axis=0)

            Var = np.sqrt(R1_var / 2 + R2_var / 2)

            R1 = np.mean(R1, axis=0)
            R2 = np.mean(R2, axis=0)

            phase1 = np.angle(S1)  # Phase
            phase2 = np.angle(S2)  # Phase

            phase1 = signal.detrend(np.unwrap(phase1))
            phase2 = signal.detrend(np.unwrap(phase2))
            phase1 = np.mean(phase1, axis=0)
            phase2 = np.mean(phase2, axis=0)
            #
            S1 = np.mean(S1, axis=0)
            S2 = np.mean(S2, axis=0)
            VarS = np.sqrt(np.abs(np.var(S1) / 2 + np.var(S2) / 2))
            progress_counter(
                iteration, self.options.n_avg, start_time=self.results.get_start_time()
            )

        fig = plt.figure()
        diff = np.abs(S1 - S2) / VarS
        # diff = np.abs(R1 - R2)/Var
        # diff = np.abs(phase1 - phase2)/np.pi/2

        res_freq = frequencies[np.argmax(diff)]
        print("Resonator  freq is: ", (resonator_LO - res_freq) / 1e6, "MHz")

        plt.suptitle(
            f"Resonator spectroscopy - LO = {resonator_LO / u.GHz} GHz "
            f"\nRESONATOR : pulse amplitude = {readout_amp} , "
            f" length = {readout_len / 1e3} us"
            f"\n QUBIT : pulse amplitude = {saturation_amp}, length = {saturation_len / 1e3} us"
        )
        plt.subplot(311)
        plt.axvline(x=(resonator_LO - res_freq) / u.MHz, color="r", linestyle="--")
        plt.axvline(x=resonator_freq / u.MHz, color="g", linestyle="--")
        plt.plot((resonator_LO - frequencies) / u.MHz, R1, label="Without Drive")
        plt.plot((resonator_LO - frequencies) / u.MHz, R2, label="With Drive")
        plt.legend()
        #
        plt.ylim([0, max(R1) * 1.2])

        plt.ylabel(r"$R=\sqrt{I^2 + Q^2}$ [V]")
        plt.subplot(312)
        plt.plot((resonator_LO - frequencies) / u.MHz, phase1, label="Without Drive")
        plt.plot((resonator_LO - frequencies) / u.MHz, phase2, label="With Drive")

        # plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xlabel("Intermediate frequency [MHz]")
        plt.ylabel("Phase [rad]")
        plt.legend()
        plt.subplot(313)
        plt.ylabel("Diff ")
        plt.plot((resonator_LO - frequencies) / u.MHz, diff)
        plt.axvline(
            x=(resonator_LO - res_freq) / u.MHz,
            color="r",
            linestyle="--",
            label=f"new freq = {(resonator_LO - res_freq) / u.MHz} MHz",
        )
        plt.axvline(
            x=resonator_freq / u.MHz,
            color="g",
            linestyle="--",
            label=f"old freq = {(resonator_freq) / u.MHz} MHz",
        )
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot(self):
        pass


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

    print(experiment.results)

