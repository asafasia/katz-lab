import numpy as np
import matplotlib.pyplot as plt

from qm.qua import *
from qm import QuantumMachinesManager
from qualang_tools.results import fetching_tool, progress_counter

from katz_lab.macros.macros import readout_macro_mahalabonis, qubit_initialization

from katz_lab.experiments.base_experiment import BaseExperiment, Options


class OptionsIQBlobs(Options):
    pass


class IQBlobsExperiment(BaseExperiment):
    def __init__(
        self,
        qubit: str,
        options: OptionsIQBlobs,
        config: dict = None,
        qmm: QuantumMachinesManager = None,
    ):
        super().__init__(qubit=qubit, options=options, config=config, qmm=qmm)

    def define_program(self):
        with program() as IQ_blobs:
            n = declare(int)
            state_g_st = declare_stream()
            state_e_st = declare_stream()
            I_g_st = declare_stream()
            Q_g_st = declare_stream()
            I_e_st = declare_stream()
            Q_e_st = declare_stream()
            n_st = declare_stream()

            with for_(n, 0, n < self.options.n_avg, n + 1):
                # Ground state measurement
                state_g, I_g, Q_g = readout_macro_mahalabonis()
                qubit_initialization(
                    self.options.active_reset,
                    three_state=False,
                    n=self.options.active_reset_n,
                )

                # Excited state measurement
                align()
                play("x180", "qubit")
                align("qubit", "resonator", "qubit_ef")
                state_e, I_e, Q_e = readout_macro_mahalabonis()
                qubit_initialization(
                    self.options.active_reset,
                    three_state=False,
                    n=self.options.active_reset_n,
                )

                # Save streams
                save(I_g, I_g_st)
                save(Q_g, Q_g_st)
                save(I_e, I_e_st)
                save(Q_e, Q_e_st)
                save(state_g, state_g_st)
                save(state_e, state_e_st)
                save(n, n_st)

            with stream_processing():
                I_g_st.save_all("I_g")
                Q_g_st.save_all("Q_g")
                I_e_st.save_all("I_e")
                Q_e_st.save_all("Q_e")
                state_g_st.save_all("state_g")
                state_e_st.save_all("state_e")
                n_st.save("iteration")

        self.program = IQ_blobs

    def execute_program(self):
        # Set qubit bias
        # DC.set_voltage(qubit_flux_bias_channel, flux_bias)
        qm = self.qmm.open_qm(self.config)
        job = qm.execute(self.program)
        results = fetching_tool(
            job,
            data_list=["I_g", "Q_g", "I_e", "Q_e", "state_g", "state_e", "iteration"],
            mode="live",
        )

        while results.is_processing():
            I_g, Q_g, I_e, Q_e, state_g, state_e, iteration = results.fetch_all()
            progress_counter(
                iteration, self.options.n_avg, start_time=results.get_start_time()
            )

        # DC.set_voltage(qubit_flux_bias_channel, 0)

        # Convert to numpy arrays
        self.I_g = np.array(I_g)
        self.Q_g = np.array(Q_g)
        self.I_e = np.array(I_e)
        self.Q_e = np.array(Q_e)
        self.state_g = np.array(state_g)
        self.state_e = np.array(state_e)

        return self.I_g, self.Q_g, self.I_e, self.Q_e, self.state_g, self.state_e

    def analyze_results(self):
        pass
        # IQ discrimination
        # angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(
        #     self.I_g, self.Q_g, self.I_e, self.Q_e, b_print=True, b_plot=True
        # )

        # two_state_discrimination_calib(
        #     np.column_stack((self.I_g, self.Q_g)),
        #     np.column_stack((self.I_e, self.Q_e)),
        #     threshold=[95, 95],
        #     e2f=self.options.e2f,
        #     update_args=self.options.update_args,
        # )

        # fidelity_realtime = np.zeros((2, 2))
        # states = [self.state_g, self.state_e]
        # for i in range(2):
        #     for j in range(2):
        #         fidelity_realtime[i, j] = np.sum(states[i] == j) / len(states[i])

        # print("Fidelity matrix (realtime):")
        # print(fidelity_realtime)

        # self.fidelity_realtime = fidelity_realtime
        # return fidelity_realtime

    def plot_results(self):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(self.I_g, self.Q_g, ".", label="ground", color="C00", alpha=0.5)
        plt.plot(self.I_e, self.Q_e, ".", label="excited", color="C03", alpha=0.2)

        plt.subplot(1, 2, 2)
        plt.hist(self.I_g, bins=50, alpha=0.5, label="ground")
        plt.hist(self.I_e, bins=50, alpha=0.5, label="excited")

        plt.xlabel("I")
        plt.ylabel("Q")
        plt.legend()
        plt.title("IQ Blobs")
        plt.show()

    def save_results(self):
        pass


if __name__ == "__main__":
    qubit = "q10"
    options = OptionsIQBlobs()
    options.n_avg = 10000
    experiment = IQBlobsExperiment(
        qubit=qubit,
        options=options,
    )
    experiment.run()
