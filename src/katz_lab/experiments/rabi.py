import numpy as np
from scipy.optimize import curve_fit
from qm.qua import *
from qm import QuantumMachinesManager, SimulationConfig
import matplotlib.pyplot as plt

# from experiment_utils import DC
# from experiment_utils.change_args import modify_json
# from experiment_utils.configuration import *
# from qualang_tools.results import progress_counter, fetching_tool
# from qualang_tools.loops import from_array
# import experiment_utils.labber_util as lu
# from experiment_utils.macros import readout_macro_mahalabonis, qubit_initialization
# from experiment_utils.time_estimation import calculate_time




if __name__ == "__main__":

    from katz_lab.configuration import *
    from qualang_tools.loops import from_array

    qmm = QuantumMachinesManager(host=qm_host)

    from macros.macros import readout_macro_mahalabonis

    n_avg = 1
    amplitudes = np.arange(0.5,1,100)
    with program() as power_rabi:
        n = declare(int)  # QUA variable for the averaging loop
        a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor

        I_st = declare_stream()  # Stream for the 'I' quadrature
        Q_st = declare_stream()  # Stream for the 'Q' quadrature
        n_st = declare_stream()  # Stream for the averaging iteration 'n'
        state_st = declare_stream()  # Stream for the qubit state

        with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
            with for_(*from_array(a, amplitudes)):  # QUA for_ loop for sweeping the pulse amplitude pre-factor
                play("x180" * amp(a), "qubit")
                align("qubit", "resonator")
                state, I, Q = readout_macro_mahalabonis()
                save(I, I_st)
                save(Q, Q_st)
                save(state, state_st)
            save(n, n_st)

        with stream_processing():
            I_st.buffer(len(amplitudes)).average().save("I")
            Q_st.buffer(len(amplitudes)).average().save("Q")
            state_st.buffer(len(amplitudes)).average().save("state")
            n_st.save("iteration")

    sim_config = SimulationConfig(duration=10_00)
    job = qmm.simulate(config, power_rabi, sim_config)
    job.get_simulated_samples().con1.plot()


    plt.show()
