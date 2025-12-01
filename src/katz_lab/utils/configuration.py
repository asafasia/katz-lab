import json
from qualang_tools.units import unit
from library.pulses import *
from qualang_tools.config.waveform_tools import drag_gaussian_pulse_waveforms


args_path = "C:/Users/owner/Janis Lab Code repos/Guy/janis-lab-opx+/experiment_utils/"
optimal_weights_path = "C:/Users/owner/Janis Lab Code repos/Guy/janis-lab-opx+/experiment_utils/optimal_weights.npz"
import numpy as np

user = "Ariel"

args_path = (
    f"C:/Users/owner/Janis Lab Code repos/{user}/janis-lab-opx+/experiment_utils/"
)
optimal_weights_path = f"C:/Users/owner/Janis Lab Code repos/{user}/janis-lab-opx+/experiment_utils/optimal_weights.npz"
if user == "Asaf":
    args_path += "args_asaf.json"
elif user == "Ariel":
    args_path += "args_ariel.json"
elif user == "Guy":
    args_path += "args_guy.json"
elif user == "Harel":
    args_path += "args_harel.json"


u = unit(coerce_to_integer=True)



con = "con1"
qubit = "qubit10"
#############################################
#                  Qubits                   #
#############################################
qubit_args = args[qubit]["qubit"]

qubit_anharmonicity = qubit_args["qubit_anharmonicity"]


qubit_LO = qubit_args["qubit_LO"]

sweet_spot_flag = bool(qubit_args["sweet_spot_flag"])

qubit_freq = (
    qubit_args["qubit_freq_sweet_spot"]
    if sweet_spot_flag
    else qubit_args["qubit_freq_zero_bias"]
)
flux_bias = qubit_args["flux_sweet_spot"] if sweet_spot_flag else 0
qubit_IF = qubit_LO - qubit_freq
qubit_ef_freq = qubit_freq + qubit_anharmonicity
qubit_ef_IF = qubit_LO - qubit_ef_freq
qubit_gf_freq = qubit_freq + qubit_anharmonicity / 2
qubit_gf_IF = qubit_LO - qubit_gf_freq
qubit_flux_sweet_spot = qubit_args["flux_sweet_spot"]
qubit_flux_bias_channel = qubit_args["flux_bias_channel"]

mixer_qubit_g = (
    -0.03566545454545449
)  #  0.026705454545454553  # 0.009018181818181828  # 0.2054400000000001
mixer_qubit_phi = (
    -0.24667636363636358
)  # 0.07806545454545451  # 0.05301818181818182  # 0.19518545454545452
qubit_correction_matrix = IQ_imbalance(mixer_qubit_g, mixer_qubit_phi)

qubit_T1 = qubit_args["T1"]
thermalization_time = qubit_args["thermalization_time"]
# Saturation_pulse
saturation_len = qubit_args["saturation_length"]
saturation_amp = qubit_args["saturation_amplitude"]
# Square pi pulse
square180_len = qubit_args["square180_len"]
square180_amp = qubit_args["square180_amp"]
square90_len = qubit_args["square180_len"]
square90_amp = qubit_args["square90_amp"]
# reset_gate
reset_gate_len = qubit_args["reset_gate_len"]
reset_gate_amp = qubit_args["reset_gate_amp"]
# Square pi pulse ef
square_pi_len_ef = qubit_args["square_pi_len_ef"]
square_pi_amp_ef = qubit_args["square_pi_amp_ef"]
# Square pi pulse gf
square_pi_len_gf = qubit_args["square_pi_len_gf"]
square_pi_amp_gf = qubit_args["square_pi_amp_gf"]

# Drag pulses
drag_coef = qubit_args["drag_coef"]
anharmonicity = -qubit_anharmonicity
AC_stark_detuning = 0 * u.MHz

pulse_type = "square"
if pulse_type == "drag":
    x180_len = qubit_args["drag_len"]
    x180_sigma = x180_len / 5
    x180_amp = qubit_args["drag180_amp"]
    x90_amp = qubit_args["drag90_amp"]
    x180_wf, x180_der_wf = np.array(
        drag_gaussian_pulse_waveforms(
            x180_amp, x180_len, x180_sigma, drag_coef, anharmonicity, AC_stark_detuning
        )
    )
    x180_I_wf = x180_wf
    x180_Q_wf = x180_der_wf
    # No DRAG when alpha=0, it's just a gaussian.

    x90_len = x180_len
    x90_sigma = x90_len / 5
    x90_wf, x90_der_wf = np.array(
        drag_gaussian_pulse_waveforms(
            x90_amp, x90_len, x90_sigma, drag_coef, anharmonicity, AC_stark_detuning
        )
    )
    x90_I_wf = x90_wf
    x90_Q_wf = x90_der_wf

    minus_x90_len = x180_len
    minus_x90_sigma = minus_x90_len / 5
    minus_x90_amp = -x90_amp
    minus_x90_wf, minus_x90_der_wf = np.array(
        drag_gaussian_pulse_waveforms(
            minus_x90_amp,
            minus_x90_len,
            minus_x90_sigma,
            drag_coef,
            anharmonicity,
            AC_stark_detuning,
        )
    )
    minus_x90_I_wf = minus_x90_wf
    minus_x90_Q_wf = minus_x90_der_wf

    y180_len = x180_len
    y180_sigma = y180_len / 5
    y180_amp = x180_amp
    y180_wf, y180_der_wf = np.array(
        drag_gaussian_pulse_waveforms(
            y180_amp, y180_len, y180_sigma, drag_coef, anharmonicity, AC_stark_detuning
        )
    )
    y180_I_wf = (-1) * y180_der_wf
    y180_Q_wf = y180_wf

    y90_len = x180_len
    y90_sigma = y90_len / 5
    y90_amp = qubit_args["drag90_amp"]
    y90_wf, y90_der_wf = np.array(
        drag_gaussian_pulse_waveforms(
            y90_amp, y90_len, y90_sigma, drag_coef, anharmonicity, AC_stark_detuning
        )
    )
    y90_I_wf = (-1) * y90_der_wf
    y90_Q_wf = y90_wf

    minus_y90_len = y180_len
    minus_y90_sigma = minus_y90_len / 5
    minus_y90_amp = -y90_amp
    minus_y90_wf, minus_y90_der_wf = np.array(
        drag_gaussian_pulse_waveforms(
            minus_y90_amp,
            minus_y90_len,
            minus_y90_sigma,
            drag_coef,
            anharmonicity,
            AC_stark_detuning,
        )
    )
    minus_y90_I_wf = (-1) * minus_y90_der_wf
    minus_y90_Q_wf = minus_y90_wf
if pulse_type == "square":
    x180_len = qubit_args["square180_len"]
    x180_amp = qubit_args["square180_amp"]
    x90_amp = qubit_args["square90_amp"]
    pulse_vec = x180_amp * np.ones(x180_len)
    x180_I_wf = pulse_vec
    x180_Q_wf = pulse_vec * 0
    x90_I_wf = x90_amp * np.ones(x180_len)
    x90_Q_wf = pulse_vec * 0
    minus_x90_I_wf = -x90_amp * np.ones(x180_len)
    minus_x90_Q_wf = pulse_vec * 0
    y180_I_wf = pulse_vec * 0
    y180_Q_wf = pulse_vec
    y90_I_wf = pulse_vec * 0
    y90_Q_wf = x90_amp * np.ones(x180_len)
    minus_y90_I_wf = pulse_vec * 0
    minus_y90_Q_wf = -x90_amp * np.ones(x180_len)
    x90_len = x180_len
    minus_x90_len = x180_len
    y180_len = x180_len
    y90_len = x180_len
else:
    pass


# No DRAG when alpha=0, it's just a gaussian.


def amp_V_to_Hz(amp):
    return amp / square180_len / (2 * square180_amp * 1e-9) / 1e6


#############################################
#                Resonators                 #
#############################################
resonator_args = args[qubit]["resonator"]

resonator_LO = resonator_args["resonator_LO"]
resonator_freq = resonator_args["resonator_freq"]
# resonator_freq_ef = resonator_args['resonator_freq_ef']
resonator_IF = resonator_LO - resonator_freq
mixer_resonator_g = 0.030080000000000016  # -0.25966545454545453  # 0.1976436363636364
mixer_resonator_phi = -0.09290181818181818  # 0.0706472727272727  # 0.11934545454545453
resonator_correction_matrix = IQ_imbalance(mixer_resonator_g, mixer_resonator_phi)

readout_len = resonator_args["readout_pulse_length"]
readout_amp = resonator_args["readout_pulse_amplitude"]

time_of_flight = resonator_args["time_of_flight"]
smearing = resonator_args["smearing"]
depletion_time = 0 * u.us

fid_matrix = resonator_args["fidelity_matrix"]
fid_matrix_2 = np.array(resonator_args["fidelity_matrix_2"])
fid_matrix_3 = np.array(resonator_args["fidelity_matrix_3"])
gef_centers = resonator_args["gef_centers"]
gef_covs = [np.array(cov_mat) for cov_mat in resonator_args["gef_covariance_mats"]]
ringdown_length = 0

opt_weights = True
if opt_weights:
    from qualang_tools.config.integration_weights_tools import (
        convert_integration_weights,
    )

    weights = np.load(optimal_weights_path)
    opt_weights_real = convert_integration_weights(weights["weights_real"])
    opt_weights_minus_imag = convert_integration_weights(weights["weights_minus_imag"])
    opt_weights_imag = convert_integration_weights(weights["weights_imag"])
    opt_weights_minus_real = convert_integration_weights(weights["weights_minus_real"])
else:
    opt_weights_real = [(1.0, readout_len)]
    opt_weights_minus_imag = [(0.0, readout_len)]
    opt_weights_imag = [(0.0, readout_len)]
    opt_weights_minus_real = [(-1.0, readout_len)]

# IQ Plane
rotation_angle = resonator_args["rotation_angle"]
ge_threshold = resonator_args["threshold"]

#############################################
#                   else                    #
#############################################
const_pulse_len = 1000
const_pulse_amp = 0.3

##########################################
#               Flux line                #
##########################################
max_frequency_point = 0.0
flux_settle_time = 100 * u.ns

# Resonator frequency versus flux fit parameters according to resonator_spec_vs_flux
# amplitude * np.cos(2 * np.pi * frequency * x + phase) + offset
amplitude_fit, frequency_fit, phase_fit, offset_fit = [0, 0, 0, 0]

# FLux pulse parameters
const_flux_len = 200
const_flux_amp = 0.45

#############################################
#                  Config                   #
#############################################


config = {
    "version": 1,
    "controllers": {
        con: {
            "analog_outputs": {
                1: {"offset": resonator_args["IQ_bias"]["I"]},  # I resonator
                2: {"offset": resonator_args["IQ_bias"]["Q"]},  # Q resonator
                3: {"offset": qubit_args["IQ_bias"]["I"]},  # I qubit
                4: {"offset": qubit_args["IQ_bias"]["Q"]},  # Q qubit
                # 5: {"offset": qubit_args['IQ_bias']['I']},  # I qubit
                # 6: {"offset": qubit_args['IQ_bias']['Q']},  # Q qubit
            },
            "digital_outputs": {},
            "analog_inputs": {
                1: {"offset": 0.292664, "gain_db": 2},  # I from down-conversion
                2: {"offset": 0.277516, "gain_db": 2},  # Q from down-conversion
            },
        }
    },
    "elements": {
        "qubit": {
            "mixInputs": {
                "I": (con, qubit_args["IQ_input"]["I"]),
                "Q": (con, qubit_args["IQ_input"]["Q"]),
                "lo_frequency": qubit_LO,
                "mixer": "mixer_qubit",
            },
            "intermediate_frequency": qubit_IF,
            "operations": {
                "cw": "const_pulse",
                "saturation": "saturation_pulse",
                "pi": "square_pi_pulse",
                "pi_half": "square_pi_half_pulse",
                "x180": "x180_pulse",
                "y180": "y180_pulse",
                "x90": "x90_pulse",
                "-x90": "-x90_pulse",
                "y90": "y90_pulse",
                "-y90": "-y90_pulse",
                "y360": "y360_pulse",
                "rampup": "rampup_pulse",
                "rampdown": "rampdown_pulse",
                "gaussian": "gaussian_pulse",
            },
        },
        "qubit_ef": {
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
                "lo_frequency": qubit_LO,
                "mixer": "mixer_qubit_ef",
            },
            "intermediate_frequency": qubit_ef_IF,
            "operations": {
                "cw": "const_pulse",
                "saturation": "saturation_pulse",
                "x180_ef": "x180_ef_pulse",
            },
        },
        "qubit_gf": {
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
                "lo_frequency": qubit_LO,
                "mixer": "mixer_qubit_gf",
            },
            "intermediate_frequency": qubit_gf_IF,
            # "intermediate_frequency": 1,
            "operations": {
                "cw": "const_pulse",
                "saturation": "saturation_pulse",
                "x180_gf": "x180_gf_pulse",
            },
        },
        "qubit_reset": {
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
                "lo_frequency": qubit_LO,
                "mixer": "mixer_qubit_reset",
            },
            "intermediate_frequency": qubit_IF,
            "operations": {
                "cw": "const_pulse",
                "saturation": "saturation_pulse",
                "reset": "square_pi_pulse",
            },
        },
        "resonator": {
            "mixInputs": {
                "I": (con, 1),
                "Q": (con, 2),
                "lo_frequency": resonator_LO,
                "mixer": "mixer_resonator",
            },
            "intermediate_frequency": resonator_IF,
            "operations": {
                "cw": "const_pulse",
                "readout": "readout_pulse",
            },
            "outputs": {
                "out1": (con, 1),
                "out2": (con, 2),
            },
            "time_of_flight": time_of_flight,
            "smearing": smearing,
        },
    },
    "pulses": {
        "const_pulse": {
            "operation": "control",
            "length": 1000,
            "waveforms": {
                "I": "const_wf",
                "Q": "zero_wf",
            },
        },
        "square_pi_pulse": {
            "operation": "control",
            "length": reset_gate_len,
            "waveforms": {
                "I": "square_pi_wf",
                "Q": "zero_wf",
            },
        },
        "square_pi_half_pulse": {
            "operation": "control",
            "length": square180_len,
            "waveforms": {
                "I": "square_pi_half_wf",
                "Q": "zero_wf",
            },
        },
        "saturation_pulse": {
            "operation": "control",
            "length": saturation_len,
            "waveforms": {"I": "saturation_drive_wf", "Q": "zero_wf"},
        },
        "gaussian_pulse": {
            "operation": "control",
            "length": saturation_len,
            "waveforms": {
                "I": "I_gaussian_wf",
                "Q": "Q_gaussian_wf",
            },
        },
        "x180_pulse": {
            "operation": "control",
            "length": x180_len,
            "waveforms": {
                "I": "x180_I_wf",
                "Q": "x180_Q_wf",
            },
        },
        "x180_ef_pulse": {
            "operation": "control",
            "length": square_pi_len_ef,
            "waveforms": {
                "I": "x180_I_wf_ef",
                "Q": "x180_Q_wf_ef",
            },
        },
        "x180_gf_pulse": {
            "operation": "control",
            "length": square_pi_len_gf,
            "waveforms": {
                "I": "x180_I_wf_gf",
                "Q": "x180_Q_wf_gf",
            },
        },
        "y180_pulse": {
            "operation": "control",
            "length": x180_len,
            "waveforms": {
                "I": "y180_I_wf",
                "Q": "y180_Q_wf",
            },
        },
        "x90_pulse": {
            "operation": "control",
            "length": x90_len,
            "waveforms": {
                "I": "x90_I_wf",
                "Q": "x90_Q_wf",
            },
        },
        "-x90_pulse": {
            "operation": "control",
            "length": x90_len,
            "waveforms": {
                "I": "minus_x90_I_wf",
                "Q": "minus_x90_Q_wf",
            },
        },
        "y90_pulse": {
            "operation": "control",
            "length": x90_len,
            "waveforms": {
                "I": "y90_I_wf",
                "Q": "y90_Q_wf",
            },
        },
        "-y90_pulse": {
            "operation": "control",
            "length": x90_len,
            "waveforms": {
                "I": "minus_y90_I_wf",
                "Q": "minus_y90_Q_wf",
            },
        },
        "y360_pulse": {
            "operation": "control",
            "length": square180_len,
            "waveforms": {
                "I": "y360_I_wf",
                "Q": "y360_Q_wf",
            },
        },
        "rampup_pulse": {
            "operation": "control",
            "length": square180_len,  # this is redefined in experiment script
            "waveforms": {
                "I": "rampup_I_wf",
                "Q": "rampup_Q_wf",
            },
        },
        "rampdown_pulse": {
            "operation": "control",
            "length": square180_len,  # this is redefined in experiment script
            "waveforms": {
                "I": "rampdown_I_wf",
                "Q": "rampdown_Q_wf",
            },
        },
        "readout_pulse": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {"I": "readout_wf", "Q": "zero_wf"},
            "integration_weights": {
                "cos": "rotated_cosine_weights",
                "sin": "rotated_sine_weights",
                "minus_sin": "rotated_minus_sine_weights",
                "opt_cos": "opt_cosine_weights",
                "opt_sin": "opt_sine_weights",
                "opt_minus_sin": "opt_minus_sine_weights",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": const_pulse_amp},
        "saturation_drive_wf": {"type": "constant", "sample": saturation_amp},
        "square_pi_wf": {"type": "constant", "sample": reset_gate_amp},
        "square_pi_half_wf": {"type": "constant", "sample": square90_amp},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "I_gaussian_wf": {
            "type": "arbitrary",
            "samples": generate_gaussian_pulse(
                amplitude=saturation_amp, length=saturation_len, sigma=0.02
            ),
        },
        "Q_gaussian_wf": {"type": "arbitrary", "samples": [0] * saturation_len},
        "x90_I_wf": {"type": "arbitrary", "samples": x90_I_wf.tolist()},
        "x90_Q_wf": {"type": "arbitrary", "samples": x90_Q_wf.tolist()},
        "x180_I_wf": {"type": "arbitrary", "samples": x180_I_wf.tolist()},
        "x180_Q_wf": {"type": "arbitrary", "samples": x180_Q_wf.tolist()},
        "x180_I_wf_ef": {"type": "constant", "sample": square_pi_amp_ef},
        "x180_Q_wf_ef": {"type": "constant", "sample": 0},
        "x180_I_wf_gf": {"type": "constant", "sample": square_pi_amp_gf},
        "x180_Q_wf_gf": {"type": "constant", "sample": 0},
        "minus_x90_I_wf": {"type": "arbitrary", "samples": minus_x90_I_wf.tolist()},
        "minus_x90_Q_wf": {"type": "arbitrary", "samples": minus_x90_Q_wf.tolist()},
        "y90_Q_wf": {"type": "arbitrary", "samples": y90_Q_wf.tolist()},
        "y90_I_wf": {"type": "arbitrary", "samples": y90_I_wf.tolist()},
        "y180_Q_wf": {"type": "arbitrary", "samples": y180_Q_wf.tolist()},
        "y180_I_wf": {"type": "arbitrary", "samples": y180_I_wf.tolist()},
        "minus_y90_Q_wf": {"type": "arbitrary", "samples": minus_y90_Q_wf.tolist()},
        "minus_y90_I_wf": {"type": "arbitrary", "samples": minus_y90_I_wf.tolist()},
        "y360_I_wf": {"type": "constant", "sample": square180_amp * 2},
        "y360_Q_wf": {"type": "constant", "sample": 0},
        "readout_wf": {"type": "constant", "sample": readout_amp},
        "rampup_I_wf": {
            "type": "arbitrary",
            "samples": np.linspace(0, 0.5, square180_len).tolist(),
        },
        # this is redefined in experiment script
        "rampup_Q_wf": {
            "type": "arbitrary",
            "samples": np.zeros(square180_len).tolist(),
        },
        # this is redefined in experiment script
        "rampdown_I_wf": {
            "type": "arbitrary",
            "samples": np.linspace(0.5, 0, square180_len).tolist(),
        },
        # this is redefined in experiment script
        "rampdown_Q_wf": {
            "type": "arbitrary",
            "samples": np.zeros(square180_len).tolist(),
        },
        # this is redefined in experiment script
    },
    "mixers": {
        "mixer_qubit": [
            {
                "intermediate_frequency": qubit_IF,
                "lo_frequency": qubit_LO,
                "correction": qubit_correction_matrix,
            }
        ],
        "mixer_qubit2": [
            {
                "intermediate_frequency": qubit_IF,
                "lo_frequency": qubit_LO,
                "correction": qubit_correction_matrix,
            }
        ],
        "mixer_resonator": [
            {
                "intermediate_frequency": resonator_IF,
                "lo_frequency": resonator_LO,
                "correction": resonator_correction_matrix,
            }
        ],
        "mixer_qubit_reset": [
            {
                "intermediate_frequency": qubit_IF,
                "lo_frequency": qubit_LO,
                "correction": qubit_correction_matrix,
            }
        ],
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        "cosine_weights": {
            "cosine": [(1.0, readout_len)],
            "sine": [(0.0, readout_len)],
        },
        "sine_weights": {
            "cosine": [(0.0, readout_len)],
            "sine": [(1.0, readout_len)],
        },
        "minus_sine_weights": {
            "cosine": [(0.0, readout_len)],
            "sine": [(-1.0, readout_len)],
        },
        "opt_cosine_weights": {
            "cosine": opt_weights_real,
            "sine": opt_weights_minus_imag,
        },
        "opt_sine_weights": {
            "cosine": opt_weights_imag,
            "sine": opt_weights_real,
        },
        "opt_minus_sine_weights": {
            "cosine": opt_weights_minus_imag,
            "sine": opt_weights_minus_real,
        },
        "rotated_cosine_weights": {
            "cosine": [(np.cos(rotation_angle), readout_len - ringdown_length)],
            "sine": [(np.sin(rotation_angle), readout_len - ringdown_length)],
        },
        "rotated_sine_weights": {
            "cosine": [(-np.sin(rotation_angle), readout_len - ringdown_length)],
            "sine": [(np.cos(rotation_angle), readout_len - ringdown_length)],
        },
        "rotated_minus_sine_weights": {
            "cosine": [(np.sin(rotation_angle), readout_len - ringdown_length)],
            "sine": [(-np.cos(rotation_angle), readout_len - ringdown_length)],
        },
    },
}

from katz_lab.utils.options import Options


def load_config(options: Options):
    return config
