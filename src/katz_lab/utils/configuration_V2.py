from katz_lab.utils.options import Options
from katz_lab.utils.params import args
from collections import defaultdict


def nested_dict():
    return defaultdict(nested_dict)


def _get_default_config():
    config = nested_dict()
    config["version"] = 1
    config["controllers"] = nested_dict()
    config["elements"] = nested_dict()
    config["pulses"] = nested_dict()
    config["waveforms"] = nested_dict()
    config["mixers"] = nested_dict()
    config["digital_waveforms"] = nested_dict()
    config["integration_weights"] = nested_dict()
    return config


class QUAConfigBuilder:
    def __init__(self, qubit: str, args: dict, options: Options):
        self.con = "con1"
        self.config = _get_default_config()
        self.qubit = qubit
        self.options = options
        self.args = args
        self.resonator_args = args[qubit]["resonator"]
        self.qubit_args = args[qubit]["qubit"]

    def _compute_controllers(self):
        controller = {
            "analog_outputs": {
                1: {"offset": self.resonator_args["IQ_bias"]["I"]},  # I resonator
                2: {"offset": self.resonator_args["IQ_bias"]["Q"]},  # Q resonator
                3: {"offset": self.qubit_args["IQ_bias"]["I"]},  # I qubit
                4: {"offset": self.qubit_args["IQ_bias"]["Q"]},  # Q qubit
                # 5: {"offset": qubit_args['IQ_bias']['I']},  # I qubit
                # 6: {"offset": qubit_args['IQ_bias']['Q']},  # Q qubit
            },
            "digital_outputs": {},
            "analog_inputs": {
                1: {"offset": 0.292664, "gain_db": 2},  # I from down-conversion
                2: {"offset": 0.277516, "gain_db": 2},  # Q from down-conversion
            },
        }
        self.config["controllers"][self.con] = controller

    def _compute_elements(self):
        resonator_LO = self.resonator_args["resonator_LO"]
        resonator_RF = self.resonator_args["resonator_freq"]
        resonator_IF = resonator_LO - resonator_RF
        resonator = {
            "mixInputs": {
                "I": (self.con, 1),
                "Q": (self.con, 2),
                "lo_frequency": resonator_LO,
                "mixer": "mixer_resonator",
            },
            "intermediate_frequency": resonator_IF,
            "operations": {
                "cw": "const_pulse",
                "readout": "readout_pulse",
            },
            "outputs": {
                "out1": (self.con, 1),
                "out2": (self.con, 2),
            },
            "time_of_flight": self.resonator_args["time_of_flight"],
            "smearing": self.resonator_args["smearing"],
        }

        self.config["elements"]["resonator"] = resonator

    def _compute_digital_waveforms(self):
        pass

    def _compute_waveforms(self):
        pass

    def _compute_pulses(self):
        pass

    def _compute_mixers(self):
        pass

    def _compute_integration_weights(self):
        pass

    def build(self):
        self._compute_controllers()
        self._compute_digital_waveforms()
        self._compute_waveforms()
        self._compute_pulses()
        self._compute_elements()
        self._compute_mixers()
        self._compute_integration_weights()

        return self.config


if __name__ == "__main__":
    qubit_name = "q10"
    config = QUAConfigBuilder(qubit=qubit_name, options=Options(), args=args).build()

    from pprint import pprint

    pprint(config)
