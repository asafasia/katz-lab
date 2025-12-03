from katz_lab.utils.options import Options

config = dict()

config["version"] = 1
config["controllers"] = None
config["elements"] = None
config["pulses"] = None
config["waveforms"] = None
config["mixers"] = None
config["digital_waveforms"] = None
config["integration_weights"] = None


class Config:
    def __init__(self, qubit: str, options: Options):
        self.config = config
        self.qubit = qubit
        self.options = options

    