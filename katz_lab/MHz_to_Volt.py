from experiment_utils.configuration import *


def amp_Volt_to_MHz(amp_Volt):
    return amp_Volt / reset_gate_amp / (2 * reset_gate_len * 1e-9) / 1e6


def amp_MHz_to_Volt(amp_MHz):
    return amp_MHz * 2 * reset_gate_len * 1e-9 * reset_gate_amp * 1e6


if __name__ == "__main__":
    amp_Volt = 0.0652154995580671

    amp_MHz = amp_Volt_to_MHz(amp_Volt)

    print(f'amp_Volt = {amp_MHz} MHz')

    amp_MHz = 12.5

    amp_Volt = amp_MHz_to_Volt(amp_MHz)

    print(f'amp_Volt = {amp_Volt} V')
