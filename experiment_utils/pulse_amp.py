from matplotlib import pyplot as plt
from qutip.core.data.norm import l2_dense
from scipy.optimize import curve_fit
from experiment_utils.configuration import x180_len, x180_amp, x90_amp

import numpy as np


def square_pulse(pulse_len):
    d_180 = {
        16: 0.162101636,
        20: 0.13009,
        24: 0.108740300000,
        30: 0.1,
        40: 0.065251087952,
        72: 0.036278945982,
        100: 0.026121378388,
        200: 0.013036448000
    }
    l = list(d_180.keys())

    d_90 = {
        16: 0.162101636 / 2,
        20: 0.13009 / 2,
        24: 0.108740300000 / 2,
        30: 0.1 / 2,
        40: 0.065251087952 / 2,
        72: 0.018132123123,
        100: 0.026121378388 / 2,
        200: 0.006523532000,
    }

    v180 = np.array(list(d_180.values()))
    # v90 = np.array(list(d_90.values()))

    args = curve_fit(lambda x, a, b: a / x + b, l, v180)
    a, b = args[0]

    x180 = a * 1 / pulse_len + b
    x90 = x180 / 2

    plt.plot(l, v180, 'o')
    l_dense = np.linspace(16, 200, 1000)
    plt.plot(l_dense, a * 1 / l_dense + b)
    # plt.plot(l,  v90, 'o')

    plt.ylim([0, 0.2])
    plt.show()

    print('### Square ###')

    print(f"x180: {x180:.5f}")
    print(f"x90: {x90:.5f}")


def gaussian_pulse(pulse_len):
    l = [20, 40]

    amp_180 = [0.303090056924, 0.303090056924 / 2]

    args = curve_fit(lambda x, a, b: a / x + b, l, amp_180)
    a, b = args[0]

    x180 = a * pulse_len + b
    x90 = x180 / 2

    print('### Gaussian ###')
    print(f"x180: {x180:.5f}")
    print(f"x90: {x90:.5f}")


if __name__ == "__main__":
    square_pulse(x180_len)

    gaussian_pulse(x180_len)
