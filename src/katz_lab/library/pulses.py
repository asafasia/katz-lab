import numpy as np
from matplotlib import pyplot as plt


def generate_eco_pulse(amplitude=0.01, length=1000):
    ts = np.linspace(-1, 1, length)
    vec = (2 * np.heaviside(ts, 0) - 1) * amplitude

    # vec = -np.sin(10 * np.pi * ts) * amplitude

    return vec.tolist()


def generate_lorentzian_pulse(amplitude, length, cutoff, n, echo=False):
    sigma = (1 / ((1 / cutoff ** (1 / n)) - 1)) ** (1 / 2)
    ts = np.linspace(-1, 1, length)
    vec = amplitude / (1 + (ts / sigma) ** 2) ** n

    if echo:
        vec = vec*(2*np.heaviside(ts, 0)-1)
    return vec.tolist()


def generate_lorentzian_pulse_up(amplitude=0.01, length=1000, cutoff=0.1, n=1 / 2):
    sigma = (1 / ((1 / cutoff ** (1 / n)) - 1)) ** (1 / 2)
    ts = np.linspace(-1, 0, length)
    vec = amplitude / (1 + (ts / sigma) ** 2) ** n
    return vec.tolist()


def generate_lorentzian_pulse_down(amplitude=0.01, length=1000, cutoff=0.1, n=1 / 2):
    sigma = ((1) ** 2 / ((1 / cutoff ** (1 / n)) - 1)) ** (1 / 2)
    ts = np.linspace(0, 1, length)
    vec = amplitude / (1 + (ts / sigma) ** 2) ** n
    return vec.tolist()


def generate_half_lorentzian_pulse(amplitude=0.01, length=1000, cutoff=0.1, n=1 / 2):
    vec = np.array(generate_lorentzian_pulse(amplitude=amplitude, length=length, cutoff=cutoff, n=n))
    half = generate_eco_pulse(amplitude=1, length=length)
    return (half * vec).tolist()


def generate_gaussian_pulse(amplitude=0.01, length=1000, sigma=0.2):
    ts = np.linspace(-1, 1, length)
    vec = amplitude * np.exp(-ts ** 2 / (2 * sigma ** 2))
    return vec.tolist()


def readout_pulse(x):
    return


def generate_SL_ramp_up_and_hold(amplitude=0.01, length=1000, cutoff=0.1, n=1 / 2):
    vec = np.array(generate_lorentzian_pulse(amplitude=amplitude, length=length, cutoff=cutoff, n=n))
    ramp = np.linspace(0, amplitude, length)
    hold = np.ones(length) * amplitude
    return (ramp * vec).tolist(), hold.tolist()


if __name__ == "__main__":
    amplitude = 1
    cutoff = 0.5
    length = 10000
    n = 1 / 2

    eco_pulse_samples = generate_eco_pulse(amplitude=amplitude, length=length)
    lorentzian_pulse_samples = generate_lorentzian_pulse(
        amplitude=amplitude,
        length=length,
        cutoff=cutoff,
        n=n

    )
    lorentzian_half_pulse_samples = generate_lorentzian_pulse(
        amplitude=amplitude,
        length=length,
        cutoff=cutoff,
        n=n,
        echo=True
    )
    plt.plot(eco_pulse_samples)
    plt.plot(lorentzian_pulse_samples)
    plt.plot(lorentzian_half_pulse_samples)
    plt.axhline(cutoff, color='k', linestyle='--')
    plt.axhline(-cutoff, color='k', linestyle='--')

    plt.show()
