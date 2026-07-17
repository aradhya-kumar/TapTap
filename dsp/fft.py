import numpy as np


def magnitude(signal):

    fft = np.fft.rfft(signal)

    return np.abs(fft)


def frequencies(length, samplerate):

    return np.fft.rfftfreq(length, 1 / samplerate)


def power_spectrum(signal):

    return np.abs(np.fft.rfft(signal)) ** 2