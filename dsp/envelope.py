import numpy as np


def rms(signal):
    return np.sqrt(np.mean(signal ** 2))


def peak(signal):
    return np.max(np.abs(signal))


def energy(signal):
    return np.sum(signal ** 2)


def zero_crossings(signal):
    return np.sum(np.diff(np.sign(signal)) != 0)


def envelope(signal):

    return np.abs(signal)