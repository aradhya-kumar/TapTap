import numpy as np
from scipy.signal import butter, lfilter


def lowpass(signal, cutoff, fs, order=4):
    nyquist = fs * 0.5
    normal = cutoff / nyquist

    b, a = butter(order, normal, btype="low")

    return lfilter(b, a, signal)


def highpass(signal, cutoff, fs, order=4):
    nyquist = fs * 0.5
    normal = cutoff / nyquist

    b, a = butter(order, normal, btype="high")

    return lfilter(b, a, signal)


def bandpass(signal, low, high, fs, order=4):
    nyquist = fs * 0.5

    low /= nyquist
    high /= nyquist

    b, a = butter(order, [low, high], btype="band")

    return lfilter(b, a, signal)