import librosa
import numpy as np

from dsp.envelope import rms, peak, energy, zero_crossings
from dsp.fft import magnitude


def extract(signal, samplerate):

    spectrum = magnitude(signal)

    centroid = librosa.feature.spectral_centroid(
        y=signal,
        sr=samplerate
    )[0][0]

    bandwidth = librosa.feature.spectral_bandwidth(
        y=signal,
        sr=samplerate
    )[0][0]

    rolloff = librosa.feature.spectral_rolloff(
        y=signal,
        sr=samplerate
    )[0][0]

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=samplerate,
        n_mfcc=13
    )

    return {

        "rms": rms(signal),

        "peak": peak(signal),

        "energy": energy(signal),

        "zero_crossings": zero_crossings(signal),

        "centroid": centroid,

        "bandwidth": bandwidth,

        "rolloff": rolloff,

        "mfcc": mfcc.mean(axis=1),

        "spectrum": spectrum

    }