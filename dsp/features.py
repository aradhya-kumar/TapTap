import numpy as np
import librosa

from dsp.envelope import rms, peak, energy, zero_crossings
from dsp.fft import magnitude


def extract(signal, samplerate):
    """
    Extract acoustic features from an audio event.

    Designed for classifying short impulsive events such as:
    - Desk taps
    - Typing
    - Speech transients
    - Environmental noise
    """

    # --------------------------------------------------
    # Basic preparation
    # --------------------------------------------------

    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)

    if len(signal) == 0:
        raise ValueError("Signal is empty.")

    # Avoid numerical problems with silent signals
    eps = 1e-12

    # --------------------------------------------------
    # Basic time-domain features
    # --------------------------------------------------

    rms_value = float(rms(signal))
    peak_value = float(peak(signal))
    energy_value = float(energy(signal))
    zc_value = int(zero_crossings(signal))

    # Crest factor:
    # Sharp impulses tend to have a high peak relative to RMS.

    crest_factor = float(
        peak_value / (rms_value + eps)
    )

    # --------------------------------------------------
    # Peak position
    # --------------------------------------------------

    abs_signal = np.abs(signal)

    peak_index = int(
        np.argmax(abs_signal)
    )

    peak_position = float(
        peak_index / max(len(signal) - 1, 1)
    )

    # --------------------------------------------------
    # Attack time
    # --------------------------------------------------

    # Find the first point reaching 10% and 90%
    # of the maximum amplitude.

    threshold_10 = peak_value * 0.10
    threshold_90 = peak_value * 0.90

    indices_10 = np.where(
        abs_signal >= threshold_10
    )[0]

    indices_90 = np.where(
        abs_signal >= threshold_90
    )[0]

    if (
        len(indices_10) > 0
        and len(indices_90) > 0
    ):

        first_10 = indices_10[0]
        first_90 = indices_90[0]

        attack_samples = max(
            0,
            first_90 - first_10
        )

        attack_time = float(
            attack_samples
            / samplerate
        )

    else:

        attack_time = 0.0

    # --------------------------------------------------
    # Post-peak decay
    # --------------------------------------------------

    # Measure how much signal energy remains
    # after the strongest peak.

    post_peak = signal[
        peak_index:
    ]

    if len(post_peak) > 0:

        post_peak_rms = float(
            np.sqrt(
                np.mean(
                    post_peak ** 2
                )
            )
        )

    else:

        post_peak_rms = 0.0

    decay_ratio = float(
        post_peak_rms
        / (peak_value + eps)
    )

    # --------------------------------------------------
    # FFT
    # --------------------------------------------------

    spectrum = magnitude(signal)

    frequencies = np.fft.rfftfreq(
        len(signal),
        d=1.0 / samplerate
    )

    spectrum_sum = float(
        np.sum(spectrum)
    )

    # --------------------------------------------------
    # Spectral centroid
    # --------------------------------------------------

    centroid = float(
        np.sum(
            frequencies
            * spectrum
        )
        / (
            spectrum_sum
            + eps
        )
    )

    # --------------------------------------------------
    # Spectral bandwidth
    # --------------------------------------------------

    bandwidth = float(
        np.sqrt(
            np.sum(
                (
                    frequencies
                    - centroid
                ) ** 2
                * spectrum
            )
            / (
                spectrum_sum
                + eps
            )
        )
    )

    # --------------------------------------------------
    # Spectral rolloff (85%)
    # --------------------------------------------------

    cumulative_spectrum = np.cumsum(
        spectrum
    )

    rolloff_threshold = (
        0.85
        * cumulative_spectrum[-1]
    )

    rolloff_index = np.searchsorted(
        cumulative_spectrum,
        rolloff_threshold
    )

    rolloff_index = min(
        rolloff_index,
        len(frequencies) - 1
    )

    rolloff = float(
        frequencies[
            rolloff_index
        ]
    )

    # --------------------------------------------------
    # Spectral flatness
    # --------------------------------------------------

    power_spectrum = (
        spectrum ** 2
        + eps
    )

    geometric_mean = np.exp(
        np.mean(
            np.log(
                power_spectrum
            )
        )
    )

    arithmetic_mean = np.mean(
        power_spectrum
    )

    spectral_flatness = float(
        geometric_mean
        / (
            arithmetic_mean
            + eps
        )
    )

    # --------------------------------------------------
    # Spectral entropy
    # --------------------------------------------------

    normalized_power = (
        power_spectrum
        / (
            np.sum(
                power_spectrum
            )
            + eps
        )
    )

    spectral_entropy = float(
        -np.sum(
            normalized_power
            * np.log2(
                normalized_power
                + eps
            )
        )
    )

    # Normalize entropy to roughly 0-1

    if len(normalized_power) > 1:

        spectral_entropy /= np.log2(
            len(
                normalized_power
            )
        )

    # --------------------------------------------------
    # Frequency band energies
    # --------------------------------------------------

    def band_energy(
        low_frequency,
        high_frequency
    ):

        mask = (
            (frequencies >= low_frequency)
            &
            (frequencies < high_frequency)
        )

        if not np.any(mask):
            return 0.0

        return float(
            np.sum(
                power_spectrum[
                    mask
                ]
            )
        )

    low_energy = band_energy(
        20,
        500
    )

    low_mid_energy = band_energy(
        500,
        2000
    )

    mid_energy = band_energy(
        2000,
        5000
    )

    high_energy = band_energy(
        5000,
        10000
    )

    ultra_high_energy = band_energy(
        10000,
        min(
            20000,
            samplerate / 2
        )
    )

    total_spectral_energy = (
        low_energy
        + low_mid_energy
        + mid_energy
        + high_energy
        + ultra_high_energy
        + eps
    )

    # Convert absolute energies to ratios.
    # This helps reduce sensitivity to tap loudness.

    low_ratio = float(
        low_energy
        / total_spectral_energy
    )

    low_mid_ratio = float(
        low_mid_energy
        / total_spectral_energy
    )

    mid_ratio = float(
        mid_energy
        / total_spectral_energy
    )

    high_ratio = float(
        high_energy
        / total_spectral_energy
    )

    ultra_high_ratio = float(
        ultra_high_energy
        / total_spectral_energy
    )

    # --------------------------------------------------
    # MFCC
    # --------------------------------------------------

    mfcc_matrix = librosa.feature.mfcc(
        y=signal,
        sr=samplerate,
        n_mfcc=13
    )

    mfcc_mean = np.mean(
        mfcc_matrix,
        axis=1
    )

    mfcc_std = np.std(
        mfcc_matrix,
        axis=1
    )

    # --------------------------------------------------
    # MFCC Delta
    # --------------------------------------------------

    # Short 90 ms clips may not always have enough
    # frames for librosa's default delta width.

    if mfcc_matrix.shape[1] >= 3:

        width = min(
            9,
            mfcc_matrix.shape[1]
        )

        # Width must be odd.

        if width % 2 == 0:
            width -= 1

        if width >= 3:

            delta = librosa.feature.delta(
                mfcc_matrix,
                width=width,
                mode="nearest"
            )

            delta2 = librosa.feature.delta(
                mfcc_matrix,
                order=2,
                width=width,
                mode="nearest"
            )

            delta_mean = np.mean(
                delta,
                axis=1
            )

            delta2_mean = np.mean(
                delta2,
                axis=1
            )

        else:

            delta_mean = np.zeros(
                13
            )

            delta2_mean = np.zeros(
                13
            )

    else:

        delta_mean = np.zeros(
            13
        )

        delta2_mean = np.zeros(
            13
        )

    # --------------------------------------------------
    # Return all features
    # --------------------------------------------------

    return {

        # Basic features

        "rms":
            rms_value,

        "peak":
            peak_value,

        "energy":
            energy_value,

        "zero_crossings":
            zc_value,

        "crest_factor":
            crest_factor,

        # Impulse shape

        "peak_index":
            peak_index,

        "peak_position":
            peak_position,

        "attack_time":
            attack_time,

        "decay_ratio":
            decay_ratio,

        # Spectral features

        "centroid":
            centroid,

        "bandwidth":
            bandwidth,

        "rolloff":
            rolloff,

        "spectral_flatness":
            spectral_flatness,

        "spectral_entropy":
            spectral_entropy,

        # Frequency bands

        "low_ratio":
            low_ratio,

        "low_mid_ratio":
            low_mid_ratio,

        "mid_ratio":
            mid_ratio,

        "high_ratio":
            high_ratio,

        "ultra_high_ratio":
            ultra_high_ratio,

        # MFCC features

        "mfcc":
            mfcc_mean,

        "mfcc_std":
            mfcc_std,

        "mfcc_delta":
            delta_mean,

        "mfcc_delta2":
            delta2_mean,

        # Keep spectrum available
        # for future experiments.

        "spectrum":
            spectrum,
    }