import numpy as np
from scipy.signal import correlate


def rms(signal):
    return float(np.sqrt(np.mean(signal ** 2) + 1e-12))


def energy(signal):
    return float(np.sum(signal ** 2))


def peak(signal):
    return float(np.max(np.abs(signal)))


def peak_index(signal):
    return int(np.argmax(np.abs(signal)))


def normalized_cross_correlation(ch1, ch2):
    ch1 = ch1 - np.mean(ch1)
    ch2 = ch2 - np.mean(ch2)

    correlation = correlate(
        ch1,
        ch2,
        mode="full",
        method="fft"
    )

    denominator = (
        np.linalg.norm(ch1)
        * np.linalg.norm(ch2)
        + 1e-12
    )

    correlation = (
        correlation
        / denominator
    )

    lags = np.arange(
        -len(ch2) + 1,
        len(ch1)
    )

    index = int(
        np.argmax(
            np.abs(correlation)
        )
    )

    best_lag = int(
        lags[index]
    )

    best_correlation = float(
        correlation[index]
    )

    return (
        best_lag,
        best_correlation
    )


def extract_stereo_features(
    audio,
    samplerate
):

    if audio.ndim != 2:

        raise ValueError(
            "Expected stereo audio "
            "with shape (samples, channels)"
        )

    if audio.shape[1] < 2:

        raise ValueError(
            "Audio must contain "
            "at least 2 channels"
        )

    ch1 = (
        audio[:, 0]
        .astype(np.float32)
    )

    ch2 = (
        audio[:, 1]
        .astype(np.float32)
    )

    # ==========================================
    # Basic channel features
    # ==========================================

    rms_1 = rms(ch1)
    rms_2 = rms(ch2)

    peak_1 = peak(ch1)
    peak_2 = peak(ch2)

    energy_1 = energy(ch1)
    energy_2 = energy(ch2)

    peak_index_1 = peak_index(ch1)
    peak_index_2 = peak_index(ch2)

    # ==========================================
    # Relative amplitude features
    # ==========================================

    rms_difference = (
        rms_1
        - rms_2
    )

    peak_difference = (
        peak_1
        - peak_2
    )

    energy_difference = (
        energy_1
        - energy_2
    )

    rms_ratio = (
        rms_1
        / (
            rms_2
            + 1e-12
        )
    )

    peak_ratio = (
        peak_1
        / (
            peak_2
            + 1e-12
        )
    )

    energy_ratio = (
        energy_1
        / (
            energy_2
            + 1e-12
        )
    )

    # Log ratios are often easier
    # for ML models to learn.

    log_rms_ratio = float(
        np.log(
            (
                rms_1
                + 1e-12
            )
            /
            (
                rms_2
                + 1e-12
            )
        )
    )

    log_peak_ratio = float(
        np.log(
            (
                peak_1
                + 1e-12
            )
            /
            (
                peak_2
                + 1e-12
            )
        )
    )

    log_energy_ratio = float(
        np.log(
            (
                energy_1
                + 1e-12
            )
            /
            (
                energy_2
                + 1e-12
            )
        )
    )

    # ==========================================
    # Peak timing
    # ==========================================

    peak_sample_difference = (
        peak_index_1
        - peak_index_2
    )

    peak_time_difference = (
        peak_sample_difference
        / samplerate
    )

    # ==========================================
    # Cross-correlation / TDOA-like feature
    # ==========================================

    (
        correlation_lag,
        correlation_value
    ) = normalized_cross_correlation(
        ch1,
        ch2
    )

    correlation_time_difference = (
        correlation_lag
        / samplerate
    )

    # ==========================================
    # Direct channel similarity
    # ==========================================

    if (
        np.std(ch1) > 0
        and
        np.std(ch2) > 0
    ):

        channel_correlation = float(
            np.corrcoef(
                ch1,
                ch2
            )[0, 1]
        )

    else:

        channel_correlation = 0.0

    mean_absolute_difference = float(
        np.mean(
            np.abs(
                ch1
                - ch2
            )
        )
    )

    # ==========================================
    # Return feature dictionary
    # ==========================================

    return {

        "ch1_rms":
            rms_1,

        "ch2_rms":
            rms_2,

        "ch1_peak":
            peak_1,

        "ch2_peak":
            peak_2,

        "ch1_energy":
            energy_1,

        "ch2_energy":
            energy_2,

        "ch1_peak_index":
            peak_index_1,

        "ch2_peak_index":
            peak_index_2,

        "rms_difference":
            rms_difference,

        "peak_difference":
            peak_difference,

        "energy_difference":
            energy_difference,

        "rms_ratio":
            rms_ratio,

        "peak_ratio":
            peak_ratio,

        "energy_ratio":
            energy_ratio,

        "log_rms_ratio":
            log_rms_ratio,

        "log_peak_ratio":
            log_peak_ratio,

        "log_energy_ratio":
            log_energy_ratio,

        "peak_sample_difference":
            peak_sample_difference,

        "peak_time_difference":
            peak_time_difference,

        "correlation_lag":
            correlation_lag,

        "correlation_time_difference":
            correlation_time_difference,

        "correlation_value":
            correlation_value,

        "channel_correlation":
            channel_correlation,

        "mean_absolute_difference":
            mean_absolute_difference,
    }