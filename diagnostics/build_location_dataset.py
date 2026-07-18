from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from dsp.features import extract


DATA_FOLDER = Path("data/location")
OUTPUT_FILE = Path("data/location_features.csv")

ZONES = [
    "left",
    "right",
]


def process_file(filepath, zone):

    signal, samplerate = sf.read(filepath)

    # Convert stereo to mono if necessary
    if signal.ndim > 1:
        signal = np.mean(
            signal,
            axis=1
        )

    signal = signal.astype(
        np.float32
    )

    # Extract DSP features
    features = extract(
        signal,
        samplerate
    )

    # -----------------------------------------
    # Basic + impulse + spectral features
    # -----------------------------------------

    row = {

        "filename":
            filepath.name,

        "label":
            zone,

        "rms":
            features["rms"],

        "peak":
            features["peak"],

        "energy":
            features["energy"],

        "zero_crossings":
            features["zero_crossings"],

        "crest_factor":
            features["crest_factor"],

        "peak_index":
            features["peak_index"],

        "peak_position":
            features["peak_position"],

        "attack_time":
            features["attack_time"],

        "decay_ratio":
            features["decay_ratio"],

        "centroid":
            features["centroid"],

        "bandwidth":
            features["bandwidth"],

        "rolloff":
            features["rolloff"],

        "spectral_flatness":
            features["spectral_flatness"],

        "spectral_entropy":
            features["spectral_entropy"],

        "low_ratio":
            features["low_ratio"],

        "low_mid_ratio":
            features["low_mid_ratio"],

        "mid_ratio":
            features["mid_ratio"],

        "high_ratio":
            features["high_ratio"],

        "ultra_high_ratio":
            features["ultra_high_ratio"],
    }

    # -----------------------------------------
    # MFCC mean
    # -----------------------------------------

    for i, value in enumerate(
        features["mfcc"]
    ):

        row[
            f"mfcc_mean_{i + 1}"
        ] = value

    # -----------------------------------------
    # MFCC standard deviation
    # -----------------------------------------

    for i, value in enumerate(
        features["mfcc_std"]
    ):

        row[
            f"mfcc_std_{i + 1}"
        ] = value

    # -----------------------------------------
    # MFCC delta
    # -----------------------------------------

    for i, value in enumerate(
        features["mfcc_delta"]
    ):

        row[
            f"mfcc_delta_{i + 1}"
        ] = value

    # -----------------------------------------
    # MFCC delta-delta
    # -----------------------------------------

    for i, value in enumerate(
        features["mfcc_delta2"]
    ):

        row[
            f"mfcc_delta2_{i + 1}"
        ] = value

    return row


def main():

    rows = []

    print()
    print("=" * 60)
    print(
        "EchoDesk Location Dataset Builder"
    )
    print("=" * 60)
    print()

    for zone in ZONES:

        folder = (
            DATA_FOLDER
            / zone
        )

        files = sorted(
            folder.glob(
                "*.wav"
            )
        )

        print(
            f"{zone.upper()}: "
            f"{len(files)} files"
        )

        for filepath in files:

            try:

                row = process_file(
                    filepath,
                    zone
                )

                rows.append(
                    row
                )

                print(
                    f"Processed: "
                    f"{filepath.name}"
                )

            except Exception as error:

                print(
                    f"ERROR processing "
                    f"{filepath.name}: "
                    f"{error}"
                )

    if not rows:

        print()
        print(
            "No location recordings found."
        )

        return

    # -----------------------------------------
    # Create dataframe
    # -----------------------------------------

    dataframe = pd.DataFrame(
        rows
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # -----------------------------------------
    # Results
    # -----------------------------------------

    print()
    print("=" * 60)

    print(
        f"Dataset saved to: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Total samples: "
        f"{len(dataframe)}"
    )

    print(
        f"Features: "
        f"{len(dataframe.columns) - 2}"
    )

    print()

    print(
        "Location distribution:"
    )

    print(
        dataframe[
            "label"
        ].value_counts()
    )

    print("=" * 60)


if __name__ == "__main__":
    main()