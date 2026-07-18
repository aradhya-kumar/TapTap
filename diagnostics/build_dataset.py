from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from dsp.features import extract

print("BUILD DATASET SCRIPT STARTED")

DATA_FOLDER = Path("data/events")
OUTPUT_FILE = Path("data/event_features.csv")

LABELS = [
    "tap",
    "typing",
    "speech",
    "noise",
]


def process_file(filepath, label):

    signal, samplerate = sf.read(filepath)

    # Convert stereo to mono if necessary
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)

    signal = signal.astype(np.float32)

    features = extract(
        signal,
        samplerate
    )

    row = {
        "filename": filepath.name,
        "label": label,

        "rms": features["rms"],
        "peak": features["peak"],
        "energy": features["energy"],
        "zero_crossings": features["zero_crossings"],

        "crest_factor": features["crest_factor"],
        "peak_index": features["peak_index"],
        "peak_position": features["peak_position"],
        "attack_time": features["attack_time"],
        "decay_ratio": features["decay_ratio"],

        "centroid": features["centroid"],
        "bandwidth": features["bandwidth"],
        "rolloff": features["rolloff"],

        "spectral_flatness": features["spectral_flatness"],
        "spectral_entropy": features["spectral_entropy"],

        "low_ratio": features["low_ratio"],
        "low_mid_ratio": features["low_mid_ratio"],
        "mid_ratio": features["mid_ratio"],
        "high_ratio": features["high_ratio"],
        "ultra_high_ratio": features["ultra_high_ratio"],
    }

    # MFCC means

    for i, value in enumerate(
        features["mfcc"]
    ):

        row[
            f"mfcc_mean_{i + 1}"
        ] = value

    # MFCC standard deviations

    for i, value in enumerate(
        features["mfcc_std"]
    ):

        row[
            f"mfcc_std_{i + 1}"
        ] = value

    # Delta MFCC

    for i, value in enumerate(
        features["mfcc_delta"]
    ):

        row[
            f"mfcc_delta_{i + 1}"
        ] = value

    # Delta-Delta MFCC

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
    print("EchoDesk Feature Dataset Builder")
    print("=" * 60)
    print()

    for label in LABELS:

        folder = DATA_FOLDER / label

        files = sorted(
            folder.glob("*.wav")
        )

        print(
            f"{label.upper()}: "
            f"{len(files)} files"
        )

        for filepath in files:

            try:

                row = process_file(
                    filepath,
                    label
                )

                rows.append(row)

                print(
                    f"  Processed: "
                    f"{filepath.name}"
                )

            except Exception as error:

                print(
                    f"  ERROR: "
                    f"{filepath.name}"
                )

                print(
                    f"  {error}"
                )

    if not rows:

        print(
            "No audio files found."
        )

        return

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
        f"Total features: "
        f"{len(dataframe.columns) - 2}"
    )

    print()

    print(
        "Class distribution:"
    )

    print(
        dataframe[
            "label"
        ].value_counts()
    )

    print("=" * 60)


if __name__ == "__main__":

    main()