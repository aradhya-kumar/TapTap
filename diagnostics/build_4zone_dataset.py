from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io.wavfile import read

from dsp.stereo_features import extract_stereo_features


# ==========================================================
# CONFIGURATION
# ==========================================================

DATA_FOLDER = Path(
    "data/stereo_4zone"
)

OUTPUT_FILE = Path(
    "data/stereo_4zone_features.csv"
)

ZONES = [
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
]


# ==========================================================
# LOAD AUDIO
# ==========================================================

def load_audio(
    filepath
):

    samplerate, audio = read(
        filepath
    )

    # Convert to float32

    if audio.dtype == np.int16:

        audio = (
            audio.astype(
                np.float32
            )
            / 32768.0
        )

    elif audio.dtype == np.int32:

        audio = (
            audio.astype(
                np.float32
            )
            / 2147483648.0
        )

    else:

        audio = audio.astype(
            np.float32
        )

    # Verify stereo

    if (
        audio.ndim != 2
        or
        audio.shape[1] < 2
    ):

        raise ValueError(
            "Audio file is not stereo."
        )

    # Keep first two channels

    audio = audio[
        :,
        :2
    ]

    return (
        samplerate,
        audio
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print()
    print("=" * 60)
    print(
        "ECHODESK 4-ZONE DATASET BUILDER"
    )
    print("=" * 60)

    rows = []

    zone_counts = {
        zone: 0
        for zone in ZONES
    }

    # ======================================================
    # PROCESS EACH ZONE
    # ======================================================

    for zone in ZONES:

        folder = (
            DATA_FOLDER
            / zone
        )

        print()
        print(
            f"Processing: "
            f"{zone.upper().replace('_', ' ')}"
        )

        if not folder.exists():

            print(
                f"WARNING: Folder not found: "
                f"{folder}"
            )

            continue

        wav_files = sorted(
            folder.glob(
                "*.wav"
            )
        )

        print(
            f"Found "
            f"{len(wav_files)} "
            f"recordings."
        )

        for filepath in wav_files:

            try:

                samplerate, audio = (
                    load_audio(
                        filepath
                    )
                )

                # ==========================================
                # EXTRACT STEREO FEATURES
                # ==========================================

                features = (
                    extract_stereo_features(
                        audio,
                        samplerate
                    )
                )

                # Make copy

                row = dict(
                    features
                )

                # Add label

                row[
                    "label"
                ] = zone

                # Add source filename

                row[
                    "filename"
                ] = filepath.name

                rows.append(
                    row
                )

                zone_counts[
                    zone
                ] += 1

                print(
                    f"OK: "
                    f"{filepath.name}"
                )

            except Exception as error:

                print(
                    f"ERROR: "
                    f"{filepath.name}"
                )

                print(
                    f"       "
                    f"{error}"
                )

    # ======================================================
    # CHECK RESULTS
    # ======================================================

    if not rows:

        print()
        print(
            "ERROR: No features "
            "were extracted."
        )

        return

    # ======================================================
    # CREATE DATAFRAME
    # ======================================================

    dataframe = pd.DataFrame(
        rows
    )

    # ======================================================
    # SAVE DATASET
    # ======================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ======================================================
    # SUMMARY
    # ======================================================

    print()
    print("=" * 60)
    print(
        "DATASET COMPLETE"
    )
    print("=" * 60)

    print()

    for zone in ZONES:

        print(
            f"{zone.upper():15} : "
            f"{zone_counts[zone]}"
        )

    print()

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
        "Dataset saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print("=" * 60)


# ==========================================================
# START
# ==========================================================

if __name__ == "__main__":

    main()