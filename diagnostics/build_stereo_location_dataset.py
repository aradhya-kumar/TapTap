from pathlib import Path

import pandas as pd
import soundfile as sf

from dsp.stereo_features import (
    extract_stereo_features
)


DATA_FOLDER = Path(
    "data/filtered_stereo_location"
)

OUTPUT_FILE = Path(
    "data/stereo_location_features.csv"
)

ZONES = [
    "left",
    "right",
]


def main():

    print()
    print("=" * 60)
    print(
        "EchoDesk Stereo Spatial "
        "Feature Extraction"
    )
    print("=" * 60)

    rows = []

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

        print()
        print(
            f"{zone.upper()}: "
            f"{len(files)} files"
        )

        for index, file in enumerate(
            files,
            start=1
        ):

            try:

                audio, samplerate = (
                    sf.read(
                        file,
                        dtype="float32",
                        always_2d=True
                    )
                )

                if (
                    audio.shape[1]
                    < 2
                ):

                    print(
                        f"SKIPPED MONO: "
                        f"{file.name}"
                    )

                    continue

                features = (
                    extract_stereo_features(
                        audio,
                        samplerate
                    )
                )

                row = {
                    "filename":
                        file.name,

                    "label":
                        zone,
                }

                row.update(
                    features
                )

                rows.append(
                    row
                )

                print(
                    f"\r"
                    f"Processed "
                    f"{index}/{len(files)}",
                    end=""
                )

            except Exception as error:

                print()
                print(
                    f"ERROR: "
                    f"{file.name}"
                )

                print(
                    error
                )

        print()

    # ==========================================
    # Create dataset
    # ==========================================

    if not rows:

        print()
        print(
            "No valid stereo recordings found."
        )

        return

    dataset = pd.DataFrame(
        rows
    )

    dataset.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ==========================================
    # Summary
    # ==========================================

    print()
    print("=" * 60)
    print(
        "STEREO DATASET COMPLETE"
    )
    print("=" * 60)

    print()
    print(
        f"Total samples: "
        f"{len(dataset)}"
    )

    print()
    print(
        "Class distribution:"
    )

    print(
        dataset[
            "label"
        ].value_counts()
    )

    print()
    print(
        f"Spatial features: "
        f"{len(dataset.columns) - 2}"
    )

    print()
    print(
        f"Saved to: "
        f"{OUTPUT_FILE}"
    )

    # ==========================================
    # Quick spatial statistics
    # ==========================================

    print()
    print("=" * 60)
    print(
        "SPATIAL FEATURE SUMMARY"
    )
    print("=" * 60)

    important_features = [

        "log_rms_ratio",

        "log_energy_ratio",

        "peak_sample_difference",

        "correlation_lag",

        "channel_correlation",
    ]

    for feature in important_features:

        print()
        print(
            f"{feature}:"
        )

        print(
            dataset.groupby(
                "label"
            )[feature].mean()
        )


if __name__ == "__main__":
    main()