import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sounddevice as sd

from dsp.stereo_features import extract_stereo_features


# ==========================================================
# CONFIGURATION
# ==========================================================

DEVICE = 1
SAMPLERATE = 48000
CHANNELS = 2

RECORD_SECONDS = 0.30

MODEL_PATH = Path(
    "models/stereo_4zone_classifier.pkl"
)

FEATURES_PATH = Path(
    "models/stereo_4zone_features.pkl"
)


# ==========================================================
# ZONES
# ==========================================================

ZONES = {

    "1": "top_left",

    "2": "top_right",

    "3": "bottom_left",

    "4": "bottom_right",

}


# ==========================================================
# SELECT ACTUAL ZONE
# ==========================================================

def select_zone():

    while True:

        print()
        print("=" * 60)
        print("ECHODESK LIVE 4-ZONE TEST")
        print("=" * 60)

        print()
        print("Select the zone you will tap:")
        print()

        print("1 - TOP LEFT")
        print("2 - TOP RIGHT")
        print("3 - BOTTOM LEFT")
        print("4 - BOTTOM RIGHT")

        print()

        choice = input(
            "> "
        ).strip()

        if choice in ZONES:

            return ZONES[
                choice
            ]

        print(
            "Invalid option."
        )


# ==========================================================
# RECORD ONE TAP
# ==========================================================

def record_tap():

    frames = int(
        RECORD_SECONDS
        * SAMPLERATE
    )

    print()
    print(
        "TAP NOW!"
    )

    audio = sd.rec(

        frames,

        samplerate=
            SAMPLERATE,

        channels=
            CHANNELS,

        dtype=
            "float32",

        device=
            DEVICE

    )

    sd.wait()

    return audio


# ==========================================================
# MAIN
# ==========================================================

def main():

    print()
    print("=" * 60)
    print(
        "LOADING 4-ZONE MODEL"
    )
    print("=" * 60)

    # ======================================================
    # LOAD MODEL
    # ======================================================

    model = joblib.load(
        MODEL_PATH
    )

    feature_names = joblib.load(
        FEATURES_PATH
    )

    print()
    print(
        "4-zone classifier loaded."
    )

    print(
        f"Features: "
        f"{len(feature_names)}"
    )

    # ======================================================
    # SELECT TEST ZONE
    # ======================================================

    actual_zone = (
        select_zone()
    )

    print()
    print("=" * 60)

    print(
        f"ACTUAL ZONE: "
        f"{actual_zone.upper()}"
    )

    print("=" * 60)

    print()
    print(
        "Press ENTER before each tap."
    )

    print(
        "Type q and press ENTER "
        "to finish."
    )

    print()

    # ======================================================
    # STATISTICS
    # ======================================================

    total = 0

    correct = 0

    wrong = 0

    # ======================================================
    # TEST LOOP
    # ======================================================

    while True:

        command = input(
            "ENTER = test tap | q = quit: "
        ).strip().lower()

        if command == "q":

            break

        # ==================================================
        # RECORD TAP
        # ==================================================

        audio = record_tap()

        # ==================================================
        # VERIFY STEREO
        # ==================================================

        if (
            audio.ndim != 2
            or
            audio.shape[1] < 2
        ):

            print(
                "ERROR: Stereo audio "
                "was not received."
            )

            continue

        audio = audio[
            :,
            :2
        ]

        # ==================================================
        # EXTRACT FEATURES
        # ==================================================

        try:

            spatial_features = (
                extract_stereo_features(
                    audio,
                    SAMPLERATE
                )
            )

        except Exception as error:

            print(
                f"Feature extraction error: "
                f"{error}"
            )

            continue

        # ==================================================
        # BUILD MODEL INPUT
        # ==================================================

        try:

            row = {

                feature:
                    spatial_features[
                        feature
                    ]

                for feature
                in feature_names

            }

        except KeyError as error:

            print()
            print(
                "Missing feature:"
            )

            print(
                error
            )

            continue

        model_input = pd.DataFrame(
            [
                row
            ]
        )

        # ==================================================
        # PREDICT
        # ==================================================

        prediction = (
            model.predict(
                model_input
            )[0]
        )

        prediction = str(
            prediction
        )

        # ==================================================
        # PROBABILITIES
        # ==================================================

        probabilities = None

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = (
                model.predict_proba(
                    model_input
                )[0]
            )

        # ==================================================
        # UPDATE STATISTICS
        # ==================================================

        total += 1

        is_correct = (
            prediction
            == actual_zone
        )

        if is_correct:

            correct += 1

        else:

            wrong += 1

        # ==================================================
        # DISPLAY RESULT
        # ==================================================

        print()
        print("-" * 60)

        print(
            f"ACTUAL:    "
            f"{actual_zone.upper()}"
        )

        print(
            f"PREDICTED: "
            f"{prediction.upper()}"
        )

        if is_correct:

            print(
                "RESULT:    CORRECT"
            )

        else:

            print(
                "RESULT:    WRONG"
            )

        # ==================================================
        # SHOW CLASS PROBABILITIES
        # ==================================================

        if (
            probabilities
            is not None
        ):

            print()
            print(
                "MODEL CONFIDENCE:"
            )

            for zone, probability in zip(

                model.classes_,

                probabilities

            ):

                print(

                    f"{str(zone).upper():15} "

                    f"{probability * 100:6.2f}%"

                )

        # ==================================================
        # CURRENT SCORE
        # ==================================================

        accuracy = (

            correct
            / total
            * 100

        )

        print()

        print(
            f"SCORE: "
            f"{correct}/{total}"
        )

        print(
            f"ACCURACY: "
            f"{accuracy:.2f}%"
        )

        print("-" * 60)

    # ======================================================
    # FINAL RESULTS
    # ======================================================

    print()
    print("=" * 60)
    print(
        "4-ZONE LIVE TEST RESULTS"
    )
    print("=" * 60)

    print()

    print(
        f"Actual zone: "
        f"{actual_zone.upper()}"
    )

    print(
        f"Total taps: "
        f"{total}"
    )

    print(
        f"Correct: "
        f"{correct}"
    )

    print(
        f"Wrong: "
        f"{wrong}"
    )

    if total > 0:

        accuracy = (

            correct
            / total
            * 100

        )

        print(
            f"Accuracy: "
            f"{accuracy:.2f}%"
        )

    print()


# ==========================================================
# START
# ==========================================================

if __name__ == "__main__":

    main()