import time
from pathlib import Path

import joblib
import pandas as pd
import sounddevice as sd

from main import (
    EchoDesk,
    DEVICE,
    SAMPLERATE,
    CHANNELS,
    BLOCKSIZE,
)

from dsp.stereo_features import extract_stereo_features


# ==========================================================
# 4-ZONE MODEL PATHS
# ==========================================================

ZONE_MODEL_PATH = Path(
    "models/stereo_4zone_classifier.pkl"
)

ZONE_FEATURES_PATH = Path(
    "models/stereo_4zone_features.pkl"
)


# ==========================================================
# 4-ZONE ECHODESK
# ==========================================================

class EchoDesk4Zone(EchoDesk):

    def __init__(self):

        # Load normal EchoDesk system
        super().__init__()

        print()
        print("=" * 60)
        print("Loading EchoDesk 4-Zone Model")
        print("=" * 60)

        # --------------------------------------------------
        # Check model files
        # --------------------------------------------------

        if not ZONE_MODEL_PATH.exists():

            raise FileNotFoundError(
                f"4-zone model not found: "
                f"{ZONE_MODEL_PATH}"
            )

        if not ZONE_FEATURES_PATH.exists():

            raise FileNotFoundError(
                f"4-zone feature list not found: "
                f"{ZONE_FEATURES_PATH}"
            )

        # --------------------------------------------------
        # Load model
        # --------------------------------------------------

        self.zone_model = joblib.load(
            ZONE_MODEL_PATH
        )

        self.zone_feature_names = joblib.load(
            ZONE_FEATURES_PATH
        )

        print(
            "4-zone classifier loaded."
        )

        print(
            f"Number of zone features: "
            f"{len(self.zone_feature_names)}"
        )

        # --------------------------------------------------
        # Zone statistics
        # --------------------------------------------------

        self.zone_counts = {

            "top_left": 0,

            "top_right": 0,

            "bottom_left": 0,

            "bottom_right": 0,

        }

        # Latest 4-zone result

        self.latest_zone_result = None


    # ======================================================
    # FINISH EVENT
    # ======================================================

    def finish_event(self):

        # --------------------------------------------------
        # IMPORTANT
        #
        # We need access to the original stereo event before
        # the parent EchoDesk resets event_buffer.
        # --------------------------------------------------

        if not self.event_buffer:

            return

        try:

            # --------------------------------------------------
            # Build copy of stereo event
            # --------------------------------------------------

            import numpy as np

            stereo_event = np.concatenate(
                self.event_buffer,
                axis=0
            )

            stereo_event = stereo_event[
                :self.event_samples,
                :2
            ]

            # Pad short event if necessary

            if (
                len(stereo_event)
                < self.event_samples
            ):

                missing = (
                    self.event_samples
                    - len(stereo_event)
                )

                stereo_event = np.pad(
                    stereo_event,
                    (
                        (0, missing),
                        (0, 0)
                    )
                )

        except Exception as error:

            print()
            print(
                "ERROR COPYING STEREO EVENT:"
            )

            print(
                error
            )

            # Let normal EchoDesk clean up

            super().finish_event()

            return

        # --------------------------------------------------
        # Run normal EchoDesk processing
        #
        # This performs:
        #
        # impulse
        #   ↓
        # tap classifier
        #   ↓
        # LEFT / RIGHT classifier
        #
        # It also sets self.latest_result.
        # --------------------------------------------------

        super().finish_event()

        # --------------------------------------------------
        # If event was rejected as NOT TAP,
        # don't run 4-zone classifier.
        # --------------------------------------------------

        if (
            self.latest_result
            is None
        ):

            return

        # ==================================================
        # EXTRACT 4-ZONE FEATURES
        # ==================================================

        try:

            spatial_features = (
                extract_stereo_features(
                    stereo_event,
                    SAMPLERATE
                )
            )

            # --------------------------------------------------
            # Build model input in exact training feature order
            # --------------------------------------------------

            row = {

                feature:
                    spatial_features[
                        feature
                    ]

                for feature
                in self.zone_feature_names

            }

            model_input = pd.DataFrame(
                [
                    row
                ]
            )

            # ==================================================
            # PREDICT ZONE
            # ==================================================

            prediction = (
                self.zone_model
                .predict(
                    model_input
                )[0]
            )

            prediction = str(
                prediction
            ).lower()

            # ==================================================
            # CLASS PROBABILITIES
            # ==================================================

            probabilities = {}

            if hasattr(
                self.zone_model,
                "predict_proba"
            ):

                probability_values = (
                    self.zone_model
                    .predict_proba(
                        model_input
                    )[0]
                )

                for zone, probability in zip(

                    self.zone_model.classes_,

                    probability_values

                ):

                    probabilities[
                        str(zone).lower()
                    ] = float(
                        probability
                    )

            # ==================================================
            # PREDICTION CONFIDENCE
            # ==================================================

            confidence = (
                probabilities.get(
                    prediction
                )
            )

            # ==================================================
            # STORE RESULT
            # ==================================================

            self.latest_zone_result = {

                "prediction":
                    prediction,

                "confidence":
                    confidence,

                "probabilities":
                    probabilities,

                "spatial_features":
                    spatial_features.copy(),

            }

            # ==================================================
            # UPDATE COUNTERS
            # ==================================================

            if (
                prediction
                in self.zone_counts
            ):

                self.zone_counts[
                    prediction
                ] += 1

            # ==================================================
            # DISPLAY
            # ==================================================

            print()
            print("=" * 60)

            print(
                "4-ZONE LOCATION"
            )

            print("=" * 60)

            print()

            # --------------------------------------------------
            # Visual zone result
            # --------------------------------------------------

            if prediction == "top_left":

                print(
                    "X TOP LEFT    |   TOP RIGHT"
                )

                print(
                    "--------------+--------------"
                )

                print(
                    "  BOTTOM LEFT |   BOTTOM RIGHT"
                )

            elif prediction == "top_right":

                print(
                    "  TOP LEFT    | X TOP RIGHT"
                )

                print(
                    "--------------+--------------"
                )

                print(
                    "  BOTTOM LEFT |   BOTTOM RIGHT"
                )

            elif prediction == "bottom_left":

                print(
                    "  TOP LEFT    |   TOP RIGHT"
                )

                print(
                    "--------------+--------------"
                )

                print(
                    "X BOTTOM LEFT |   BOTTOM RIGHT"
                )

            elif prediction == "bottom_right":

                print(
                    "  TOP LEFT    |   TOP RIGHT"
                )

                print(
                    "--------------+--------------"
                )

                print(
                    "  BOTTOM LEFT | X BOTTOM RIGHT"
                )

            else:

                print(
                    f"UNKNOWN ZONE: "
                    f"{prediction}"
                )

            print()

            print(
                f"PREDICTED ZONE: "
                f"{prediction.upper()}"
            )

            # ==================================================
            # CONFIDENCE
            # ==================================================

            if (
                confidence
                is not None
            ):

                print(
                    f"CONFIDENCE: "
                    f"{confidence * 100:.2f}%"
                )

            # ==================================================
            # ALL PROBABILITIES
            # ==================================================

            if probabilities:

                print()
                print(
                    "ZONE PROBABILITIES"
                )

                print(
                    "-" * 40
                )

                for zone in [

                    "top_left",

                    "top_right",

                    "bottom_left",

                    "bottom_right",

                ]:

                    probability = (
                        probabilities.get(
                            zone
                        )
                    )

                    if (
                        probability
                        is not None
                    ):

                        print(

                            f"{zone.upper():15} "

                            f"{probability * 100:6.2f}%"

                        )

            # ==================================================
            # COUNTERS
            # ==================================================

            print()
            print(
                "SESSION ZONE COUNTS"
            )

            print(
                "-" * 40
            )

            print(
                f"Top Left:     "
                f"{self.zone_counts['top_left']}"
            )

            print(
                f"Top Right:    "
                f"{self.zone_counts['top_right']}"
            )

            print(
                f"Bottom Left:  "
                f"{self.zone_counts['bottom_left']}"
            )

            print(
                f"Bottom Right: "
                f"{self.zone_counts['bottom_right']}"
            )

            print("=" * 60)

        except KeyError as error:

            print()
            print("=" * 60)

            print(
                "4-ZONE FEATURE ERROR"
            )

            print("=" * 60)

            print(
                "The model expects a feature "
                "that stereo_features.py did "
                "not return:"
            )

            print(
                error
            )

        except Exception as error:

            print()
            print("=" * 60)

            print(
                "4-ZONE PREDICTION ERROR"
            )

            print("=" * 60)

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print()
    print("=" * 60)

    print(
        "ECHODESK LIVE 4-ZONE SYSTEM"
    )

    print("=" * 60)

    # ======================================================
    # MICROPHONE INFO
    # ======================================================

    device_info = sd.query_devices(
        DEVICE,
        "input"
    )

    print()

    print(
        f"Microphone: "
        f"{device_info['name']}"
    )

    print(
        f"Channels: "
        f"{CHANNELS}"
    )

    print(
        f"Sample rate: "
        f"{SAMPLERATE}"
    )

    # ======================================================
    # CREATE SYSTEM
    # ======================================================

    echodesk = EchoDesk4Zone()

    print()
    print(
        "Stay quiet for "
        "2 seconds during calibration..."
    )

    # ======================================================
    # START STREAM
    # ======================================================

    stream = sd.InputStream(

        device=
            DEVICE,

        samplerate=
            SAMPLERATE,

        channels=
            CHANNELS,

        dtype=
            "float32",

        blocksize=
            BLOCKSIZE,

        callback=
            echodesk.audio_callback

    )

    stream.start()

    try:

        while True:

            time.sleep(
                1
            )

    except KeyboardInterrupt:

        print()
        print(
            "Stopping EchoDesk 4-Zone..."
        )

    finally:

        stream.stop()

        stream.close()

        # ==================================================
        # FINAL SUMMARY
        # ==================================================

        print()
        print("=" * 60)

        print(
            "4-ZONE SESSION SUMMARY"
        )

        print("=" * 60)

        print()

        print(
            f"Candidate events: "
            f"{echodesk.candidates}"
        )

        print(
            f"Rejected events: "
            f"{echodesk.rejected}"
        )

        print(
            f"Valid taps: "
            f"{echodesk.valid_taps}"
        )

        print()

        print(
            f"TOP LEFT:     "
            f"{echodesk.zone_counts['top_left']}"
        )

        print(
            f"TOP RIGHT:    "
            f"{echodesk.zone_counts['top_right']}"
        )

        print(
            f"BOTTOM LEFT:  "
            f"{echodesk.zone_counts['bottom_left']}"
        )

        print(
            f"BOTTOM RIGHT: "
            f"{echodesk.zone_counts['bottom_right']}"
        )

        print()

        print("=" * 60)


# ==========================================================
# START
# ==========================================================

if __name__ == "__main__":

    main()