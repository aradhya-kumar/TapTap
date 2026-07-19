import time
import csv
from pathlib import Path
from datetime import datetime

import sounddevice as sd

from main import (
    EchoDesk,
    DEVICE,
    SAMPLERATE,
    CHANNELS,
    BLOCKSIZE,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

LOG_FOLDER = Path("logs")

LOG_FILE = (
    LOG_FOLDER
    / "location_test_results.csv"
)

LOG_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# TEST LOGGER
# ==========================================================

class TestLogger:

    def __init__(self):

        self.fieldnames = [
            "timestamp",
            "actual_location",
            "predicted_location",
            "correct",
            "tap_probability",
            "location_probability",
            "left_probability",
            "right_probability",
            "log_rms_ratio",
            "log_energy_ratio",
            "peak_difference",
            "peak_sample_difference",
            "correlation_lag",
            "correlation_value",
            "channel_correlation",
        ]

        # Create CSV if it does not exist

        if not LOG_FILE.exists():

            with open(
                LOG_FILE,
                "w",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=self.fieldnames
                )

                writer.writeheader()

    # ======================================================
    # LOG RESULT
    # ======================================================

    def log(
        self,
        actual_location,
        result
    ):

        prediction = result[
            "prediction"
        ]

        spatial = result[
            "spatial_features"
        ]

        correct = (
            prediction
            == actual_location
        )

        row = {

            "timestamp":
                datetime.now().isoformat(),

            "actual_location":
                actual_location,

            "predicted_location":
                prediction,

            "correct":
                correct,

            "tap_probability":
                result.get(
                    "tap_probability"
                ),

            "location_probability":
                result.get(
                    "location_probability"
                ),

            "left_probability":
                result.get(
                    "left_probability"
                ),

            "right_probability":
                result.get(
                    "right_probability"
                ),

            "log_rms_ratio":
                spatial.get(
                    "log_rms_ratio"
                ),

            "log_energy_ratio":
                spatial.get(
                    "log_energy_ratio"
                ),

            "peak_difference":
                spatial.get(
                    "peak_difference"
                ),

            "peak_sample_difference":
                spatial.get(
                    "peak_sample_difference"
                ),

            "correlation_lag":
                spatial.get(
                    "correlation_lag"
                ),

            "correlation_value":
                spatial.get(
                    "correlation_value"
                ),

            "channel_correlation":
                spatial.get(
                    "channel_correlation"
                ),
        }

        with open(
            LOG_FILE,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=self.fieldnames
            )

            writer.writerow(
                row
            )


# ==========================================================
# TEST VERSION OF ECHODESK
# ==========================================================

class EchoDeskTest(
    EchoDesk
):

    def __init__(
        self,
        actual_location
    ):

        # Initialize main EchoDesk first

        super().__init__()

        # Actual side being tested

        self.actual_location = (
            actual_location
        )

        # Detailed test logger

        self.test_logger = (
            TestLogger()
        )

        # Test statistics

        self.test_total = 0

        self.test_correct = 0

        self.test_wrong = 0


    # ======================================================
    # PROCESS EVENT
    # ======================================================

    def finish_event(
        self
    ):

        # Run normal EchoDesk processing first

        super().finish_event()

        # --------------------------------------------------
        # If event was rejected as NOT TAP,
        # latest_result will be None.
        # --------------------------------------------------

        if (
            self.latest_result
            is None
        ):

            return

        result = (
            self.latest_result
        )

        prediction = (
            result[
                "prediction"
            ]
        )

        correct = (
            prediction
            == self.actual_location
        )

        # --------------------------------------------------
        # Update statistics
        # --------------------------------------------------

        self.test_total += 1

        if correct:

            self.test_correct += 1

        else:

            self.test_wrong += 1

        # --------------------------------------------------
        # Save detailed result
        # --------------------------------------------------

        self.test_logger.log(
            self.actual_location,
            result
        )

        # --------------------------------------------------
        # Calculate accuracy
        # --------------------------------------------------

        accuracy = (

            self.test_correct
            / self.test_total
            * 100

        )

        # --------------------------------------------------
        # Display result
        # --------------------------------------------------

        print()
        print("-" * 60)

        print(
            f"ACTUAL:    "
            f"{self.actual_location.upper()}"
        )

        print(
            f"PREDICTED: "
            f"{prediction.upper()}"
        )

        if correct:

            print(
                "RESULT:    CORRECT"
            )

        else:

            print(
                "RESULT:    WRONG"
            )

        print()

        print(
            f"CORRECT: "
            f"{self.test_correct}"
        )

        print(
            f"WRONG:   "
            f"{self.test_wrong}"
        )

        print(
            f"TOTAL:   "
            f"{self.test_total}"
        )

        print(
            f"ACCURACY: "
            f"{accuracy:.1f}%"
        )

        print("-" * 60)


# ==========================================================
# SELECT ACTUAL LOCATION
# ==========================================================

def select_location():

    while True:

        print()
        print("=" * 60)

        print(
            "ECHODESK LIVE LOCATION TEST"
        )

        print("=" * 60)

        print()
        print(
            "Which side will you tap?"
        )

        print()
        print(
            "1 - LEFT"
        )

        print(
            "2 - RIGHT"
        )

        print()

        choice = input(
            "> "
        ).strip()

        if choice == "1":

            return "left"

        elif choice == "2":

            return "right"

        else:

            print()
            print(
                "Invalid option."
            )


# ==========================================================
# MAIN
# ==========================================================

def main():

    # ------------------------------------------------------
    # Select actual test side
    # ------------------------------------------------------

    actual_location = (
        select_location()
    )

    print()
    print("=" * 60)

    print(
        f"TESTING ACTUAL LOCATION: "
        f"{actual_location.upper()}"
    )

    print("=" * 60)

    print()
    print(
        "Stay quiet during "
        "the 2-second calibration."
    )

    print()

    print(
        "When ECHODESK READY appears,"
    )

    print(
        f"tap ONLY the "
        f"{actual_location.upper()} side."
    )

    print()

    print(
        "Recommended: 20 taps."
    )

    print()

    print(
        "Press Ctrl+C when finished."
    )

    print()

    # ------------------------------------------------------
    # Create test system
    # ------------------------------------------------------

    echodesk = EchoDeskTest(
        actual_location
    )

    # ------------------------------------------------------
    # Open microphone stream
    # ------------------------------------------------------

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
            "Stopping location test..."
        )

    finally:

        stream.stop()

        stream.close()

        # ==================================================
        # FINAL TEST RESULTS
        # ==================================================

        print()
        print("=" * 60)

        print(
            "LOCATION TEST RESULTS"
        )

        print("=" * 60)

        print()

        print(
            f"Actual location: "
            f"{actual_location.upper()}"
        )

        print(
            f"Valid taps tested: "
            f"{echodesk.test_total}"
        )

        print(
            f"Correct: "
            f"{echodesk.test_correct}"
        )

        print(
            f"Wrong: "
            f"{echodesk.test_wrong}"
        )

        if (
            echodesk.test_total
            > 0
        ):

            accuracy = (

                echodesk.test_correct
                / echodesk.test_total
                * 100

            )

            print(
                f"Accuracy: "
                f"{accuracy:.2f}%"
            )

        print()

        print(
            "Detailed results saved to:"
        )

        print(
            LOG_FILE
        )

        print()

        print("=" * 60)


# ==========================================================
# START
# ==========================================================

if __name__ == "__main__":

    main()