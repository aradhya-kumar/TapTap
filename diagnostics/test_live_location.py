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

TEST_LOG_FOLDER = Path("logs")

TEST_LOG_FILE = (
    TEST_LOG_FOLDER
    / "location_test_results.csv"
)

TEST_LOG_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# TEST LOGGER
# ==========================================================

class LocationTestLogger:

    def __init__(self):

        self.fieldnames = [
            "timestamp",
            "actual_location",
            "predicted_location",
            "correct",
            "tap_probability",
            "location_probability",
            "log_rms_ratio",
            "log_energy_ratio",
            "peak_difference",
            "peak_sample_difference",
            "correlation_lag",
            "correlation_value",
            "channel_correlation",
        ]

        if not TEST_LOG_FILE.exists():

            with open(
                TEST_LOG_FILE,
                "w",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=self.fieldnames
                )

                writer.writeheader()

    def log(
        self,
        actual_location,
        predicted_location,
        tap_probability,
        location_probability,
        spatial_features
    ):

        correct = (
            actual_location
            == predicted_location
        )

        row = {

            "timestamp":
                datetime.now().isoformat(),

            "actual_location":
                actual_location,

            "predicted_location":
                predicted_location,

            "correct":
                correct,

            "tap_probability":
                tap_probability,

            "location_probability":
                location_probability,

            "log_rms_ratio":
                spatial_features.get(
                    "log_rms_ratio"
                ),

            "log_energy_ratio":
                spatial_features.get(
                    "log_energy_ratio"
                ),

            "peak_difference":
                spatial_features.get(
                    "peak_difference"
                ),

            "peak_sample_difference":
                spatial_features.get(
                    "peak_sample_difference"
                ),

            "correlation_lag":
                spatial_features.get(
                    "correlation_lag"
                ),

            "correlation_value":
                spatial_features.get(
                    "correlation_value"
                ),

            "channel_correlation":
                spatial_features.get(
                    "channel_correlation"
                ),
        }

        with open(
            TEST_LOG_FILE,
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

        super().__init__()

        self.actual_location = (
            actual_location
        )

        self.test_logger = (
            LocationTestLogger()
        )

        self.test_total = 0

        self.test_correct = 0

        self.test_wrong = 0

        self.last_logged_valid_taps = 0

        self.last_left_count = 0

        self.last_right_count = 0

    # ======================================================
    # WATCH FOR NEW PREDICTIONS
    # ======================================================

    def finish_event(
        self
    ):

        # Save counters before
        # parent processes event

        old_valid = (
            self.valid_taps
        )

        old_left = (
            self.left_taps
        )

        old_right = (
            self.right_taps
        )

        # Run normal EchoDesk pipeline

        super().finish_event()

        # --------------------------------------------------
        # Check if a valid tap was detected
        # --------------------------------------------------

        if (
            self.valid_taps
            == old_valid
        ):

            return

        # --------------------------------------------------
        # Determine prediction from counters
        # --------------------------------------------------

        if (
            self.left_taps
            > old_left
        ):

            prediction = (
                "left"
            )

        elif (
            self.right_taps
            > old_right
        ):

            prediction = (
                "right"
            )

        else:

            return

        self.test_total += 1

        # --------------------------------------------------
        # Accuracy
        # --------------------------------------------------

        correct = (
            prediction
            == self.actual_location
        )

        if correct:

            self.test_correct += 1

            result = (
                "CORRECT"
            )

        else:

            self.test_wrong += 1

            result = (
                "WRONG"
            )

        # --------------------------------------------------
        # Display result
        # --------------------------------------------------

        print()
        print(
            "-" * 60
        )

        print(
            f"ACTUAL:    "
            f"{self.actual_location.upper()}"
        )

        print(
            f"PREDICTED: "
            f"{prediction.upper()}"
        )

        print(
            f"RESULT:    "
            f"{result}"
        )

        accuracy = (
            self.test_correct
            / self.test_total
            * 100
        )

        print()
        print(
            f"TEST SCORE: "
            f"{self.test_correct}"
            f"/"
            f"{self.test_total}"
        )

        print(
            f"ACCURACY: "
            f"{accuracy:.1f}%"
        )

        print(
            "-" * 60
        )


# ==========================================================
# SELECT TEST LOCATION
# ==========================================================

def select_location():

    while True:

        print()
        print("=" * 60)
        print(
            "ECHODESK LOCATION TEST"
        )
        print("=" * 60)

        print()
        print(
            "Which side are you "
            "going to tap?"
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

        if choice == "2":

            return "right"

        print(
            "Invalid option."
        )


# ==========================================================
# MAIN
# ==========================================================

def main():

    actual_location = (
        select_location()
    )

    print()
    print("=" * 60)

    print(
        f"TESTING: "
        f"{actual_location.upper()}"
    )

    print("=" * 60)

    print()
    print(
        "All taps in this session "
        "should be on the:"
    )

    print()
    print(
        f">>> "
        f"{actual_location.upper()} "
        f"SIDE <<<"
    )

    print()

    print(
        "Stay quiet for "
        "2 seconds during calibration."
    )

    print()

    # ======================================================
    # CREATE TEST SYSTEM
    # ======================================================

    echodesk = EchoDeskTest(
        actual_location
    )

    # ======================================================
    # START MICROPHONE
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
            "Stopping test..."
        )

    finally:

        stream.stop()

        stream.close()

        # ==================================================
        # FINAL RESULTS
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
            "Test results:"
        )

        print(
            TEST_LOG_FILE
        )

        print("=" * 60)


# ==========================================================
# START
# ==========================================================

if __name__ == "__main__":

    main()