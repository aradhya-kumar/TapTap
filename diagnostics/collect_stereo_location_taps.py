import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


# ==========================================================
# Configuration
# ==========================================================

DEVICE = 1
SAMPLERATE = 48000
CHANNELS = 2

RECORD_DURATION = 0.12
PRE_DELAY = 0.15

BASE_FOLDER = Path(
    "data/stereo_location"
)

ZONES = [
    "left",
    "right",
]


# ==========================================================
# Create folders
# ==========================================================

for zone in ZONES:

    (
        BASE_FOLDER
        / zone
    ).mkdir(
        parents=True,
        exist_ok=True
    )


# ==========================================================
# Find next filename number
# ==========================================================

def get_next_number(zone):

    folder = (
        BASE_FOLDER
        / zone
    )

    files = list(
        folder.glob(
            f"{zone}_*.wav"
        )
    )

    numbers = []

    for file in files:

        try:

            number = int(
                file.stem.split(
                    "_"
                )[-1]
            )

            numbers.append(
                number
            )

        except ValueError:

            pass

    if not numbers:

        return 0

    return max(numbers) + 1


# ==========================================================
# Record one stereo tap
# ==========================================================

def record_tap(zone):

    number = get_next_number(
        zone
    )

    filename = (
        BASE_FOLDER
        / zone
        / f"{zone}_{number:04d}.wav"
    )

    print()
    print(
        f"Get ready to tap "
        f"{zone.upper()}..."
    )

    time.sleep(
        PRE_DELAY
    )

    print(
        "TAP NOW!"
    )

    recording = sd.rec(
        int(
            RECORD_DURATION
            * SAMPLERATE
        ),
        samplerate=SAMPLERATE,
        channels=CHANNELS,
        dtype="float32",
        device=DEVICE,
    )

    sd.wait()

    # ======================================================
    # Verify stereo
    # ======================================================

    if (
        recording.ndim != 2
        or
        recording.shape[1] != 2
    ):

        print(
            "ERROR: Recording "
            "is not stereo."
        )

        return

    # ======================================================
    # Basic signal check
    # ======================================================

    channel_1 = (
        recording[:, 0]
    )

    channel_2 = (
        recording[:, 1]
    )

    peak_1 = float(
        np.max(
            np.abs(
                channel_1
            )
        )
    )

    peak_2 = float(
        np.max(
            np.abs(
                channel_2
            )
        )
    )

    # Ignore recordings that are
    # essentially silent

    if (
        peak_1 < 0.001
        and
        peak_2 < 0.001
    ):

        print()
        print(
            "Recording appears silent."
        )

        print(
            "Tap was NOT saved."
        )

        return

    # ======================================================
    # Save stereo WAV
    # ======================================================

    sf.write(
        filename,
        recording,
        SAMPLERATE,
    )

    print()
    print(
        f"SAVED: "
        f"{filename}"
    )

    print(
        f"Shape: "
        f"{recording.shape}"
    )

    print(
        f"Channel 1 peak: "
        f"{peak_1:.6f}"
    )

    print(
        f"Channel 2 peak: "
        f"{peak_2:.6f}"
    )


# ==========================================================
# Show sample counts
# ==========================================================

def show_counts():

    print()
    print("=" * 50)
    print(
        "CURRENT STEREO DATASET"
    )
    print("=" * 50)

    for zone in ZONES:

        count = len(
            list(
                (
                    BASE_FOLDER
                    / zone
                ).glob(
                    "*.wav"
                )
            )
        )

        print(
            f"{zone.upper()}: "
            f"{count}"
        )

    print("=" * 50)


# ==========================================================
# Main
# ==========================================================

def main():

    print()
    print("=" * 60)
    print(
        "EchoDesk Stereo Location Collector"
    )
    print("=" * 60)

    print()
    print(
        "Microphone device:"
    )

    device_info = (
        sd.query_devices(
            DEVICE,
            "input"
        )
    )

    print(
        device_info[
            "name"
        ]
    )

    print(
        f"Channels: "
        f"{CHANNELS}"
    )

    print(
        f"Sample rate: "
        f"{SAMPLERATE}"
    )

    show_counts()

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Keep the laptop in the same "
        "position during collection."
    )

    print(
        "Tap naturally and vary the exact "
        "tap position within each zone."
    )

    print()

    while True:

        print()
        print(
            "Choose an option:"
        )

        print(
            "1 - Record LEFT tap"
        )

        print(
            "2 - Record RIGHT tap"
        )

        print(
            "3 - Show counts"
        )

        print(
            "Q - Quit"
        )

        choice = input(
            "> "
        ).strip().lower()

        if choice == "1":

            record_tap(
                "left"
            )

        elif choice == "2":

            record_tap(
                "right"
            )

        elif choice == "3":

            show_counts()

        elif choice == "q":

            break

        else:

            print(
                "Invalid option."
            )

    print()
    print(
        "Collection finished."
    )

    show_counts()


if __name__ == "__main__":
    main()