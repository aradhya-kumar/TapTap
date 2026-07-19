import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write


# ==========================================================
# CONFIGURATION
# ==========================================================

DEVICE = 1
SAMPLERATE = 48000
CHANNELS = 2

RECORD_SECONDS = 0.30

OUTPUT_FOLDER = Path(
    "data/stereo_4zone"
)

ZONES = {
    "1": "top_left",
    "2": "top_right",
    "3": "bottom_left",
    "4": "bottom_right",
}


# ==========================================================
# SELECT ZONE
# ==========================================================

def select_zone():

    while True:

        print()
        print("=" * 60)
        print("ECHODESK 4-ZONE STEREO DATA COLLECTOR")
        print("=" * 60)

        print()
        print("Select the zone you are collecting:")
        print()
        print("1 - TOP LEFT")
        print("2 - TOP RIGHT")
        print("3 - BOTTOM LEFT")
        print("4 - BOTTOM RIGHT")
        print()

        choice = input("> ").strip()

        if choice in ZONES:

            return ZONES[choice]

        print()
        print("Invalid option.")


# ==========================================================
# FIND NEXT FILE NUMBER
# ==========================================================

def get_next_index(
    folder
):

    existing_files = list(
        folder.glob("*.wav")
    )

    if not existing_files:

        return 1

    numbers = []

    for file in existing_files:

        try:

            number = int(
                file.stem.split("_")[-1]
            )

            numbers.append(
                number
            )

        except ValueError:

            continue

    if not numbers:

        return 1

    return max(
        numbers
    ) + 1


# ==========================================================
# RECORD TAP
# ==========================================================

def record_tap():

    frames = int(
        RECORD_SECONDS
        * SAMPLERATE
    )

    audio = sd.rec(
        frames,
        samplerate=SAMPLERATE,
        channels=CHANNELS,
        dtype="float32",
        device=DEVICE
    )

    sd.wait()

    return audio


# ==========================================================
# SAVE TAP
# ==========================================================

def save_tap(
    audio,
    filepath
):

    # Convert float32 audio to int16 WAV

    audio = np.clip(
        audio,
        -1.0,
        1.0
    )

    audio_int16 = (
        audio
        * 32767
    ).astype(
        np.int16
    )

    write(
        filepath,
        SAMPLERATE,
        audio_int16
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    zone = select_zone()

    zone_folder = (
        OUTPUT_FOLDER
        / zone
    )

    zone_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    next_index = get_next_index(
        zone_folder
    )

    print()
    print("=" * 60)

    print(
        f"COLLECTING: "
        f"{zone.upper().replace('_', ' ')}"
    )

    print("=" * 60)

    print()
    print(
        f"Output folder: "
        f"{zone_folder}"
    )

    print()
    print(
        "Instructions:"
    )

    print()
    print(
        "- Press ENTER"
        " to record one tap."
    )

    print(
        "- Tap immediately"
        " after pressing ENTER."
    )

    print(
        "- Each recording is"
        " 0.30 seconds."
    )

    print(
        "- Type q and press ENTER"
        " when finished."
    )

    print()
    print(
        "Recommended:"
        " 50 taps per zone."
    )

    print()
    print(
        "Vary the taps slightly:"
    )

    print(
        "- Soft taps"
    )

    print(
        "- Normal taps"
    )

    print(
        "- Slightly harder taps"
    )

    print(
        "- Different positions"
        " inside the same zone"
    )

    print()

    count = 0

    while True:

        command = input(
            "Press ENTER to record "
            "or q to quit: "
        ).strip().lower()

        if command == "q":

            break

        print(
            "RECORDING... TAP NOW!"
        )

        audio = record_tap()

        filename = (
            f"{zone}_"
            f"{next_index:04d}.wav"
        )

        filepath = (
            zone_folder
            / filename
        )

        save_tap(
            audio,
            filepath
        )

        count += 1

        print(
            f"SAVED: {filepath}"
        )

        print(
            f"Session count: "
            f"{count}"
        )

        print()

        next_index += 1

        time.sleep(
            0.1
        )

    print()
    print("=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)

    print(
        f"Zone: "
        f"{zone.upper().replace('_', ' ')}"
    )

    print(
        f"Taps collected this session: "
        f"{count}"
    )

    print(
        f"Saved in: "
        f"{zone_folder}"
    )

    print("=" * 60)


# ==========================================================
# START
# ==========================================================

if __name__ == "__main__":

    main()