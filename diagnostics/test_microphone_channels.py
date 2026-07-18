import sounddevice as sd
import numpy as np
import time


SAMPLERATE = 48000
DEVICE = 1
DURATION = 5


def main():

    print()
    print("=" * 60)
    print("EchoDesk Microphone Channel Test")
    print("=" * 60)

    # Get device information

    device_info = sd.query_devices(
        DEVICE,
        "input"
    )

    max_channels = int(
        device_info[
            "max_input_channels"
        ]
    )

    print()
    print(
        f"Device: "
        f"{device_info['name']}"
    )

    print(
        f"Maximum input channels: "
        f"{max_channels}"
    )

    print(
        f"Default sample rate: "
        f"{device_info['default_samplerate']}"
    )

    if max_channels < 2:

        print()
        print(
            "Only one input channel is available."
        )

        print(
            "Multi-channel localization "
            "cannot be tested on this device."
        )

        return

    print()
    print(
        f"Recording {DURATION} seconds "
        f"using {max_channels} channels."
    )

    print()
    print(
        "During recording:"
    )

    print(
        "1. Tap LEFT side 3 times."
    )

    print(
        "2. Wait about 1 second."
    )

    print(
        "3. Tap RIGHT side 3 times."
    )

    print()
    print(
        "Starting in 3 seconds..."
    )

    time.sleep(3)

    print()
    print("RECORDING NOW!")
    print()

    # Record all available channels

    recording = sd.rec(
        int(
            DURATION
            * SAMPLERATE
        ),
        samplerate=SAMPLERATE,
        channels=max_channels,
        dtype="float32",
        device=DEVICE,
    )

    sd.wait()

    print()
    print(
        "Recording complete."
    )

    print()
    print("=" * 60)
    print("CHANNEL ANALYSIS")
    print("=" * 60)

    # Analyze each channel

    for channel in range(
        max_channels
    ):

        signal = recording[
            :,
            channel
        ]

        rms = float(
            np.sqrt(
                np.mean(
                    signal ** 2
                )
            )
        )

        peak = float(
            np.max(
                np.abs(
                    signal
                )
            )
        )

        energy = float(
            np.sum(
                signal ** 2
            )
        )

        print()
        print(
            f"Channel {channel + 1}"
        )

        print(
            f"  RMS:    "
            f"{rms:.8f}"
        )

        print(
            f"  Peak:   "
            f"{peak:.8f}"
        )

        print(
            f"  Energy: "
            f"{energy:.8f}"
        )

    # ==========================================
    # Compare channels
    # ==========================================

    print()
    print("=" * 60)
    print("CHANNEL DIFFERENCE TEST")
    print("=" * 60)

    for i in range(
        max_channels
    ):

        for j in range(
            i + 1,
            max_channels
        ):

            channel_a = recording[
                :,
                i
            ]

            channel_b = recording[
                :,
                j
            ]

            # Mean absolute difference

            difference = float(
                np.mean(
                    np.abs(
                        channel_a
                        - channel_b
                    )
                )
            )

            # Correlation

            if (
                np.std(channel_a) > 0
                and
                np.std(channel_b) > 0
            ):

                correlation = float(
                    np.corrcoef(
                        channel_a,
                        channel_b
                    )[0, 1]
                )

            else:

                correlation = 0.0

            print()
            print(
                f"Channel "
                f"{i + 1} "
                f"vs "
                f"Channel "
                f"{j + 1}"
            )

            print(
                f"  Mean difference: "
                f"{difference:.10f}"
            )

            print(
                f"  Correlation: "
                f"{correlation:.6f}"
            )

    print()
    print("=" * 60)

    print(
        "Interpretation:"
    )

    print()

    print(
        "If channels have nearly identical "
        "values and correlation near 1.0,"
    )

    print(
        "they may be duplicate signals."
    )

    print()

    print(
        "If channels have measurable differences "
        "and lower correlation,"
    )

    print(
        "the microphone array may provide useful "
        "spatial information."
    )

    print("=" * 60)


if __name__ == "__main__":
    main()