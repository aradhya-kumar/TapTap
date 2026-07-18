import time
from pathlib import Path
from collections import deque

import numpy as np
import soundfile as sf

from audio.stream import AudioStream


class EventCollector:
    """
    Collects labeled acoustic events for EchoDesk.

    Labels:
        tap
        typing
        speech
        noise

    The collector:
        1. Learns the room noise.
        2. Detects candidate acoustic events.
        3. Saves a 90 ms WAV clip.
        4. Stores it under the selected label.
    """

    def __init__(
        self,
        samplerate=48000,
        pre_ms=20,
        post_ms=70,
        warmup_sec=2.0,
        threshold_multiplier=4.0,
        peak_multiplier=5.0,
        cooldown_ms=300,
    ):

        self.samplerate = samplerate

        # --------------------------------
        # Event window configuration
        # --------------------------------

        self.pre_samples = int(
            samplerate * pre_ms / 1000
        )

        self.post_samples = int(
            samplerate * post_ms / 1000
        )

        self.event_samples = (
            self.pre_samples
            + self.post_samples
        )

        # --------------------------------
        # Detection configuration
        # --------------------------------

        self.warmup_sec = warmup_sec

        self.threshold_multiplier = (
            threshold_multiplier
        )

        self.peak_multiplier = (
            peak_multiplier
        )

        self.cooldown_sec = (
            cooldown_ms / 1000
        )

        # --------------------------------
        # Noise calibration
        # --------------------------------

        self.noise_rms_values = []

        self.noise_peak_values = []

        self.warmup_samples = 0

        self.calibrated = False

        self.noise_rms = None

        self.noise_peak = None

        self.rms_threshold = None

        self.peak_threshold = None

        # --------------------------------
        # Ring buffer
        # --------------------------------

        self.pre_buffer = deque(
            maxlen=self.pre_samples
        )

        # --------------------------------
        # Event collection
        # --------------------------------

        self.collecting = False

        self.event_buffer = []

        self.samples_needed = 0

        # --------------------------------
        # Current label
        # --------------------------------

        self.current_label = None

        # --------------------------------
        # Detection timing
        # --------------------------------

        self.last_detection = 0

        # --------------------------------
        # Storage
        # --------------------------------

        self.base_folder = Path(
            "data/events"
        )

        self.labels = [
            "tap",
            "typing",
            "speech",
            "noise",
        ]

        for label in self.labels:

            folder = (
                self.base_folder
                / label
            )

            folder.mkdir(
                parents=True,
                exist_ok=True
            )

        self.counters = {}

        for label in self.labels:

            self.counters[label] = (
                self._get_next_counter(
                    label
                )
            )

    # ==========================================
    # Find next filename number
    # ==========================================

    def _get_next_counter(
        self,
        label,
    ):

        folder = (
            self.base_folder
            / label
        )

        files = list(
            folder.glob(
                f"{label}_*.wav"
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

    # ==========================================
    # Select label
    # ==========================================

    def set_label(
        self,
        label,
    ):

        if label not in self.labels:

            print(
                f"Unknown label: {label}"
            )

            return

        self.current_label = label

        print()
        print("=" * 50)
        print(
            f"COLLECTING: "
            f"{label.upper()}"
        )
        print("=" * 50)
        print()

    # ==========================================
    # Main processing
    # ==========================================

    def process(
        self,
        audio,
    ):

        samples = (
            audio[:, 0]
            .astype(
                np.float32
            )
        )

        # --------------------------------
        # Initial calibration
        # --------------------------------

        if not self.calibrated:

            self._learn_noise(
                samples
            )

            return

        # --------------------------------
        # Finish current event
        # --------------------------------

        if self.collecting:

            self._collect_event(
                samples
            )

            return

        # --------------------------------
        # Keep rolling history
        # --------------------------------

        self.pre_buffer.extend(
            samples
        )

        # Do not save anything if
        # no label has been selected

        if self.current_label is None:

            return

        # --------------------------------
        # Calculate signal statistics
        # --------------------------------

        rms = float(
            np.sqrt(
                np.mean(
                    samples ** 2
                )
            )
        )

        peak = float(
            np.max(
                np.abs(
                    samples
                )
            )
        )

        # --------------------------------
        # Cooldown
        # --------------------------------

        if (
            time.time()
            - self.last_detection
            < self.cooldown_sec
        ):

            return

        # --------------------------------
        # Candidate event detection
        # --------------------------------

        if (
            rms
            > self.rms_threshold
            and peak
            > self.peak_threshold
        ):

            self._start_event(
                samples,
                rms,
                peak,
            )

    # ==========================================
    # Learn background noise
    # ==========================================

    def _learn_noise(
        self,
        samples,
    ):

        rms = float(
            np.sqrt(
                np.mean(
                    samples ** 2
                )
            )
        )

        peak = float(
            np.max(
                np.abs(
                    samples
                )
            )
        )

        self.noise_rms_values.append(
            rms
        )

        self.noise_peak_values.append(
            peak
        )

        self.warmup_samples += len(
            samples
        )

        self.pre_buffer.extend(
            samples
        )

        required = int(
            self.samplerate
            * self.warmup_sec
        )

        if (
            self.warmup_samples
            >= required
        ):

            self.noise_rms = float(
                np.median(
                    self.noise_rms_values
                )
            )

            self.noise_peak = float(
                np.median(
                    self.noise_peak_values
                )
            )

            self.rms_threshold = max(
                self.noise_rms
                * self.threshold_multiplier,
                0.0005,
            )

            self.peak_threshold = max(
                self.noise_peak
                * self.peak_multiplier,
                0.002,
            )

            self.calibrated = True

            print()
            print("=" * 50)
            print(
                "CALIBRATION COMPLETE"
            )
            print("=" * 50)

            print(
                f"Noise RMS: "
                f"{self.noise_rms:.6f}"
            )

            print(
                f"Noise Peak: "
                f"{self.noise_peak:.6f}"
            )

            print(
                f"RMS Threshold: "
                f"{self.rms_threshold:.6f}"
            )

            print(
                f"Peak Threshold: "
                f"{self.peak_threshold:.6f}"
            )

            print("=" * 50)
            print()

    # ==========================================
    # Start event
    # ==========================================

    def _start_event(
        self,
        samples,
        rms,
        peak,
    ):

        self.last_detection = (
            time.time()
        )

        pre_audio = np.array(
            self.pre_buffer,
            dtype=np.float32,
        )

        self.event_buffer = list(
            pre_audio
        )

        self.event_buffer.extend(
            samples
        )

        self.samples_needed = (
            self.event_samples
            - len(
                self.event_buffer
            )
        )

        print(
            f"Candidate "
            f"{self.current_label} "
            f"detected "
            f"| RMS={rms:.5f} "
            f"| Peak={peak:.5f}"
        )

        if (
            self.samples_needed
            <= 0
        ):

            self._finish_event()

        else:

            self.collecting = True

    # ==========================================
    # Collect post-trigger samples
    # ==========================================

    def _collect_event(
        self,
        samples,
    ):

        take = min(
            len(samples),
            self.samples_needed,
        )

        self.event_buffer.extend(
            samples[:take]
        )

        self.samples_needed -= take

        if (
            self.samples_needed
            <= 0
        ):

            self._finish_event()

    # ==========================================
    # Save event
    # ==========================================

    def _finish_event(
        self,
    ):

        event = np.array(
            self.event_buffer,
            dtype=np.float32,
        )

        event = event[
            :self.event_samples
        ]

        if (
            len(event)
            < self.event_samples
        ):

            event = np.pad(
                event,
                (
                    0,
                    self.event_samples
                    - len(event),
                ),
            )

        label = (
            self.current_label
        )

        number = (
            self.counters[
                label
            ]
        )

        filename = (
            self.base_folder
            / label
            / f"{label}_"
              f"{number:04d}.wav"
        )

        sf.write(
            filename,
            event,
            self.samplerate,
        )

        self.counters[
            label
        ] += 1

        print(
            f"Saved → "
            f"{filename}"
        )

        print(
            f"{label.upper()} "
            f"samples: "
            f"{self.counters[label]}"
        )

        print()

        self.collecting = False

        self.event_buffer = []

        self.samples_needed = 0

        self.pre_buffer.clear()


# ==============================================
# Main
# ==============================================


def main():

    collector = (
        EventCollector()
    )

    stream = (
        AudioStream()
    )

    stream.start(
        collector.process
    )

    print()
    print("=" * 50)
    print(
        "EchoDesk Event Collector"
    )
    print("=" * 50)

    print()
    print(
        "Stay quiet for 2 seconds."
    )
    print()

    # Wait until automatic
    # calibration finishes

    while (
        not collector.calibrated
    ):

        time.sleep(
            0.1
        )

    print(
        "Ready to collect data."
    )

    try:

        while True:

            print()
            print(
                "Choose event type:"
            )

            print(
                "1 - Desk Tap"
            )

            print(
                "2 - Typing"
            )

            print(
                "3 - Speech"
            )

            print(
                "4 - Other Noise"
            )

            print(
                "Q - Quit"
            )

            print()

            choice = input(
                "> "
            ).strip().lower()

            if choice == "1":

                collector.set_label(
                    "tap"
                )

                input(
                    "Perform desk taps. "
                    "Press ENTER when done..."
                )

                collector.current_label = None

            elif choice == "2":

                collector.set_label(
                    "typing"
                )

                input(
                    "Type normally. "
                    "Press ENTER when done..."
                )

                collector.current_label = None

            elif choice == "3":

                collector.set_label(
                    "speech"
                )

                input(
                    "Speak normally. "
                    "Press ENTER when done..."
                )

                collector.current_label = None

            elif choice == "4":

                collector.set_label(
                    "noise"
                )

                input(
                    "Create normal desk noise. "
                    "Press ENTER when done..."
                )

                collector.current_label = None

            elif choice == "q":

                break

            else:

                print(
                    "Invalid choice."
                )

    except KeyboardInterrupt:

        pass

    finally:

        stream.stop()

        print()
        print(
            "Event collection stopped."
        )


if __name__ == "__main__":

    main()