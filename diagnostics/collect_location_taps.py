import time
from pathlib import Path
from collections import deque

import numpy as np
import soundfile as sf

from audio.stream import AudioStream


class LocationTapCollector:

    def __init__(
        self,
        samplerate=48000,
        pre_ms=20,
        post_ms=70,
        warmup_sec=2.0,
        threshold_multiplier=4.0,
        peak_multiplier=5.0,
        cooldown_ms=400,
    ):

        self.samplerate = samplerate

        # ------------------------------------------
        # Event duration
        # ------------------------------------------

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

        # ------------------------------------------
        # Detection settings
        # ------------------------------------------

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

        # ------------------------------------------
        # Calibration
        # ------------------------------------------

        self.noise_rms_values = []

        self.noise_peak_values = []

        self.warmup_samples = 0

        self.calibrated = False

        self.noise_rms = None

        self.noise_peak = None

        self.rms_threshold = None

        self.peak_threshold = None

        # ------------------------------------------
        # Ring buffer
        # ------------------------------------------

        self.pre_buffer = deque(
            maxlen=self.pre_samples
        )

        # ------------------------------------------
        # Event collection
        # ------------------------------------------

        self.collecting = False

        self.event_buffer = []

        self.samples_needed = 0

        self.last_detection = 0

        # ------------------------------------------
        # Current location
        # ------------------------------------------

        self.current_zone = None

        # ------------------------------------------
        # Storage
        # ------------------------------------------

        self.base_folder = Path(
            "data/location"
        )

        self.zones = [
            "left",
            "right",
        ]

        for zone in self.zones:

            (
                self.base_folder
                / zone
            ).mkdir(
                parents=True,
                exist_ok=True
            )

        self.counters = {}

        for zone in self.zones:

            self.counters[zone] = (
                self._get_next_counter(
                    zone
                )
            )

    # ==============================================
    # Get next filename
    # ==============================================

    def _get_next_counter(
        self,
        zone,
    ):

        folder = (
            self.base_folder
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

    # ==============================================
    # Select zone
    # ==============================================

    def set_zone(
        self,
        zone,
    ):

        if zone not in self.zones:

            return

        self.current_zone = zone

        print()
        print("=" * 50)
        print(
            f"COLLECTING: "
            f"{zone.upper()}"
        )
        print("=" * 50)

        print(
            f"Current samples: "
            f"{self.counters[zone]}"
        )

        print()

    # ==============================================
    # Audio processing
    # ==============================================

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

        # ------------------------------------------
        # Calibration
        # ------------------------------------------

        if not self.calibrated:

            self._learn_noise(
                samples
            )

            return

        # ------------------------------------------
        # Continue current event
        # ------------------------------------------

        if self.collecting:

            self._collect_event(
                samples
            )

            return

        # ------------------------------------------
        # Calculate signal values
        # ------------------------------------------

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

        # ------------------------------------------
        # Detect candidate tap
        # ------------------------------------------

        if (
            self.current_zone
            is not None
        ):

            cooldown_finished = (
                time.time()
                - self.last_detection
                >= self.cooldown_sec
            )

            if (
                cooldown_finished
                and
                rms
                > self.rms_threshold
                and
                peak
                > self.peak_threshold
            ):

                self._start_event(
                    samples
                )

                return

        # ------------------------------------------
        # Update history
        # ------------------------------------------

        self.pre_buffer.extend(
            samples
        )

    # ==============================================
    # Noise calibration
    # ==============================================

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
                0.0005
            )

            self.peak_threshold = max(
                self.noise_peak
                * self.peak_multiplier,
                0.002
            )

            self.calibrated = True

            print()
            print("=" * 50)
            print(
                "CALIBRATION COMPLETE"
            )
            print("=" * 50)

            print(
                f"RMS threshold: "
                f"{self.rms_threshold:.6f}"
            )

            print(
                f"Peak threshold: "
                f"{self.peak_threshold:.6f}"
            )

            print()

    # ==============================================
    # Start event
    # ==============================================

    def _start_event(
        self,
        samples,
    ):

        self.last_detection = (
            time.time()
        )

        pre_audio = np.array(
            self.pre_buffer,
            dtype=np.float32
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

        if (
            self.samples_needed
            <= 0
        ):

            self._finish_event()

        else:

            self.collecting = True

    # ==============================================
    # Continue event
    # ==============================================

    def _collect_event(
        self,
        samples,
    ):

        take = min(
            len(samples),
            self.samples_needed
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

    # ==============================================
    # Save event
    # ==============================================

    def _finish_event(
        self,
    ):

        event = np.array(
            self.event_buffer,
            dtype=np.float32
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
                    - len(event)
                )
            )

        zone = self.current_zone

        if zone is not None:

            number = (
                self.counters[
                    zone
                ]
            )

            filename = (
                self.base_folder
                / zone
                / f"{zone}_"
                  f"{number:04d}.wav"
            )

            sf.write(
                filename,
                event,
                self.samplerate
            )

            self.counters[
                zone
            ] += 1

            print(
                f"{zone.upper()} "
                f"tap saved → "
                f"{self.counters[zone]}"
            )

        self.collecting = False

        self.event_buffer = []

        self.samples_needed = 0

        self.pre_buffer.clear()


# ==================================================
# Main
# ==================================================


def main():

    collector = (
        LocationTapCollector()
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
        "EchoDesk Location Tap Collector"
    )
    print("=" * 50)

    print()
    print(
        "Stay quiet for 2 seconds..."
    )

    while (
        not collector.calibrated
    ):

        time.sleep(
            0.1
        )

    try:

        while True:

            print()
            print(
                "Choose tap location:"
            )

            print(
                "1 - LEFT"
            )

            print(
                "2 - RIGHT"
            )

            print(
                "Q - Quit"
            )

            choice = input(
                "> "
            ).strip().lower()

            if choice == "1":

                collector.set_zone(
                    "left"
                )

                input(
                    "Tap LEFT side repeatedly. "
                    "Press ENTER when finished..."
                )

                collector.current_zone = None

            elif choice == "2":

                collector.set_zone(
                    "right"
                )

                input(
                    "Tap RIGHT side repeatedly. "
                    "Press ENTER when finished..."
                )

                collector.current_zone = None

            elif choice == "q":

                break

    except KeyboardInterrupt:

        pass

    finally:

        stream.stop()

        print()
        print(
            "Collection finished."
        )

        print(
            f"LEFT samples: "
            f"{collector.counters['left']}"
        )

        print(
            f"RIGHT samples: "
            f"{collector.counters['right']}"
        )


if __name__ == "__main__":
    main()