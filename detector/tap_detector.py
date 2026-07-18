import time
from collections import deque
from pathlib import Path

import numpy as np
import soundfile as sf

from audio.stream import AudioStream


class TapDetector:
    """
    EchoDesk streaming tap detector.

    Pipeline:
        Microphone
            -> Audio blocks
            -> Adaptive noise estimation
            -> Peak / RMS / crest-factor detection
            -> Pre-trigger buffer
            -> Post-trigger collection
            -> Exact 90 ms event
            -> WAV file
    """

    def __init__(
        self,
        samplerate=48000,
        pre_ms=20,
        post_ms=70,
        warmup_sec=2.0,
        threshold_multiplier=5.0,
        peak_multiplier=6.0,
        min_crest_factor=2.0,
        cooldown_ms=250,
        device=1,
    ):
        # -------------------------
        # Audio configuration
        # -------------------------

        self.samplerate = samplerate
        self.device = device

        # -------------------------
        # Event window
        # -------------------------

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

        # -------------------------
        # Detection configuration
        # -------------------------

        self.warmup_sec = warmup_sec

        self.threshold_multiplier = (
            threshold_multiplier
        )

        self.peak_multiplier = (
            peak_multiplier
        )

        self.min_crest_factor = (
            min_crest_factor
        )

        self.cooldown_sec = (
            cooldown_ms / 1000
        )

        # -------------------------
        # Noise estimation
        # -------------------------

        self.noise_rms_values = []
        self.noise_peak_values = []

        self.noise_rms = None
        self.noise_peak = None

        self.rms_threshold = None
        self.peak_threshold = None

        self.warmup_samples = 0

        self.calibrated = False

        # -------------------------
        # Ring buffer
        # -------------------------

        self.pre_buffer = deque(
            maxlen=self.pre_samples
        )

        # -------------------------
        # Event collection
        # -------------------------

        self.collecting = False

        self.event_buffer = []

        self.samples_needed = 0

        # -------------------------
        # Timing
        # -------------------------

        self.last_detection_time = 0

        # -------------------------
        # Output
        # -------------------------

        self.output_folder = Path(
            "data/raw"
        )

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        self.event_counter = self._get_next_counter()

    # ==================================================
    # Find next available WAV number
    # ==================================================

    def _get_next_counter(self):

        files = list(
            self.output_folder.glob(
                "tap_*.wav"
            )
        )

        if not files:
            return 0

        numbers = []

        for file in files:

            try:

                number = int(
                    file.stem.split("_")[1]
                )

                numbers.append(number)

            except (ValueError, IndexError):
                pass

        if not numbers:
            return 0

        return max(numbers) + 1

    # ==================================================
    # Main audio processing
    # ==================================================

    def process(self, audio):

        samples = audio[:, 0].astype(
            np.float32
        )

        # -------------------------
        # Warm-up / calibration
        # -------------------------

        if not self.calibrated:

            self._learn_noise(samples)

            return

        # -------------------------
        # Currently collecting event
        # -------------------------

        if self.collecting:

            self._collect_post_trigger(
                samples
            )

            return

        # -------------------------
        # Analyze current block
        # -------------------------

        rms = self._calculate_rms(
            samples
        )

        peak = self._calculate_peak(
            samples
        )

        crest_factor = (
            peak / (rms + 1e-12)
        )

        # -------------------------
        # Check cooldown
        # -------------------------

        current_time = time.time()

        cooldown_active = (
            current_time
            - self.last_detection_time
            < self.cooldown_sec
        )

        # -------------------------
        # Detection
        # -------------------------

        if not cooldown_active:

            detected = self._is_tap(
                rms,
                peak,
                crest_factor,
            )

            if detected:

                self._start_event(
                    samples,
                    rms,
                    peak,
                    crest_factor,
                )

                return

        # -------------------------
        # Update rolling buffer
        # -------------------------

        self._update_pre_buffer(
            samples
        )

        # -------------------------
        # Slowly adapt noise floor
        # -------------------------

        self._update_noise_floor(
            rms,
            peak,
        )

    # ==================================================
    # Learn initial room noise
    # ==================================================

    def _learn_noise(self, samples):

        rms = self._calculate_rms(
            samples
        )

        peak = self._calculate_peak(
            samples
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

        self._update_pre_buffer(
            samples
        )

        required_samples = int(
            self.warmup_sec
            * self.samplerate
        )

        if (
            self.warmup_samples
            >= required_samples
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

            self._update_thresholds()

            self.calibrated = True

            print()
            print("=" * 50)
            print(
                "EchoDesk calibration complete"
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
            print(
                "Listening for taps..."
            )
            print()

    # ==================================================
    # Tap decision
    # ==================================================

    def _is_tap(
        self,
        rms,
        peak,
        crest_factor,
    ):

        rms_pass = (
            rms
            > self.rms_threshold
        )

        peak_pass = (
            peak
            > self.peak_threshold
        )

        crest_pass = (
            crest_factor
            >= self.min_crest_factor
        )

        clipping = (
            peak >= 0.99
        )

        return (
            rms_pass
            and peak_pass
            and crest_pass
            and not clipping
        )

    # ==================================================
    # Start event collection
    # ==================================================

    def _start_event(
        self,
        samples,
        rms,
        peak,
        crest_factor,
    ):

        self.last_detection_time = (
            time.time()
        )

        print(
            "TAP DETECTED"
        )

        print(
            f"RMS: {rms:.6f} | "
            f"Peak: {peak:.6f} | "
            f"Crest: "
            f"{crest_factor:.2f}"
        )

        # Copy exact pre-trigger history

        pre_audio = np.array(
            self.pre_buffer,
            dtype=np.float32,
        )

        # Start event with pre-trigger

        self.event_buffer = list(
            pre_audio
        )

        # Add current block

        self.event_buffer.extend(
            samples
        )

        self.samples_needed = (
            self.event_samples
            - len(self.event_buffer)
        )

        # Enough audio already collected

        if self.samples_needed <= 0:

            self._finish_event()

        else:

            self.collecting = True

    # ==================================================
    # Collect audio after trigger
    # ==================================================

    def _collect_post_trigger(
        self,
        samples,
    ):

        if self.samples_needed <= 0:

            self._finish_event()

            return

        take = min(
            len(samples),
            self.samples_needed,
        )

        self.event_buffer.extend(
            samples[:take]
        )

        self.samples_needed -= take

        if self.samples_needed <= 0:

            self._finish_event()

    # ==================================================
    # Finish event
    # ==================================================

    def _finish_event(self):

        event = np.array(
            self.event_buffer,
            dtype=np.float32,
        )

        # Guarantee exactly 90 ms

        if len(event) > self.event_samples:

            event = event[
                :self.event_samples
            ]

        elif len(event) < self.event_samples:

            padding = (
                self.event_samples
                - len(event)
            )

            event = np.pad(
                event,
                (0, padding),
            )

        self._save_event(
            event
        )

        # Reset

        self.collecting = False

        self.event_buffer = []

        self.samples_needed = 0

        # Clear old pre-trigger audio

        self.pre_buffer.clear()

    # ==================================================
    # Save WAV
    # ==================================================

    def _save_event(
        self,
        event,
    ):

        filename = (
            self.output_folder
            / f"tap_"
              f"{self.event_counter:04d}"
              f".wav"
        )

        sf.write(
            filename,
            event,
            self.samplerate,
        )

        duration_ms = (
            len(event)
            / self.samplerate
            * 1000
        )

        print(
            f"Saved: {filename}"
        )

        print(
            f"Duration: "
            f"{duration_ms:.1f} ms"
        )

        print()

        self.event_counter += 1

    # ==================================================
    # Ring buffer
    # ==================================================

    def _update_pre_buffer(
        self,
        samples,
    ):

        self.pre_buffer.extend(
            samples
        )

    # ==================================================
    # Adaptive noise floor
    # ==================================================

    def _update_noise_floor(
        self,
        rms,
        peak,
    ):

        # Only adapt when signal is below
        # detection threshold

        if rms < self.rms_threshold:

            alpha = 0.005

            self.noise_rms = (
                (1 - alpha)
                * self.noise_rms
                + alpha
                * rms
            )

            self.noise_peak = (
                (1 - alpha)
                * self.noise_peak
                + alpha
                * peak
            )

            self._update_thresholds()

    # ==================================================
    # Threshold calculation
    # ==================================================

    def _update_thresholds(self):

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

    # ==================================================
    # DSP helpers
    # ==================================================

    @staticmethod
    def _calculate_rms(
        samples,
    ):

        return float(
            np.sqrt(
                np.mean(
                    samples ** 2
                )
            )
        )

    @staticmethod
    def _calculate_peak(
        samples,
    ):

        return float(
            np.max(
                np.abs(
                    samples
                )
            )
        )


# ======================================================
# Main
# ======================================================


def main():

    print()
    print("=" * 50)
    print("EchoDesk Tap Detector")
    print("=" * 50)

    print()
    print(
        "Keep quiet for 2 seconds "
        "while EchoDesk learns "
        "the room noise."
    )

    print()

    detector = TapDetector()

    stream = AudioStream()

    stream.start(
        detector.process
    )

    try:

        while True:

            time.sleep(1)

    except KeyboardInterrupt:

        print()
        print(
            "Stopping EchoDesk..."
        )

        stream.stop()


if __name__ == "__main__":

    main()