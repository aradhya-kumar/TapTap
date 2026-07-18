import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from audio.stream import AudioStream
from dsp.features import extract


class LiveTapClassifier:

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

        # ==========================================
        # Load trained model
        # ==========================================

        model_path = Path(
            "models/tap_classifier.pkl"
        )

        feature_path = Path(
            "models/tap_classifier_features.pkl"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                "models/tap_classifier.pkl not found."
            )

        if not feature_path.exists():
            raise FileNotFoundError(
                "models/tap_classifier_features.pkl not found."
            )

        print("Loading tap classifier...")

        self.model = joblib.load(
            model_path
        )

        self.feature_names = joblib.load(
            feature_path
        )

        print(
            f"Model loaded with "
            f"{len(self.feature_names)} features."
        )

        # ==========================================
        # Event window
        # ==========================================

        self.pre_samples = int(
            samplerate
            * pre_ms
            / 1000
        )

        self.post_samples = int(
            samplerate
            * post_ms
            / 1000
        )

        self.event_samples = (
            self.pre_samples
            + self.post_samples
        )

        # ==========================================
        # Detection configuration
        # ==========================================

        self.warmup_sec = (
            warmup_sec
        )

        self.threshold_multiplier = (
            threshold_multiplier
        )

        self.peak_multiplier = (
            peak_multiplier
        )

        self.cooldown_sec = (
            cooldown_ms
            / 1000
        )

        # ==========================================
        # Noise calibration
        # ==========================================

        self.noise_rms_values = []

        self.noise_peak_values = []

        self.warmup_samples = 0

        self.calibrated = False

        self.noise_rms = None

        self.noise_peak = None

        self.rms_threshold = None

        self.peak_threshold = None

        # ==========================================
        # Pre-trigger buffer
        # ==========================================

        self.pre_buffer = deque(
            maxlen=self.pre_samples
        )

        # ==========================================
        # Event collection
        # ==========================================

        self.collecting = False

        self.event_buffer = []

        self.samples_needed = 0

        # ==========================================
        # Timing
        # ==========================================

        self.last_detection = 0

        # ==========================================
        # Statistics
        # ==========================================

        self.total_events = 0

        self.tap_events = 0

        self.not_tap_events = 0

    # ==============================================
    # Main audio callback
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
        # Initial calibration
        # ------------------------------------------

        if not self.calibrated:

            self._learn_noise(
                samples
            )

            return

        # ------------------------------------------
        # Continue collecting event
        # ------------------------------------------

        if self.collecting:

            self._collect_event(
                samples
            )

            return

        # ------------------------------------------
        # Calculate current signal
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
        # Cooldown
        # ------------------------------------------

        cooldown_active = (
            time.time()
            - self.last_detection
            < self.cooldown_sec
        )

        # ------------------------------------------
        # Candidate detection
        # ------------------------------------------

        if not cooldown_active:

            if (
                rms > self.rms_threshold
                and
                peak > self.peak_threshold
            ):

                self._start_event(
                    samples
                )

                return

        # ------------------------------------------
        # Update pre-trigger buffer
        # ------------------------------------------

        self.pre_buffer.extend(
            samples
        )

    # ==============================================
    # Initial noise calibration
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

        required_samples = int(
            self.samplerate
            * self.warmup_sec
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
            print("=" * 60)
            print(
                "LIVE CLASSIFIER READY"
            )
            print("=" * 60)

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

            print()
            print(
                "Listening for candidate events..."
            )

            print()

    # ==============================================
    # Start collecting event
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
    # Collect remainder
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
    # Finish and classify event
    # ==============================================

    def _finish_event(
        self,
    ):

        event = np.array(
            self.event_buffer,
            dtype=np.float32
        )

        # Guarantee exact 90 ms

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

        try:

            self._classify_event(
                event
            )

        except Exception as error:

            print()
            print(
                "Classification error:"
            )

            print(
                error
            )

            print()

        # Reset collection state

        self.collecting = False

        self.event_buffer = []

        self.samples_needed = 0

        self.pre_buffer.clear()

    # ==============================================
    # ML classification
    # ==============================================

    def _classify_event(
        self,
        event,
    ):

        # Extract DSP features

        features = extract(
            event,
            self.samplerate
        )

        # Convert nested feature output into
        # same flat format used during training.

        feature_row = {
            "rms":
                features["rms"],

            "peak":
                features["peak"],

            "energy":
                features["energy"],

            "zero_crossings":
                features["zero_crossings"],

            "crest_factor":
                features["crest_factor"],

            "peak_index":
                features["peak_index"],

            "peak_position":
                features["peak_position"],

            "attack_time":
                features["attack_time"],

            "decay_ratio":
                features["decay_ratio"],

            "centroid":
                features["centroid"],

            "bandwidth":
                features["bandwidth"],

            "rolloff":
                features["rolloff"],

            "spectral_flatness":
                features[
                    "spectral_flatness"
                ],

            "spectral_entropy":
                features[
                    "spectral_entropy"
                ],

            "low_ratio":
                features["low_ratio"],

            "low_mid_ratio":
                features[
                    "low_mid_ratio"
                ],

            "mid_ratio":
                features["mid_ratio"],

            "high_ratio":
                features["high_ratio"],

            "ultra_high_ratio":
                features[
                    "ultra_high_ratio"
                ],
        }

        # ------------------------------------------
        # MFCC means
        # ------------------------------------------

        for i, value in enumerate(
            features["mfcc"]
        ):

            feature_row[
                f"mfcc_mean_{i + 1}"
            ] = value

        # ------------------------------------------
        # MFCC standard deviations
        # ------------------------------------------

        for i, value in enumerate(
            features["mfcc_std"]
        ):

            feature_row[
                f"mfcc_std_{i + 1}"
            ] = value

        # ------------------------------------------
        # Delta MFCC
        # ------------------------------------------

        for i, value in enumerate(
            features["mfcc_delta"]
        ):

            feature_row[
                f"mfcc_delta_{i + 1}"
            ] = value

        # ------------------------------------------
        # Delta-delta MFCC
        # ------------------------------------------

        for i, value in enumerate(
            features["mfcc_delta2"]
        ):

            feature_row[
                f"mfcc_delta2_{i + 1}"
            ] = value

        # ------------------------------------------
        # Guarantee same feature order
        # as training
        # ------------------------------------------

        missing = [
            feature
            for feature
            in self.feature_names
            if feature
            not in feature_row
        ]

        if missing:

            raise ValueError(
                "Missing features: "
                + str(missing)
            )

        X = pd.DataFrame(
            [
                {
                    name:
                    feature_row[name]

                    for name
                    in self.feature_names
                }
            ]
        )

        # ------------------------------------------
        # Prediction
        # ------------------------------------------

        prediction = int(
            self.model.predict(
                X
            )[0]
        )

        # Get probability if supported

        probability = None

        if hasattr(
            self.model,
            "predict_proba"
        ):

            probabilities = (
                self.model.predict_proba(
                    X
                )[0]
            )

            classes = list(
                self.model.classes_
            )

            if 1 in classes:

                tap_index = (
                    classes.index(1)
                )

                probability = float(
                    probabilities[
                        tap_index
                    ]
                )

        # ------------------------------------------
        # Statistics
        # ------------------------------------------

        self.total_events += 1

        if prediction == 1:

            self.tap_events += 1

            print()
            print(
                ">>> VALID TAP <<<"
            )

            if probability is not None:

                print(
                    f"Tap probability: "
                    f"{probability * 100:.1f}%"
                )

        else:

            self.not_tap_events += 1

            print()
            print(
                "NOT TAP - ignored"
            )

            if probability is not None:

                print(
                    f"Tap probability: "
                    f"{probability * 100:.1f}%"
                )

        print(
            f"Events: "
            f"{self.total_events} | "
            f"Taps: "
            f"{self.tap_events} | "
            f"Ignored: "
            f"{self.not_tap_events}"
        )

        print()


# ==================================================
# Main
# ==================================================


def main():

    print()
    print("=" * 60)
    print(
        "EchoDesk Live ML Tap Classifier"
    )
    print("=" * 60)

    print()
    print(
        "Stay quiet for the first "
        "2 seconds."
    )

    print()

    classifier = (
        LiveTapClassifier()
    )

    stream = (
        AudioStream()
    )

    stream.start(
        classifier.process
    )

    try:

        while True:

            time.sleep(
                1
            )

    except KeyboardInterrupt:

        print()
        print(
            "Stopping EchoDesk..."
        )

        stream.stop()

        print()
        print(
            "Final statistics:"
        )

        print(
            f"Candidate events: "
            f"{classifier.total_events}"
        )

        print(
            f"Valid taps: "
            f"{classifier.tap_events}"
        )

        print(
            f"Ignored: "
            f"{classifier.not_tap_events}"
        )


if __name__ == "__main__":
    main()