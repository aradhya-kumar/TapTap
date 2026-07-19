import time
from pathlib import Path
from collections import deque

import joblib
import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf

from dsp.features import extract


# ==========================================================
# CONFIGURATION
# ==========================================================

DEVICE = 1
SAMPLERATE = 48000
CHANNELS = 2

PRE_MS = 20
POST_MS = 100

CALIBRATION_SECONDS = 2.0

RMS_MULTIPLIER = 4.0
PEAK_MULTIPLIER = 5.0

MIN_RMS_THRESHOLD = 0.0005
MIN_PEAK_THRESHOLD = 0.002

COOLDOWN_SECONDS = 0.4

BLOCKSIZE = 256

# Probability required before saving.
# Start low because the current tap classifier
# has imperfect recall.
TAP_PROBABILITY_THRESHOLD = 0.50

# Warn when signal is close to digital clipping.
CLIPPING_THRESHOLD = 0.98


# ==========================================================
# PATHS
# ==========================================================

BASE_FOLDER = Path(
    "data/filtered_stereo_location"
)

TAP_MODEL_PATH = Path(
    "models/tap_classifier.pkl"
)

TAP_FEATURES_PATH = Path(
    "models/tap_classifier_features.pkl"
)

ZONES = [
    "left",
    "right",
]


# ==========================================================
# COLLECTOR
# ==========================================================

class FilteredStereoCollector:

    def __init__(self):

        # --------------------------------------------------
        # Load tap classifier
        # --------------------------------------------------

        print()
        print("Loading tap classifier...")

        if not TAP_MODEL_PATH.exists():

            raise FileNotFoundError(
                f"Model not found: "
                f"{TAP_MODEL_PATH}"
            )

        if not TAP_FEATURES_PATH.exists():

            raise FileNotFoundError(
                f"Feature list not found: "
                f"{TAP_FEATURES_PATH}"
            )

        self.tap_model = joblib.load(
            TAP_MODEL_PATH
        )

        self.tap_feature_names = joblib.load(
            TAP_FEATURES_PATH
        )

        print(
            "Tap classifier loaded."
        )

        print(
            f"Expected features: "
            f"{len(self.tap_feature_names)}"
        )

        # --------------------------------------------------
        # Create folders
        # --------------------------------------------------

        for zone in ZONES:

            (
                BASE_FOLDER
                / zone
            ).mkdir(
                parents=True,
                exist_ok=True
            )

        # --------------------------------------------------
        # Event sizes
        # --------------------------------------------------

        self.pre_samples = int(
            SAMPLERATE
            * PRE_MS
            / 1000
        )

        self.post_samples = int(
            SAMPLERATE
            * POST_MS
            / 1000
        )

        self.event_samples = (
            self.pre_samples
            + self.post_samples
        )

        # --------------------------------------------------
        # Stereo pre-trigger buffer
        # --------------------------------------------------

        self.pre_buffer = deque(
            maxlen=self.pre_samples
        )

        # --------------------------------------------------
        # Calibration
        # --------------------------------------------------

        self.calibrated = False

        self.calibration_samples = 0

        self.noise_rms_values = []

        self.noise_peak_values = []

        self.rms_threshold = None

        self.peak_threshold = None

        # --------------------------------------------------
        # Event state
        # --------------------------------------------------

        self.collecting = False

        self.event_buffer = []

        self.samples_needed = 0

        self.last_detection = 0

        # --------------------------------------------------
        # Active label
        # --------------------------------------------------

        self.current_zone = None

        # --------------------------------------------------
        # Counters
        # --------------------------------------------------

        self.counters = {}

        for zone in ZONES:

            self.counters[zone] = (
                self.get_next_number(
                    zone
                )
            )

        # --------------------------------------------------
        # Session statistics
        # --------------------------------------------------

        self.candidates = 0

        self.accepted_taps = 0

        self.rejected_events = 0

        self.clipped_events = 0

    # ======================================================
    # Find next file number
    # ======================================================

    def get_next_number(
        self,
        zone
    ):

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

    # ======================================================
    # Select zone
    # ======================================================

    def set_zone(
        self,
        zone
    ):

        self.current_zone = zone

        print()
        print("=" * 60)

        print(
            f"COLLECTING FILTERED "
            f"{zone.upper()} TAPS"
        )

        print("=" * 60)

        print()
        print(
            "Tap naturally inside this zone."
        )

        print(
            "Wait about half a second "
            "between taps."
        )

        print()
        print(
            "Typing or other impulses may trigger "
            "the detector, but the Tap SVM will "
            "try to reject them."
        )

        print()

    # ======================================================
    # Audio callback
    # ======================================================

    def audio_callback(
        self,
        indata,
        frames,
        time_info,
        status
    ):

        if status:

            print(
                f"Audio status: "
                f"{status}"
            )

        samples = (
            indata.copy()
            .astype(
                np.float32
            )
        )

        if (
            samples.ndim != 2
            or
            samples.shape[1] < 2
        ):

            return

        samples = samples[
            :,
            :2
        ]

        # --------------------------------------------------
        # Calibration
        # --------------------------------------------------

        if not self.calibrated:

            self.learn_noise(
                samples
            )

            return

        # --------------------------------------------------
        # Continue active event
        # --------------------------------------------------

        if self.collecting:

            self.collect_event(
                samples
            )

            return

        # --------------------------------------------------
        # Detection values
        # --------------------------------------------------

        ch1 = samples[
            :,
            0
        ]

        ch2 = samples[
            :,
            1
        ]

        rms_1 = float(
            np.sqrt(
                np.mean(
                    ch1 ** 2
                )
            )
        )

        rms_2 = float(
            np.sqrt(
                np.mean(
                    ch2 ** 2
                )
            )
        )

        peak_1 = float(
            np.max(
                np.abs(
                    ch1
                )
            )
        )

        peak_2 = float(
            np.max(
                np.abs(
                    ch2
                )
            )
        )

        current_rms = max(
            rms_1,
            rms_2
        )

        current_peak = max(
            peak_1,
            peak_2
        )

        cooldown_finished = (
            time.time()
            - self.last_detection
            >= COOLDOWN_SECONDS
        )

        # --------------------------------------------------
        # Candidate impulse
        # --------------------------------------------------

        if (
            self.current_zone
            is not None
            and
            cooldown_finished
            and
            current_rms
            > self.rms_threshold
            and
            current_peak
            > self.peak_threshold
        ):

            self.start_event(
                samples
            )

            return

        # --------------------------------------------------
        # Update stereo pre-buffer
        # --------------------------------------------------

        for sample in samples:

            self.pre_buffer.append(
                sample.copy()
            )

    # ======================================================
    # Calibration
    # ======================================================

    def learn_noise(
        self,
        samples
    ):

        ch1 = samples[
            :,
            0
        ]

        ch2 = samples[
            :,
            1
        ]

        rms_1 = float(
            np.sqrt(
                np.mean(
                    ch1 ** 2
                )
            )
        )

        rms_2 = float(
            np.sqrt(
                np.mean(
                    ch2 ** 2
                )
            )
        )

        peak_1 = float(
            np.max(
                np.abs(
                    ch1
                )
            )
        )

        peak_2 = float(
            np.max(
                np.abs(
                    ch2
                )
            )
        )

        self.noise_rms_values.append(
            max(
                rms_1,
                rms_2
            )
        )

        self.noise_peak_values.append(
            max(
                peak_1,
                peak_2
            )
        )

        self.calibration_samples += len(
            samples
        )

        for sample in samples:

            self.pre_buffer.append(
                sample.copy()
            )

        required = int(
            SAMPLERATE
            * CALIBRATION_SECONDS
        )

        if (
            self.calibration_samples
            >= required
        ):

            noise_rms = float(
                np.median(
                    self.noise_rms_values
                )
            )

            noise_peak = float(
                np.median(
                    self.noise_peak_values
                )
            )

            self.rms_threshold = max(
                noise_rms
                * RMS_MULTIPLIER,
                MIN_RMS_THRESHOLD
            )

            self.peak_threshold = max(
                noise_peak
                * PEAK_MULTIPLIER,
                MIN_PEAK_THRESHOLD
            )

            self.calibrated = True

            print()
            print("=" * 60)
            print(
                "CALIBRATION COMPLETE"
            )
            print("=" * 60)

            print(
                f"Noise RMS: "
                f"{noise_rms:.6f}"
            )

            print(
                f"Noise Peak: "
                f"{noise_peak:.6f}"
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

    # ======================================================
    # Start event
    # ======================================================

    def start_event(
        self,
        samples
    ):

        self.last_detection = (
            time.time()
        )

        self.candidates += 1

        if len(
            self.pre_buffer
        ) > 0:

            pre_audio = np.array(
                self.pre_buffer,
                dtype=np.float32
            )

        else:

            pre_audio = np.empty(
                (
                    0,
                    2
                ),
                dtype=np.float32
            )

        event = np.concatenate(
            [
                pre_audio,
                samples
            ],
            axis=0
        )

        self.event_buffer = [
            event
        ]

        self.samples_needed = (
            self.event_samples
            - len(event)
        )

        if (
            self.samples_needed
            <= 0
        ):

            self.finish_event()

        else:

            self.collecting = True

    # ======================================================
    # Continue event
    # ======================================================

    def collect_event(
        self,
        samples
    ):

        take = min(
            len(samples),
            self.samples_needed
        )

        self.event_buffer.append(
            samples[
                :take
            ].copy()
        )

        self.samples_needed -= take

        if (
            self.samples_needed
            <= 0
        ):

            self.finish_event()

    # ======================================================
    # Build feature row for tap model
    # ======================================================

    def build_tap_feature_row(
        self,
        mono_event
    ):

        features = extract(
            mono_event,
            SAMPLERATE
        )

        row = {

            "rms":
                features["rms"],

            "peak":
                features["peak"],

            "energy":
                features["energy"],

            "zero_crossings":
                features[
                    "zero_crossings"
                ],

            "crest_factor":
                features[
                    "crest_factor"
                ],

            "peak_index":
                features[
                    "peak_index"
                ],

            "peak_position":
                features[
                    "peak_position"
                ],

            "attack_time":
                features[
                    "attack_time"
                ],

            "decay_ratio":
                features[
                    "decay_ratio"
                ],

            "centroid":
                features[
                    "centroid"
                ],

            "bandwidth":
                features[
                    "bandwidth"
                ],

            "rolloff":
                features[
                    "rolloff"
                ],

            "spectral_flatness":
                features[
                    "spectral_flatness"
                ],

            "spectral_entropy":
                features[
                    "spectral_entropy"
                ],

            "low_ratio":
                features[
                    "low_ratio"
                ],

            "low_mid_ratio":
                features[
                    "low_mid_ratio"
                ],

            "mid_ratio":
                features[
                    "mid_ratio"
                ],

            "high_ratio":
                features[
                    "high_ratio"
                ],

            "ultra_high_ratio":
                features[
                    "ultra_high_ratio"
                ],
        }

        for i, value in enumerate(
            features["mfcc"]
        ):

            row[
                f"mfcc_mean_{i + 1}"
            ] = value

        for i, value in enumerate(
            features["mfcc_std"]
        ):

            row[
                f"mfcc_std_{i + 1}"
            ] = value

        for i, value in enumerate(
            features["mfcc_delta"]
        ):

            row[
                f"mfcc_delta_{i + 1}"
            ] = value

        for i, value in enumerate(
            features["mfcc_delta2"]
        ):

            row[
                f"mfcc_delta2_{i + 1}"
            ] = value

        return row

    # ======================================================
    # Finish event
    # ======================================================

    def finish_event(
        self
    ):

        event = np.concatenate(
            self.event_buffer,
            axis=0
        )

        event = event[
            :self.event_samples,
            :2
        ]

        if (
            len(event)
            < self.event_samples
        ):

            missing = (
                self.event_samples
                - len(event)
            )

            event = np.pad(
                event,
                (
                    (0, missing),
                    (0, 0)
                )
            )

        # --------------------------------------------------
        # Check clipping
        # --------------------------------------------------

        peak_1 = float(
            np.max(
                np.abs(
                    event[:, 0]
                )
            )
        )

        peak_2 = float(
            np.max(
                np.abs(
                    event[:, 1]
                )
            )
        )

        clipping = (
            peak_1
            >= CLIPPING_THRESHOLD
            or
            peak_2
            >= CLIPPING_THRESHOLD
        )

        if clipping:

            self.clipped_events += 1

        # --------------------------------------------------
        # Create mono COPY for existing tap classifier
        #
        # IMPORTANT:
        # The original stereo event is preserved.
        # --------------------------------------------------

        mono_event = np.mean(
            event,
            axis=1
        ).astype(
            np.float32
        )

        try:

            feature_row = (
                self.build_tap_feature_row(
                    mono_event
                )
            )

            # Build model input in EXACT
            # training feature order.

            model_input = pd.DataFrame([
                {
                    name:
                        feature_row[name]

                    for name
                    in self.tap_feature_names
                }
            ])

            # --------------------------------------------------
            # Tap probability
            # --------------------------------------------------

            tap_probability = None

            if hasattr(
                self.tap_model,
                "predict_proba"
            ):

                probabilities = (
                    self.tap_model
                    .predict_proba(
                        model_input
                    )[0]
                )

                classes = list(
                    self.tap_model.classes_
                )

                if 1 in classes:

                    tap_index = (
                        classes.index(1)
                    )

                    tap_probability = float(
                        probabilities[
                            tap_index
                        ]
                    )

            # --------------------------------------------------
            # Decision
            # --------------------------------------------------

            if (
                tap_probability
                is not None
            ):

                is_tap = (
                    tap_probability
                    >=
                    TAP_PROBABILITY_THRESHOLD
                )

            else:

                prediction = int(
                    self.tap_model.predict(
                        model_input
                    )[0]
                )

                is_tap = (
                    prediction == 1
                )

        except Exception as error:

            print()
            print(
                "Classification error:"
            )

            print(
                error
            )

            self.reset_event()

            return

        # --------------------------------------------------
        # Reject non-tap
        # --------------------------------------------------

        if not is_tap:

            self.rejected_events += 1

            if (
                tap_probability
                is not None
            ):

                print(
                    f"REJECTED → NOT TAP "
                    f"| Confidence: "
                    f"{tap_probability * 100:.1f}%"
                )

            else:

                print(
                    "REJECTED → NOT TAP"
                )

            self.reset_event()

            return

        # --------------------------------------------------
        # Valid tap
        # --------------------------------------------------

        zone = self.current_zone

        if zone is None:

            self.reset_event()

            return

        number = (
            self.counters[
                zone
            ]
        )

        filename = (
            BASE_FOLDER
            / zone
            / (
                f"{zone}_"
                f"{number:04d}.wav"
            )
        )

        # Save ORIGINAL STEREO audio

        sf.write(
            filename,
            event,
            SAMPLERATE
        )

        self.counters[
            zone
        ] += 1

        self.accepted_taps += 1

        print()

        print(
            f"{zone.upper()} TAP SAVED "
            f"#{self.counters[zone]}"
        )

        if (
            tap_probability
            is not None
        ):

            print(
                f"Tap confidence: "
                f"{tap_probability * 100:.1f}%"
            )

        print(
            f"CH1 Peak: "
            f"{peak_1:.3f}"
        )

        print(
            f"CH2 Peak: "
            f"{peak_2:.3f}"
        )

        if clipping:

            print(
                "WARNING: CLIPPING DETECTED"
            )

        print()

        self.reset_event()

    # ======================================================
    # Reset
    # ======================================================

    def reset_event(
        self
    ):

        self.collecting = False

        self.event_buffer = []

        self.samples_needed = 0

        self.pre_buffer.clear()

    # ======================================================
    # Statistics
    # ======================================================

    def show_stats(
        self
    ):

        print()
        print("=" * 60)
        print(
            "FILTERED STEREO DATASET"
        )
        print("=" * 60)

        print(
            f"LEFT saved: "
            f"{self.counters['left']}"
        )

        print(
            f"RIGHT saved: "
            f"{self.counters['right']}"
        )

        print()

        print(
            f"Candidate events: "
            f"{self.candidates}"
        )

        print(
            f"Accepted taps: "
            f"{self.accepted_taps}"
        )

        print(
            f"Rejected events: "
            f"{self.rejected_events}"
        )

        print(
            f"Clipping warnings: "
            f"{self.clipped_events}"
        )

        print("=" * 60)


# ==========================================================
# MAIN
# ==========================================================

def main():

    print()
    print("=" * 60)

    print(
        "EchoDesk Filtered "
        "Stereo Location Collector"
    )

    print("=" * 60)

    device_info = sd.query_devices(
        DEVICE,
        "input"
    )

    print()
    print(
        f"Device: "
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

    collector = (
        FilteredStereoCollector()
    )

    collector.show_stats()

    print()
    print(
        "Stay quiet for "
        "2 seconds..."
    )

    stream = sd.InputStream(
        device=DEVICE,
        samplerate=SAMPLERATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=BLOCKSIZE,
        callback=collector.audio_callback
    )

    stream.start()

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
                "Choose zone:"
            )

            print(
                "1 - Collect LEFT"
            )

            print(
                "2 - Collect RIGHT"
            )

            print(
                "3 - Show statistics"
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
                    "Press ENTER when "
                    "LEFT collection is finished..."
                )

                collector.current_zone = None

            elif choice == "2":

                collector.set_zone(
                    "right"
                )

                input(
                    "Press ENTER when "
                    "RIGHT collection is finished..."
                )

                collector.current_zone = None

            elif choice == "3":

                collector.show_stats()

            elif choice == "q":

                break

    except KeyboardInterrupt:

        pass

    finally:

        collector.current_zone = None

        stream.stop()

        stream.close()

        print()
        print(
            "Collection finished."
        )

        collector.show_stats()


if __name__ == "__main__":
    main()