import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sounddevice as sd

from dsp.features import extract
from dsp.stereo_features import extract_stereo_features
from diagnostics.live_logger import LiveLogger


# ==========================================================
# CONFIGURATION
# ==========================================================

DEVICE = 1
SAMPLERATE = 48000
CHANNELS = 2
BLOCKSIZE = 256

PRE_MS = 20
POST_MS = 100

CALIBRATION_SECONDS = 2.0

RMS_MULTIPLIER = 4.0
PEAK_MULTIPLIER = 5.0

MIN_RMS_THRESHOLD = 0.0005
MIN_PEAK_THRESHOLD = 0.002

COOLDOWN_SECONDS = 0.4

# Minimum confidence required for the
# existing tap classifier to accept an event.
TAP_THRESHOLD = 0.50


# ==========================================================
# MODEL PATHS
# ==========================================================

TAP_MODEL_PATH = Path(
    "models/tap_classifier.pkl"
)

TAP_FEATURES_PATH = Path(
    "models/tap_classifier_features.pkl"
)

LOCATION_MODEL_PATH = Path(
    "models/stereo_location_classifier.pkl"
)

LOCATION_FEATURES_PATH = Path(
    "models/stereo_location_features.pkl"
)


# ==========================================================
# ECHODESK
# ==========================================================

class EchoDesk:

    def __init__(self):

        print()
        print("=" * 60)
        print("Loading EchoDesk Models")
        print("=" * 60)

        # ==================================================
        # LOAD TAP MODEL
        # ==================================================

        self.tap_model = joblib.load(
            TAP_MODEL_PATH
        )

        self.tap_feature_names = joblib.load(
            TAP_FEATURES_PATH
        )

        print(
            "Tap classifier loaded."
        )

        # ==================================================
        # LOAD STEREO LOCATION MODEL
        # ==================================================

        self.location_model = joblib.load(
            LOCATION_MODEL_PATH
        )

        self.location_feature_names = joblib.load(
            LOCATION_FEATURES_PATH
        )

        print(
            "Stereo location classifier loaded."
        )

        # ==================================================
        # LIVE LOGGER
        # ==================================================

        self.logger = LiveLogger()

        print(
            "Live prediction logger ready."
        )

        # ==================================================
        # EVENT WINDOW
        # ==================================================

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

        # ==================================================
        # PRE-TRIGGER BUFFER
        # ==================================================

        self.pre_buffer = deque(
            maxlen=self.pre_samples
        )

        # ==================================================
        # CALIBRATION
        # ==================================================

        self.calibrated = False

        self.calibration_samples = 0

        self.noise_rms_values = []

        self.noise_peak_values = []

        self.rms_threshold = None

        self.peak_threshold = None

        # ==================================================
        # EVENT COLLECTION
        # ==================================================

        self.collecting = False

        self.event_buffer = []

        self.samples_needed = 0

        self.last_detection = 0

        # ==================================================
        # STATISTICS
        # ==================================================

        self.candidates = 0

        self.rejected = 0

        self.valid_taps = 0

        self.left_taps = 0

        self.right_taps = 0

    # ======================================================
    # AUDIO CALLBACK
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

        # --------------------------------------------------
        # Verify stereo input
        # --------------------------------------------------

        if (
            samples.ndim != 2
            or
            samples.shape[1] < 2
        ):

            print(
                "ERROR: Expected "
                "2-channel microphone input."
            )

            return

        # Keep exactly two channels

        samples = samples[
            :,
            :2
        ]

        # ==================================================
        # CALIBRATION
        # ==================================================

        if not self.calibrated:

            self.learn_noise(
                samples
            )

            return

        # ==================================================
        # CONTINUE CURRENT EVENT
        # ==================================================

        if self.collecting:

            self.collect_event(
                samples
            )

            return

        # ==================================================
        # CANDIDATE DETECTION
        # ==================================================

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

        # ==================================================
        # UPDATE PRE-TRIGGER BUFFER
        # ==================================================

        for sample in samples:

            self.pre_buffer.append(
                sample.copy()
            )

    # ======================================================
    # NOISE CALIBRATION
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

        rms = max(
            rms_1,
            rms_2
        )

        peak = max(
            peak_1,
            peak_2
        )

        self.noise_rms_values.append(
            rms
        )

        self.noise_peak_values.append(
            peak
        )

        self.calibration_samples += len(
            samples
        )

        # Store stereo pre-trigger history

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
                "ECHODESK READY"
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
            print(
                "Listening for taps..."
            )

            print()

    # ======================================================
    # START EVENT
    # ======================================================

    def start_event(
        self,
        samples
    ):

        self.last_detection = (
            time.time()
        )

        self.candidates += 1

        # --------------------------------------------------
        # Retrieve stereo pre-trigger audio
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Combine history with trigger block
        # --------------------------------------------------

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
    # CONTINUE EVENT
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
    # BUILD TAP CLASSIFIER FEATURES
    # ======================================================

    def build_tap_features(
        self,
        mono
    ):

        features = extract(
            mono,
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

        # --------------------------------------------------
        # MFCC means
        # --------------------------------------------------

        for i, value in enumerate(
            features["mfcc"]
        ):

            row[
                f"mfcc_mean_{i + 1}"
            ] = value

        # --------------------------------------------------
        # MFCC standard deviations
        # --------------------------------------------------

        for i, value in enumerate(
            features["mfcc_std"]
        ):

            row[
                f"mfcc_std_{i + 1}"
            ] = value

        # --------------------------------------------------
        # MFCC delta
        # --------------------------------------------------

        for i, value in enumerate(
            features["mfcc_delta"]
        ):

            row[
                f"mfcc_delta_{i + 1}"
            ] = value

        # --------------------------------------------------
        # MFCC delta-delta
        # --------------------------------------------------

        for i, value in enumerate(
            features["mfcc_delta2"]
        ):

            row[
                f"mfcc_delta2_{i + 1}"
            ] = value

        return row

    # ======================================================
    # FINISH AND ANALYZE EVENT
    # ======================================================

    def finish_event(
        self
    ):

        try:

            # ==============================================
            # BUILD STEREO EVENT
            # ==============================================

            event = np.concatenate(
                self.event_buffer,
                axis=0
            )

            event = event[
                :self.event_samples,
                :2
            ]

            # Pad if required

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

            # ==============================================
            # MONO COPY FOR TAP CLASSIFIER
            # ==============================================

            mono = np.mean(
                event,
                axis=1
            ).astype(
                np.float32
            )

            # ==============================================
            # EXTRACT TAP FEATURES
            # ==============================================

            tap_features = (
                self.build_tap_features(
                    mono
                )
            )

            tap_input = pd.DataFrame([
                {
                    name:
                        tap_features[
                            name
                        ]

                    for name
                    in self.tap_feature_names
                }
            ])

            # ==============================================
            # TAP PROBABILITY
            # ==============================================

            tap_probability = None

            if hasattr(
                self.tap_model,
                "predict_proba"
            ):

                probabilities = (
                    self.tap_model
                    .predict_proba(
                        tap_input
                    )[0]
                )

                classes = list(
                    self.tap_model.classes_
                )

                if 1 in classes:

                    tap_index = (
                        classes.index(
                            1
                        )
                    )

                    tap_probability = float(
                        probabilities[
                            tap_index
                        ]
                    )

            # ==============================================
            # TAP DECISION
            # ==============================================

            if (
                tap_probability
                is not None
            ):

                is_tap = (
                    tap_probability
                    >= TAP_THRESHOLD
                )

            else:

                prediction = (
                    self.tap_model.predict(
                        tap_input
                    )[0]
                )

                is_tap = (
                    int(
                        prediction
                    )
                    == 1
                )

            # ==============================================
            # REJECT NON-TAP
            # ==============================================

            if not is_tap:

                self.rejected += 1

                if (
                    tap_probability
                    is not None
                ):

                    print(
                        f"NOT TAP → Ignored "
                        f"({tap_probability * 100:.1f}%)"
                    )

                else:

                    print(
                        "NOT TAP → Ignored"
                    )

                return

            # ==============================================
            # VALID TAP
            # ==============================================

            self.valid_taps += 1

            # ==============================================
            # EXTRACT STEREO SPATIAL FEATURES
            # ==============================================

            spatial_features = (
                extract_stereo_features(
                    event,
                    SAMPLERATE
                )
            )

            location_input = pd.DataFrame([
                {
                    name:
                        spatial_features[
                            name
                        ]

                    for name
                    in self.location_feature_names
                }
            ])

            # ==============================================
            # LOCATION PREDICTION
            # ==============================================

            location = (
                self.location_model.predict(
                    location_input
                )[0]
            )

            # ==============================================
            # LOCATION CONFIDENCE
            # ==============================================

            location_probability = None

            if hasattr(
                self.location_model,
                "predict_proba"
            ):

                probabilities = (
                    self.location_model
                    .predict_proba(
                        location_input
                    )[0]
                )

                classes = list(
                    self.location_model.classes_
                )

                location_index = (
                    classes.index(
                        location
                    )
                )

                location_probability = float(
                    probabilities[
                        location_index
                    ]
                )

            # ==============================================
            # SAVE TO CSV LOG
            # ==============================================

            self.logger.log(

                tap_probability=
                    tap_probability,

                prediction=
                    location,

                location_probability=
                    location_probability,

                spatial_features=
                    spatial_features

            )

            # ==============================================
            # DISPLAY RESULT
            # ==============================================

            print()

            if (
                location
                == "left"
            ):

                self.left_taps += 1

                print(
                    "<<< LEFT TAP <<<"
                )

            elif (
                location
                == "right"
            ):

                self.right_taps += 1

                print(
                    ">>> RIGHT TAP >>>"
                )

            else:

                print(
                    f"TAP: "
                    f"{location}"
                )

            # ==============================================
            # CONFIDENCE
            # ==============================================

            if (
                tap_probability
                is not None
            ):

                print(
                    f"Tap confidence: "
                    f"{tap_probability * 100:.1f}%"
                )

            if (
                location_probability
                is not None
            ):

                print(
                    f"Location confidence: "
                    f"{location_probability * 100:.1f}%"
                )

            # ==============================================
            # SPATIAL DIAGNOSTICS
            # ==============================================

            print(
                f"RMS ratio: "
                f"{spatial_features['log_rms_ratio']:.3f}"
            )

            print(
                f"Energy ratio: "
                f"{spatial_features['log_energy_ratio']:.3f}"
            )

            print(
                f"Peak difference: "
                f"{spatial_features['peak_difference']:.3f}"
            )

            print(
                f"Peak sample difference: "
                f"{spatial_features['peak_sample_difference']}"
            )

            print(
                f"Correlation lag: "
                f"{spatial_features['correlation_lag']}"
            )

            print(
                f"Correlation value: "
                f"{spatial_features['correlation_value']:.3f}"
            )

            print(
                f"Channel correlation: "
                f"{spatial_features['channel_correlation']:.3f}"
            )

            print()

            print(
                f"Valid: "
                f"{self.valid_taps}"
                f" | "
                f"Left: "
                f"{self.left_taps}"
                f" | "
                f"Right: "
                f"{self.right_taps}"
            )

            print()

        except Exception as error:

            print()
            print(
                "ERROR PROCESSING EVENT:"
            )

            print(
                error
            )

        finally:

            self.reset_event()

    # ======================================================
    # RESET EVENT
    # ======================================================

    def reset_event(
        self
    ):

        self.collecting = False

        self.event_buffer = []

        self.samples_needed = 0

        self.pre_buffer.clear()


# ==========================================================
# MAIN
# ==========================================================

def main():

    print()
    print("=" * 60)

    print(
        "EchoDesk Stereo "
        "Acoustic Surface System"
    )

    print("=" * 60)

    # ======================================================
    # MICROPHONE INFORMATION
    # ======================================================

    device_info = sd.query_devices(
        DEVICE,
        "input"
    )

    print()

    print(
        f"Microphone: "
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

    # ======================================================
    # CREATE SYSTEM
    # ======================================================

    echodesk = EchoDesk()

    print()
    print(
        "Stay quiet for "
        "2 seconds..."
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
            "Stopping EchoDesk..."
        )

    finally:

        stream.stop()

        stream.close()

        # ==================================================
        # SESSION SUMMARY
        # ==================================================

        print()
        print("=" * 60)

        print(
            "SESSION SUMMARY"
        )

        print("=" * 60)

        print(
            f"Candidate events: "
            f"{echodesk.candidates}"
        )

        print(
            f"Rejected events: "
            f"{echodesk.rejected}"
        )

        print(
            f"Valid taps: "
            f"{echodesk.valid_taps}"
        )

        print(
            f"LEFT predictions: "
            f"{echodesk.left_taps}"
        )

        print(
            f"RIGHT predictions: "
            f"{echodesk.right_taps}"
        )

        print()

        print(
            "Prediction log saved to:"
        )

        print(
            "logs/live_predictions.csv"
        )

        print("=" * 60)


# ==========================================================
# START
# ==========================================================

if __name__ == "__main__":

    main()