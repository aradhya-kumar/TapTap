import csv
from datetime import datetime
from pathlib import Path


LOG_FOLDER = Path("logs")
LOG_FILE = LOG_FOLDER / "live_predictions.csv"


class LiveLogger:

    def __init__(self):

        LOG_FOLDER.mkdir(
            parents=True,
            exist_ok=True
        )

        self.fieldnames = [
            "timestamp",
            "tap_probability",
            "prediction",
            "location_probability",
            "log_rms_ratio",
            "log_energy_ratio",
            "peak_difference",
            "peak_sample_difference",
            "correlation_lag",
            "correlation_value",
            "channel_correlation",
        ]

        if not LOG_FILE.exists():

            with open(
                LOG_FILE,
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
        tap_probability,
        prediction,
        location_probability,
        spatial_features
    ):

        row = {

            "timestamp":
                datetime.now().isoformat(),

            "tap_probability":
                tap_probability,

            "prediction":
                prediction,

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
            LOG_FILE,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=self.fieldnames
            )

            writer.writerow(row)