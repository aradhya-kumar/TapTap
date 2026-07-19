from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

DATASET_PATH = Path(
    "data/stereo_4zone_features.csv"
)

MODEL_FOLDER = Path(
    "models"
)

MODEL_PATH = (
    MODEL_FOLDER
    / "stereo_4zone_classifier.pkl"
)

FEATURES_PATH = (
    MODEL_FOLDER
    / "stereo_4zone_features.pkl"
)

RANDOM_STATE = 42


# ==========================================================
# MAIN
# ==========================================================

def main():

    print()
    print("=" * 60)
    print(
        "ECHODESK 4-ZONE CLASSIFIER TRAINING"
    )
    print("=" * 60)

    # ======================================================
    # LOAD DATASET
    # ======================================================

    if not DATASET_PATH.exists():

        print()
        print(
            f"ERROR: Dataset not found:"
        )

        print(
            DATASET_PATH
        )

        return

    dataframe = pd.read_csv(
        DATASET_PATH
    )

    print()
    print(
        f"Dataset loaded: "
        f"{DATASET_PATH}"
    )

    print(
        f"Total samples: "
        f"{len(dataframe)}"
    )

    # ======================================================
    # CHECK LABEL
    # ======================================================

    if "label" not in dataframe.columns:

        print()
        print(
            "ERROR: 'label' column "
            "not found."
        )

        return

    # ======================================================
    # SHOW CLASS DISTRIBUTION
    # ======================================================

    print()
    print(
        "Class distribution:"
    )

    print()

    print(
        dataframe[
            "label"
        ].value_counts()
    )

    # ======================================================
    # PREPARE FEATURES
    # ======================================================

    columns_to_remove = [
        "label",
        "filename",
    ]

    feature_names = [

        column

        for column
        in dataframe.columns

        if column
        not in columns_to_remove

    ]

    X = dataframe[
        feature_names
    ]

    y = dataframe[
        "label"
    ]

    print()
    print(
        f"Number of features: "
        f"{len(feature_names)}"
    )

    print()

    print(
        "Features:"
    )

    for feature in feature_names:

        print(
            f" - {feature}"
        )

    # ======================================================
    # CHECK FOR MISSING VALUES
    # ======================================================

    if X.isnull().any().any():

        print()
        print(
            "WARNING: Missing values found."
        )

        print(
            "Replacing missing values with 0."
        )

        X = X.fillna(
            0
        )

    # ======================================================
    # TRAIN / TEST SPLIT
    # ======================================================

    X_train, X_test, y_train, y_test = (
        train_test_split(

            X,

            y,

            test_size=0.20,

            random_state=
                RANDOM_STATE,

            stratify=
                y

        )
    )

    print()
    print(
        f"Training samples: "
        f"{len(X_train)}"
    )

    print(
        f"Test samples: "
        f"{len(X_test)}"
    )

    # ======================================================
    # CREATE RANDOM FOREST
    # ======================================================

    model = RandomForestClassifier(

        n_estimators=500,

        max_depth=None,

        min_samples_split=2,

        min_samples_leaf=1,

        max_features="sqrt",

        class_weight="balanced",

        random_state=
            RANDOM_STATE,

        n_jobs=-1

    )

    # ======================================================
    # CROSS VALIDATION
    # ======================================================

    print()
    print("=" * 60)
    print(
        "5-FOLD CROSS VALIDATION"
    )
    print("=" * 60)

    cross_validation = (
        StratifiedKFold(

            n_splits=5,

            shuffle=True,

            random_state=
                RANDOM_STATE

        )
    )

    cv_scores = (
        cross_val_score(

            model,

            X,

            y,

            cv=
                cross_validation,

            scoring=
                "accuracy",

            n_jobs=-1

        )
    )

    print()

    for index, score in enumerate(
        cv_scores,
        start=1
    ):

        print(
            f"Fold {index}: "
            f"{score * 100:.2f}%"
        )

    print()

    print(
        f"Mean CV accuracy: "
        f"{cv_scores.mean() * 100:.2f}%"
    )

    print(
        f"CV standard deviation: "
        f"{cv_scores.std() * 100:.2f}%"
    )

    # ======================================================
    # TRAIN MODEL
    # ======================================================

    print()
    print("=" * 60)
    print(
        "TRAINING MODEL"
    )
    print("=" * 60)

    model.fit(
        X_train,
        y_train
    )

    # ======================================================
    # TEST MODEL
    # ======================================================

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print()
    print(
        f"Holdout accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    # ======================================================
    # CLASSIFICATION REPORT
    # ======================================================

    print()
    print("=" * 60)
    print(
        "CLASSIFICATION REPORT"
    )
    print("=" * 60)

    print()

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    # ======================================================
    # CONFUSION MATRIX
    # ======================================================

    labels = sorted(
        y.unique()
    )

    matrix = confusion_matrix(

        y_test,

        predictions,

        labels=
            labels

    )

    print()
    print("=" * 60)
    print(
        "CONFUSION MATRIX"
    )
    print("=" * 60)

    print()

    matrix_dataframe = pd.DataFrame(

        matrix,

        index=[
            f"Actual {label}"
            for label
            in labels
        ],

        columns=[
            f"Predicted {label}"
            for label
            in labels
        ]

    )

    print(
        matrix_dataframe
    )

    # ======================================================
    # FEATURE IMPORTANCE
    # ======================================================

    importance_dataframe = pd.DataFrame({

        "feature":
            feature_names,

        "importance":
            model.feature_importances_

    })

    importance_dataframe = (
        importance_dataframe
        .sort_values(

            by=
                "importance",

            ascending=
                False

        )
    )

    print()
    print("=" * 60)
    print(
        "TOP FEATURE IMPORTANCES"
    )
    print("=" * 60)

    print()

    print(
        importance_dataframe.head(
            15
        ).to_string(
            index=False
        )
    )

    # ======================================================
    # RETRAIN ON FULL DATASET
    # ======================================================

    print()
    print(
        "Training final model "
        "on full dataset..."
    )

    model.fit(
        X,
        y
    )

    # ======================================================
    # SAVE MODEL
    # ======================================================

    MODEL_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    joblib.dump(
        feature_names,
        FEATURES_PATH
    )

    # ======================================================
    # COMPLETE
    # ======================================================

    print()
    print("=" * 60)
    print(
        "TRAINING COMPLETE"
    )
    print("=" * 60)

    print()

    print(
        "Model saved to:"
    )

    print(
        MODEL_PATH
    )

    print()

    print(
        "Feature list saved to:"
    )

    print(
        FEATURES_PATH
    )

    print()


# ==========================================================
# START
# ==========================================================

if __name__ == "__main__":

    main()