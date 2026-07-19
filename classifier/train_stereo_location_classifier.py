print("STEREO TRAINING SCRIPT STARTED")

from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
)

from sklearn.svm import SVC


# ==========================================================
# PATHS
# ==========================================================

DATASET = Path(
    "data/stereo_location_features.csv"
)

MODEL_FOLDER = Path(
    "models"
)

MODEL_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset():

    if not DATASET.exists():

        raise FileNotFoundError(
            f"Dataset not found: "
            f"{DATASET}"
        )

    df = pd.read_csv(
        DATASET
    )

    print()
    print("=" * 60)
    print(
        "EchoDesk Stereo Location Classifier"
    )
    print("=" * 60)

    print()
    print(
        f"Total samples: "
        f"{len(df)}"
    )

    print()
    print(
        "Class distribution:"
    )

    print(
        df["label"]
        .value_counts()
    )

    # Remove metadata columns

    X = df.drop(
        columns=[
            "filename",
            "label",
        ]
    )

    y = df[
        "label"
    ]

    print()
    print(
        f"Number of features: "
        f"{X.shape[1]}"
    )

    print()
    print(
        "Features:"
    )

    for feature in X.columns:

        print(
            f"  {feature}"
        )

    return (
        X,
        y
    )


# ==========================================================
# EVALUATE MODEL
# ==========================================================

def evaluate_model(
    name,
    model,
    X,
    y,
    cv
):

    print()
    print("=" * 60)
    print(
        f"Training: {name}"
    )
    print("=" * 60)

    predictions = (
        cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            method="predict"
        )
    )

    accuracy = (
        accuracy_score(
            y,
            predictions
        )
    )

    precision = (
        precision_score(
            y,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    recall = (
        recall_score(
            y,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    f1 = (
        f1_score(
            y,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    matrix = (
        confusion_matrix(
            y,
            predictions,
            labels=[
                "left",
                "right",
            ]
        )
    )

    print()
    print(
        f"Accuracy:  "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall:    "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1 Score:  "
        f"{f1 * 100:.2f}%"
    )

    print()
    print(
        "Confusion Matrix:"
    )

    print()
    print(
        "             Predicted"
    )

    print(
        "             LEFT  RIGHT"
    )

    print(
        f"Actual LEFT  "
        f"{matrix[0][0]:4d}  "
        f"{matrix[0][1]:5d}"
    )

    print(
        f"Actual RIGHT "
        f"{matrix[1][0]:4d}  "
        f"{matrix[1][1]:5d}"
    )

    print()
    print(
        "Classification Report:"
    )

    print(
        classification_report(
            y,
            predictions,
            labels=[
                "left",
                "right",
            ],
            zero_division=0
        )
    )

    return {

        "name":
            name,

        "model":
            model,

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,
    }


# ==========================================================
# MAIN
# ==========================================================

def main():

    X, y = (
        load_dataset()
    )

    # ======================================================
    # CROSS VALIDATION
    # ======================================================

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # ======================================================
    # RANDOM FOREST
    # ======================================================

    random_forest = (
        RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            class_weight="balanced",
            random_state=42
        )
    )

    # ======================================================
    # SVM
    # ======================================================

    svm = Pipeline([

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=42
            )
        ),

    ])

    # ======================================================
    # GRADIENT BOOSTING
    # ======================================================

    gradient_boosting = (
        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
    )

    # ======================================================
    # MODEL LIST
    # ======================================================

    models = [

        (
            "Random Forest",
            random_forest
        ),

        (
            "SVM",
            svm
        ),

        (
            "Gradient Boosting",
            gradient_boosting
        ),

    ]

    results = []

    # ======================================================
    # BENCHMARK
    # ======================================================

    for name, model in models:

        result = (
            evaluate_model(
                name,
                model,
                X,
                y,
                cv
            )
        )

        results.append(
            result
        )

    # ======================================================
    # FINAL COMPARISON
    # ======================================================

    print()
    print("=" * 60)
    print(
        "FINAL STEREO LOCATION MODEL COMPARISON"
    )
    print("=" * 60)

    for result in results:

        print()

        print(
            result["name"]
        )

        print(
            f"Accuracy:  "
            f"{result['accuracy'] * 100:.2f}%"
        )

        print(
            f"Precision: "
            f"{result['precision'] * 100:.2f}%"
        )

        print(
            f"Recall:    "
            f"{result['recall'] * 100:.2f}%"
        )

        print(
            f"F1:        "
            f"{result['f1'] * 100:.2f}%"
        )

    # ======================================================
    # SELECT BEST MODEL
    # ======================================================

    best = max(
        results,
        key=lambda result:
        result["f1"]
    )

    best_model = (
        best["model"]
    )

    print()
    print("=" * 60)

    print(
        f"Best stereo model: "
        f"{best['name']}"
    )

    print(
        f"Best accuracy: "
        f"{best['accuracy'] * 100:.2f}%"
    )

    print(
        f"Best F1: "
        f"{best['f1'] * 100:.2f}%"
    )

    # ======================================================
    # TRAIN WINNER ON FULL DATASET
    # ======================================================

    print()
    print(
        "Training best stereo model "
        "on full dataset..."
    )

    best_model.fit(
        X,
        y
    )

    # ======================================================
    # SAVE MODEL
    # ======================================================

    model_path = (
        MODEL_FOLDER
        / "stereo_location_classifier.pkl"
    )

    feature_path = (
        MODEL_FOLDER
        / "stereo_location_features.pkl"
    )

    joblib.dump(
        best_model,
        model_path
    )

    joblib.dump(
        list(
            X.columns
        ),
        feature_path
    )

    print()
    print(
        f"Model saved: "
        f"{model_path}"
    )

    print(
        f"Feature list saved: "
        f"{feature_path}"
    )

    print()
    print("=" * 60)
    print(
        "STEREO LOCATION TRAINING COMPLETE"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()