from pathlib import Path

import joblib
import numpy as np
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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DATASET = Path("data/event_features.csv")
MODEL_FOLDER = Path("models")

MODEL_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)


def load_dataset():

    df = pd.read_csv(DATASET)

    # Convert original 4 labels into:
    #
    # tap      -> 1
    # typing   -> 0
    # speech   -> 0
    # noise    -> 0

    df["target"] = (
        df["label"] == "tap"
    ).astype(int)

    X = df.drop(
        columns=[
            "filename",
            "label",
            "target",
        ]
    )

    y = df["target"]

    print()
    print("=" * 60)
    print("EchoDesk Tap vs Not-Tap Benchmark")
    print("=" * 60)

    print()
    print(
        f"Total samples: {len(df)}"
    )

    print(
        f"Taps: {(y == 1).sum()}"
    )

    print(
        f"Not-Taps: {(y == 0).sum()}"
    )

    print(
        f"Features: {X.shape[1]}"
    )

    return X, y


def evaluate_model(
    name,
    model,
    X,
    y,
    cv,
):

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict",
    )

    accuracy = accuracy_score(
        y,
        predictions,
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y,
        predictions,
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

    print(
        matrix
    )

    print()
    print(
        "Classification Report:"
    )

    print(
        classification_report(
            y,
            predictions,
            target_names=[
                "Not Tap",
                "Tap",
            ],
            zero_division=0,
        )
    )

    return {
        "name": name,
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():

    X, y = load_dataset()

    # ----------------------------------------
    # 5-fold stratified cross-validation
    # ----------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # ----------------------------------------
    # Model 1: Random Forest
    # ----------------------------------------

    random_forest = (
        RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
        )
    )

    # ----------------------------------------
    # Model 2: SVM
    # ----------------------------------------

    svm = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=42,
            ),
        ),
    ])

    # ----------------------------------------
    # Model 3: Gradient Boosting
    # ----------------------------------------

    gradient_boosting = (
        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )
    )

    models = [
        (
            "Random Forest",
            random_forest,
        ),
        (
            "SVM",
            svm,
        ),
        (
            "Gradient Boosting",
            gradient_boosting,
        ),
    ]

    results = []

    for name, model in models:

        result = evaluate_model(
            name,
            model,
            X,
            y,
            cv,
        )

        results.append(
            result
        )

    # ----------------------------------------
    # Compare models
    # ----------------------------------------

    print()
    print("=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    for result in results:

        print()

        print(
            result["name"]
        )

        print(
            f"  Accuracy:  "
            f"{result['accuracy'] * 100:.2f}%"
        )

        print(
            f"  Precision: "
            f"{result['precision'] * 100:.2f}%"
        )

        print(
            f"  Recall:    "
            f"{result['recall'] * 100:.2f}%"
        )

        print(
            f"  F1:        "
            f"{result['f1'] * 100:.2f}%"
        )

    # ----------------------------------------
    # Select best model by F1 score
    # ----------------------------------------

    best = max(
        results,
        key=lambda x: x["f1"],
    )

    best_model = best["model"]

    print()
    print("=" * 60)

    print(
        f"Best model: "
        f"{best['name']}"
    )

    print(
        f"Best F1: "
        f"{best['f1'] * 100:.2f}%"
    )

    # ----------------------------------------
    # Train best model on ALL available data
    # ----------------------------------------

    print()
    print(
        "Training best model "
        "on full dataset..."
    )

    best_model.fit(
        X,
        y,
    )

    # ----------------------------------------
    # Save model
    # ----------------------------------------

    model_path = (
        MODEL_FOLDER
        / "tap_classifier.pkl"
    )

    joblib.dump(
        best_model,
        model_path,
    )

    # Save feature names too.
    # This is important when we integrate
    # the model into the live detector.

    feature_path = (
        MODEL_FOLDER
        / "tap_classifier_features.pkl"
    )

    joblib.dump(
        list(X.columns),
        feature_path,
    )

    print()
    print(
        f"Model saved to: "
        f"{model_path}"
    )

    print(
        f"Feature list saved to: "
        f"{feature_path}"
    )

    print()
    print("=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()