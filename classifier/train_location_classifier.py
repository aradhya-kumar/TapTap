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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DATASET = Path("data/location_features.csv")
MODEL_FOLDER = Path("models")

MODEL_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


def load_dataset():

    df = pd.read_csv(DATASET)

    print()
    print("=" * 60)
    print("EchoDesk LEFT vs RIGHT Location Classifier")
    print("=" * 60)

    print()
    print(f"Total samples: {len(df)}")

    print()
    print("Location distribution:")
    print(df["label"].value_counts())

    X = df.drop(
        columns=[
            "filename",
            "label",
        ]
    )

    y = df["label"]

    print()
    print(f"Features: {X.shape[1]}")

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

    # Macro averages treat LEFT and RIGHT equally

    precision = precision_score(
        y,
        predictions,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0,
    )

    matrix = confusion_matrix(
        y,
        predictions,
        labels=[
            "left",
            "right",
        ],
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
    print("Confusion Matrix:")
    print()
    print("          Predicted")
    print("          LEFT RIGHT")

    print(
        f"LEFT      "
        f"{matrix[0][0]:4d} "
        f"{matrix[0][1]:5d}"
    )

    print(
        f"RIGHT     "
        f"{matrix[1][0]:4d} "
        f"{matrix[1][1]:5d}"
    )

    print()
    print("Classification Report:")

    print(
        classification_report(
            y,
            predictions,
            labels=[
                "left",
                "right",
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

    # ==========================================
    # Cross-validation
    # ==========================================

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # ==========================================
    # Random Forest
    # ==========================================

    random_forest = (
        RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=42,
        )
    )

    # ==========================================
    # SVM
    # ==========================================

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

    # ==========================================
    # Gradient Boosting
    # ==========================================

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

    # ==========================================
    # Benchmark all models
    # ==========================================

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

    # ==========================================
    # Final comparison
    # ==========================================

    print()
    print("=" * 60)
    print("FINAL LOCATION MODEL COMPARISON")
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

    # ==========================================
    # Choose best model
    # ==========================================

    best = max(
        results,
        key=lambda result:
        result["f1"],
    )

    best_model = best["model"]

    print()
    print("=" * 60)

    print(
        f"Best location model: "
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

    # ==========================================
    # Train best model on full dataset
    # ==========================================

    print()
    print(
        "Training best model "
        "on full location dataset..."
    )

    best_model.fit(
        X,
        y,
    )

    # ==========================================
    # Save model
    # ==========================================

    model_path = (
        MODEL_FOLDER
        / "location_classifier.pkl"
    )

    feature_path = (
        MODEL_FOLDER
        / "location_classifier_features.pkl"
    )

    joblib.dump(
        best_model,
        model_path,
    )

    joblib.dump(
        list(X.columns),
        feature_path,
    )

    print()
    print(
        f"Model saved: "
        f"{model_path}"
    )

    print(
        f"Features saved: "
        f"{feature_path}"
    )

    print()
    print("=" * 60)
    print(
        "LOCATION TRAINING COMPLETE"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()