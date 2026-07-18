from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


DATASET = Path("data/event_features.csv")
MODEL_FOLDER = Path("models")

MODEL_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


def load_dataset():

    df = pd.read_csv(DATASET)

    print("=" * 60)
    print("EchoDesk Event Classifier")
    print("=" * 60)

    print()
    print(f"Total samples: {len(df)}")

    print()
    print("Class distribution:")
    print(df["label"].value_counts())

    # Remove columns that should not
    # be used for training

    X = df.drop(
        columns=[
            "filename",
            "label"
        ]
    )

    y = df["label"]

    return X, y


def evaluate_model(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test
):

    print()
    print("=" * 60)
    print(f"Training: {name}")
    print("=" * 60)

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print()
    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print()
    print(
        "Classification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    print(
        "Confusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    return accuracy


def main():

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y
        )
    )

    print()
    print(
        f"Training samples: "
        f"{len(X_train)}"
    )

    print(
        f"Testing samples: "
        f"{len(X_test)}"
    )

    # ---------------------------------
    # Random Forest
    # ---------------------------------

    random_forest = (
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        )
    )

    rf_accuracy = evaluate_model(
        "Random Forest",
        random_forest,
        X_train,
        X_test,
        y_train,
        y_test
    )

    # ---------------------------------
    # SVM
    # ---------------------------------

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
                probability=True,
                class_weight="balanced"
            )
        )
    ])

    svm_accuracy = evaluate_model(
        "SVM",
        svm,
        X_train,
        X_test,
        y_train,
        y_test
    )

    # ---------------------------------
    # Select best model
    # ---------------------------------

    print()
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        f"Random Forest: "
        f"{rf_accuracy * 100:.2f}%"
    )

    print(
        f"SVM: "
        f"{svm_accuracy * 100:.2f}%"
    )

    if svm_accuracy > rf_accuracy:

        best_model = svm
        best_name = "SVM"

    else:

        best_model = random_forest
        best_name = "Random Forest"

    model_path = (
        MODEL_FOLDER
        / "event_classifier.pkl"
    )

    joblib.dump(
        best_model,
        model_path
    )

    print()
    print(
        f"Best model: "
        f"{best_name}"
    )

    print(
        f"Saved model: "
        f"{model_path}"
    )


if __name__ == "__main__":
    main()