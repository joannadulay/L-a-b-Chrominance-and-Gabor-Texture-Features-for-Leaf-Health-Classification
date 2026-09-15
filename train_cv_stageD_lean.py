"""
train_cv_stageD_lean.py

Article 2 — lean Stage D: baseline + L*a*b* chrominance + Gabor
(multi-scale LBP dropped as the lowest per-feature-importance addition
in Stages B/C). Same structure as train_cv_v2.py / train_cv_stageB.py.
"""

from pathlib import Path
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

import feature_extractor_stageD_lean as feature_extractor
from feature_extractor_stageD_lean import extract_features

print(f"Using feature_extractor module from: {feature_extractor.__file__}")

DATASET_DIR = Path("dataset")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs_stageD_lean")
MODEL_PATH = MODEL_DIR / "gbm_model_cv_stageD_lean.pkl"

CLASS_NAMES = ["healthy", "discolored", "diseased"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

EXPECTED_FEATURE_LENGTH = feature_extractor.EXPECTED_FEATURE_LENGTH

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def load_dataset():
    X, y, paths = [], [], []
    label_to_idx = {label: idx for idx, label in enumerate(CLASS_NAMES)}

    for label in CLASS_NAMES:
        class_dir = DATASET_DIR / label
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing folder: {class_dir}")

        n_loaded = 0
        for file in class_dir.iterdir():
            if file.suffix.lower() in VALID_EXTENSIONS:
                try:
                    feat = extract_features(file)
                    X.append(feat)
                    y.append(label_to_idx[label])
                    paths.append(str(file))
                    n_loaded += 1
                except Exception as e:
                    print(f"[WARNING] Skipped {file}: {e}")

        print(f"  {label}: {n_loaded} images")

    if len(X) == 0:
        raise ValueError("No valid images found in dataset.")

    return np.array(X, dtype=np.float32), np.array(y), paths, label_to_idx


def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def plot_confusion_matrix(cm, class_names, path, normalize=False, title=""):
    if normalize:
        cm_display = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_display, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format=fmt, colorbar=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def report_pairwise_confusion(cm, class_names, label_a, label_b, out_lines):
    idx_a = class_names.index(label_a)
    idx_b = class_names.index(label_b)

    a_total = cm[idx_a].sum()
    b_total = cm[idx_b].sum()
    a_as_b = cm[idx_a, idx_b]
    b_as_a = cm[idx_b, idx_a]

    out_lines.append(f"\n--- {label_a} vs {label_b} confusion ---")
    out_lines.append(
        f"{label_a} predicted as {label_b}: {a_as_b}/{a_total} "
        f"({(a_as_b / a_total * 100 if a_total else 0):.1f}%)"
    )
    out_lines.append(
        f"{label_b} predicted as {label_a}: {b_as_a}/{b_total} "
        f"({(b_as_a / b_total * 100 if b_total else 0):.1f}%)"
    )


def main():
    print("Loading dataset (lean Stage D: baseline + L*a*b* + Gabor)...")
    X, y, paths, label_to_idx = load_dataset()
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    print(f"\nTotal samples: {len(X)}")
    print(f"Feature length: {X.shape[1]}")

    if X.shape[1] != EXPECTED_FEATURE_LENGTH:
        raise RuntimeError(
            f"Feature length mismatch: got {X.shape[1]}, expected "
            f"{EXPECTED_FEATURE_LENGTH}. Check that feature_extractor_stageD_lean.py "
            f"is the one being imported (see path printed above)."
        )

    X_dev, X_test, y_dev, y_test, paths_dev, test_paths = train_test_split(
        X, y, paths,
        test_size=0.15,
        random_state=42,
        stratify=y
    )

    print(f"Development set: {len(X_dev)}")
    print(f"Final test set:   {len(X_test)}")

    sample_weight_dev = compute_sample_weight(class_weight="balanced", y=y_dev)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    param_dist = {
        "n_estimators": [25, 30, 35, 40, 45, 50, 60, 80, 100],
        "learning_rate": [0.03, 0.05, 0.1, 0.15],
        "max_depth": [2, 3, 4],
        "subsample": [0.8, 0.9, 1.0],
        "max_features": [None, "sqrt", "log2"],
        "min_samples_leaf": [2, 4, 8, 12],
        "min_samples_split": [4, 8, 16],
    }

    gbm = GradientBoostingClassifier(
        random_state=42,
        validation_fraction=0.15,
        n_iter_no_change=10,
        tol=1e-4,
    )

    search = RandomizedSearchCV(
        estimator=gbm,
        param_distributions=param_dist,
        n_iter=30,
        scoring="f1_macro",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    print("Running cross-validated hyperparameter search...")
    search.fit(X_dev, y_dev, sample_weight=sample_weight_dev)

    best_model = search.best_estimator_

    print("\nBest parameters:")
    print(search.best_params_)
    print(f"\nBest CV macro F1: {search.best_score_:.4f}")

    results = search.cv_results_
    ranked_idx = np.argsort(results["rank_test_score"])
    lines = [
        f"Best CV macro F1: {search.best_score_:.4f}",
        f"Best parameters: {search.best_params_}",
        "\nTop 10 parameter sets (sorted by CV score):\n",
    ]
    for rank_pos, idx in enumerate(ranked_idx[:10], start=1):
        mean_test = results["mean_test_score"][idx]
        std_test = results["std_test_score"][idx]
        mean_train = results["mean_train_score"][idx]
        params = results["params"][idx]
        gap = mean_train - mean_test
        lines.append(
            f"{rank_pos}. mean_cv_f1={mean_test:.4f} ± {std_test:.4f} | "
            f"mean_train_f1={mean_train:.4f} | train_cv_gap={gap:.4f} | params={params}"
        )
    save_text(OUTPUT_DIR / "cv_results_summary.txt", "\n".join(lines))

    y_test_pred = best_model.predict(X_test)

    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1_macro = f1_score(y_test, y_test_pred, average="macro")
    test_report = classification_report(
        y_test, y_test_pred, target_names=CLASS_NAMES, digits=4
    )
    cm = confusion_matrix(y_test, y_test_pred)

    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(f"Final Test Macro F1: {test_f1_macro:.4f}")
    print(test_report)
    print("Confusion Matrix:\n", cm)

    report_lines = [
        f"Final Test Accuracy: {test_acc:.4f}",
        f"Final Test Macro F1: {test_f1_macro:.4f}",
        "",
        test_report,
    ]
    report_pairwise_confusion(cm, CLASS_NAMES, "discolored", "diseased", report_lines)

    save_text(OUTPUT_DIR / "final_test_report.txt", "\n".join(report_lines))
    save_text(OUTPUT_DIR / "final_test_confusion_matrix.txt", str(cm))

    plot_confusion_matrix(
        cm, CLASS_NAMES,
        OUTPUT_DIR / "final_test_confusion_matrix.png",
        normalize=False,
        title="Lean Stage D Final Test Confusion Matrix (counts)"
    )
    plot_confusion_matrix(
        cm, CLASS_NAMES,
        OUTPUT_DIR / "final_test_confusion_matrix_normalized.png",
        normalize=True,
        title="Lean Stage D Final Test Confusion Matrix (row-normalized)"
    )

    print("\nMisclassified final test images:")
    mis_lines = []
    for i in range(len(y_test)):
        if y_test[i] != y_test_pred[i]:
            true_label = CLASS_NAMES[y_test[i]]
            pred_label = CLASS_NAMES[y_test_pred[i]]
            line = f"- {test_paths[i]} | TRUE={true_label} -> PRED={pred_label}"
            print(line)
            mis_lines.append(line)
    save_text(OUTPUT_DIR / "final_test_misclassified.txt", "\n".join(mis_lines))

    save_obj = {
        "model": best_model,
        "class_names": CLASS_NAMES,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "feature_length": X.shape[1],
        "best_params": search.best_params_,
        "best_cv_macro_f1": search.best_score_
    }
    joblib.dump(save_obj, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    print("\n" + "=" * 60)
    print("Comparison to Article 1 baseline and Stages B/C (same test split):")
    print(f"  Article 1     — Accuracy: 0.9353 | Macro-F1: 0.9363")
    print(f"  Stage B       — Accuracy: 0.9424 | Macro-F1: 0.9432")
    print(f"  Stage C       — Accuracy: 0.9424 | Macro-F1: 0.9434")
    print(f"  Lean Stage D  — Accuracy: {test_acc:.4f} | Macro-F1: {test_f1_macro:.4f}")
    print(f"  Delta vs A1   — Accuracy: {test_acc - 0.9353:+.4f} | Macro-F1: {test_f1_macro - 0.9363:+.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
