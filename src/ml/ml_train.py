"""
ml_train.py

Model training utilities for chess outcome prediction (classical ML).
Priority:
1) Load existing models
2) If missing -> try Xy NPZ (flat)
3) If NPZ missing -> try CSV to build Xy
"""

from __future__ import annotations
from pathlib import Path
import os
import time
import numpy as np
import joblib
import matplotlib
import matplotlib.pyplot as plt
import re
matplotlib.use("Agg")

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_utils import load_or_build_xy, extract_size_from_filename, read_xy_npz

# =========================
# Configuration (Path-based)
# =========================
BASE: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = BASE / "data"
CACHE_DIR: Path = BASE / "cache"
FIGURES_DIR: Path = BASE / "figures"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH: Path = DATA_DIR / "200K_filtered_games.csv"


# ================== TRAINER ==================
class Trainer:
    def __init__(self, model, test_size: float = 0.2, random_state: int = 42, stratify: bool = True):
        self.model = model
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None

    def split(self, X: np.ndarray, y: np.ndarray):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if self.stratify else None
        )
        print(f"X_train: {self.X_train.shape}, y_train: {self.y_train.shape}")
        print(f"X_test : {self.X_test.shape}, y_test : {self.y_test.shape}")

    def fit(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self) -> dict:
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1_score_val = f1_score(self.y_test, y_pred, zero_division=0)
        confusion_mat = confusion_matrix(self.y_test, y_pred)

        print("\nEvaluation Metrics:")
        print(f"Accuracy : {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall   : {recall:.2f}")
        print(f"F1 Score : {f1_score_val:.2f}")
        print("Confusion Matrix:\n", confusion_mat)
        print("----\n")

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1_score_val),
            "confusion_matrix": confusion_mat.tolist(),
        }

    def save(self, path: Path):
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, os.fspath(path))
        print(f"Model saved to {path}")


# ================== DATA LOADERS ==================
def load_dataset_npz_then_csv(csv_path: Path, cache_dir: Path, suffix: str = "flat"):
    """
    Try NPZ (cache) first, then CSV (build).
    Returns (X, y, cache_path_str) or (None, None, None) if neither is available.
    """
    size_tag = extract_size_from_filename(csv_path)
    npz_path = cache_dir / f"xy_{size_tag}_{suffix}.npz"

    # 1) NPZ present -> load directly (no CSV required)
    if npz_path.exists():
        print(f"Loading dataset from NPZ cache: {npz_path}")
        X, y = read_xy_npz(os.fspath(npz_path), x_dtype=np.float32, y_dtype=np.int64, mmap_mode=None)
        return X, y, os.fspath(npz_path)

    # 2) NPZ missing -> try CSV build (if present)
    if csv_path.exists():
        print(f"NPZ cache not found. Building from CSV: {csv_path}")
        X, y, cache_path = load_or_build_xy(
            csv_path=os.fspath(csv_path),
            cache_dir=os.fspath(cache_dir),
            force_rebuild=False,
            chunksize=50_000,
            parallel=True,
            n_jobs=-1,
        )
        return X, y, cache_path

    # 3) Neither available
    return None, None, None


def ensure_dataset_loaded(X, y, dataset_loaded: bool, *, csv_path: Path, cache_dir: Path):
    """
    Lazily load dataset in order NPZ -> CSV.
    Returns: (X, y, dataset_loaded, cache_path_or_None)
    """
    if dataset_loaded:
        return X, y, dataset_loaded, None

    X_new, Y_new, cache_path = load_dataset_npz_then_csv(csv_path, cache_dir, suffix="flat")
    if X_new is None or Y_new is None:
        return X, y, False, None

    print(f"Dataset: X={X_new.shape}, y={Y_new.shape} (source: {cache_path})")
    X_new = X_new.astype(np.float32, copy=False)
    return X_new, Y_new, True, cache_path


# ================== SINGLE-SIZE PROGRESS (BAR CHART) ==================
SIZE_RE = re.compile(r"([0-9]+)K", re.IGNORECASE)


def _plot_confusion_matrix_heatmap(cm: np.ndarray, title: str, out_path):
    """
    Save a heatmap of a 2x2 confusion matrix with integer count annotations.
    """
    plt.figure(figsize=(4.2, 3.8))
    plt.imshow(cm, interpolation="nearest", aspect="auto")
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    plt.colorbar()

    # annotate counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion heatmap → {out_path}")


def evaluate_and_plot_single_size(size_tag: str, X: np.ndarray, y: np.ndarray):
    """
    Evaluate all saved classic-ML models for a single dataset size (e.g., '200K')
    using the PROVIDED X, y, and for each model:
      • compute metrics (acc/prec/recall/F1 + confusion matrix)
      • save a confusion-matrix HEATMAP image with counts
    Also prints the confusion matrix values to the console.
    """
    # --- validate inputs ---
    if X is None or y is None:
        raise ValueError("X and y must be provided (got None).")
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}.")
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)

    # Normalize tag like '200K'
    m = SIZE_RE.fullmatch(size_tag.upper())
    if not m:
        raise ValueError(f"size_tag must look like '200K', got: {size_tag}")

    # Discover models of that size
    lr_path = CACHE_DIR / f"logistic_regression_{size_tag}.joblib"
    knn_paths = sorted(CACHE_DIR.glob(f"knn_*n_{size_tag}.joblib"),
                       key=lambda p: int(p.name.split('_')[1][:-1]))  # knn_{k}n_{size}.joblib

    model_entries = []
    if lr_path.exists():
        model_entries.append(("Logistic Regression", lr_path))
    for p in knn_paths:
        try:
            k_str = p.name.split('_')[1]  # e.g., "101n"
            k = int(k_str[:-1])
            model_entries.append((f"KNN k={k}", p))
        except Exception:
            continue

    if not model_entries:
        print(f"No saved models found for size_tag={size_tag} in {CACHE_DIR}")
        return

    # Evaluate each model and emit a heatmap
    for label, path in model_entries:
        print(f"\nEvaluating {label} from {path} ...")
        model = joblib.load(os.fspath(path))
        trainer = Trainer(model)
        trainer.split(X, y)  # consistent stratified split per model
        metrics = trainer.evaluate()  # returns dict with 'confusion_matrix', 'accuracy', 'precision', 'recall', 'f1'

        cm = np.array(metrics["confusion_matrix"], dtype=np.int64)

        out_png = FIGURES_DIR / f"confusion_matrix_{label.replace(' ', '_').replace('=', '')}_{size_tag}.png"
        _plot_confusion_matrix_heatmap(cm, title=f"{label} @ {size_tag}", out_path=out_png)


# ================== KNN comparison diagram (single figure) ==================
K_RE = re.compile(r"^knn_(\d+)n_([0-9]+K)\.joblib$", re.IGNORECASE)


def evaluate_knn_and_plot_metrics(size_tag: str, X: np.ndarray, y: np.ndarray):
    """
    Find all saved KNN models for the given size_tag (e.g. '200K'), evaluate each
    on the same split, and plot Accuracy/Precision/Recall/F1 vs k in ONE chart.

    Output: figures/knn_comparison_{size_tag}.png
    """
    if X is None or y is None:
        raise ValueError("X and y must be provided.")
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}.")
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)

    # discover available k from files
    ks = []
    paths = []
    for p in CACHE_DIR.glob(f"knn_*n_{size_tag}.joblib"):
        m = K_RE.match(p.name)
        if m and m.group(2).upper() == size_tag.upper():
            ks.append(int(m.group(1)))
            paths.append(p)
    if not ks:
        print(f"No saved KNN models for size_tag={size_tag} in {CACHE_DIR}")
        return

    # sort by k
    pairs = sorted(zip(ks, paths), key=lambda t: t[0])
    ks = [k for k, _ in pairs]
    paths = [p for _, p in pairs]

    # evaluate each KNN
    acc, prec, rec, f1 = [], [], [], []
    for k, path in zip(ks, paths):
        print(f"Evaluating KNN(k={k}) from {path} ...")
        model = joblib.load(os.fspath(path))
        trainer = Trainer(model)
        trainer.split(X, y)                    # same random_state -> consistent split
        metrics = trainer.evaluate()           # dict with accuracy/precision/recall/f1
        acc.append(metrics["accuracy"])
        prec.append(metrics["precision"])
        rec.append(metrics["recall"])
        f1.append(metrics["f1"])

    # make sure figures/ exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # plot one figure (no seaborn, no custom colors)
    plt.figure(figsize=(8, 5))
    plt.plot(ks, acc, marker="o", label="Accuracy")
    plt.plot(ks, prec, marker="o", label="Precision")
    plt.plot(ks, rec, marker="o", label="Recall")
    plt.plot(ks, f1, marker="o", label="F1")

    plt.xlabel("k (neighbors)")
    plt.ylabel("Score")
    plt.title(f"KNN Comparison @ {size_tag}")
    plt.grid(True)
    plt.legend()

    out_png = FIGURES_DIR / f"knn_comparison_{size_tag}.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved KNN comparison chart → {out_png}")

    # optional: print compact table
    print("\nKNN summary:")
    for k, a, p, r, f in zip(ks, acc, prec, rec, f1):
        print(f"  k={k:<4}  Acc={a:.2f}  Prec={p:.2f}  Rec={r:.2f}  F1={f:.2f}")


# ================== MAIN ==================
def main():
    size_tag = extract_size_from_filename(os.fspath(CSV_PATH))

    # -------------------------------
    # Load/Build dataset ONCE (NPZ -> CSV)
    # -------------------------------
    X = None  # np.ndarray | None
    y = None  # np.ndarray | None

    X, y, dataset_loaded, cache_path = ensure_dataset_loaded(
        X, y, False, csv_path=CSV_PATH, cache_dir=CACHE_DIR
    )
    if dataset_loaded:
        print(f"Dataset is ready.")
    else:
        print("\nNo NPZ cache and no CSV available — running in model-only mode (evaluation skipped).")

    # -------------------------------
    # Option 1: Logistic Regression
    # -------------------------------
    print("\n------- Option 1: Logistic Regression -------\n")
    lr_model_path = CACHE_DIR / f"logistic_regression_{size_tag}.joblib"

    if lr_model_path.exists():
        print(f"Logistic Regression model found with Dataset of size: {size_tag}. Loading...")
        lr_loaded = joblib.load(os.fspath(lr_model_path))
        if dataset_loaded:
            lr_trainer = Trainer(lr_loaded)
            lr_trainer.split(X, y)
            lr_trainer.evaluate()
        else:
            print("(No dataset available — skipping LR evaluation.)")

    else:
        print("Logistic Regression model not found.")
        if not dataset_loaded:
            print("No dataset to train LR — skipping.")
        else:
            logistic_regression = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(
                    solver="lbfgs",
                    max_iter=400,
                    tol=1e-3,
                    random_state=42
                )),
            ])
            lr_trainer = Trainer(logistic_regression)
            lr_trainer.split(X, y)
            t0 = time.perf_counter()
            lr_trainer.fit()
            print(f"(LR fit time: {time.perf_counter() - t0:.2f}s)")
            lr_trainer.evaluate()
            lr_trainer.save(lr_model_path)

    # -------------------
    # Option 2: KNN sweep
    # -------------------
    print("\n------- Option 2: KNN -------\n")

    # for k in [5, 11, 51, 101, 501, 1001]:
    # Long to run, we keep only one number as running example
    for k in [51]:  # best was K=51
        model_path = CACHE_DIR / f"knn_{k}n_{size_tag}.joblib"

        if model_path.exists():
            print(f"KNN model K={k} found with Dataset of size: {size_tag}. Loading...")
            loaded_model = joblib.load(os.fspath(model_path))
            if dataset_loaded:
                trainer_knn = Trainer(loaded_model)
                trainer_knn.split(X, y)
                trainer_knn.evaluate()
            else:
                print(f"(No dataset available — skipping evaluation for k={k}.)")

        else:
            print(f"KNN model (k={k}) not found.")
            if not dataset_loaded:
                print(f"No dataset to train KNN k={k} — skipping.")
                continue

            print(f"Training new KNN (k={k})")
            knn = KNeighborsClassifier(n_neighbors=k)
            trainer_knn = Trainer(knn)
            trainer_knn.split(X, y)

            t0 = time.perf_counter()
            trainer_knn.fit()
            print(f"(KNN fit time for k={k}: {time.perf_counter() - t0:.2f}s)")
            trainer_knn.evaluate()
            trainer_knn.save(model_path)

    # single-size progress plot
    if dataset_loaded:
        evaluate_and_plot_single_size("200K", X, y)

    evaluate_knn_and_plot_metrics(size_tag, X, y)


if __name__ == "__main__":
    main()
