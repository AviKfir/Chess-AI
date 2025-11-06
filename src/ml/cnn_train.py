"""
cnn_train.py

Training script for a chess policy CNN that predicts a probability distribution
over a fixed move-encoding of size 4672 (see cnn_model.OUTPUT_DIM).
---------------------
What this script does:

1) Loads (or builds & caches) ONE dataset from CSV_PATH via `load_or_build_policy_xy`.
    • Uses a per-file index cache (cache/xy_idx_*.npz) to skip argmax when available.
    • Otherwise converts Y_onehot → class indices with a chunked argmax and frees Y_onehot.

2) Splits by indices into train/test (80/20) without copying large arrays.

3) Builds memory-light Dataset/DataLoader objects over (X, y_idx).

4) Creates `cnn_model.ChessModel` and optimizer.
    • Optional: if RESUME_WEIGHTS=True and cache/best_cnn_policy.pt exists, loads weights to continue training.

5) Trains for EPOCHS:
    • Loss: `CrossEntropyLoss` with `label_smoothing=LABEL_SMOOTH`.
    • ETA/throughput prints during training.
    • After each epoch, evaluates accuracy, macro precision/recall/F1, and Top-K accuracy.

6) Saves the best checkpoint by macro-F1 to `cache/best_cnn_policy.pt`
-------------------
Tensor shape legend:
    • B = batch size (number of samples processed in parallel).
    • For inputs:  (B, 13, 8, 8)
    • For outputs: (B, 4672) raw logits (one logit per encoded move).

-------------------
Usage examples:

Choose one CSV: "07_20.csv", "08_20.csv", "09_20.csv", "07_23.csv", "08_23.csv", "09_23.csv"

Train from scratch:
    - Set CSV_PATH
    - RESUME_WEIGHTS = False

Continue training:
    - Set CSV_PATH
    - Set RESUME_WEIGHTS = True

Run in Terminal:
python cnn_train.py
"""

from __future__ import annotations
import os
import gc
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from cnn_utils import load_or_build_policy_xy
from cnn_model import ChessModel
from features import OUTPUT_DIM
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

# =========================
# Configuration
# =========================

BASE = Path(__file__).resolve().parents[2]  # project root = scripts_src
CSV_PATH = BASE / "data" / "07_23.csv"
CACHE_DIR = BASE / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
BATCH_SIZE = 1024
EPOCHS = 6
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_CLASSES = OUTPUT_DIM  # 4672-way policy classification

# if True and cache/best_cnn_policy.pt exists, load weights and continue training
RESUME_WEIGHTS = True
BEST_MODEL_PATH = os.path.join(CACHE_DIR, "best_cnn_policy.pt")

# Regularization / reporting
LABEL_SMOOTH = 0.10
TOP_K = 5  # Top-K accuracy (Top-5)


# Dataset (memory-light: index-based)
class PolicyDataset(Dataset):
    """
    Keeps references to full X (float32 planes) and y_idx (int labels),
    and selects rows via an indices array (no large copies).
    """

    def __init__(self, X_full: np.ndarray, y_idx_full: np.ndarray, indices: np.ndarray):
        self.X = X_full
        self.y = y_idx_full
        self.indices = indices

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        j = int(self.indices[i])
        x = torch.from_numpy(self.X[j]).to(dtype=torch.float32)  # (13, 8, 8)
        y = torch.tensor(int(self.y[j]), dtype=torch.long)
        return x, y


def build_index_cache(X: np.ndarray,
                      y_onehot: np.ndarray,
                      idx_cache_path,
                      block: int = 8192,
                      progress_every: int = 20) -> np.ndarray:
    """
    Convert dense one-hot targets into compact class indices and save a small cache.

    Purpose
    -------
    Training only needs class IDs, not full one-hot vectors. This helper:
      1) Converts y_onehot of shape (N, C) → y_idx of shape (N,)
      2) Saves a compact cache with fields:
         - 'X'     : float32 array, shape (N, 13, 8, 8)
         - 'y_idx' : uint16 array,  shape (N,)
            • Range: 0 … 65,535 (= 2^16 - 1). We only have NUM_CLASSES = 4672.
            • uint16 = unsigned 16-bit integer → uses all 16 bits for the value.

    Why this exists
    ---------------
    • Massive memory/I/O savings. For C=4672 and N≈279K (samples):
        y_onehot  ≈ 5.2 GB (float32)
        y_idx     ≈ 0.56 MB (uint16)
      Future runs can load {X, y_idx} directly and skip argmax + the huge one-hot file.

    Returns
    -------
    y_idx : np.ndarray
        Class indices (int64), shape (N,). Also written to disk as uint16.

    Tiny example (before → after)
    -----------------------------
    Suppose C = 5 classes, and we look at a single sample i:

        # BEFORE (in big one-hot):
        y_onehot[i] = [0, 0, 1, 0, 0]   # class 2 is the target

        # AFTER (compact index):
        y_idx[i] = 2
    """
    print("Converting one-hot y → class indices...", flush=True)
    N = y_onehot.shape[0]
    y_idx = np.empty((N,), dtype=np.int64)

    for start in range(0, N, block):
        end = min(start + block, N)
        y_idx[start:end] = np.argmax(y_onehot[start:end], axis=1)
        if end == N or (start // block) % max(progress_every, 1) == 0:
            print(f"  {end}/{N} rows ({100.0 * end / N:.1f}%)", flush=True)

    # Save compact cache (X + indices as uint16 to shrink file size)
    cache_dir = os.path.dirname(idx_cache_path)
    if cache_dir and not os.path.isdir(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    np.savez_compressed(idx_cache_path, X=X, y_idx=y_idx.astype(np.uint16))
    print(f"Saved index-only cache: {idx_cache_path}", flush=True)

    return y_idx


# =========================
# Metrics
# =========================
def topK_accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """
    Compute Top-K accuracy for single-label multi-class classification.
    Gotchas:
      - Ensure k ≤ C; we clamp with k=min(k, logits.size(1)).
      - You don’t need softmax; logits ranking == probs ranking.
    """
    k = min(k, logits.size(1))
    topK = logits.topk(k, dim=1).indices  # (B, k)
    correct_any = topK.eq(targets.view(-1, 1)).any(1)  # (B,)
    return correct_any.float().mean().item()


def plot_progress(num_models: int, device: str, test_loader, model_class):
    """
    Evaluate and plot Top-5 accuracy for sequentially trained models.

    Args:
        num_models (int): Number of checkpoints (e.g. 6).
        device (str): 'cpu' or 'cuda'.
        test_loader: DataLoader for test set.
        model_class: The CNN model class (e.g. ChessModel).
    """
    scores = []

    for i in range(1, num_models + 1):
        path = CACHE_DIR / f"best_cnn_{i}.pt"
        model = model_class(num_classes=4672).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

        # evaluate Top-5
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                top5 = logits.topk(5, dim=1).indices
                correct += top5.eq(y.view(-1, 1)).any(1).sum().item()
                total += y.size(0)

        acc = correct / total
        scores.append(acc)
        print(f"Model {i}: Top-5 = {acc:.4f}")

    # plot progress
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, num_models + 1), scores, marker="o")
    plt.xlabel("Training round (model number)")
    plt.ylabel("Top-5 accuracy")
    plt.title("CNN Training Progress (Top-5 accuracy)")
    plt.grid(True)
    plt.savefig(CACHE_DIR / "progress_top5.png", dpi=200)
    plt.close()


# =========================
# Train / Eval
# =========================
def train_one_epoch(model, loader, optimizer, device, epoch, scaler, use_amp, log_every: int = 50) -> float:
    model.train()
    running_loss = 0.0
    total_steps = len(loader)
    start = time.time()

    for step, (xb, yb) in enumerate(loader, 1):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        # AMP autocast (cuda only when use_amp=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(xb)
            # Label smoothing (small quality bump)
            loss = torch.nn.functional.cross_entropy(
                logits, yb,
                label_smoothing=LABEL_SMOOTH
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item())

        # Minimal ETA print
        if step % log_every == 0 or step == total_steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - start
            avg_step = elapsed / step
            eta = avg_step * (total_steps - step)
            seen = step * loader.batch_size
            ips = int(seen / max(elapsed, 1e-9))
            print(f"[E{epoch:02d}] {step}/{total_steps} "
                  f"{avg_step:.3f}s/step | ETA {int(eta)}s | ~{ips} samples/s | "
                  f"loss {running_loss / step:.4f}")

    return running_loss / max(total_steps, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, float, float, float]:
    model.eval()
    all_pred, all_true = [], []
    topK_sum, n_total = 0.0, 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)  # (B, C)
        preds = torch.argmax(logits, 1)  # (B,)
        all_pred.append(preds.cpu().numpy())
        all_true.append(yb.cpu().numpy())

        # accumulate batch-weighted Top-K accuracy
        topK_batch = topK_accuracy_from_logits(logits, yb, k=TOP_K)
        topK_sum += topK_batch * xb.size(0)
        n_total += xb.size(0)

    y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64)
    y_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.int64)

    accuracy = accuracy_score(y_true, y_pred) if y_true.size else 0.0
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0) if y_true.size else 0.0
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0) if y_true.size else 0.0
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true.size else 0.0
    topK = topK_sum / max(n_total, 1)
    return accuracy, precision, recall, f1, topK


# =========================
# Main
# =========================
def main():
    print("\ncnn_train.py", flush=True)

    os.makedirs(CACHE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # AMP = Automatic Mixed Precision
    use_amp = (device.type == "cuda")
    if use_amp:
        torch.backends.cudnn.benchmark = True  # optional speed-up for convs
    # pass device type as positional arg for broad compatibility
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # 1) Load (or build) dataset for ONE CSV
    # FAST PATH: load compact index-only cache if present
    csv_stem = os.path.splitext(os.path.basename(CSV_PATH))[0]  # "07_23"
    idx_fname = f"xy_idx_{csv_stem}.npz"
    idx_cache = os.path.join(CACHE_DIR, idx_fname)

    print("idx:", idx_fname, flush=True)  # for pretty log
    print("full path:", idx_cache, flush=True)  # optional debug

    if os.path.exists(idx_cache):
        z = np.load(idx_cache)
        X = z["X"]
        Y_class_idx = z["y_idx"].astype(np.int64)
        print(f"Loaded index-only cache: {idx_cache} | X={X.shape}, y_idx={Y_class_idx.shape}", flush=True)

    else:
        print("else", flush=True)
        # build one-hot, then save index cache
        X, y_onehot, cache_path = load_or_build_policy_xy(
            csv_path=str(CSV_PATH),
            cache_dir=str(CACHE_DIR),
            force_rebuild=False,
            chunksize=50_000,
            parallel=True,
            n_jobs=-1,
        )
        print(f"X={X.shape}, y={y_onehot.shape} (cache: {cache_path})", flush=True)
        # Build and SAVE compact cache (X + y_idx) under the short name
        Y_class_idx = build_index_cache(
            X=X,
            y_onehot=y_onehot,
            idx_cache_path=idx_cache,  # string path is fine
            block=8192,
            progress_every=20,
        )
        # Free the huge one-hot
        del y_onehot
        gc.collect()
        # --- End else ---

    # 2) Train/test split by INDICES (no large array copies)
    N = len(Y_class_idx)
    rng = np.random.default_rng(42)
    perm = rng.permutation(N)
    cut = int(0.8 * N)
    train_idx, test_idx = perm[:cut], perm[cut:]
    print(f"Index split: train={len(train_idx)}, test={len(test_idx)}", flush=True)

    # 3) Datasets & loaders
    train_dataset = PolicyDataset(X, Y_class_idx, train_idx)
    test_dataset = PolicyDataset(X, Y_class_idx, test_idx)

    num_workers = 2
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 4) Model + optimizer
    model = ChessModel(num_classes=NUM_CLASSES).to(device)

    # Param groups: apply weight decay to weight, not to biases / norm params
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias") or "bn" in name.lower() or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)

    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=LR,
    )

    # (Optional) resume weights and continue training with best_cnn_policy
    if RESUME_WEIGHTS and os.path.exists(BEST_MODEL_PATH):
        try:
            state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
            # strict=False tolerates minor arch diffs; set True if checkpoint matches exactly
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"[resume] Loaded weights from {BEST_MODEL_PATH}")
            if missing or unexpected:
                print(f"[resume] (strict=False) missing={missing}, unexpected={unexpected}")
        except Exception as e:
            print(f"[resume] Could not load weights: {e}; starting from scratch.")
    else:
        print("[resume] Starting from randomly initialized weights.")

    print(f"Train steps/epoch: {len(train_loader)} | Test steps: {len(test_loader)}", flush=True)

    # 5) Training loop
    best_top_K = 0.0
    try:
        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, scaler, use_amp, log_every=50)
            acc, prec, rec, f1, topK = evaluate(model, test_loader, device)
            print(f"Epoch {epoch:02d} | loss {train_loss:.4f} | "
                  f"accuracy {acc:.4f} | top{TOP_K} {topK:.4f} | "
                  f"precision {prec:.4f} | recall {rec:.4f} | F1 {f1:.4f}")

            if topK > best_top_K:
                best_top_K = topK
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"  Saved new best model (top{TOP_K} = {best_top_K:.4f})", flush=True)

    except KeyboardInterrupt:
        torch.save(model.state_dict(), os.path.join(CACHE_DIR, "interrupt_cnn_policy.pt"))
        print("\nInterrupted. Saved checkpoint to interrupt_cnn_policy.pt", flush=True)

    print(f"Done. Best top{TOP_K} = {best_top_K:.4f}")

    plot_progress(
        num_models=6,
        device=str(device),
        test_loader=test_loader,
        model_class=ChessModel
    )


if __name__ == "__main__":
    main()
