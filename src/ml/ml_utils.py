"""
ml_utils.py

Shared dataset utilities (classic ML + helpers reused by CNN code).
- Classic ML build_xy keeps labels as win/loss and features as flat 13x8x8.
- Also provides generic cache loader/builder to avoid duplication elsewhere.
"""

from __future__ import annotations
from typing import Optional, Tuple, Callable, Dict, Any
import os
import io
import re
import numpy as np
import pandas as pd
import chess.pgn
from joblib import Parallel, delayed

import features  # board_at_move_20, board_to_13_planes

ENCODING = "latin1"
PGN_COL = "PGN"
RESULT_COL = "Result"


# -------------------------
# Generic, reusable helpers
# -------------------------

def extract_size_from_filename(path: str) -> str:
    """
    Return a size tag like '80K' from the filename (case-insensitive).
    If no size tag is found, return the filename without its extension.
    """
    base = os.path.basename(path)  # e.g. "07_23.csv"
    m = re.search(r"([0-9]+k)", base, re.IGNORECASE)
    if m:
        return m.group(1).upper()  # normalize '80k' -> '80K'
    name, _ = os.path.splitext(base)
    return name  # ensures "07_23.csv" -> "07_23"


def read_xy_npz(
        path: str,
        x_dtype: np.dtype = np.float32,
        y_dtype: Optional[np.dtype] = None,
        mmap_mode: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read cached NPZ with arrays 'X' and 'y' and cast dtypes consistently.
    """
    data = np.load(path, allow_pickle=False, mmap_mode=mmap_mode)
    X = data["X"].astype(x_dtype, copy=False)
    y = data["y"]
    if y_dtype is not None:
        y = y.astype(y_dtype, copy=False)
    return X, y


def cache_or_build_xy(
        *,
        csv_path: str,
        cache_dir: str,
        build_xy_function: Callable[..., Tuple[np.ndarray, np.ndarray]],
        build_kwargs: Optional[Dict[str, Any]] = None,
        suffix: str,  # suffix = "flat" or "cnn"
        force_rebuild: bool = False,
        mmap_mode: Optional[str] = None,
        x_dtype: np.dtype = np.float32,
        y_dtype: Optional[np.dtype] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Generic 'load-or-build' with caching (centralized).
    The file name will be: xy_{size_tag}_{suffix}.npz
    """
    os.makedirs(cache_dir, exist_ok=True)
    size_tag = extract_size_from_filename(csv_path)
    cache_path = os.path.join(cache_dir, f"xy_{size_tag}_{suffix}.npz")

    # Case A: CSV missing but cache exists -> load cache
    if not os.path.exists(csv_path) and os.path.exists(cache_path) and not force_rebuild:
        print(f"CSV not found ({csv_path}); loading cached dataset: {cache_path}")
        X, Y = read_xy_npz(cache_path, x_dtype=x_dtype, y_dtype=y_dtype, mmap_mode=mmap_mode)
        return X, Y, cache_path

    # Freshness check (only possible if both exist)
    fresh = False
    try:
        if os.path.exists(cache_path) and os.path.exists(csv_path):
            fresh = os.path.getmtime(cache_path) >= os.path.getmtime(csv_path)
    except OSError:
        fresh = False

    if (not force_rebuild) and fresh:
        X, Y = read_xy_npz(cache_path, x_dtype=x_dtype, y_dtype=y_dtype, mmap_mode=mmap_mode)
        return X, Y, cache_path

    # Case B: Need to build but CSV is missing
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No CSV at {csv_path}, and cache {cache_path} is missing or rebuild is forced.")

    # Build and save
    build_kwargs = build_kwargs or {}
    X, Y = build_xy_function(**build_kwargs)
    np.savez_compressed(cache_path, X=X, y=Y)

    X = X.astype(x_dtype, copy=False)
    if y_dtype is not None:
        Y = Y.astype(y_dtype, copy=False)
    return X, Y, cache_path


# ---------------------------------
# Classic ML-specific implementation
# ---------------------------------

def _encode_single_classic(pgn_text: str, result: str) -> Optional[tuple[np.ndarray, int]]:
    """
    Build (flat_13x8x8, label) where label is 1 for '1-0' (White) and 0 for '0-1' (Black).
    Draws are skipped.
    """
    result = (result or "").strip()
    if result not in {"1-0", "0-1"}:
        return None

    game = chess.pgn.read_game(io.StringIO(str(pgn_text)))
    if game is None:
        return None

    board = features.board_at_move_20(game)
    planes = features.board_to_13_planes(board)
    x = planes.reshape(-1).astype(np.float32)
    y = 1 if result == "1-0" else 0
    return x, y


def _process_chunk_classic(df_chunk: pd.DataFrame, n_jobs: int) -> tuple[np.ndarray, np.ndarray]:
    rows = list(zip(df_chunk[PGN_COL].astype(str).tolist(),
                    df_chunk[RESULT_COL].astype(str).tolist()))
    results = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
        delayed(_encode_single_classic)(pgn, res) for (pgn, res) in rows
    )
    results = [r for r in results if r is not None]
    if not results:
        return np.empty((0, 13 * 8 * 8), dtype=np.float32), np.empty((0,), dtype=np.int64)
    Xc = np.stack([r[0] for r in results]).astype(np.float32)
    yc = np.fromiter((r[1] for r in results), dtype=np.int64, count=len(results))
    return Xc, yc


def build_xy(
        csv_path: str,
        chunksize: int = 100_000,
        parallel: bool = True,
        n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build classic ML dataset:
      X: (N, 832) float32 (flattened 13x8x8)
      y: (N,) int64 in {0,1} (White vs Black), draws removed.
    """
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    reader = pd.read_csv(csv_path, chunksize=chunksize, dtype=str,
                         usecols=[PGN_COL, RESULT_COL], encoding=ENCODING)
    for df_chunk in reader:
        if parallel:
            Xc, yc = _process_chunk_classic(df_chunk, n_jobs=n_jobs)
        else:
            accum = []
            for pgn, res in zip(df_chunk[PGN_COL].astype(str), df_chunk[RESULT_COL].astype(str)):
                r = _encode_single_classic(pgn, res)
                if r is not None:
                    accum.append(r)
            if accum:
                Xc = np.stack([t[0] for t in accum]).astype(np.float32)
                yc = np.asarray([t[1] for t in accum], dtype=np.int64)
            else:
                Xc = np.empty((0, 13 * 8 * 8), dtype=np.float32)
                yc = np.empty((0,), dtype=np.int64)

        if Xc.size > 0:
            X_parts.append(Xc)
            y_parts.append(yc)

    if not X_parts:
        return np.empty((0, 13 * 8 * 8), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    return X, y


def load_or_build_xy(
        csv_path: str,
        cache_dir: str,
        force_rebuild: bool = False,
        chunksize: int = 100_000,
        parallel: bool = True,
        n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray, str]:
    return cache_or_build_xy(
        csv_path=csv_path,
        cache_dir=cache_dir,
        build_xy_function=build_xy,
        build_kwargs={
            "csv_path": csv_path,
            "chunksize": chunksize,
            "parallel": parallel,
            "n_jobs": n_jobs,
        },
        suffix="flat",
        x_dtype=np.float32,
        y_dtype=np.int64,
        force_rebuild=force_rebuild,
    )
