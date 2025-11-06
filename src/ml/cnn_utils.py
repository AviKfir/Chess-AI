"""
cnn_utils.py

Dataset building & caching utilities for the CNN policy head.
Random-ply supervision and one-hot move labels (size 4672).
"""

from __future__ import annotations
from typing import Optional
import io
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import chess.pgn

import features  # board_to_13_planes, random_position_and_next_move, move_to_index, index_to_onehot
from ml_utils import cache_or_build_xy  # centralized caching logic

ENCODING = "latin1"
PGN_COL = "PGN"


# -------------------------------
# Encoder: random ply + one-hot
# -------------------------------

def _encode_single_policy(pgn_text: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    game = chess.pgn.read_game(io.StringIO(str(pgn_text)))
    if game is None:
        return None

    res = features.random_position_and_next_move(game)
    if res is None:
        return None

    board, next_move = res
    planes = features.board_to_13_planes(board)

    move_idx = features.move_to_index(next_move)
    if move_idx is None:
        return None

    onehot = features.index_to_onehot(move_idx)
    return planes.astype(np.float32), onehot.astype(np.float32)


def _process_chunk_policy(df_chunk: pd.DataFrame, n_jobs: int) -> tuple[np.ndarray, np.ndarray]:
    rows = df_chunk[PGN_COL].astype(str).tolist()
    results = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
        delayed(_encode_single_policy)(pgn) for pgn in rows
    )
    results = [r for r in results if r is not None]
    if not results:
        return np.empty((0, 13, 8, 8), dtype=np.float32), np.empty((0, 4672), dtype=np.float32)
    Xc = np.stack([r[0] for r in results]).astype(np.float32)
    yc = np.stack([r[1] for r in results]).astype(np.float32)
    return Xc, yc


# -------------------------------
# Public builder + cache wrapper
# -------------------------------

def build_policy_xy(
        csv_path: str,
        chunksize: int = 100_000,
        parallel: bool = True,
        n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    reader = pd.read_csv(csv_path, chunksize=chunksize, dtype=str, usecols=[PGN_COL], encoding=ENCODING)
    for df_chunk in reader:
        if parallel:
            Xc, yc = _process_chunk_policy(df_chunk, n_jobs=n_jobs)
        else:
            accum = []
            for pgn in df_chunk[PGN_COL].astype(str):
                r = _encode_single_policy(pgn)
                if r is not None:
                    accum.append(r)
            if accum:
                Xc = np.stack([t[0] for t in accum]).astype(np.float32)
                yc = np.stack([t[1] for t in accum]).astype(np.float32)
            else:
                Xc = np.empty((0, 13, 8, 8), dtype=np.float32)
                yc = np.empty((0, 4672), dtype=np.float32)

        if Xc.size > 0:
            X_parts.append(Xc)
            y_parts.append(yc)

    if not X_parts:
        return np.empty((0, 13, 8, 8), dtype=np.float32), np.empty((0, 4672), dtype=np.float32)

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    return X, y


def load_or_build_policy_xy(
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
        build_xy_function=build_policy_xy,
        build_kwargs={
            "csv_path": csv_path,
            "chunksize": chunksize,
            "parallel": parallel,
            "n_jobs": n_jobs,
        },
        suffix="cnn",
        x_dtype=np.float32,
        y_dtype=np.float32,
        force_rebuild=force_rebuild,
    )

