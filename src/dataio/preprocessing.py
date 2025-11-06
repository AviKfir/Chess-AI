"""
preprocessing.py

Utilities to filter and preprocess a large chess games CSV
that can be used for model training and feature extraction.

Workflow:
  1) Load a large CSV.
  2) Stream INPUT_CSV in chunks, clean rows, and keep only decisive games (1-0, 0-1) with at least 20 full moves.
  3) Append surviving rows to OUTPUT_CSV.
"""

import pandas as pd
import chess
import chess.pgn
import re
import os
import csv
import sys
from typing import Iterator, TextIO

INPUT_CSV = "80K_games.csv"
OUTPUT_CSV = "80K_filtered_games.csv"
ENCODING = "latin1"
CHUNK_SIZE = 100_000

RESULT_TOKEN_AT_END = re.compile(r"\s*(1-0|0-1|1/2-1/2)\s*$")
# Regex: matches move numbers like "1.", "23." etc.
NUMBERS_REGEX = re.compile(r"\b\d+\.")
WS = re.compile(r"\s+")


def is_non_draw(game):
    """
    Return True if the PGN game has a decisive result (White or Black wins).
    """
    result = (game.headers.get("Result") or "").strip()
    return result in {"1-0", "0-1"}


def write_game(game, out_path):
    """
    Write a single PGN game to disk.
    """
    with out_path.open("w", encoding="utf-8") as f:
        exporter = chess.pgn.FileExporter(f)
        game.accept(exporter)


def has_20_or_more_full_moves(pgn_str: str) -> bool:
    """
    Check if a PGN text appears to contain at least 20 full moves.
    """
    # Quick & cheap: count "N." occurrences
    return len(NUMBERS_REGEX.findall(str(pgn_str))) >= 20


def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a chunk of the raw CSV and keep only plausible decisive games.
    """
    # Drop obvious empties
    chunk = chunk.dropna(subset=["Result", "PGN"])

    # Normalize to string and strip whitespace
    chunk["Result"] = chunk["Result"].astype(str).str.strip()
    chunk["PGN"] = chunk["PGN"].astype(str)

    # Remove accidental header rows like: "Result,PGN" that got duplicated mid-file
    chunk = chunk[~chunk["Result"].str.fullmatch(r"\s*Result\s*", na=False)]

    # Keep only decisive games
    chunk = chunk[chunk["Result"].isin(["1-0", "0-1"])]

    # Keep only rows that look like a real PGN (contain "1.")
    chunk = chunk[chunk["PGN"].str.contains(r"\b1\.", na=False)]

    return chunk


def iter_games(fp: TextIO) -> Iterator[chess.pgn.Game]:
    """
    Stream PGN games one by one from an open text file.
    Returns until EOF.
    """
    while True:
        game = chess.pgn.read_game(fp)
        if game is None:
            break
        yield game


def movetext_sans_headers(game: chess.pgn.Game) -> str:
    """
    Export only the movetext (SAN moves). Removes comments, variations and the
    trailing game result token from the movetext to avoid duplication with the
    'Result' CSV column.
    """
    exporter = chess.pgn.StringExporter(
        headers=False, variations=False, comments=False
    )
    s = game.accept(exporter).strip()
    # Drop the trailing result token (e.g., " ... 64. Kf5 Rf3+ 0-1")
    s = RESULT_TOKEN_AT_END.sub("", s)
    # Normalize whitespace to single spaces
    s = WS.sub(" ", s).strip()
    return s


def process_reader(reader, output_csv=OUTPUT_CSV):
    wrote_header = False
    num_all_games = 0
    num_filtered_games = 0

    for chunk in reader:
        num_all_games += len(chunk)

        # cleaning step
        chunk = clean_chunk(chunk)
        if chunk.empty:
            continue

        # length filter
        candidate = chunk[chunk["PGN"].map(has_20_or_more_full_moves)]
        num_filtered_games += len(candidate)
        if candidate.empty:
            continue

        # Append survivors to output (write header only once)
        candidate.to_csv(
            output_csv,
            mode="a",
            index=False,
            header=not wrote_header
        )
        wrote_header = True

    return num_all_games, num_filtered_games


def pgn_file_to_minimal_csv(
        pgn_path: str,
        output_csv: str,
        *,
        encoding: str = "utf-8",
        include_draws: bool = True,
        progress_every: int = 0,
) -> None:
    """
    Convert a multi-game .pgn file into a 2-column CSV: Result, PGN.

    - Result ∈ {"1-0", "0-1", "1/2-1/2"}
    - PGN is move-text only (no headers/comments/variations, result token removed)

    Args:
        pgn_path: path to the source PGN file (e.g., "07_23.pgn")
        output_csv: path to write the CSV (e.g., "07_23_min.csv")
        encoding: file encoding for the PGN (defaults to 'utf-8')
        include_draws: if False, drops "1/2-1/2" games
        progress_every: if >0, prints a progress line every N games
    """
    valid_results = {"1-0", "0-1"}
    if include_draws:
        valid_results.add("1/2-1/2")

    count = 0
    kept = 0

    with open(pgn_path, "r", encoding=encoding, errors="ignore") as f_in, \
            open(output_csv, "w", newline="", encoding="utf-8") as f_out:

        writer = csv.writer(f_out)
        writer.writerow(["Result", "PGN"])

        for game in iter_games(f_in):
            count += 1
            result = (game.headers.get("Result") or "").strip()

            if result not in valid_results:
                # Skip malformed/unknown results
                continue

            move_text = movetext_sans_headers(game)
            writer.writerow([result, move_text])
            kept += 1

            if progress_every and (count % progress_every == 0):
                print(f"Processed {count:,} games … kept {kept:,}")

    if progress_every:
        print(f"Done. Processed {count:,} games; wrote {kept:,} rows to {output_csv!r}.")


def main():
    print("All's ready.")
    # df = pd.read_csv("../milestone/chess_games.csv", encoding="latin1")
    # df.head(80_000).to_csv(INPUT_CSV, index=False)
    #
    # # Delete output CSV (if exists) from previous run (because we will append, using mode="a").
    # # mode="a" → append mode. The new rows get added to the end of the existing file.
    # # mode="w" (the default) → write mode. It overwrites the file completely each time.
    # if os.path.exists(OUTPUT_CSV):
    #     os.remove(OUTPUT_CSV)
    #
    # reader = pd.read_csv(
    #     INPUT_CSV,
    #     usecols=["Result", "PGN"],
    #     encoding=ENCODING,
    #     chunksize=CHUNK_SIZE,
    #     on_bad_lines="skip",
    # )
    #
    # num_all_games, num_filtered_games = process_reader(reader)
    #
    # print(
    #     f"Done, wrote filtered rows to {OUTPUT_CSV}.\n"
    #     f"rows_in={num_all_games}, after_clean={num_filtered_games}"
    # )

    # pgn_file_to_minimal_csv("07_23.pgn", "07_23.csv", progress_every=10_000)
    # pgn_file_to_minimal_csv("08_23.pgn", "08_23.csv", progress_every=10_000)
    # pgn_file_to_minimal_csv("09_23.pgn", "09_23.csv", progress_every=10_000)
    # pgn_file_to_minimal_csv("07_20.pgn", "07_20.csv", progress_every=10_000)
    # pgn_file_to_minimal_csv("08_20.pgn", "08_20.csv", progress_every=10_000)
    pgn_file_to_minimal_csv("09_20.pgn", "09_20.csv", progress_every=10_000)


if __name__ == "__main__":
    main()
