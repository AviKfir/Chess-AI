"""
ui.py

Chess Game in Terminal

Play vs. the CNN model

Overview
--------
A lightweight command-line chess match interface for a human vs. CNN model.
You pick your color. The program handles rendering the board,
reading your move, and asking the engine for its reply.

Supported engines
-----------------
There is also a Minimax engine because I used Minimax first to sanity check
that chess rules, moves, and I/O worked, and only then switched the engine to the CNN.

1) CNN (by default)
   • Loads a trained CNN policy (address: cache/best_cnn_policy.pt)
   • Chooses moves via cnn_move(...)

2) Minimax
   • Alpha–beta–pruned minimax search algorithm
   • Chooses moves via minimax_move(...)

Key features
------------
• Choose side to play (White/Black).
• Pretty Unicode Board with ANSI colors.
• Last move highlighting.
• Validates and applies only legal moves.
• Delegates engine turns to the selected move generator.
• Hint: on your turn, type 'hint' or '?' to see the engine's best move (does not play it).

Typical usage
-------------
# Play vs. CNN (it's by default. You can also add: --engine cnn):
python -m src.engine.ui

# Play vs. Minimax:
python -m src.engine.ui --engine minimax
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ml"))

import chess
import torch
import argparse
from typing import Optional
from src.ml.cnn_model import ChessModel, predict_masked
from src.ml.features import OUTPUT_DIM  # num classes = 4672 (all possible chess moves)
from src.engine.move_search import minimax_move  # used only when: --engine minimax

# --- ANSI COLOR CONSTANTS ---
RESET = "\x1b[0m"
LABEL_STYLE = "\x1b[0;30;107m"
LIGHT_SQUARE = "\x1b[48;5;253m"
DARK_SQUARE = "\x1b[48;5;255m"
HIGHLIGHT_MOVE = "\x1b[48;5;153m"  # blue highlight for last move

# --- DEFAULTS ---
DEFAULT_MODEL_PATH = os.path.join("cache", "best_cnn_policy.pt")
DEFAULT_TEMPERATURE = 1.0
DEFAULT_DEPTH = 2


# ---------------------------------------------------------------------
# CNN policy helpers
# ---------------------------------------------------------------------
def load_policy(device: torch.device, model_path: str) -> ChessModel:
    """
    Instantiate and load the trained policy model.
    """
    model = ChessModel(num_classes=OUTPUT_DIM).to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. Train first (cnn_train.py), "
            f"or pass --model-path to an existing .pt."
        )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def cnn_move(
        board: chess.Board,
        model: ChessModel,
        device: torch.device,
        temperature: float = DEFAULT_TEMPERATURE,
        sample: bool = True,
) -> chess.Move:
    """
    Choose the next move using the CNN policy head.
    Falls back to a random legal move if something unexpected happens.
    """
    try:
        move, _probs = predict_masked(
            model,
            board,
            device=device,
            temperature=temperature,
            sample=sample,
            return_probs=True,
        )

        # Safety: ensure legality (predict_masked should already do this)
        if move not in board.legal_moves:
            move = next(iter(board.legal_moves))
            print("did: next(iter(board.legal_moves))", flush=True)
        return move

    except Exception as e:
        print(f"[cnn_move] Policy error ({e}); using fallback.")
        return next(iter(board.legal_moves))  # random-ish legal move


# ---------------------------------------------------------------------
# Hint helpers (new)
# ---------------------------------------------------------------------
def engine_suggest_move(
        board: chess.Board,
        params: argparse.Namespace,
        model: Optional[ChessModel],
        device: torch.device,
) -> chess.Move:
    """
    Return the engine's best move for the current position without modifying the board.

    • Minimax: calls minimax_move(depth, board).
    • CNN: deterministic suggestion via predict_masked(..., sample=False).
    """
    if params.engine == "minimax":
        return minimax_move(params.depth, board)
    else:
        # Deterministic argmax suggestion for clarity
        try:
            move, _probs = predict_masked(
                model,
                board,
                device=device,
                temperature=params.temperature,
                sample=False,  # <— force argmax for hints
                return_probs=True,
            )
            if move not in board.legal_moves:
                return next(iter(board.legal_moves))
            return move
        except Exception as e:
            print(f"[hint] Policy error ({e}); falling back to a legal move.")
            return next(iter(board.legal_moves))


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
def print_unicode_board(board: chess.Board, perspective: chess.Color = chess.WHITE):
    """
    Print the chessboard with Unicode pieces and ANSI-colored squares.
    Highlights the most recent move and adjusts rendering
    based on the chosen perspective.
    """
    for r in range(8) if perspective == chess.BLACK else range(7, -1, -1):
        line = [f"{LABEL_STYLE} {r + 1}"]
        for c in range(8) if perspective == chess.WHITE else range(7, -1, -1):
            color = DARK_SQUARE if (r + c) % 2 == 1 else LIGHT_SQUARE

            if board.move_stack:
                last_move = board.move_stack[-1]
                sq = 8 * r + c
                if last_move.to_square == sq or last_move.from_square == sq:
                    color = HIGHLIGHT_MOVE

            piece = board.piece_at(8 * r + c)
            symbol = chess.UNICODE_PIECE_SYMBOLS[piece.symbol()] if piece else " "
            line.append(color + symbol)

        print(" " + " ".join(line) + f" {LABEL_STYLE} {RESET}")

    if perspective == chess.WHITE:
        print(f" {LABEL_STYLE}   a b c d e f g h  {RESET}\n")
    else:
        print(f" {LABEL_STYLE}   h g f e d c b a  {RESET}\n")


def get_move(
        board: chess.Board,
        params: argparse.Namespace,
        model: Optional[ChessModel],
        device: torch.device,
) -> chess.Move:
    """
    Ask the user to input a move until a legal one is provided.
    Type 'hint', 'h', or '?' to see the engine's suggested move (does not make a move).
    """
    while True:
        user_inp = input(
            f"\n(type 'hint' to get a suggestion for next best move)\nYour move:\n"
        ).strip().lower()

        # --- NEW: hint pathway ---
        if user_inp in {"hint", "?", "h"}:
            suggestion = engine_suggest_move(board, params, model, device)
            print(f"Hint: try {suggestion}")
            # Loop continues; we do not alter the board.
            continue

        # Normal move validation
        for legal_move in board.legal_moves:
            if user_inp == str(legal_move):
                return legal_move
        print("Illegal move, try again (or type 'hint').")


def play_game(params: argparse.Namespace):
    """
    Main game loop: user vs chosen engine (minimax or cnn).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nProcessing Unit: {device}")

    if params.engine == "minimax":
        print("Engine: Minimax\n")
        model = None
    else:
        print("Engine: CNN\n")
        model_path = params.model_path or DEFAULT_MODEL_PATH
        model = load_policy(device, model_path)

    board = chess.Board()
    valid = False
    user_side = chess.WHITE
    print("\nPlease maximize your terminal window so the chessboard prints clearly without being cut :)\n")
    print("\nEnter your move in UCI coordinates (from-square + to-square)\n"
          "For example: e2e4, g1f3\n"
          "Captures use the same format: e4d5\n"
          "Promotions add the piece letter: e7e8q\n"
          "Castling is e1g1/e1c1 (or e8g8/e8c8)\n")
    while not valid:
        color = input("Do you want to be white or black?\n")
        if color.lower().startswith("w"):
            valid = True
        elif color.lower().startswith("b"):
            user_side = chess.BLACK
            valid = True

    print_unicode_board(board, user_side)

    # If user plays White, user moves first
    if user_side == chess.WHITE:
        board.push(get_move(board, params, model, device))

    # Loop until game over
    while not board.is_game_over():
        if params.engine == "minimax":
            # --- Minimax engine turn ---
            board.push(minimax_move(params.depth, board))
        else:
            # --- CNN engine turn ---
            board.push(cnn_move(board, model, device, params.temperature, params.sample))

        print_unicode_board(board, user_side)

        if board.is_game_over():
            break

        # User turn
        board.push(get_move(board, params, model, device))
        print_unicode_board(board, user_side)

    print(f"\nResult: [w] {board.result()} [b]")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Play vs. the engine from the terminal.")
    p.add_argument(
        "--engine",
        choices=["minimax", "cnn"],
        default="cnn",
        help="Which engine to use for the computer player.",
    )
    p.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to CNN weights (.pt). Default: {DEFAULT_MODEL_PATH}",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Softmax temperature for CNN move selection (>=0). 1.0 is typical.",
    )
    p.add_argument(
        "--depth",
        type=int,
        default=DEFAULT_DEPTH,
        help="Search depth for the minimax engine (ignored in CNN mode).",
    )
    p.add_argument(
        "--no-sample",
        dest="sample",
        action="store_false",
        help="Disable sampling. Use argmax."
    )
    p.set_defaults(sample=True)  # Sample from the CNN policy

    return p.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        play_game(args)
    except KeyboardInterrupt:
        pass
