"""
features.py

Board feature extraction utilities and move-encoding helpers for chess ML/CNN.

This module provides:
- Encode a 'chess.Board' at a fixed point in a game as 13×8×8 (=832) float32 planes:
  • 12 planes for piece occupancy (6 per side)
  • 1 plane for side-to-move indicator (all ones if White to move, else zeros)
- Move encoding to an index in a fixed 4672-way policy space (OUTPUT_DIM)
- pick a random ply and the next move for supervision
- convert a class index to one-hot vectors

Encoding convention (13 planes):
  [0–5]   : White pieces [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]
  [6–11]  : Black pieces [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]
  [12]    : Side-to-move indicator (all ones if White to move, else zeros)

Array orientation:
  • NumPy rows are rank numbers flipped so that White is at the bottom.
  • Row 0 = rank 8 (Black’s back rank, top of board)
  • Row 7 = rank 1 (White’s back rank, bottom of board)

Fixed policy encoding of size 4672 (OUTPUT_DIM):
    • 3584 queen-like (64 squares × 8 dirs × 7 steps)
    • 512 knight-like (64 × 8)
    • 576 under-promotions (64 × 3 directions × 3 pieces [N,B,R])
Move_to_index(board, move): map chess.Move -> [0..4671] or None if not encodable.
Legal_policy_mask(board): bool mask over 4672 entries for current position.
"""

from __future__ import annotations
import numpy as np
import chess
import chess.pgn

# =========================
# Constants
# =========================

# 8x8 board
BOARD_H = 8
BOARD_W = 8

# 12 piece planes + 1 side-to-move plane = 13 channels total
NUM_PLANES = 13

# policy space size (for all possible moves)
OUTPUT_DIM = 4672  # 3584 queen-like + 512 knight + 576 under-promotions (N/B/R only)

# Direction packing order for queen-like moves (8 dirs):
DIR_INDEX = {
    (-1, 0): 0,  # N
    (+1, 0): 1,  # S
    (0, -1): 2,  # W
    (0, +1): 3,  # E
    (-1, -1): 4,  # NW
    (-1, +1): 5,  # NE
    (+1, -1): 6,  # SW
    (+1, +1): 7,  # SE
}

KNIGHT_INDEX = {
    (-2, -1): 0, (-2, +1): 1,
    (-1, -2): 2, (-1, +2): 3,
    (+1, -2): 4, (+1, +2): 5,
    (+2, -1): 6, (+2, +1): 7,
}


# =========================
# Plane encoders
# =========================

def board_to_13_planes(board: chess.Board) -> np.ndarray:
    """
    Encode the given board into (13, 8, 8) float32 planes:
      planes[0..5]     : white P,N,B,R,Q,K occupancy
      planes[6..11]    : black P,N,B,R,Q,K occupancy
      planes[12]       : side-to-move indicator (all ones if White to move, else zeros)

    Returns:
      np.ndarray shape (13, 8, 8), dtype float32

    Orientation:
      Row 0 = rank 8 (top, Black's back rank), Row 7 = rank 1 (bottom, White's back rank).
      Implemented via: row = 7 - chess.square_rank(sq); col = chess.square_file(sq).

    This representation is friendly for ML & CNN.
    """
    planes = np.zeros((13, 8, 8), dtype=np.float32)

    # piece maps for occupancy
    piece_map = board.piece_map()
    for sq, piece in piece_map.items():
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        r = 7 - rank
        c = file
        base = 0 if piece.color else 6  # True=white -> planes[0..5], False=black -> planes[6..11]
        planes[base + (piece.piece_type - 1), r, c] = 1.0

    # side-to-move plane (13th plane, index 12)
    if board.turn:  # True if White to move
        planes[12, :, :] = 1.0
    # else: leave zeros for Black to move

    return planes


# =========================
# Move <-> index encoding
# =========================
# We map any legal chess move to an integer 0...OUTPUT_DIM-1 in a fixed space.
# The exact scheme must match cnn_model. Ensure consistency!

# Directions for queen-like moves (horizontal/vertical/diagonals), max length 7.
QUEEN_DIRS = [
    (+1, 0), (-1, 0), (0, +1), (0, -1),  # rook-like
    (+1, +1), (+1, -1), (-1, +1), (-1, -1)  # bishop-like
]
MAX_QUEEN_STEP = 7  # up to 7 squares in any queen-like direction

# Knight move deltas (8 possibilities)
KNIGHT_DELTAS = [
    (+2, +1), (+2, -1), (-2, +1), (-2, -1),
    (+1, +2), (+1, -2), (-1, +2), (-1, -2)
]


def _square_to_rc(sq: int) -> tuple[int, int]:
    """Row/col with row 0 = rank 8 (top), row 7 = rank 1 (bottom)."""
    rank = chess.square_rank(sq)
    row = 7 - rank
    col = chess.square_file(sq)
    return row, col


def _rc_to_square(r: int, c: int) -> int:
    return (7 - r) * 8 + c


def _inside(r: int, c: int) -> bool:
    return 0 <= r < 8 and 0 <= c < 8


def _sign(x: int) -> int:
    """
    Return the sign of an integer as -1, 0, or +1.
    It exploits Python's boolean-to-int behavior:
    True == 1 and False == 0. Thus (x > 0) - (x < 0) yields:
      • +1 if x > 0
      • -1 if x < 0
      •  0 if x == 0
    """
    return (x > 0) - (x < 0)


def move_to_index(move: chess.Move) -> int | None:
    """
    Map python-chess Move -> [0 .. OUTPUT_DIM-1] under a fixed policy encoding:
      0..3583  : queen-like lines (rook/bishop/queen/king/pawns/en passant/castling king step=2)
                 idx = from*56 + dir*7 + (step-1)
      3584..4095: knight jumps
                 idx = 3584 + from*8 + knight_dir
      4096..4671: under-promotions tail (to N/B/R only), 64 * (3 dirs) * (3 pieces)
                 idx = 4096 + from*9 + dir3*3 + promo_i
                 dir3 ∈ {FL=0, F=1, FR=2} relative to the mover's forward (handled symmetrically)
                 promo_i: {N=0, B=1, R=2}
    Returns None if the move is outside this encoding.
    """
    from_sq = move.from_square
    to_sq = move.to_square

    r0, c0 = _square_to_rc(from_sq)
    r1, c1 = _square_to_rc(to_sq)
    dr, dc = r1 - r0, c1 - c0
    adr, adc = abs(dr), abs(dc)

    # ---------- Under-promotions (tail) ----------
    # Handle first so N/B/R promotions don't get absorbed by the queen-like block.
    if move.promotion in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
        # Map (dr,dc) into {FL,F,FR} for either color using symmetric patterns:
        # White forward is dr=-1; Black forward is dr=+1. We cover both explicitly.
        dir3_map = {
            (-1, -1): 0, (-1, 0): 1, (-1, +1): 2,  # white forward-left/forward/forward-right
            (+1, +1): 0, (+1, 0): 1, (+1, -1): 2,  # black forward-left/forward/forward-right (in our row system)
        }
        if (dr, dc) in dir3_map:
            dir3 = dir3_map[(dr, dc)]
            promo_i = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}[move.promotion]
            return 4096 + from_sq * 9 + dir3 * 3 + promo_i

    # ---------- Knights ----------
    if (dr, dc) in KNIGHT_INDEX:
        return 3584 + from_sq * 8 + KNIGHT_INDEX[(dr, dc)]

    # ---------- Queen-like lines (rook/bishop/queen/king/pawns/en-passant/castling king step=2) ----------
    # Horizontal/vertical: one of dr/dc is 0 and the other is 1..7
    # Diagonal: |dr| == |dc| in 1..7
    if (dr == 0 and 1 <= adc <= 7) or (dc == 0 and 1 <= adr <= 7) or (adr == adc and 1 <= adr <= 7):
        d = (_sign(dr), _sign(dc))  # unit direction
        if d in DIR_INDEX:
            dir_i = DIR_INDEX[d]
            # step: diagonal uses adr; straight uses the non-zero of adr/adc
            step = adr if adr == adc else (adr + adc)
            # Pack: 64 * (8*7) = 3584 total
            idx = from_sq * (8 * 7) + dir_i * 7 + (step - 1)
            if 0 <= idx < 3584:
                return idx

    # Not encodable under this scheme
    return None


# =========================
# Classic ML
# =========================

def board_at_move_20(game: chess.pgn.Game) -> chess.Board:
    """
    Return the board after 40 plies (20 full moves) along the mainline.
    Used by classic ML parts (do not change).
    """
    board = game.board()
    for i, mv in enumerate(game.mainline_moves()):
        if i >= 40:
            break
        board.push(mv)
    return board


# =========================
# CNN-specific helpers (random ply + one-hot)
# =========================

def board_after_k_halfmoves(game: chess.pgn.Game, k: int) -> chess.Board:
    """
    Return the board after exactly k half-moves (plies) along the mainline.
    k must be >= 0 and <= len(moves). If k == len(moves), it's the final position.
    """
    assert k >= 0
    board = game.board()
    for i, mv in enumerate(game.mainline_moves()):
        if i >= k:
            break
        board.push(mv)
    return board


def _sample_segmented_k(end_k: int, rng: np.random.Generator,
                        weights=(0.02, 0.5, 0.48)) -> int:
    """
    Pick k in [1, end_k] with segment weights:
      - 2% from early game
      - 50% from middle game
      - 48% from end game

    Segments are defined by terciles of the game's ply count:
      start  = [1, a]
      middle = [a+1, b]
      end    = [b+1, end_k]
    where a = floor(end_k/3), b = floor(2*end_k/3).

    Always returns a valid k in [1, end_k].
    """
    if end_k <= 1:
        return 1
    if end_k <= 3:
        # For very short games, uniform over valid ks
        return int(rng.integers(1, end_k + 1))

    a = max(1, end_k // 3)
    b = max(a + 1, (2 * end_k) // 3)
    b = min(b, end_k - 1)  # ensure the end segment is non-empty

    segments = [(1, a), (a + 1, b), (b + 1, end_k)]
    valid = [(i, (lo, hi)) for i, (lo, hi) in enumerate(segments) if lo <= hi]
    if not valid:
        return int(rng.integers(1, end_k + 1))

    # Reweight to the existing segments only
    w = np.array([weights[i] for i, _ in valid], dtype=float)
    w = w / w.sum()
    seg_idx = int(rng.choice(len(valid), p=w))
    lo, hi = valid[seg_idx][1]
    return int(rng.integers(lo, hi + 1))


def random_position_and_next_move(game: chess.pgn.Game, rng=None,
                                  weights=(0.02, 0.5, 0.48)):
    """
    Select a random (board, next_move) supervision pair from a single PGN game
    with biased sampling: 2% early-game, 50% mid-game, 48% end-game.

    Returns:
        (board_at_k, moves[k]) where k ∈ [1, len(moves)-1].
    """
    moves = list(game.mainline_moves())

    # Need at least two plies to have a "next" move label.
    if len(moves) < 2:
        return None

    rng = rng if rng is not None else np.random.default_rng()
    end_k = len(moves) - 1
    k = _sample_segmented_k(end_k, rng, weights)

    board = game.board()
    for mv in moves[:k]:
        board.push(mv)
    next_move = moves[k]
    return board, next_move


def index_to_onehot(idx: int, num_classes: int = OUTPUT_DIM) -> np.ndarray:
    """
    Convert a class index into a 1-of-K one-hot vector (float32) of length num_classes.
    """
    vec = np.zeros((num_classes,), dtype=np.float32)
    if idx is not None and 0 <= idx < num_classes:
        vec[idx] = 1.0
    return vec

# ---------------------------


def legal_policy_mask(board: chess.Board) -> np.ndarray:
    """
    Boolean mask over the policy space where True = legal move.
    Uses your move_to_index(move) (no board arg).
    """
    mask = np.zeros((OUTPUT_DIM,), dtype=bool)
    for mv in board.legal_moves:
        idx = move_to_index(mv)
        if idx is not None and 0 <= idx < OUTPUT_DIM:
            mask[idx] = True
    return mask


def best_legal_move_from_logits(logits: np.ndarray, board: chess.Board) -> chess.Move | None:
    """
    Choose the legal move with the highest logit.
    """
    best_mv, best_val = None, -np.inf
    for mv in board.legal_moves:
        idx = move_to_index(mv)
        if idx is None:
            continue
        val = logits[idx]
        if val > best_val:
            best_mv, best_val = mv, val
    return best_mv


def sample_legal_move_from_probs(
    probs: np.ndarray, board: chess.Board, rng: np.random.Generator | None = None
) -> chess.Move | None:
    """
    Sample a legal move proportionally to its probability.
    """
    rng = rng or np.random.default_rng()
    legal_moves, weights = [], []
    for mv in board.legal_moves:
        idx = move_to_index(mv)
        if idx is None:
            continue
        p = probs[idx]
        if p > 0:
            legal_moves.append(mv)
            weights.append(p)
    if not legal_moves:
        return None
    w = np.asarray(weights, dtype=np.float64)
    w /= w.sum()
    return legal_moves[rng.choice(len(legal_moves), p=w)]
