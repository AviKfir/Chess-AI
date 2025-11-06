# src/preprocessing.py
import re
import pandas as pd
from typing import Optional, Tuple


def _find_pgn_col(df: pd.DataFrame) -> Optional[str]:
    # common names seen across chess datasets
    for c in ['AN', 'Moves', 'SAN', 'pgn', 'PGN']:
        if c in df.columns:
            return c
    return None


def compute_full_moves(df: pd.DataFrame) -> pd.Series:
    col = _find_pgn_col(df)
    if col is None:
        return pd.Series([0] * len(df), index=df.index)

    def _count(pgn: str) -> int:
        if not isinstance(pgn, str) or not pgn:
            return 0
        nums = re.findall(r"\b(\d+)\.", pgn)
        if nums:
            return int(nums[-1])
        tokens = [t for t in pgn.split() if not t.endswith('.')]
        return max(0, len(tokens) // 2)

    return df[col].apply(_count)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['WhiteElo'] = pd.to_numeric(out.get('WhiteElo'), errors='coerce')
    out['BlackElo'] = pd.to_numeric(out.get('BlackElo'), errors='coerce')
    out['elo_diff'] = out['WhiteElo'] - out['BlackElo']
    out['game_len'] = compute_full_moves(out)

    res = out.get('Result', '').fillna('')
    out['white_win'] = (res == '1-0').astype(int)
    out['black_win'] = (res == '0-1').astype(int)
    out['draw'] = (res == '1/2-1/2').astype(int)
    return out


# ---------- MIRRORING FEATURES ----------


_MOVE_NUM_RE = re.compile(r"\s*\d+\.(?:\.\.)?\s*")  # strip "1." and "1..." etc.


def _clean_an(an: str) -> str:
    if not isinstance(an, str):
        return ""
    # remove move numbers like "1." or "12..."
    s = _MOVE_NUM_RE.sub(" ", an)
    # remove result tokens
    s = s.replace("1-0", " ").replace("0-1", " ").replace("1/2-1/2", " ").replace("*", " ")
    return " ".join(s.split())


def parse_san_to_plies(an: str) -> list[str]:
    """Tokenize AN (SAN movetext) into plies: ['e4','e5','Nf3','Nf6', ...]."""
    return _clean_an(an).split()


_SQUARE_RE = re.compile(r"([a-h][1-8])")


def _canon_piece_and_dest(san: str) -> Optional[Tuple[str, str]]:
    """
    Convert a SAN token to canonical (Piece, destination-square).
    Piece in {'K','Q','R','B','N','P'}. For castles returns ('K','O-O') / ('K','O-O-O').
    Ignores checks, mates, captures, promotions, disambiguation — keeps only piece letter + final square.
    """
    san = san.replace("+", "").replace("#", "")
    if san in ("O-O", "0-0"):
        return "K", "O-O"
    if san in ("O-O-O", "0-0-0"):
        return "K", "O-O-O"

    piece = "P"
    if san and san[0] in "KQRBN":
        piece = san[0]

    # find last destination square in token
    m_all = _SQUARE_RE.findall(san)
    if not m_all:
        return None
    dest = m_all[-1]
    return piece, dest


def _mirror_white_dest_to_black(dest_white: str) -> str:
    """
    Mirror a destination square from White's move to Black's 'copied' move by
    flipping the rank: r -> 9-r (keep the file the same). e4 -> e5, f3 -> f6.
    """
    file_, rank = dest_white[0], int(dest_white[1])
    return f"{file_}{9 - rank}"


def _is_pair_mirrored(white_san: str, black_san: str) -> bool:
    """Compare a (W,B) ply pair in canonical form under our mirroring rule."""
    cw = _canon_piece_and_dest(white_san)
    cb = _canon_piece_and_dest(black_san)
    if cw is None or cb is None:
        return False

    # handle castles explicitly (K O-O ↔ K O-O, same for O-O-O)
    if cw[1] in ("O-O", "O-O-O"):
        return cb[0] == "K" and cb[1] == cw[1]

    # same piece type, mirrored destination rank, same file
    w_piece, w_dst = cw
    b_piece, b_dst = cb
    if w_piece != b_piece:
        return False
    return b_dst == _mirror_white_dest_to_black(w_dst)


def longest_mirrored_prefix(plies: list[str], max_pairs: int = 30) -> int:
    """
    Return the length (in plies) of the longest strict mirrored prefix:
    e.g., ['e4','e5','Nf3','Nf6','Nc3','Nc6','a3','a6','h3','h6'] -> 10
    """
    k_pairs = min(len(plies) // 2, max_pairs)
    length = 0
    for i in range(k_pairs):
        w, b = plies[2 * i], plies[2 * i + 1]
        if not _is_pair_mirrored(w, b):
            break
        length += 2
    return length


# --------- STREAMING HELPERS (per-row mirroring) ----------
def compute_mirror_fields_from_an(an: str, ks=(4, 6, 8)) -> dict:
    """
    Return {'mirror_prefix_len': int, 'is_mirror_4':0/1, ...} for a single AN string.
    Fast and GC-friendly for streaming.
    """
    plies = parse_san_to_plies(an)
    mlen = longest_mirrored_prefix(plies)
    out = {"mirror_prefix_len": mlen}
    for k in ks:
        out[f"is_mirror_{k}"] = int(mlen >= k)
    return out


# --------- /STREAMING HELPERS ----------


# --- in src/preprocessing.py ---

def mirrored_segments(plies: list[str]) -> list[tuple[int, int]]:
    """
    Return a list of mirrored segments across the whole game.
    Each item = (start_plies_index, length_in_plies).
    A mirrored 'pair' (W,B) counts as 2 plies.
    """
    segs = []
    cur_start = None
    cur_len = 0
    n_pairs = len(plies) // 2

    for i in range(n_pairs):
        w, b = plies[2 * i], plies[2 * i + 1]
        if _is_pair_mirrored(w, b):
            if cur_len == 0:
                cur_start = 2 * i  # start index in plies
            cur_len += 2  # add 2 plies for each mirrored pair
        else:
            if cur_len > 0:
                segs.append((cur_start, cur_len))
                cur_start, cur_len = None, 0

    if cur_len > 0:
        segs.append((cur_start, cur_len))
    return segs


def add_mirroring_features(df: pd.DataFrame, ks=(4, 6, 8)) -> pd.DataFrame:
    plies = df["PGN"].apply(parse_san_to_plies)
    # existing prefix-only feature
    df["mirror_prefix_len"] = plies.apply(longest_mirrored_prefix)

    # segments anywhere in the game
    segs = plies.apply(mirrored_segments)
    df["mirror_max_streak_len"] = segs.apply(lambda L: max((l for _, l in L), default=0))
    df["mirror_total_len"] = segs.apply(lambda L: sum(l for _, l in L))
    df["mirror_any_midgame"] = segs.apply(lambda L: int(any(start > 0 for start, _ in L)))

    # keep k-threshold flags (prefix-based)
    for k in ks:
        df[f"is_mirror_{k}"] = (df["mirror_prefix_len"] >= k).astype(int)
    return df
