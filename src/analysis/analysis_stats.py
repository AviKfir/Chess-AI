from typing import Dict
import pandas as pd
import numpy as np
from collections import defaultdict
from src.analysis.preprocessing import compute_mirror_fields_from_an


def game_length_summary(df: pd.DataFrame) -> dict:
    s = df['game_len'].dropna()
    return {
        "mean": float(s.mean()),       # average length
        "median": float(s.median()),   # middle value
        "p90": float(s.quantile(0.9))  # 90th percentile
    }


def win_rates(df: pd.DataFrame) -> Dict[str, float]:
    total = len(df)
    if total == 0:
        return {"total": 0, "white_pct": 0.0, "black_pct": 0.0, "draw_pct": 0.0}

    w = (df['white_win'] == 1).sum()
    b = (df['black_win'] == 1).sum()
    d = (df['draw'] == 1).sum()

    return {
        "total": int(total),
        "white_pct": float(round((w / total) * 100, 2)),
        "black_pct": float(round((b / total) * 100, 2)),
        "draw_pct": float(round((d / total) * 100, 2)),
    }


def win_rates_stream(reader) -> Dict[str, float]:
    total = w = b = d = 0
    for chunk in reader:
        res = chunk.get("Result").fillna("")
        n = len(chunk)
        total += n
        w += (res == "1-0").sum()
        b += (res == "0-1").sum()
        d += (res == "1/2-1/2").sum()

    if total == 0:
        return {"total": 0, "white_pct": 0.0, "black_pct": 0.0, "draw_pct": 0.0}

    return {
        "total": int(total),
        "white_pct": float(round((w / total) * 100, 2)),
        "black_pct": float(round((b / total) * 100, 2)),
        "draw_pct": float(round((d / total) * 100, 2)),
    }


# ---------- MIRROR ANALYSIS ----------

def _white_win_flag(result: str) -> int:
    if result == "1-0":
        return 1
    if result == "0-1":
        return 0
    return 0  # treat draws as 0 for white-win rate


def _elo_bin(avg_elo: float) -> str:
    if pd.isna(avg_elo):
        return "unknown"
    if avg_elo < 1200: return "Beginner"
    if avg_elo < 1800: return "Intermediate"
    if avg_elo < 2200: return "Advanced"
    return "Expert"


def mirror_win_rates(df: pd.DataFrame, k: int = 4) -> dict:
    """
    Returns dict with white-win rate for mirrored vs non-mirrored games.
    Needs columns: 'Result', f'is_mirror_{k}'
    """
    x = df.copy()
    if "WhiteElo" in x.columns and "BlackElo" in x.columns:
        x["avg_elo"] = (pd.to_numeric(x["WhiteElo"], errors="coerce") +
                        pd.to_numeric(x["BlackElo"], errors="coerce")) / 2
    x["white_win"] = x["Result"].apply(_white_win_flag)

    g = x.groupby(x[f"is_mirror_{k}"])["white_win"].mean()
    return {
        "k": k,
        "white_win_non_mirror": float(g.get(0, np.nan)),
        "white_win_mirror": float(g.get(1, np.nan)),
        "delta": float(g.get(1, 0.0) - g.get(0, 0.0)),
    }


def mirror_win_rates_by_skill(df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """
    White-win rate by skill bin and mirroring indicator.
    """
    x = df.copy()
    x["white_win"] = x["Result"].apply(_white_win_flag)
    if "WhiteElo" in x.columns and "BlackElo" in x.columns:
        x["avg_elo"] = (pd.to_numeric(x["WhiteElo"], errors="coerce") +
                        pd.to_numeric(x["BlackElo"], errors="coerce")) / 2
        x["skill"] = x["avg_elo"].apply(_elo_bin)
    else:
        x["skill"] = "unknown"

    out = (
        x.groupby(["skill", x[f"is_mirror_{k}"]])["white_win"]
        .mean()
        .rename("white_win_rate")
        .reset_index()
        .rename(columns={f"is_mirror_{k}": "is_mirror"})
    )

    # enforce skill order
    order = ["Beginner", "Intermediate", "Advanced", "Expert", "unknown"]
    out["skill"] = pd.Categorical(out["skill"], categories=order, ordered=True)
    return out.sort_values(["skill", "is_mirror"]).reset_index(drop=True)

# ---------- /MIRROR ANALYSIS ----------


# ---------- MIRROR STATS (STREAMING) ----------


def mirror_stats_stream(chunks, ks=(4,6,8), hist_cap=30):
    """
    Iterate over CSV chunks (from stream_csv) and aggregate:
      - histogram of mirror_prefix_len (0..hist_cap, with >cap bucket)
      - white-win rate by skill bin and mirroring flag (k=4 by default, also others)
    Returns: {"hist": pandas.Series, "by_skill": {k: pandas.DataFrame}}
    """
    # histogram buckets
    hist = defaultdict(int)  # len -> count
    # per-skill, per-mirror sums
    sums = {k: defaultdict(int) for k in ks}   # key: (skill, mirror)-> white wins
    cnts = {k: defaultdict(int) for k in ks}   # key: (skill, mirror)-> total

    for df in chunks:
        # ensure columns exist
        if "AN" not in df.columns or "Result" not in df.columns:
            continue

        # rating features
        if "WhiteElo" in df.columns and "BlackElo" in df.columns:
            w = pd.to_numeric(df["WhiteElo"], errors="coerce")
            b = pd.to_numeric(df["BlackElo"], errors="coerce")
            avg = (w + b) / 2
            skill = avg.apply(_elo_bin)
        else:
            skill = pd.Series(["unknown"] * len(df))

        # row-wise compute mirroring features
        feats = df["AN"].apply(lambda s: compute_mirror_fields_from_an(s, ks=ks))
        mlen = feats.apply(lambda d: d["mirror_prefix_len"]).astype(int)

        # histogram update
        for L, c in mlen.value_counts().items():
            bucket = L if L <= hist_cap else (hist_cap + 1)
            hist[bucket] += int(c)

        # white-win flags
        ww = df["Result"].apply(_white_win_flag).astype(int)

        # per-k aggregation
        for k in ks:
            mir = feats.apply(lambda d, kk=k: d[f"is_mirror_{kk}"])
            for (s, m), grp in pd.DataFrame({"s": skill, "m": mir, "w": ww}).groupby(["s","m"]):
                sums[k][(s, int(m))] += int(grp["w"].sum())
                cnts[k][(s, int(m))] += int(len(grp))

    # pack outputs
    # histogram series\
    idx = list(range(0, hist_cap+1)) + [hist_cap+1]
    ser = pd.Series([hist.get(i, 0) for i in idx], index=idx, name="count")
    ser.index.name = "mirror_prefix_len"

    by_skill = {}
    for k in ks:
        rows = []
        for (s, m), n in cnts[k].items():
            wwins = sums[k].get((s, m), 0)
            rate = wwins / n if n else float("nan")
            rows.append({"skill": s, "is_mirror": m, "white_win_rate": rate, "n": n})
        by_skill[k] = pd.DataFrame(rows).sort_values(["skill","is_mirror","n"], ascending=[True,True,False])

    return {"hist": ser, "by_skill": by_skill}
# ---------- /MIRROR STATS (STREAMING) ----------
