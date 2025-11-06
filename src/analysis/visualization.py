from __future__ import annotations

import warnings
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

# Optional dependency
try:
    import seaborn as sns

    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

# ---------------------------- Globals & helpers ---------------------------- #
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def _ensure_parent(p: Path | str) -> Path:
    path = Path(p)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_fig(p: Path | str) -> None:
    path = _ensure_parent(p)
    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[saved] {path}")


def _find_col(df: pd.DataFrame, candidates: list[str | None]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c and c in cols:
            return c
    return None


def _detect_common_columns(df: pd.DataFrame) -> dict[str, Optional[str]]:
    return {
        "opening_col": _find_col(df, [
            "Opening", "opening_name", "opening", "ECO", "eco", "opening_full",
        ]),
        "white_elo_col": _find_col(df, [
            "WhiteElo", "white_elo", "white_rating", "whiteRating", "WhiteRating",
        ]),
        "moves_col": _find_col(df, [
            "moves_san", "AN", "moves", "pgn", "moves_list", "moves_san_str",
        ]),
        "ply_col": _find_col(df, [
            "ply_count", "game_len", "moves_count", "fullmoves", "ply",
        ]),
        "result_col": _find_col(df, [
            "Result", "result", "outcome", "winner", "result_str",
        ]),
        "date_col": _find_col(df, [
            "UTCDate", "date", "created_at", "timestamp", "time",
        ]),
    }


def _is_white_win(x: object) -> int:
    if pd.isna(x):
        return 0
    s = str(x).strip().lower()
    return int(s.startswith("1-0") or s.startswith("white"))


# ------------------------------ Basic plots ------------------------------ #

def pie_outcomes(stats: Mapping[str, float], out_path: str | Path = FIG_DIR / "outcomes_pie.png") -> None:
    labels = ["White", "Black", "Draw"]
    vals = [stats.get("white_pct", 0.0), stats.get("black_pct", 0.0), stats.get("draw_pct", 0.0)]
    plt.figure(figsize=(5, 5))
    plt.pie(vals, labels=labels, autopct="%1.1f%%")
    plt.title("Overall Outcomes")
    _save_fig(out_path)


def hist_game_lengths(df: pd.DataFrame, out_path: str | Path = FIG_DIR / "game_len_hist.png", bins: int = 40) -> None:
    detected = _detect_common_columns(df)
    ply_col = detected["ply_col"] or "game_len"
    if ply_col not in df.columns:
        warnings.warn(f"hist_game_lengths: cannot find length column; tried {detected['ply_col']} and 'game_len'.")
        return
    s = pd.to_numeric(df[ply_col], errors="coerce").dropna()
    plt.figure(figsize=(7, 4))
    plt.hist(s, bins=bins)
    plt.xlabel("Moves")
    plt.ylabel("#Games")
    plt.title("Game Length Distribution")
    _save_fig(out_path)


# plot histogram from precomputed bin counts
def plot_hist_from_edges_counts(
        bin_edges: np.ndarray,
        bin_counts: np.ndarray,
        out_path: str | Path = FIG_DIR / "game_len_hist_FULL.png",
        title: str = "Game Length Distribution (FULL)",
        xlabel: str = "Moves",
        ylabel: str = "#Games",
) -> None:
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths = (bin_edges[1:] - bin_edges[:-1])

    plt.figure(figsize=(9, 4))
    plt.bar(centers, bin_counts, width=widths, align="center")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    _save_fig(out_path)


# ----------------------------- Mirroring plots ---------------------------- #

def hist_mirror_prefix_len(df: pd.DataFrame, out_path: str | Path = FIG_DIR / "mirror_prefix_hist.png",
                           bins: int = 20) -> None:
    if "mirror_prefix_len" not in df.columns:
        warnings.warn("hist_mirror_prefix_len: 'mirror_prefix_len' not in DataFrame; skipping.")
        return
    s = pd.to_numeric(df["mirror_prefix_len"], errors="coerce").dropna().astype(int)
    plt.figure(figsize=(6, 4))
    plt.hist(s, bins=bins)
    plt.xlabel("Mirrored prefix length (plies)")
    plt.ylabel("Games")
    plt.title("Distribution of Mirrored Opening Length")
    _save_fig(out_path)


def bar_skill_vs_mirroring(
        mdf: pd.DataFrame,
        out_path: str | Path = FIG_DIR / "mirror_skill_bar.png",
        order: list[str] | None = None,
        annotate_n: bool = False,
        k: int | None = None,
) -> None:
    if order is None:
        order = ["Beginner", "Intermediate", "Advanced", "Expert", "unknown"]

    mdf = mdf.copy()
    mdf["skill"] = pd.Categorical(mdf["skill"], categories=order, ordered=True)
    pv = (
        mdf.pivot(index="skill", columns="is_mirror", values="white_win_rate").rename(
            columns={0: "non_mirror", 1: "mirror"}).reindex(order)
    )

    plt.figure(figsize=(6, 4))
    ax = pv.plot(kind="bar", ax=plt.gca())
    ax.set_ylabel("White win rate")
    title = "White Win Rate: Mirroring vs Non-Mirroring by Skill"
    if k is not None:
        title += f" (k={k} plies)"
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    if annotate_n and "n" in mdf.columns:
        count_map = {(r["skill"], r["is_mirror"]): r["n"] for _, r in mdf.iterrows()}
        for i, skill in enumerate(pv.index):
            for j, col in enumerate(pv.columns):
                val = pv.loc[skill, col]
                if pd.isna(val):
                    continue
                n = count_map.get((skill, 0 if col == "non_mirror" else 1))
                if n is not None:
                    ax.text(i + (j - 0.5) * 0.35, float(val) + 0.005, f"n={n}",
                            ha="center", va="bottom", fontsize=8)
    _save_fig(out_path)


def hist_from_series(hist_series: pd.Series, out_path: str | Path = FIG_DIR / "mirror_prefix_hist_FULL.png") -> None:
    plt.figure(figsize=(6, 4))
    hist_series.plot(kind="bar")
    plt.xlabel("Mirrored prefix length (plies)\n(last bin = > cap)")
    plt.ylabel("Games")
    plt.title("Distribution of Mirrored Opening Length (FULL)")
    _save_fig(out_path)


# ----------------------------- Clusters & centrality ----------------------------- #

def plot_central_openings(
        centrality: Mapping[str, float],
        top_n: int = 15,
        title: str = "Top openings by centrality",
        out_path: str | Path = FIG_DIR / "opening_centrality.png",
) -> None:
    items = sorted(centrality.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    idx = list(range(len(labels)))

    plt.figure(figsize=(10, 6))
    plt.barh(idx[::-1], list(vals)[::-1])
    plt.yticks(idx[::-1], labels[::-1])
    plt.title(title)
    _save_fig(out_path)


# ----------------------------- Reliability / calibration ----------------------------- #

def plot_reliability_curves(
        curves: Mapping[str, Mapping[str, list[float] | float]],
        out_path: str | Path = FIG_DIR / "reliability.png",
        title: str = "Reliability (Calibration)",
) -> None:
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    for name, d in curves.items():
        mp = np.asarray(d.get("mean_pred", []), dtype=float)
        fp = np.asarray(d.get("frac_pos", []), dtype=float)
        m = ~np.isnan(mp) & ~np.isnan(fp)
        if m.any():
            ece = float(d.get("ece", np.nan))
            label = f"{name} (ECE={ece:.3f})" if np.isfinite(ece) else name
            plt.plot(mp[m], fp[m], marker="o", label=label)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(title)
    plt.legend()
    _save_fig(out_path)


# ----------------------------- Heatmap: Opening × Elo ----------------------------- #

def heatmap_opening_by_elo(
        df: pd.DataFrame,
        opening_col: Optional[str] = None,
        elo_col: Optional[str] = None,
        result_col: Optional[str] = None,
        top_n_openings: int = 20,
        bins: list[int] | None = None,
        agg: str = "count",  # 'count' or 'wr'
        out_path: str | Path = FIG_DIR / "heatmap_opening_elo.png",
) -> None:
    if not HAS_SEABORN:
        warnings.warn("heatmap_opening_by_elo requires seaborn; skipping.")
        return

    if bins is None:
        bins = [0, 1000, 1400, 1800, 2200, 3000]

    df = df.copy()
    detected = _detect_common_columns(df)

    op_col = opening_col or detected["opening_col"] or _find_col(df,
                                                                 ["opening_name", "opening", "ECO", "eco", "Opening"])
    el_col = elo_col or detected["white_elo_col"] or _find_col(
        df, ["WhiteElo", "white_elo", "white_rating", "whiteRating", "avg_elo", "elo", "rating"])
    r_col = result_col or detected["result_col"] or _find_col(df, ["Result", "result", "outcome", "winner"])

    missing: list[str] = []
    if op_col is None:
        missing.append("opening")
    if el_col is None:
        missing.append("elo")
    if agg == "wr" and r_col is None:
        missing.append("result")
    if missing:
        warnings.warn(f"heatmap_opening_by_elo: missing columns: {missing}. df.columns={list(df.columns)}")
        return

    required = [op_col, el_col] if agg != "wr" else [op_col, el_col, r_col]
    df = df.dropna(subset=required)

    top_open = df[op_col].value_counts().nlargest(top_n_openings).index
    df = df[df[op_col].isin(top_open)]  # type: ignore[index]

    df[el_col] = pd.to_numeric(df[el_col], errors="coerce")
    df = df.dropna(subset=[el_col])
    bins[0] = 0
    df["elo_bucket"] = pd.cut(df[el_col], bins=bins, include_lowest=True)
    df["elo_bucket"] = df["elo_bucket"].apply(lambda b: f"{int(b.left)}-{int(b.right)}")

    if agg == "count":
        pivot = df.pivot_table(index=op_col, columns="elo_bucket", values=el_col,
                               aggfunc="count", observed=False).fillna(0)
        cbar_label = "count"
        title = "Opening counts by Elo bucket"
    else:
        df["white_win"] = df[r_col].apply(_is_white_win)  # type: ignore[index]
        pivot = df.pivot_table(index=op_col, columns="elo_bucket", values="white_win",
                               aggfunc="mean", observed=False).fillna(0)
        cbar_label = "white-win-rate"
        title = "White win-rate by Opening & Elo bucket"

    plt.figure(figsize=(10, max(4, int(len(pivot) * 0.4))))
    ax = sns.heatmap(
        pivot,
        fmt="g",
        cmap="viridis",
        linewidths=0.5,
        linecolor="black",
        cbar_kws={"label": cbar_label},
    )
    ax.set_title(title)
    ax.set_xlabel("Elo bucket")
    _save_fig(out_path)


# ----------------------------- Violin plots ----------------------------- #

def violin_game_length(
        df: pd.DataFrame,
        ply_col: Optional[str] = None,
        result_col: Optional[str] = None,
        skill_col: str = "skill",
        out_path_result: str | Path = FIG_DIR / "violin_game_length_result.png",
        out_path_skill: str | Path = FIG_DIR / "violin_game_length_skill.png",
) -> None:
    detected = _detect_common_columns(df)
    ply_cands: list[str | None] = [ply_col, detected["ply_col"], "ply_count", "game_len", "moves_count", "fullmoves",
                                   "ply"]
    ply_col_found = _find_col(df, ply_cands)
    if ply_col_found is None:
        warnings.warn(f"violin_game_length: missing ply-like column; tried {ply_cands}.")
        return

    df = df.copy()
    df["ply_count"] = pd.to_numeric(df[ply_col_found], errors="coerce").astype("Int64")
    df = df.dropna(subset=["ply_count"])

    if result_col is None:
        result_col = _find_col(df, ["Result", "result", "outcome", "winner"]) or ""
    if result_col in df.columns:
        def _map_res(x: object) -> str:
            s = str(x).strip()
            if s in ["1-0", "White", "white"]:
                return "White wins"
            if s in ["0-1", "Black", "black"]:
                return "Black wins"
            if s in ["1/2-1/2", "Draw", "draw", "½-½"]:
                return "Draw"
            return s

        df["result_label"] = df[result_col].apply(_map_res)

        if HAS_SEABORN:
            plt.figure(figsize=(6, 3))
            sns.violinplot(x="result_label", y="ply_count", data=df, inner="quartile")
        else:
            warnings.warn("seaborn not installed; falling back to matplotlib violinplot (no grouping).")
            plt.figure(figsize=(6, 3))
            plt.violinplot([df["ply_count"].values])
            plt.xticks([1], ["All results"])
        plt.title("Game length distribution by result")
        plt.ylabel("Moves")
        plt.xlabel("")
        _save_fig(out_path_result)

    if skill_col in df.columns:
        if HAS_SEABORN:
            plt.figure(figsize=(8, 3))
            order = ["Beginner", "Intermediate", "Advanced", "Expert"]
            sns.violinplot(x=skill_col, y="ply_count", data=df, inner="quartile", order=order)
            plt.title("Game length distribution by skill")
            plt.ylabel("Moves")
            plt.xlabel("")
            _save_fig(out_path_skill)
        else:
            warnings.warn("seaborn not installed; skipping skill violin plot.")
    else:
        warnings.warn(f"violin_game_length: skill column '{skill_col}' not found; skipping skill violin.")


# ----------------------------- Survival curve by skill ----------------------------- #

def survival_curve_by_skill(
        df: pd.DataFrame,
        ply_col: str = "ply_count",
        skill_col: str = "skill",
        max_ply: int = 200,
        out_path: str | Path = FIG_DIR / "survival_ply_by_skill.png",
) -> None:
    if ply_col not in df.columns or skill_col not in df.columns:
        warnings.warn(f"survival_curve_by_skill: columns '{ply_col}' or '{skill_col}' not found; skipping.")
        return

    d = df.dropna(subset=[ply_col, skill_col]).copy()
    d[ply_col] = pd.to_numeric(d[ply_col], errors="coerce")
    d = d.dropna(subset=[ply_col])

    skills = ["Beginner", "Intermediate", "Advanced", "Expert"]
    plt.figure(figsize=(8, 4))
    for skill in skills:
        sub = d[d[skill_col] == skill]
        total = len(sub)
        if total == 0:
            continue
        counts = sub[ply_col].value_counts().sort_index()
        idx = np.arange(1, max_ply + 1)
        surv = np.array([(counts[counts.index >= t].sum()) / total for t in idx], dtype=float)
        plt.step(idx, surv, where="post", label=skill)

    plt.xlabel("Ply (move number)")
    plt.ylabel("Fraction of games still running")
    plt.title("Survival curve by skill")
    plt.legend(title="Skill")
    plt.grid(True, alpha=0.3)
    _save_fig(out_path)


__all__ = [
    # basics
    "pie_outcomes",
    "hist_game_lengths",
    # mirroring
    "hist_mirror_prefix_len",
    "bar_skill_vs_mirroring",
    "hist_from_series",
    # clusters & centrality
    "plot_central_openings",
    # reliability
    "plot_reliability_curves",
    # opening × elo
    "heatmap_opening_by_elo",
    # violin
    "violin_game_length",
    # survival
    "survival_curve_by_skill",
]
