from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.dataio.load_data import read_sample, stream_csv
from src.analysis.preprocessing import add_engineered_features, add_mirroring_features
from src.analysis.analysis_stats import (
    win_rates,
    game_length_summary,
    win_rates_stream,
    mirror_stats_stream,
    mirror_win_rates,
    mirror_win_rates_by_skill,
)
from src.analysis.analysis_openings import (
    openings_win_rates,
    recommend_openings_by_skill,
)
from src.analysis.opening_graph import (
    transitions_prefix_to_opening,
    opening_projection_from_transitions,
    communities_and_centrality,
)
from src.analysis.evaluation import (
    permutation_test_mirror_white,
    logistic_mirror_effect,
)
from src.analysis.visualization import (
    pie_outcomes,
    hist_game_lengths,
    hist_mirror_prefix_len,
    bar_skill_vs_mirroring,
    plot_central_openings,
    heatmap_opening_by_elo,
    violin_game_length,
    survival_curve_by_skill,
)


# --------------------------- helpers --------------------------- #

def _elo_to_skill(elo: float) -> str:
    if elo < 1200:
        return "Beginner"
    elif elo < 1600:
        return "Intermediate"
    elif elo < 2000:
        return "Advanced"
    else:
        return "Expert"


def prepare_df_sample(path: str, n: int) -> pd.DataFrame:
    """Read a sample, engineer features, add 'skill' and 'ply_count'."""
    df = read_sample(path=path, n=n)
    df = add_engineered_features(df)
    df['skill'] = df['WhiteElo'].apply(_elo_to_skill)

    # ensure ply_count exists
    ply_candidates = ['ply_count', 'game_len', 'moves_count', 'fullmoves', 'ply']
    for col in ply_candidates:
        if col in df.columns:
            df['ply_count'] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            break
    else:
        print("Warning: no ply-like column found; survival curve will be skipped.")
        df['ply_count'] = pd.Series([], dtype=int)

    return df


def make_basic_figures(df: pd.DataFrame) -> None:
    """Core figures + diversified visuals for the sample."""
    wr = win_rates(df)
    gl = game_length_summary(df)
    print("Win rates (sample):", wr)
    print("Game length (sample):", gl)

    ow = openings_win_rates(df, min_count=200).reset_index(drop=True)
    print("Top openings by count (sample):")
    print(ow.head(10).to_string())

    # Basic
    pie_outcomes(wr, "figures/outcomes_pie.png")
    hist_game_lengths(df, "figures/game_len_hist.png")

    # Diversified (no Sankey/HTML)
    try:
        heatmap_opening_by_elo(
            df, agg='wr', top_n_openings=15, out_path="figures/heatmap_opening_elo.png"
        )
        violin_game_length(
            df,
            out_path_result="figures/violin_game_length_result.png",
            out_path_skill="figures/violin_game_length_skill.png",
        )
        survival_curve_by_skill(
            df, ply_col='ply_count', skill_col='skill', max_ply=200,
            out_path="figures/survival_ply_by_skill.png",
        )
    except Exception as e:
        print("Error while producing diversified figures:", e)


def opening_network_and_clusters(df: pd.DataFrame) -> None:
    # transitions
    tr_sample = transitions_prefix_to_opening(df, ply=4)
    print("\nTop prefix→Opening transitions (sample):")
    print(tr_sample.head(10).to_string(index=True))

    # projection & centrality
    G_open = opening_projection_from_transitions(tr_sample, min_shared=50, top_per_node=8)
    comms, membership, strength, pr = communities_and_centrality(G_open)
    print(
        f"\n[Network] openings: {G_open.number_of_nodes()}, "
        f"edges: {G_open.number_of_edges()}, communities: {len(comms)}"
    )

    top_strength = dict(sorted(strength.items(), key=lambda kv: kv[1], reverse=True)[:15])
    plot_central_openings(
        top_strength,
        title="Top openings by weighted degree (projection)",
        out_path="figures/opening_centrality_strength.png",
    )

    # (keep or remove this PageRank section as you prefer)
    top_pr = dict(sorted(pr.items(), key=lambda kv: kv[1], reverse=True)[:15])
    plot_central_openings(
        top_pr,
        title="Top openings by PageRank (projection)",
        out_path="figures/opening_centrality_pagerank.png",
    )

    # simple recommender
    for skill in ["Beginner", "Intermediate", "Advanced", "Expert"]:
        rec_w = recommend_openings_by_skill(df, side="white", skill=skill, min_count=80, top_n=8)
        rec_b = recommend_openings_by_skill(df, side="black", skill=skill, min_count=80, top_n=8)
        print(f"\n[Recommend] {skill} — WHITE (top by expected score):\n{rec_w.to_string(index=False)}")
        print(f"\n[Recommend] {skill} — BLACK (top by expected score):\n{rec_b.to_string(index=False)}")


def mirroring_and_eval_sample(df: pd.DataFrame) -> None:
    # mirror features + overall eval
    df = add_mirroring_features(df, ks=[4, 6, 8])

    print("\nMirroring distribution & effects (sample):")
    for k in (4, 6, 8):
        print(mirror_win_rates(df, k))

    hist_mirror_prefix_len(df, "figures/mirror_prefix_hist.png")
    mskill = mirror_win_rates_by_skill(df, k=4)
    print("\nWhite win rate by skill (mirror k=4 plies):")
    print(mskill.to_string(index=False))
    bar_skill_vs_mirroring(mskill, "figures/mirror_skill_bar.png")

    # inferential
    perm = permutation_test_mirror_white(df, k=4, reps=2000, seed=0)
    print(
        f"\n[Permutation test] k=4: Δ white-win (mirror - non) = {perm['obs_delta']:.3f}, "
        f"two-sided p = {perm['p_value']:.4f}"
    )
    logi = logistic_mirror_effect(df, k=4)
    print(
        f"[Logistic] coef(is_mirror_4) = {logi['coef_is_mirror']:.3f} "
        f"(log-odds; >0 ⇒ mirroring helps White after controls)"
    )


BASE = Path(__file__).resolve().parents[1]  # project root
FIGS = BASE / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def streaming_full(path: str, chunksize: int = 200_000) -> None:
    full_wr = win_rates_stream(stream_csv(path, chunksize=chunksize))
    print("\nWin rates (FULL, streamed):", full_wr)

    # NEW: full-dataset game-length histogram
    game_len_hist_full(path, chunksize=chunksize, out_path=str(FIGS / "game_len_hist_FULL.png"),
                       bins=40, cap_max_ply=120)

    print("\n[FULL] Computing mirroring stats by streaming...")
    ms = mirror_stats_stream(
        stream_csv(path, chunksize=chunksize), ks=(4, 6, 8), hist_cap=30)

    # White-only, sensitivity by k
    for k in (4, 6, 8):
        dfk = ms["by_skill"][k]
        print(f"\n[FULL] White win rate by skill (k={k} plies):")
        print(dfk[["skill", "is_mirror", "n", "white_win_rate"]].to_string(index=False))

        # Plot bars with sample sizes annotated
        bar_skill_vs_mirroring(dfk, out_path=FIGS / f"mirror_white_bar_FULL_k{k}.png", annotate_n=True, k=k)

    # Build the full heatmap without loading all rows
    heatmap_opening_by_elo_stream(path, chunksize=chunksize, top_n_openings=15,
                                  out_path="figures/heatmap_opening_elo_FULL.png")


def game_len_hist_full(path: str,
                       chunksize: int = 200_000,
                       out_path: str = "figures/game_len_hist_FULL.png",
                       bins: int = 40,
                       cap_max_ply: int = 120) -> None:
    import re
    # remove tokens like "12." from SAN
    _SAN_NUM_RE = re.compile(r"\b\d+\.")

    def _len_from_moves(series: pd.Series) -> pd.Series:
        # split by whitespace, remove move numbers, count tokens
        def _count_tokens(s: object) -> float:
            if not isinstance(s, str) or not s:
                return np.nan
            toks = [t for t in s.split() if not _SAN_NUM_RE.fullmatch(t)]
            return float(len(toks))

        return series.apply(_count_tokens)

    bin_edges = np.linspace(0, cap_max_ply, bins + 1).astype(float)
    bin_counts = np.zeros(bins, dtype=np.int64)

    ply_candidates = ["ply_count", "game_len", "moves_count", "fullmoves", "ply"]
    moves_candidates = ["moves_san", "AN", "moves", "pgn", "moves_list", "moves_san_str"]

    found_any = False  # for a helpful warning

    for chunk in stream_csv(path, chunksize=chunksize):
        # 1) Try a numeric ply column
        ply_col = next((c for c in ply_candidates if c in chunk.columns), None)
        if ply_col is not None:
            s = pd.to_numeric(chunk[ply_col], errors="coerce")
        else:
            # 2) Fallback: derive from a moves/PGN column
            mv_col = next((c for c in moves_candidates if c in chunk.columns), None)
            if mv_col is None:
                continue
            s = _len_from_moves(chunk[mv_col])

        s = s.dropna().astype(float)
        if s.empty:
            continue

        found_any = True
        s = s.clip(lower=0, upper=(cap_max_ply - 1e-9))

        h, _ = np.histogram(s.values, bins=bin_edges)
        bin_counts += h

    from src.analysis.visualization import plot_hist_from_edges_counts
    if not found_any:
        print("[game_len_hist_full] No length-like or moves columns found in any chunk; produced empty histogram.")
    plot_hist_from_edges_counts(bin_edges, bin_counts, out_path=out_path)


from collections import Counter, defaultdict


def _bucket_labels(s: pd.Series, bins: list[int]) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(-1).astype(float).clip(lower=0)
    iv = pd.cut(s, bins=bins, include_lowest=True)
    return iv.apply(lambda b: f"{max(0, int(b.left))}-{int(b.right)}")


def player_counts_by_whiteelo_bucket_full(
        path: str,
        chunksize: int = 200_000,
        bins: list[int] | None = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame with:
      - elo_bucket
      - games_in_bucket
      - unique_white_players_in_bucket
    Buckets are by WhiteElo, streamed over the full dataset.
    """
    if bins is None:
        bins = [0, 1000, 1400, 1800, 2200, 3000]

    game_counts = Counter()
    players_by_bucket: dict[str, set] = defaultdict(set)

    # Accept common column names
    white_col_cands = ["White", "white", "WhitePlayer", "white_player", "WhiteUser"]
    white_elo_cands = ["WhiteElo", "white_elo", "whiteRating", "WhiteRating"]

    for chunk in stream_csv(path, chunksize=chunksize):
        # find columns present in this chunk
        wcol = next((c for c in white_col_cands if c in chunk.columns), None)
        wecol = next((c for c in white_elo_cands if c in chunk.columns), None)
        if wecol is None:
            continue  # can't bucket without WhiteElo

        # bucket by WhiteElo
        labels = _bucket_labels(chunk[wecol], bins)

        # games per bucket
        vc = labels.value_counts()
        for k, v in vc.items():
            game_counts[str(k)] += int(v)

        # unique white players per bucket (if we have a name/ID column)
        if wcol is not None:
            for label, name in zip(labels, chunk[wcol]):
                if pd.isna(label) or pd.isna(name):
                    continue
                players_by_bucket[str(label)].add(str(name))

    # build result table
    rows = []
    for b in sorted(game_counts.keys(),
                    key=lambda s: int(str(s).split("-")[0])):  # sort by left edge
        rows.append({
            "elo_bucket": b,
            "games_in_bucket": game_counts[b],
            "unique_white_players_in_bucket": len(players_by_bucket.get(b, set())),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


# --------------------------- subcommands --------------------------- #

def cmd_sample(args: argparse.Namespace) -> None:
    df = prepare_df_sample(args.path, args.n)
    make_basic_figures(df)
    if not args.no_network:
        opening_network_and_clusters(df)
    mirroring_and_eval_sample(df)


def cmd_full(args: argparse.Namespace) -> None:
    streaming_full(args.path, args.chunksize)
    player_counts_by_whiteelo_bucket_full(args.path, chunksize=args.chunksize)


def cmd_eval(args: argparse.Namespace) -> None:
    df = prepare_df_sample(args.path, args.n)
    mirroring_and_eval_sample(df)


def heatmap_opening_by_elo_stream(path: str, chunksize: int = 200_000,
                                  top_n_openings: int = 20,
                                  out_path: str = "figures/heatmap_opening_elo_FULL.png") -> None:
    import seaborn as sns
    import matplotlib.pyplot as plt
    from collections import defaultdict

    bins = [0, 1000, 1400, 1800, 2200, 3000]

    def is_white_win(x: object) -> int:
        if pd.isna(x): return 0
        s = str(x).strip().lower()
        return int(s.startswith("1-0") or s.startswith("white"))

    # global counters
    cnt = defaultdict(int)  # (opening, bucket) -> count
    wins = defaultdict(int)  # (opening, bucket) -> white-win count
    total_by_open = defaultdict(int)

    for chunk in stream_csv(path, chunksize=chunksize):
        # auto-detect columns (same candidates as visualization.py)
        op_col = next((c for c in ["Opening", "opening_name", "opening", "ECO", "eco", "opening_full"] if c in chunk),
                      None)
        el_col = next(
            (c for c in ["WhiteElo", "white_elo", "white_rating", "whiteRating", "avg_elo", "elo", "rating"] if
             c in chunk), None)
        r_col = next((c for c in ["Result", "result", "outcome", "winner", "result_str"] if c in chunk), None)
        if not op_col or not el_col or not r_col:
            print("[heatmap_full] missing columns in this chunk; skipping it")
            continue

        chunk = chunk[[op_col, el_col, r_col]].dropna()
        chunk[el_col] = pd.to_numeric(chunk[el_col], errors="coerce")
        chunk = chunk.dropna(subset=[el_col])

        # bucket Elo
        # chunk["elo_bucket"] = pd.cut(chunk[el_col], bins=bins, include_lowest=True).astype(str)
        chunk["elo_bucket"] = pd.cut(chunk[el_col], bins=bins, include_lowest=True)
        chunk["elo_bucket"] = chunk["elo_bucket"].apply(
            lambda b: f"{max(0, int(b.left))}-{int(b.right)}"
        )

        chunk["white_win"] = chunk[r_col].map(is_white_win)

        g = chunk.groupby([op_col, "elo_bucket"], observed=False)["white_win"].agg(["count", "sum"]).reset_index()
        for _, row in g.iterrows():
            key = (row[op_col], row["elo_bucket"])
            c = int(row["count"])
            s = int(row["sum"])
            cnt[key] += c
            wins[key] += s
            total_by_open[row[op_col]] += c

    # keep top openings by volume
    top = set(sorted(total_by_open, key=total_by_open.get, reverse=True)[:top_n_openings])

    # build pivot of white-win rate
    rows = []
    for (op, bucket), c in cnt.items():
        if op in top and c > 0:
            rows.append({"Opening": op, "elo_bucket": bucket, "wr": wins[(op, bucket)] / c})

    if not rows:
        print("[heatmap_full] nothing to plot")
        return

    small = pd.DataFrame(rows)
    pivot = small.pivot(index="Opening", columns="elo_bucket", values="wr").fillna(0.0)

    plt.figure(figsize=(10, max(4, int(len(pivot) * 0.4))))
    ax = sns.heatmap(pivot, cmap="viridis", linewidths=0.5, linecolor="black",
                     cbar_kws={"label": "white-win-rate"})
    ax.set_title("White win-rate by Opening & Elo bucket (FULL)")
    ax.set_xlabel("Elo bucket")
    plt.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGS / "heatmap_opening_elo_FULL.png", dpi=200)
    print(f"[saved] {out_path}")


# --------------------------- CLI --------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chess project pipelines")
    sub = parser.add_subparsers(dest="cmd")  # allow default

    # sample
    p_sample = sub.add_parser("sample", help="Run sample pipeline: figures, networks, eval")
    p_sample.add_argument("--path", default="data/chess_games.csv")
    p_sample.add_argument("-n", "--n", type=int, default=20_000)
    p_sample.add_argument("--no-network", action="store_true", help="Skip opening network + clustering")
    p_sample.set_defaults(func=cmd_sample)

    # full
    p_full = sub.add_parser("full", help="Run streaming stats on full dataset")
    p_full.add_argument("--path", default="data/chess_games.csv")
    p_full.add_argument("--chunksize", type=int, default=200_000)
    p_full.set_defaults(func=cmd_full)

    # eval
    p_eval = sub.add_parser("eval", help="Run evaluation & calibration on a sample")
    p_eval.add_argument("--path", default="data/chess_games.csv")
    p_eval.add_argument("-n", "--n", type=int, default=20_000)
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()

    # Default to 'sample' if no subcommand provided
    if not hasattr(args, "func"):
        print("[run_all] No subcommand provided; defaulting to: sample")
        args = parser.parse_args(["sample"])

    args.func(args)


if __name__ == "__main__":
    main()
