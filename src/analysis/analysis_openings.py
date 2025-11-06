import pandas as pd


def openings_win_rates(df: pd.DataFrame, min_count: int = 1000) -> pd.DataFrame:
    # Opening may be missing; fill for grouping
    d = df.copy()
    d['Opening'] = d.get('Opening', 'Unknown').fillna('Unknown')
    agg = d.groupby('Opening').agg(
        n=('Result', 'size'),
        white_win_rate=('white_win', 'mean'),
        black_win_rate=('black_win', 'mean'),
        draw_rate=('draw', 'mean'),
        avg_len=('game_len', 'mean'),
    ).reset_index()
    return agg[agg['n'] >= min_count].sort_values('n', ascending=False)


# ========= OPENING FEATURES, CLUSTERS, RECOMMENDER =========
import pandas as pd
import numpy as np


def _expected_score_for_side(g: pd.DataFrame, side: str) -> float:
    # expected score in [0,1] (win=1, draw=0.5, loss=0).
    if side == "white":
        return g["white_win"].mean() + 0.5 * g["draw"].mean()
    else:
        return g["black_win"].mean() + 0.5 * g["draw"].mean()


def opening_feature_table(df: pd.DataFrame, min_count: int = 300) -> pd.DataFrame:
    """
    One row per opening with behavioral stats.
    Requires columns: Opening, white_win, black_win, draw, game_len, mirror_prefix_len, mirror_any_midgame (optional).
    """
    cols_needed = {"Opening", "white_win", "black_win", "draw", "game_len"}
    missing = cols_needed - set(df.columns)
    if missing:
        raise ValueError(f"opening_feature_table: missing columns {missing}")

    g = (df
         .groupby("Opening")
         .agg(n=("Opening", "size"),
              white_win_rate=("white_win", "mean"),
              black_win_rate=("black_win", "mean"),
              draw_rate=("draw", "mean"),
              avg_len=("game_len", "mean"),
              mirror_prefix_len_mean=("mirror_prefix_len", "mean") if "mirror_prefix_len" in df.columns else (
              "game_len", "size"),
              mirror_any_midgame_rate=("mirror_any_midgame", "mean") if "mirror_any_midgame" in df.columns else (
              "game_len", "size"))
         .reset_index())

    # clean up names (when mirrors not present)
    if "mirror_prefix_len_mean" in g and isinstance(g["mirror_prefix_len_mean"].iloc[0], tuple):
        g = g.drop(columns=["mirror_prefix_len_mean", "mirror_any_midgame_rate"], errors="ignore")

    g = g[g["n"] >= min_count].copy()
    # expected scores for convenience
    g["white_score"] = g["white_win_rate"] + 0.5 * g["draw_rate"]
    g["black_score"] = g["black_win_rate"] + 0.5 * g["draw_rate"]
    return g.sort_values("n", ascending=False).reset_index(drop=True)


def recommend_openings_by_skill(
        df: pd.DataFrame,
        side: str = "white",
        skill: str = "Beginner",
        min_count: int = 300,
        top_n: int = 10
) -> pd.DataFrame:
    """
    Simple content-based 'recommender':
      - filter games to a skill bin
      - aggregate per opening
      - rank by expected score for the requested side
    Needs: columns Opening, Result, white_win, black_win, draw, avg_elo (or WhiteElo+BlackElo), and 'skill' (bin).
    """
    x = df.copy()
    # make skill if not present
    if "skill" not in x.columns:
        if "WhiteElo" in x.columns and "BlackElo" in x.columns:
            w = pd.to_numeric(x["WhiteElo"], errors="coerce")
            b = pd.to_numeric(x["BlackElo"], errors="coerce")
            avg = (w + b) / 2

            def _bin(v):
                if pd.isna(v):
                    return "unknown"
                if v < 1200:
                    return "Beginner"
                if v < 1800:
                    return "Intermediate"
                if v < 2200:
                    return "Advanced"
                return "Expert"

            x["skill"] = avg.apply(_bin)
        else:
            x["skill"] = "unknown"

    x = x[x["skill"] == skill]
    ft = opening_feature_table(x, min_count=min_count)
    score_col = "white_score" if side == "white" else "black_score"
    out = (ft.sort_values([score_col, "n"], ascending=[False, False])
           .head(top_n)
           .loc[:, ["Opening", "n", "white_win_rate", "black_win_rate", "draw_rate", "avg_len", score_col]]
           .rename(columns={score_col: "expected_score"}))
    return out.reset_index(drop=True)


def cluster_openings(features: pd.DataFrame, k: int = 6, random_state: int = 0) -> pd.DataFrame:
    """
    KMeans clusters of openings using behavioral features.
    Returns the same DF with columns: cluster, pca_x, pca_y.
    """
    use_cols = [c for c in ["white_win_rate", "black_win_rate", "draw_rate", "avg_len",
                            "mirror_prefix_len_mean", "mirror_any_midgame_rate"] if c in features.columns]
    X = features[use_cols].values
    # scale and cluster
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        Xs = StandardScaler().fit_transform(X)
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto").fit(Xs)
        features = features.copy()
        features["cluster"] = km.labels_
        pca = PCA(n_components=2, random_state=random_state).fit(Xs)
        Z = pca.transform(Xs)
        features["pca_x"], features["pca_y"] = Z[:, 0], Z[:, 1]
        return features
    except Exception as e:
        print(f"[cluster_openings] scikit-learn unavailable or failed ({e}). Returning features without clusters.")
        return features.assign(cluster=-1, pca_x=np.nan, pca_y=np.nan)
# ========= /OPENING FEATURES, CLUSTERS, RECOMMENDER =========
