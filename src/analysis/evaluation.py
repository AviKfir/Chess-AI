import pandas as pd
import numpy as np


def permutation_test_mirror_white(
        df: pd.DataFrame,
        k: int = 4,
        reps: int = 2000,
        seed: int = 0
) -> dict:
    """
    Stratified permutation test of the white win-rate delta between mirrored and non-mirrored games.
    Stratify by skill bin to reduce confounding.
    Returns: {'obs_delta': float, 'p_value': float}
    """
    rng = np.random.default_rng(seed)
    x = df.copy()
    x["white_win"] = (x["Result"] == "1-0").astype(int)

    # ensure skill bin
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

    x["mirror"] = (x[f"is_mirror_{k}"] == 1).astype(int)

    def _delta(df_):
        # weighted delta across strata
        parts = []
        for s, g in df_.groupby("skill"):
            if g["mirror"].nunique() < 2:  # no contrast in this stratum
                continue
            m = g[g["mirror"] == 1]["white_win"].mean()
            nm = g[g["mirror"] == 0]["white_win"].mean()
            parts.append((len(g), m - nm))
        if not parts:
            return np.nan
        # weight by stratum size
        n_tot = sum(n for n, _ in parts)
        return sum(n * d for n, d in parts) / n_tot

    obs = _delta(x)

    # permutations: shuffle mirror label WITHIN each stratum
    deltas = []
    for _ in range(reps):
        xx = x.copy()
        xx["mirror"] = (xx.groupby("skill")["mirror"]
                        .transform(lambda s: rng.permutation(s.values)))
        deltas.append(_delta(xx))
    deltas = np.array([d for d in deltas if not pd.isna(d)])
    if len(deltas) == 0:
        return {"obs_delta": float(obs), "p_value": float("nan")}

    # two-sided p-value
    p = float((np.sum(np.abs(deltas) >= abs(obs)) + 1) / (len(deltas) + 1))
    return {"obs_delta": float(obs), "p_value": p}


def logistic_mirror_effect(df: pd.DataFrame, k: int = 4):
    """
    Simple logistic regression: WhiteWin ~ is_mirror_k + avg_elo + elo_diff
    Returns dict with coefficient for is_mirror_k.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as e:
        print(f"[logistic_mirror_effect] scikit-learn unavailable ({e}).")
        return {"coef_is_mirror": float("nan")}

    x = df.copy()
    x["WhiteWin"] = (x["Result"] == "1-0").astype(int)
    x["is_mirror"] = (x[f"is_mirror_{k}"] == 1).astype(int)
    if "WhiteElo" in x.columns and "BlackElo" in x.columns:
        w = pd.to_numeric(x["WhiteElo"], errors="coerce")
        b = pd.to_numeric(x["BlackElo"], errors="coerce")
        x["avg_elo"] = (w + b) / 2
        x["elo_diff"] = w - b
    else:
        x["avg_elo"] = 1500.0
        x["elo_diff"] = 0.0

    X = x[["is_mirror", "avg_elo", "elo_diff"]].fillna(0.0).values
    y = x["WhiteWin"].values

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    coef = float(model.coef_[0][0])  # coefficient on is_mirror
    return {"coef_is_mirror": coef}
