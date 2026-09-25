from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from .io import stable_seed

CONFOUNDS = ["sem_dist", "nll_mean", "n_tokens", "edit_ratio", "zipf_mean", "nli_min"]


def matrix(rows):
    return np.asarray([[r[k] for k in CONFOUNDS] for r in rows], dtype=float)


class ResidualScorer:
    """A fixed degree-two ridge nuisance model fitted ONLY on dev parents."""
    def fit(self, rows, alpha=10.0):
        if not rows or any(r["split"] != "dev" for r in rows):
            raise ValueError("residual fit requires nonempty dev-only rows")
        self.scaler = StandardScaler().fit(matrix(rows))
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        x = self.poly.fit_transform(self.scaler.transform(matrix(rows)))
        self.model = Ridge(alpha=alpha).fit(x, [r["behavior_dist"] for r in rows])
        return self

    def score(self, rows):
        if not rows:
            return np.empty(0)
        x = self.poly.transform(self.scaler.transform(matrix(rows)))
        return np.array([r["behavior_dist"] for r in rows]) - self.model.predict(x)

    def describe(self):
        return {"fit_split": "dev", "confounds": CONFOUNDS,
                "mean": self.scaler.mean_.tolist(), "scale": self.scaler.scale_.tolist(),
                "degree": 2, "alpha": self.model.alpha,
                "coef": self.model.coef_.tolist(), "intercept": float(self.model.intercept_)}


def eligible(row, cfg):
    f = cfg["filter"]
    return (row["sem_dist"] <= 1 - f["min_cosine"]
            and row["nli_min"] >= f["min_entailment"]
            and f["min_tokens"] <= row["n_tokens"] <= f["max_tokens"]
            and row["edit_ratio"] <= f["max_edit_ratio"]
            and row["ascii_printable"])


def select_groups(rows, residuals, cfg, parent_id):
    """Pair within parent under absolute calipers. Fail, never relax calipers.

    Outliers = top fraction by residual, controls = all remaining candidates.
    Assignment maximizes number of valid pairs first, then minimizes distance.
    Every reported method receives exactly keep candidates or all fail together.
    """
    n, keep = len(rows), cfg["selection"]["keep"]
    if n < 2 * keep:
        return None, {"reason": "too_few_eligible", "eligible": n, "pairs": 0}
    order = np.argsort(-np.asarray(residuals), kind="stable")
    top_n = max(keep, int(np.ceil(n * cfg["selection"]["outlier_fraction"])))
    top_n = min(top_n, n - keep)
    high, low = order[:top_n], order[top_n:]
    x = matrix(rows)
    calipers = np.array([cfg["selection"]["calipers"][k] for k in CONFOUNDS])
    if np.any(calipers <= 0):
        raise ValueError("all calipers must be positive")
    diff = np.abs(x[high, None, :] - x[None, low, :]) / calipers
    valid = np.all(diff <= 1, axis=-1)
    cost = diff.mean(axis=-1)
    # A penalty larger than the total possible valid assignment cost.
    a, b = linear_sum_assignment(np.where(valid, cost, n + 1.0))
    pairs = [(int(high[i]), int(low[j]), float(cost[i, j]))
             for i, j in zip(a, b) if valid[i, j]]
    pairs.sort(key=lambda p: (p[2], p[0], p[1]))
    if len(pairs) < keep:
        return None, {"reason": "caliper_match_failed", "eligible": n, "pairs": len(pairs)}
    pairs = pairs[:keep]
    oi, ci = [p[0] for p in pairs], [p[1] for p in pairs]
    rng = np.random.default_rng(stable_seed(cfg["seed"], parent_id, "random-control"))
    random = rng.choice(n, keep, replace=False).tolist()
    nearest = sorted(range(n), key=lambda i: (rows[i]["sem_dist"], rows[i]["candidate_id"]))[:keep]
    # These sets are selected without knowing any projection, key, or target bit.
    groups = {"outlier": oi, "matched": ci, "random": random, "nearest": nearest}
    delta = x[oi].mean(axis=0) - x[ci].mean(axis=0)
    diagnostic = {"reason": "ok", "eligible": n, "pairs": len(pairs),
                  "outlier_residual_mean": float(np.mean(np.asarray(residuals)[oi])),
                  "matched_residual_mean": float(np.mean(np.asarray(residuals)[ci])),
                  "outlier_minus_matched": dict(zip(CONFOUNDS, delta.tolist()))}
    return groups, diagnostic
