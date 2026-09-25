from pathlib import Path
import csv


def plot(run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .io import read_json
    run = Path(run)
    cfg = read_json(run / "manifest.json")["config"]
    primary = cfg["evaluation"]["primary_margin"]
    with open(run / "summary.csv", newline="") as f: rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, split in zip(axes, ("test", "ood")):
        for method, color in zip(("outlier", "matched", "random", "nearest"), ("#476f95", "#bd785c", "#78917a", "#978ba5")):
            rr = [r for r in rows if r["split"] == split and r["method"] == method and float(r["margin"]) == primary]
            ax.plot([int(r["bits"]) for r in rr], [float(r["coverage_all"]) for r in rr], marker="o", color=color, label=method)
        ax.set(title=split, xlabel="Bits per sentence", ylim=(0, 1.02))
        ax.grid(alpha=.18); ax.spines[["right", "top"]].set_visible(False)
    axes[0].set_ylabel("Target-symbol coverage (all parents)")
    axes[1].legend(frameon=False, fontsize=8)
    synthetic = read_json(run / "feature_signature.json")["backend"] == "synthetic"
    fig.suptitle("SYNTHETIC TEST — NOT LLM RESULTS" if synthetic else f"Fixed margin = {primary}", fontsize=10)
    fig.tight_layout(); fig.savefig(run / "coverage.png", dpi=180); fig.savefig(run / "coverage.pdf")
    plt.close(fig)
