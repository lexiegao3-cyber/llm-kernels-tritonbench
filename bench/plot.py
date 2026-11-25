"""Plot a CSV from bench_rmsnorm.py: bandwidth vs token count, one panel per hidden size.

Usage: python bench/plot.py results/rmsnorm_bf16_*.csv
"""

import csv
import sys
from collections import defaultdict

import matplotlib.pyplot as plt


def main(path: str):
    data = defaultdict(lambda: defaultdict(list))  # H -> backend -> [(M, gbps)]
    with open(path) as f:
        for row in csv.DictReader(f):
            data[int(row["H"])][row["backend"]].append((int(row["M"]), float(row["gbps"])))

    hs = sorted(data)
    fig, axes = plt.subplots(1, len(hs), figsize=(6 * len(hs), 4), squeeze=False)
    for ax, H in zip(axes[0], hs):
        for backend, pts in sorted(data[H].items()):
            pts.sort()
            ax.plot([m for m, _ in pts], [g for _, g in pts], marker="o", label=backend)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("tokens (M)")
        ax.set_ylabel("achieved bandwidth (GB/s)")
        ax.set_title(f"hidden = {H}")
        ax.grid(alpha=0.3)
        ax.legend()
    out = path.rsplit(".", 1)[0] + ".png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
