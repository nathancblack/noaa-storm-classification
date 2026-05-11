"""Build reports/figures/class_distribution.png from the combined-dataset
top-10 EVENT_TYPE counts printed by the data-loading cell of weather.ipynb.

Hard-coded counts (rather than re-loading the CSVs) so the figure is
reproducible from a frozen reference and does not require the data files."""

from pathlib import Path

import matplotlib.pyplot as plt

CLASS_COUNTS = {
    "Thunderstorm Wind": 41177,
    "Hail": 20702,
    "Drought": 9288,
    "Flash Flood": 8286,
    "Excessive Heat": 7684,
    "High Wind": 7485,
    "Heat": 6494,
    "Winter Weather": 6448,
    "Flood": 5318,
    "Marine Thunderstorm Wind": 4835,
}

OUT = Path(__file__).resolve().parent.parent / "reports" / "figures" / "class_distribution.png"


def main() -> None:
    names = list(CLASS_COUNTS.keys())
    counts = list(CLASS_COUNTS.values())
    total = sum(counts)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(range(len(names)), counts, color="#3b6aa0")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("count")
    ax.set_title(f"Class distribution, NOAA Storm Events top 10 (2023 + 2024, N = {total:,})")
    for bar, n in zip(bars, counts):
        ax.text(bar.get_width() + total * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{n:,} ({100 * n / total:.1f}%)", va="center", fontsize=9)
    ax.set_xlim(0, max(counts) * 1.18)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
