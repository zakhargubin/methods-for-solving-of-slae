import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def read_history(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                {
                    "method": row["method"],
                    "iteration": int(row["iteration"]),
                    "elapsed_us": float(row["elapsed_us"]),
                    "residual_norm": float(row["residual_norm"]),
                }
            )
    return rows


def unique_methods(rows):
    return sorted({row["method"] for row in rows})


def filter_method(rows, method):
    out = [row for row in rows if row["method"] == method]
    out.sort(key=lambda row: row["iteration"])
    return out


def save_iteration_plot(rows, out_dir: Path):
    plt.figure(figsize=(8, 5))
    for method in unique_methods(rows):
        subset = filter_method(rows, method)
        iterations = [row["iteration"] for row in subset]
        residuals = [row["residual_norm"] for row in subset]
        plt.plot(iterations, residuals, linestyle="--", label=f"{method}")

    plt.yscale("log")
    plt.xlabel("N iter")
    plt.ylabel("Residual norm")
    plt.title("Convergence by iterations")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "iterative_by_iterations.png", dpi=150)
    plt.close()


def save_time_plot(rows, out_dir: Path):
    plt.figure(figsize=(8, 5))
    for method in unique_methods(rows):
        subset = filter_method(rows, method)
        times = [row["elapsed_us"] for row in subset]
        residuals = [row["residual_norm"] for row in subset]
        plt.plot(times, residuals, linestyle="--", label=f"{method}")

    plt.yscale("log")
    plt.xlabel("Time (µs)")
    plt.ylabel("Residual norm")
    plt.title("Convergence by time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "iterative_by_time.png", dpi=150)
    plt.close()


def main():
    history_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benchmark/iterative_history.csv")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("benchmark/iterative_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_history(history_path)
    save_iteration_plot(rows, out_dir)
    save_time_plot(rows, out_dir)


if __name__ == "__main__":
    main()
