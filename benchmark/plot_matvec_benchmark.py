import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "size": int(row["size"]),
                    "density": float(row["density"]),
                    "nnz": int(row["nnz"]),
                    "dense_ms": float(row["dense_ms"]),
                    "csr_ms": float(row["csr_ms"]),
                }
            )
    return rows


def unique_densities(rows):
    return sorted({row["density"] for row in rows})


def save_density_plot(rows, density, out_dir: Path):
    subset = [row for row in rows if row["density"] == density]
    subset.sort(key=lambda row: row["size"])

    sizes = [row["size"] for row in subset]
    dense = [row["dense_ms"] for row in subset]
    csr = [row["csr_ms"] for row in subset]

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, dense, marker="o", label="DenseMatrix")
    plt.plot(sizes, csr, marker="o", label="CSRMatrix")
    plt.xlabel("Размер матрицы")
    plt.ylabel("Среднее время, мс")
    plt.title(f"Умножение матрицы на вектор, density = {density:.2f}")
    plt.grid(True)
    plt.legend()
    filename = f"benchmark_density_{str(density).replace('.', '_')}.png"
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=150)
    plt.close()


def save_ratio_plot(rows, out_dir: Path):
    plt.figure(figsize=(8, 5))
    for density in unique_densities(rows):
        subset = [row for row in rows if row["density"] == density]
        subset.sort(key=lambda row: row["size"])
        sizes = [row["size"] for row in subset]
        ratio = [row["dense_ms"] / row["csr_ms"] for row in subset]
        plt.plot(sizes, ratio, marker="o", label=f"density = {density:.2f}")

    plt.xlabel("Размер матрицы")
    plt.ylabel("Dense / CSR")
    plt.title("Отношение времени DenseMatrix к CSRMatrix")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "benchmark_ratio.png", dpi=150)
    plt.close()


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benchmark/matvec_results.csv")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("benchmark/matvek_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(csv_path)
    for density in unique_densities(rows):
        save_density_plot(rows, density, out_dir)
    save_ratio_plot(rows, out_dir)


if __name__ == "__main__":
    main()
