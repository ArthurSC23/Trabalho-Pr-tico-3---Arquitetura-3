import glob
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt


def find_relevant_csv():
    """Localiza automaticamente o CSV que contém dados de OpenMP e blocked_openmp."""
    csv_files = sorted(glob.glob("*.csv"))
    candidates = []

    for csv_file in csv_files:
        try:
            sample = pd.read_csv(csv_file, nrows=20)
        except Exception:
            continue

        if not {"version", "N", "threads"}.issubset(sample.columns):
            continue

        if sample["version"].isin(["openmp", "blocked_openmp"]).any():
            candidates.append(csv_file)

    if not candidates:
        return None

    if "cpu_results.csv" in candidates:
        return "cpu_results.csv"

    return candidates[0]


def load_experiment_data(csv_path):
    """Carrega o arquivo CSV e ajusta os tipos de dados principais."""
    data = pd.read_csv(csv_path)
    if "version" not in data.columns or "N" not in data.columns or "threads" not in data.columns:
        raise ValueError("O CSV precisa conter as colunas 'version', 'N' e 'threads'.")
    data = data.copy()
    data["N"] = data["N"].astype(int)
    data["threads"] = data["threads"].astype(int)
    return data


def plot_strong_scaling(data, output_files):
    """Gera um gráfico de strong scaling para cada tamanho de matriz disponível."""
    versions = ["openmp", "blocked_openmp"]
    strong_data = data[data["version"].isin(versions)]
    if strong_data.empty:
        print("Não há dados de OpenMP ou Blocked OpenMP para strong scaling.")
        return

    for matrix_size in sorted(strong_data["N"].unique()):
        figure_name = f"strong_scaling_N{matrix_size}.png"
        subset = strong_data[strong_data["N"] == matrix_size]
        if subset.empty:
            continue

        plt.figure(figsize=(8, 6))
        plt.style.use("classic")

        for version in versions:
            version_data = subset[subset["version"] == version]
            if version_data.empty:
                continue
            version_data = version_data.sort_values(by="threads")
            marker = "o" if version == "openmp" else "s"
            label = "OpenMP" if version == "openmp" else "Blocked OpenMP"
            plt.plot(
                version_data["threads"],
                version_data["best_time_s"],
                marker=marker,
                linestyle="-",
                linewidth=1.8,
                markersize=6,
                label=label,
            )

        plt.xlabel("Número de threads")
        plt.ylabel("Melhor tempo de execução (s)")
        plt.title(f"Strong scaling para multiplicação de matrizes (N = {matrix_size})")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_name, dpi=300)
        plt.close()
        output_files.append(figure_name)


def detect_weak_scaling_candidates(data):
    """Detecta se há dados de weak scaling válidos por implementação."""
    versions = ["openmp", "blocked_openmp"]
    candidates = {}

    for version in versions:
        version_data = data[data["version"] == version]
        if version_data.empty:
            continue

        counts_per_thread = version_data.groupby("threads")["N"].nunique()
        if counts_per_thread.empty:
            continue
        if (counts_per_thread > 1).any():
            continue

        unique_thread_counts = sorted(version_data["threads"].unique())
        if len(unique_thread_counts) < 2:
            continue

        collapsed = version_data.groupby("threads").first().reset_index()
        collapsed = collapsed.sort_values(by="threads")
        n_values = collapsed["N"].tolist()
        if not all(earlier < later for earlier, later in zip(n_values, n_values[1:])):
            continue

        candidates[version] = collapsed

    return candidates


def plot_weak_scaling(weak_candidates, output_files):
    """Gera gráfico de weak scaling se houver dados válidos."""
    if not weak_candidates:
        print(
            "Não há dados suficientes para construir um gráfico de weak scaling de forma confiável. "
            "O CSV não apresenta uma associação única entre threads e tamanho de matriz para os dados de OpenMP/Blocked OpenMP."
        )
        return

    plt.figure(figsize=(8, 6))
    plt.style.use("classic")

    for version, candidate_data in weak_candidates.items():
        label = "OpenMP" if version == "openmp" else "Blocked OpenMP"
        plt.plot(
            candidate_data["threads"],
            candidate_data["best_time_s"],
            marker="o" if version == "openmp" else "s",
            linestyle="-",
            linewidth=1.8,
            markersize=6,
            label=label,
        )

    plt.xlabel("Número de threads")
    plt.ylabel("Melhor tempo de execução (s)")
    plt.title("Weak scaling para multiplicação de matrizes")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    output_name = "weak_scaling.png"
    plt.savefig(output_name, dpi=300)
    plt.close()
    output_files.append(output_name)


def main():
    csv_path = find_relevant_csv()
    if csv_path is None:
        available = ", ".join(sorted(glob.glob("*.csv"))) or "nenhum arquivo CSV encontrado"
        print(
            "Não foi possível localizar automaticamente um arquivo CSV com dados de OpenMP e Blocked OpenMP."
        )
        print(f"Arquivos CSV disponíveis: {available}")
        sys.exit(1)

    print(f"Usando o arquivo de dados correto: {csv_path}")
    data = load_experiment_data(csv_path)
    output_files = []

    plot_strong_scaling(data, output_files)
    weak_candidates = detect_weak_scaling_candidates(data)
    plot_weak_scaling(weak_candidates, output_files)

    if output_files:
        print("Gráficos gerados com sucesso:")
        for path in output_files:
            print(f"- {path}")
    else:
        print("Nenhum gráfico foi gerado. Verifique se há dados válidos no arquivo selecionado.")


if __name__ == "__main__":
    main()
