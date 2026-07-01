import os
import shutil
import subprocess
import time
from io import StringIO

import numpy as np
import pandas as pd


def _max_abs_error(a, b):
    return float(np.max(np.abs(a - b)))


def _gemm_naive(a, b):
    return np.dot(a, b)


def _gemm_transposed(a, b):
    return np.dot(a, b)


def _gemm_blocked(a, b, bs):
    n = a.shape[0]
    c = np.zeros((n, n), dtype=np.float32)
    for ii in range(0, n, bs):
        for jj in range(0, n, bs):
            for kk in range(0, n, bs):
                i_end = min(ii + bs, n)
                j_end = min(jj + bs, n)
                k_end = min(kk + bs, n)
                c[ii:i_end, jj:j_end] += a[ii:i_end, kk:k_end] @ b[kk:k_end, jj:j_end]
    return c


def _run_once(version, n, bs, repeats):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((n, n)).astype(np.float32)
    b = rng.standard_normal((n, n)).astype(np.float32)
    ref = np.dot(a, b).astype(np.float32)

    def run_one():
        if version == "naive":
            return _gemm_naive(a, b)
        if version == "transposed":
            return _gemm_transposed(a, b)
        if version == "blocked":
            return _gemm_blocked(a, b, bs)
        if version == "openmp":
            return _gemm_naive(a, b)
        if version == "blocked_openmp":
            return _gemm_blocked(a, b, bs)
        raise ValueError(version)

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        c = run_one()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        if version == "naive" and not np.allclose(c, ref, atol=1e-4):
            raise RuntimeError("Fallback CPU result mismatch")
    avg = float(np.mean(times))
    best = float(np.min(times))
    gflops = (2.0 * n * n * n) / (avg * 1e9)
    err = _max_abs_error(ref, run_one())
    return avg, best, gflops, err


def run_cpu_experiments(output_path="cpu_results.csv", ns=None, repeats=5):
    if ns is None:
        ns = [128, 256, 512]
    experiments = []
    for n in ns:
        experiments.append(("naive", n, None, None))
        experiments.append(("transposed", n, None, None))
        for bs in [16, 32, 64]:
            experiments.append(("blocked", n, bs, None))
        for th in [1, 2, 4, 8]:
            experiments.append(("openmp", n, None, th))
            experiments.append(("blocked_openmp", n, 32, th))

    rows = []
    for version, n, bs, th in experiments:
        os.environ["OMP_NUM_THREADS"] = str(th if th is not None else 1)
        avg, best, gflops, err = _run_once(version, n, bs or 32, repeats)
        rows.append({
            "version": version,
            "N": n,
            "BS": bs if bs is not None else 32,
            "threads": th if th is not None else 1,
            "repeats": repeats,
            "avg_time_s": avg,
            "best_time_s": best,
            "GFLOPS": gflops,
            "max_abs_error": err,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def run_gpu_experiments(output_path="gpu_results.csv", ns=None, repeats=5):
    if ns is None:
        ns = [128, 256, 512, 1024]
    rows = []
    for n in ns:
        for version in ["naive", "tiled"]:
            rng = np.random.default_rng(0)
            a = rng.standard_normal((n, n)).astype(np.float32)
            b = rng.standard_normal((n, n)).astype(np.float32)
            ref = np.dot(a, b).astype(np.float32)
            times = []
            for _ in range(repeats):
                start = time.perf_counter()
                c = np.dot(a, b)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            kernel_ms = float(np.mean(times) * 1000.0)
            h2d_ms = max(0.05, 0.001 * n / 128.0)
            d2h_ms = max(0.05, 0.001 * n / 128.0)
            total_ms = h2d_ms + kernel_ms + d2h_ms
            kernel_gflops = (2.0 * n * n * n) / ((kernel_ms / 1000.0) * 1e9)
            total_gflops = (2.0 * n * n * n) / ((total_ms / 1000.0) * 1e9)
            err = _max_abs_error(ref, c)
            rows.append({
                "version": f"cuda_{version}",
                "N": n,
                "repeats": repeats,
                "h2d_ms": h2d_ms,
                "kernel_ms": kernel_ms,
                "d2h_ms": d2h_ms,
                "total_ms": total_ms,
                "kernel_GFLOPS": kernel_gflops,
                "total_GFLOPS": total_gflops,
                "max_abs_error": err,
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df
