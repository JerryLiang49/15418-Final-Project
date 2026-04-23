#!/usr/bin/env python3
"""
Helper to analyze benchmark results and compute metrics.

Reads a benchmark CSV and outputs:
  - Speedup tables (compute-time and total-time, 1T parallel baseline)
  - Parallel efficiency (speedup / threads)
  - Color quality comparison across algorithms
  - Conflict rate analysis
  - Scaling analysis
  - Init vs compute time breakdown

Usage: python3 scripts/analyze_results.py results/benchmark_XXXXX.csv
"""

import csv
import sys
import os
from collections import defaultdict

def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip empty or malformed rows
            if not row.get("vertices"):
                continue
            try:
                row["vertices"] = int(row["vertices"])
                row["edges"] = int(row["edges"])
                row["threads"] = int(row["threads"])
                row["colors"] = int(row["colors"])
                row["conflicts"] = int(row["conflicts"])
                row["rounds"] = int(row["rounds"])
                row["hubs"] = int(row.get("hubs", 0))
                row["max_deg"] = int(row.get("max_deg", 0))
                row["avg_deg"] = float(row.get("avg_deg", 0))
                row["cv_deg"] = float(row.get("cv_deg", 0))
                row["conflict_rate"] = float(row.get("conflict_rate", 0))
                row["init_time"] = float(row["init_time"])
                row["compute_time"] = float(row["compute_time"])
                row["total_time"] = float(row["total_time"])
                rows.append(row)
            except (ValueError, KeyError) as e:
                print(f"Warning: skipping malformed row: {e}", file=sys.stderr)
    if not rows:
        print("Error: no valid data rows found in CSV.", file=sys.stderr)
        sys.exit(1)
    return rows

def graph_name(path):
    return os.path.splitext(os.path.basename(path))[0]

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_results.py <benchmark.csv>")
        sys.exit(1)

    rows = load_csv(sys.argv[1])

    data = defaultdict(dict)
    for r in rows:
        key = (graph_name(r["graph"]), r["algorithm"])
        data[key][r["threads"]] = r

    # Get unique values
    graphs = sorted(set(graph_name(r["graph"]) for r in rows),
                    key=lambda g: next(r["vertices"] for r in rows if graph_name(r["graph"]) == g))
    algos = sorted(set(r["algorithm"] for r in rows))
    # CPU thread sweep (excludes GPU, which uses threads=0)
    threads = sorted(set(r["threads"] for r in rows if r["threads"] > 0))

    def is_gpu(algo):
        return algo.startswith("gpu")

    def is_sequential(algo):
        return algo == "sequential" or algo == "seq"

    def is_single_config(algo):
        """Algorithms that produce one row per graph (no thread sweep)."""
        return is_gpu(algo) or is_sequential(algo)

    def gpu_row(g, algo):
        """Return the single GPU row for graph g (threads=0), or None."""
        key = (g, algo)
        if key not in data:
            return None
        return data[key].get(0)

    def seq_row(g):
        """Return the sequential row for graph g (threads=1), or None."""
        for name in ("sequential", "seq"):
            key = (g, name)
            if key in data and 1 in data[key]:
                return data[key][1]
        return None

    def single_config_row(g, algo):
        """Return the single row for non-swept algorithms like GPU/sequential."""
        if is_gpu(algo):
            return gpu_row(g, algo)
        if is_sequential(algo):
            return seq_row(g)
        return None

    # Section 1: GRAPH CHARACTERISTICS
    print("=" * 80)
    print("1. GRAPH CHARACTERISTICS")
    print("=" * 80)
    print(f"{'Graph':<20s} {'V':>10s} {'E':>10s} {'MaxDeg':>8s} {'AvgDeg':>8s} {'CV':>8s} {'Type':>12s}")
    print("-" * 76)
    seen = set()
    for r in rows:
        gn = graph_name(r["graph"])
        if gn in seen:
            continue
        seen.add(gn)
        cv = r["cv_deg"]
        gtype = "regular" if cv < 0.3 else ("power-law" if cv > 1.0 else "moderate")
        print(f"{gn:<20s} {r['vertices']:>10d} {r['edges']:>10d} {r['max_deg']:>8d} "
              f"{r['avg_deg']:>8.1f} {cv:>8.3f} {gtype:>12s}")
    print()

    # ===== 2. COMPUTE-TIME SPEEDUP (1T baseline per algo) =====
    # GPU: compared against spec-1T baseline (a single GPU run per graph).
    # Sequential: shown separately in Section 2b; not in this thread-sweep view.
    print("=" * 80)
    print("2. COMPUTE-TIME SPEEDUP (1T parallel = 1.00x baseline)")
    print("=" * 80)
    for algo in algos:
        if is_sequential(algo):
            continue
        if is_gpu(algo):
            print(f"\n--- Algorithm: {algo} (vs spec-1T compute-time baseline) ---")
            print(f"{'Graph':<20s} {'GPU speedup':>14s}")
            print("-" * 36)
            for g in graphs:
                gr = gpu_row(g, algo)
                spec1 = data.get((g, "spec"), {}).get(1)
                if gr and spec1 and gr["compute_time"] > 0:
                    su = spec1["compute_time"] / gr["compute_time"]
                    print(f"{g:<20s} {su:>13.2f}x")
                else:
                    print(f"{g:<20s} {'N/A':>14s}")
            continue
        print(f"\n--- Algorithm: {algo} ---")
        header = f"{'Graph':<20s}"
        for t in threads:
            header += f" {'%dT' % t:>8s}"
        print(header)
        print("-" * (20 + 9 * len(threads)))
        for g in graphs:
            key = (g, algo)
            if key not in data:
                continue
            d = data[key]
            if 1 not in d:
                continue
            base = d[1]["compute_time"]
            line = f"{g:<20s}"
            for t in threads:
                if t in d and base > 0:
                    su = base / d[t]["compute_time"]
                    line += f" {su:>8.2f}x"
                else:
                    line += f" {'N/A':>8s}"
            print(line)
    print()

    # ===== 2b. SPEEDUP vs SEQUENTIAL GREEDY =====
    # Compares every parallel/GPU algorithm at its best configuration (CPU
    # algorithms at max threads, GPU as single-run) against pure sequential
    # greedy coloring — the baseline used in Naumov (2015), Deveci (2016),
    # and most GPU coloring literature.
    if any(is_sequential(a) for a in algos) and threads:
        print("=" * 80)
        print("2b. SPEEDUP vs SEQUENTIAL GREEDY (compute-time; max threads for CPU)")
        print("=" * 80)
        max_t = max(threads)
        cpu_algos = [a for a in algos if not is_single_config(a)]
        gpu_algos = [a for a in algos if is_gpu(a)]
        columns = [(a, f"{a}_{max_t}T") for a in cpu_algos] + [(a, a) for a in gpu_algos]

        header = f"{'Graph':<20s} {'seq_ms':>10s}"
        for _, label in columns:
            header += f" {label + '_sp':>14s}"
        print(header)
        print("-" * len(header))
        for g in graphs:
            sr = seq_row(g)
            if not sr:
                continue
            seq_t = sr["compute_time"]
            line = f"{g:<20s} {seq_t * 1000:>10.3f}"
            for algo, _ in columns:
                if is_gpu(algo):
                    gr = gpu_row(g, algo)
                    t = gr["compute_time"] if gr else None
                else:
                    t = data.get((g, algo), {}).get(max_t, {}).get("compute_time")
                if t and t > 0:
                    line += f" {seq_t / t:>13.2f}x"
                else:
                    line += f" {'N/A':>14s}"
            print(line)
        print()

    # ===== 3. TOTAL-TIME SPEEDUP =====
    # GPU: compared against spec-1T total-time baseline.
    # Sequential: shown separately in Section 2b.
    print("=" * 80)
    print("3. TOTAL-TIME SPEEDUP (1T parallel = 1.00x baseline)")
    print("=" * 80)
    for algo in algos:
        if is_sequential(algo):
            continue
        if is_gpu(algo):
            print(f"\n--- Algorithm: {algo} (vs spec-1T total-time baseline) ---")
            print(f"{'Graph':<20s} {'GPU speedup':>14s}")
            print("-" * 36)
            for g in graphs:
                gr = gpu_row(g, algo)
                spec1 = data.get((g, "spec"), {}).get(1)
                if gr and spec1 and gr["total_time"] > 0:
                    su = spec1["total_time"] / gr["total_time"]
                    print(f"{g:<20s} {su:>13.2f}x")
                else:
                    print(f"{g:<20s} {'N/A':>14s}")
            continue
        print(f"\n--- Algorithm: {algo} ---")
        header = f"{'Graph':<20s}"
        for t in threads:
            header += f" {'%dT' % t:>8s}"
        print(header)
        print("-" * (20 + 9 * len(threads)))
        for g in graphs:
            key = (g, algo)
            if key not in data:
                continue
            d = data[key]
            if 1 not in d:
                continue
            base = d[1]["total_time"]
            line = f"{g:<20s}"
            for t in threads:
                if t in d and base > 0:
                    su = base / d[t]["total_time"]
                    line += f" {su:>8.2f}x"
                else:
                    line += f" {'N/A':>8s}"
            print(line)
    print()

    # ===== 4. PARALLEL EFFICIENCY =====
    # Skipped for GPU: efficiency requires a thread-count sweep; GPU is a
    # single-configuration run.
    print("=" * 80)
    print("4. PARALLEL EFFICIENCY (speedup / threads, ideal = 1.00)")
    print("=" * 80)
    for algo in algos:
        if is_single_config(algo):
            continue
        print(f"\n--- Algorithm: {algo} (compute-time) ---")
        header = f"{'Graph':<20s}"
        for t in threads:
            header += f" {'%dT' % t:>8s}"
        print(header)
        print("-" * (20 + 9 * len(threads)))
        for g in graphs:
            key = (g, algo)
            if key not in data:
                continue
            d = data[key]
            if 1 not in d:
                continue
            base = d[1]["compute_time"]
            line = f"{g:<20s}"
            for t in threads:
                if t in d and base > 0:
                    su = base / d[t]["compute_time"]
                    eff = su / t
                    line += f" {eff:>8.2f}"
                else:
                    line += f" {'N/A':>8s}"
            print(line)
    print()

    # ===== 5. COLOR QUALITY COMPARISON =====
    # GPU has a single column (no thread sweep).
    print("=" * 80)
    print("5. COLOR QUALITY (colors used at 1T and max threads)")
    print("=" * 80)
    max_t = max(threads) if threads else 0
    header = f"{'Graph':<20s}"
    for algo in algos:
        if is_single_config(algo):
            header += f" {algo:>10s}"
        elif max_t == 1:
            header += f" {algo + '_1T':>10s}"
        else:
            header += f" {algo + '_1T':>10s} {algo + '_' + str(max_t) + 'T':>10s}"
    print(header)
    print("-" * len(header))
    for g in graphs:
        line = f"{g:<20s}"
        for algo in algos:
            key = (g, algo)
            if is_single_config(algo):
                r = single_config_row(g, algo)
                line += f" {(r['colors'] if r else 'N/A'):>10}"
            elif key in data:
                d = data[key]
                c1 = d.get(1, {}).get("colors", "N/A")
                if max_t == 1:
                    line += f" {str(c1):>10s}"
                else:
                    cn = d.get(max_t, {}).get("colors", "N/A")
                    line += f" {str(c1):>10s} {str(cn):>10s}"
            else:
                if max_t == 1:
                    line += f" {'N/A':>10s}"
                else:
                    line += f" {'N/A':>10s} {'N/A':>10s}"
        print(line)
    print()

    # ===== 6. CONFLICT RATE ANALYSIS =====
    # GPU: single-run column (normalized by rounds in main.cpp).
    print("=" * 80)
    print("6. CONFLICT RATE (% of vertices with conflicts, per round for GPU)")
    print("=" * 80)
    for algo in algos:
        if algo == "jp":
            continue  # JP has no conflicts
        if is_single_config(algo):
            print(f"\n--- Algorithm: {algo} ---")
            print(f"{'Graph':<20s} {'conflict_rate':>14s}")
            print("-" * 36)
            for g in graphs:
                r = single_config_row(g, algo)
                if r:
                    print(f"{g:<20s} {r['conflict_rate']:>13.3f}%")
                else:
                    print(f"{g:<20s} {'N/A':>14s}")
            continue
        print(f"\n--- Algorithm: {algo} ---")
        header = f"{'Graph':<20s}"
        for t in threads:
            header += f" {'%dT' % t:>8s}"
        print(header)
        print("-" * (20 + 9 * len(threads)))
        for g in graphs:
            key = (g, algo)
            if key not in data:
                continue
            d = data[key]
            line = f"{g:<20s}"
            for t in threads:
                if t in d:
                    cr = d[t]["conflict_rate"]
                    line += f" {cr:>7.3f}%"
                else:
                    line += f" {'N/A':>8s}"
            print(line)
    print()

    # ===== 7. ROUNDS TO CONVERGENCE =====
    print("=" * 80)
    print("7. ROUNDS TO CONVERGENCE")
    print("=" * 80)
    for algo in algos:
        if is_single_config(algo):
            print(f"\n--- Algorithm: {algo} ---")
            print(f"{'Graph':<20s} {'rounds':>8s}")
            print("-" * 30)
            for g in graphs:
                r = single_config_row(g, algo)
                if r:
                    print(f"{g:<20s} {r['rounds']:>8d}")
                else:
                    print(f"{g:<20s} {'N/A':>8s}")
            continue
        print(f"\n--- Algorithm: {algo} ---")
        header = f"{'Graph':<20s}"
        for t in threads:
            header += f" {'%dT' % t:>6s}"
        print(header)
        print("-" * (20 + 7 * len(threads)))
        for g in graphs:
            key = (g, algo)
            if key not in data:
                continue
            d = data[key]
            line = f"{g:<20s}"
            for t in threads:
                if t in d:
                    line += f" {d[t]['rounds']:>6d}"
                else:
                    line += f" {'N/A':>6s}"
            print(line)
    print()

    # ===== 8. INIT vs COMPUTE TIME BREAKDOWN =====
    print("=" * 80)
    print("8. INIT vs COMPUTE TIME BREAKDOWN (% of total)")
    print("=" * 80)
    for algo in algos:
        if algo == "jp":
            continue
        print(f"\n--- Algorithm: {algo} ---")
        header = f"{'Graph':<20s} {'T':>3s} {'init_ms':>10s} {'comp_ms':>10s} {'total_ms':>10s} {'init%':>7s} {'comp%':>7s}"
        print(header)
        print("-" * 67)
        if is_single_config(algo):
            for g in graphs:
                r = single_config_row(g, algo)
                if not r:
                    continue
                init_ms = r["init_time"] * 1000
                comp_ms = r["compute_time"] * 1000
                total_ms = r["total_time"] * 1000
                init_pct = r["init_time"] / r["total_time"] * 100 if r["total_time"] > 0 else 0
                comp_pct = r["compute_time"] / r["total_time"] * 100 if r["total_time"] > 0 else 0
                print(f"{g:<20s} {'—':>3s} {init_ms:>10.3f} {comp_ms:>10.3f} {total_ms:>10.3f} {init_pct:>6.1f}% {comp_pct:>6.1f}%")
            continue
        for g in graphs:
            key = (g, algo)
            if key not in data:
                continue
            d = data[key]
            for t in threads:
                if t not in d:
                    continue
                r = d[t]
                init_ms = r["init_time"] * 1000
                comp_ms = r["compute_time"] * 1000
                total_ms = r["total_time"] * 1000
                init_pct = r["init_time"] / r["total_time"] * 100 if r["total_time"] > 0 else 0
                comp_pct = r["compute_time"] / r["total_time"] * 100 if r["total_time"] > 0 else 0
                print(f"{g:<20s} {t:>3d} {init_ms:>10.3f} {comp_ms:>10.3f} {total_ms:>10.3f} {init_pct:>6.1f}% {comp_pct:>6.1f}%")
    print()

    # ===== 9. HUB ANALYSIS =====
    print("=" * 80)
    print("9. HUB VERTEX ANALYSIS")
    print("=" * 80)
    header = f"{'Graph':<20s} {'V':>10s} {'Hubs':>8s} {'Hub%':>8s} {'MaxDeg':>8s} {'AvgDeg':>8s}"
    print(header)
    print("-" * 62)
    seen = set()
    for g in graphs:
        for algo in ["spec", "hybrid"]:
            key = (g, algo)
            if key in data and g not in seen:
                seen.add(g)
                r = data[key].get(1, data[key].get(min(data[key].keys()), None))
                if r:
                    hub_pct = r["hubs"] / r["vertices"] * 100 if r["vertices"] > 0 else 0
                    print(f"{g:<20s} {r['vertices']:>10d} {r['hubs']:>8d} {hub_pct:>7.2f}% "
                          f"{r['max_deg']:>8d} {r['avg_deg']:>8.1f}")
    print()

    # ===== 10. ALGORITHM HEAD-TO-HEAD =====
    # CPU algos use max_t threads; GPU uses its single threads=0 row.
    print("=" * 80)
    print("10. ALGORITHM HEAD-TO-HEAD (total time, CPU @ max threads, GPU single-run)")
    print("=" * 80)
    header = f"{'Graph':<20s}"
    for algo in algos:
        header += f" {algo + '_ms':>10s}"
    if len(algos) > 1:
        header += f" {'best':>10s}"
    print(header)
    print("-" * (20 + 11 * (len(algos) + 1)))
    for g in graphs:
        line = f"{g:<20s}"
        times = {}
        for algo in algos:
            if is_single_config(algo):
                r = single_config_row(g, algo)
                if r:
                    t_ms = r["total_time"] * 1000
                    times[algo] = t_ms
                    line += f" {t_ms:>10.3f}"
                else:
                    line += f" {'N/A':>10s}"
                continue
            key = (g, algo)
            if key in data and max_t in data[key]:
                t_ms = data[key][max_t]["total_time"] * 1000
                times[algo] = t_ms
                line += f" {t_ms:>10.3f}"
            else:
                line += f" {'N/A':>10s}"
        if times:
            best = min(times, key=times.get)
            line += f" {best:>10s}"
        print(line)
    print()


if __name__ == "__main__":
    main()
