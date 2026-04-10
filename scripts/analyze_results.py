#!/usr/bin/env python3
"""
Analyze benchmark results and compute derived metrics for the report.

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

    # Index: (graph, algo) -> {threads: row}
    data = defaultdict(dict)
    for r in rows:
        key = (graph_name(r["graph"]), r["algorithm"])
        data[key][r["threads"]] = r

    # Get unique values
    graphs = sorted(set(graph_name(r["graph"]) for r in rows),
                    key=lambda g: next(r["vertices"] for r in rows if graph_name(r["graph"]) == g))
    algos = sorted(set(r["algorithm"] for r in rows))
    threads = sorted(set(r["threads"] for r in rows))

    # ===== 1. GRAPH CHARACTERISTICS =====
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
    print("=" * 80)
    print("2. COMPUTE-TIME SPEEDUP (1T parallel = 1.00x baseline)")
    print("=" * 80)
    for algo in algos:
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

    # ===== 3. TOTAL-TIME SPEEDUP =====
    print("=" * 80)
    print("3. TOTAL-TIME SPEEDUP (1T parallel = 1.00x baseline)")
    print("=" * 80)
    for algo in algos:
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
    print("=" * 80)
    print("4. PARALLEL EFFICIENCY (speedup / threads, ideal = 1.00)")
    print("=" * 80)
    for algo in algos:
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
    print("=" * 80)
    print("5. COLOR QUALITY (colors used at 1T and max threads)")
    print("=" * 80)
    max_t = max(threads)
    header = f"{'Graph':<20s}"
    for algo in algos:
        header += f" {algo + '_1T':>10s} {algo + '_' + str(max_t) + 'T':>10s}"
    print(header)
    print("-" * (20 + 21 * len(algos)))
    for g in graphs:
        line = f"{g:<20s}"
        for algo in algos:
            key = (g, algo)
            if key in data:
                d = data[key]
                c1 = d.get(1, {}).get("colors", "N/A")
                cn = d.get(max_t, {}).get("colors", "N/A")
                line += f" {str(c1):>10s} {str(cn):>10s}"
            else:
                line += f" {'N/A':>10s} {'N/A':>10s}"
        print(line)
    print()

    # ===== 6. CONFLICT RATE ANALYSIS =====
    print("=" * 80)
    print("6. CONFLICT RATE (% of vertices with conflicts)")
    print("=" * 80)
    for algo in algos:
        if algo == "jp":
            continue  # JP has no conflicts
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
    print("=" * 80)
    print("10. ALGORITHM HEAD-TO-HEAD (total time at max threads)")
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
