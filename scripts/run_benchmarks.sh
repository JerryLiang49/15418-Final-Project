#!/usr/bin/env bash
# Run graph coloring benchmarks across thread counts, graph types, and algorithms.
# Usage: ./scripts/run_benchmarks.sh [max_threads]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="$PROJECT_DIR/graphcolor"
RESULTS_DIR="$PROJECT_DIR/results"

# Pin threads to cores for consistent results
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Rebuild
echo "Building..."
make -C "$PROJECT_DIR" clean >/dev/null 2>&1
make -C "$PROJECT_DIR" all 2>&1 | grep -v "^\[WARNING\]" || true
echo ""

if [ ! -x "$BINARY" ]; then
    echo "ERROR: $BINARY not found or not executable"
    exit 1
fi

MAX_THREADS="${1:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}"

THREADS=()
t=1
while [ "$t" -le "$MAX_THREADS" ]; do
    THREADS+=("$t")
    t=$((t * 2))
done
last_idx=$((${#THREADS[@]} - 1))
if [ "${THREADS[$last_idx]}" -ne "$MAX_THREADS" ]; then
    THREADS+=("$MAX_THREADS")
fi

# Algorithms to benchmark (JP removed — poor scaling, see report)
ALGORITHMS=("spec" "hybrid")

GRAPHS=(
    "graphs/er_10k.txt"
    "graphs/rmat_10k.txt"
    "graphs/grid_100.txt"
    "graphs/email-enron.txt"
    "graphs/er_100k.txt"
    "graphs/rmat_100k.txt"
    "graphs/grid_100k.txt"
    "graphs/amazon0302.txt"
    "graphs/roadnet-pa.txt"
    "graphs/er_1m.txt"
    "graphs/rmat_1m.txt"
)

AVAILABLE_GRAPHS=()
for g in "${GRAPHS[@]}"; do
    if [ -f "$PROJECT_DIR/$g" ]; then
        AVAILABLE_GRAPHS+=("$g")
    else
        echo "SKIP: $g (not found)"
    fi
done

mkdir -p "$RESULTS_DIR"
OUTFILE="$RESULTS_DIR/benchmark_$(date +%Y%m%d_%H%M%S).csv"

echo "graph,vertices,edges,threads,algorithm,max_deg,avg_deg,cv_deg,colors,conflicts,conflict_rate,rounds,hubs,init_time,compute_time,total_time,valid" > "$OUTFILE"

echo "Running benchmarks..."
echo "Thread counts: ${THREADS[*]}"
echo "Algorithms: ${ALGORITHMS[*]}"
echo "Graphs: ${#AVAILABLE_GRAPHS[@]}"
echo "OMP_PROC_BIND=$OMP_PROC_BIND  OMP_PLACES=$OMP_PLACES"
echo "Output: $OUTFILE"
echo ""

FAIL=0
for algo in "${ALGORITHMS[@]}"; do
    echo "--- Algorithm: $algo ---"
    for graph in "${AVAILABLE_GRAPHS[@]}"; do
        for threads in "${THREADS[@]}"; do
            echo -n "  $graph (t=$threads, $algo) ... "
            result=$("$BINARY" "$graph" "$threads" --csv --algorithm "$algo" 2>&1) || true
            echo "$result" >> "$OUTFILE"
            valid=$(echo "$result" | awk -F',' '{print $NF}')
            if [ "$valid" = "yes" ]; then
                echo "OK"
            else
                echo "INVALID COLORING!"
                FAIL=1
            fi
        done
    done
    echo ""
done

# GPU runs (single run per graph, no thread variation)
echo "--- Algorithm: gpu ---"
for graph in "${AVAILABLE_GRAPHS[@]}"; do
    echo -n "  $graph (gpu) ... "
    result=$("$BINARY" "$graph" 1 --csv --algorithm gpu 2>&1) || true
    if echo "$result" | grep -q ",yes$\|,NO$"; then
        echo "$result" >> "$OUTFILE"
        valid=$(echo "$result" | awk -F',' '{print $NF}')
        if [ "$valid" = "yes" ]; then
            echo "OK"
        else
            echo "INVALID COLORING!"
            FAIL=1
        fi
    else
        echo "SKIP (no GPU)"
        break  # if first graph fails, skip the rest
    fi
done
echo ""

echo "Results saved to $OUTFILE"

# Pretty-print summary
echo ""
echo "=== Benchmark Summary ==="
printf "%-20s %6s %3s %5s %6s %7s %4s %4s %10s %10s %10s %5s\n" \
    "graph" "algo" "T" "clrs" "confl" "c_rate" "rnds" "hubs" "init" "compute" "total" "ok"
echo "-----------------------------------------------------------------------------------------------------------------------"
tail -n +2 "$OUTFILE" | while IFS=',' read -r g v e t a md ad cv c co cr r h it ct tt val; do
    gname=$(basename "$g" .txt)
    printf "%-20s %6s %3s %5s %6s %7s %4s %4s %10s %10s %10s %5s\n" \
        "$gname" "$a" "$t" "$c" "$co" "$cr" "$r" "$h" "$it" "$ct" "$tt" "$val"
done

echo ""
echo "To analyze: python3 scripts/analyze_results.py $OUTFILE"

if [ "$FAIL" -ne 0 ]; then
    echo ""
    echo "WARNING: Some colorings were INVALID!"
    exit 1
fi
