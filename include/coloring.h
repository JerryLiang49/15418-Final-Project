#ifndef COLORING_H
#define COLORING_H

#include "graph.h"
#include <vector>

// ---------------------------------------------------------------------------
// ColoringResult: output of any graph coloring algorithm.
//
// Captures both the coloring itself and performance instrumentation data
// (timing breakdown, conflict counts, hub counts) for analysis.
// ---------------------------------------------------------------------------
struct ColoringResult {
    std::vector<int> colors;   // color assigned to each vertex (0-indexed)
    int num_colors;            // total distinct colors used
    int num_conflicts;         // conflicts detected and resolved
    int num_rounds;            // number of coloring rounds to convergence
    int num_hubs;              // hub vertices colored sequentially (multi-phase)
    double init_seconds;       // initialization: degree stats, hub partition, LDF sort
    double compute_seconds;    // parallel coloring + conflict resolution
    double elapsed_seconds;    // total wall-clock time (init + compute)
};

// ---------------------------------------------------------------------------
// Sequential greedy coloring.
//
// Processes vertices 0..n-1 in order, assigning each the smallest color
// not used by any already-colored neighbor. Optimal greedy baseline.
// Time: O(V + E). Colors used: at most max_degree + 1.
// ---------------------------------------------------------------------------
ColoringResult color_sequential(const Graph& g);

// ---------------------------------------------------------------------------
// Parallel speculative coloring with multi-phase hub preprocessing (OpenMP).
//
// Two-phase algorithm:
//   Phase 1 (sequential): Color high-degree "hub" vertices optimally.
//   Phase 2 (parallel):   Speculative coloring with conflict resolution
//                          on remaining vertices, using LDF ordering.
//
// Key optimizations:
//   - Precomputed degree array for cache-friendly sorting and lookups
//   - Parallel init (reductions, partitioning, sort)
//   - Persistent parallel region (single fork/join)
//   - Parallel worklist compaction via atomic fetch_add
//   - Adaptive static/dynamic scheduling based on degree CV
// ---------------------------------------------------------------------------
ColoringResult color_parallel(const Graph& g, int num_threads);

// ---------------------------------------------------------------------------
// Hybrid speculative + Jones-Plassmann coloring (OpenMP).
//
// Three-phase algorithm:
//   Phase 1 (sequential): Color high-degree hub vertices optimally.
//   Phase 2 (parallel):   One round of speculative coloring on remaining
//                          vertices — colors the vast majority correctly.
//   Phase 3 (parallel):   Jones-Plassmann independent-set refinement on
//                          the small conflict set — resolves all remaining
//                          conflicts without generating new ones.
//
// Combines speculative's speed for the bulk with JP's conflict-free
// guarantee for the few remaining conflicts.
// ---------------------------------------------------------------------------
ColoringResult color_hybrid(const Graph& g, int num_threads);

// ---------------------------------------------------------------------------
// GPU speculative coloring (CUDA).
//
// Same Gebremedhin-Manne algorithm mapped to GPU kernels:
//   Phase 1 (CPU):  Sequential hub coloring (same as CPU version).
//   Phase 2 (GPU):  Iterative tentative-color + conflict-detect kernels
//                    using 128-bit register bitmask (4 x uint32_t).
//
// Only available when compiled with CUDA (nvcc detected by Makefile).
// ---------------------------------------------------------------------------
#ifdef CUDA_ENABLED
ColoringResult color_gpu(const Graph& g);
#endif

// ---------------------------------------------------------------------------
// Verify that a coloring is valid: no adjacent vertices share a color,
// and every vertex has been assigned a color (>= 0).
// ---------------------------------------------------------------------------
bool verify_coloring(const Graph& g, const std::vector<int>& colors);

#endif // COLORING_H
