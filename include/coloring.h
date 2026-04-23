#ifndef COLORING_H
#define COLORING_H

#include "graph.h"
#include <vector>

// Struct for output of any graph coloring algorithm.
// Captures both the coloring itself and performance instrumentation data
// like timing breakdown, conflict counts, hub counts for analysis.
// ---------------------------------------------------------------------------
struct ColoringResult {
    std::vector<int> colors;    // color assigned to each vertex (0-indexed)
    int num_colors;             // total distinct colors used
    int num_conflicts;          // conflicts detected (and resolved)
    int num_rounds;             // number of coloring rounds to convergence
    int num_hubs;               // hub vertices colored sequentially (multi-phase)
    double init_seconds;        // initialization time: degree stats, hub partition, LDF sort
    double compute_seconds;     // parallel coloring + conflict resolution
    double elapsed_seconds;     // total wall-clock time (init + compute)
};

// Sequential greedy coloring results 
// Algorithm: Processes vertices in order, assigning each the smallest color
// not used by any already-colored neighbor. 
ColoringResult color_sequential(const Graph& g);

// Parallel speculative coloring with multi-phase hub preprocessing (OpenMP)
// Basically the Gebremedhin-Manne with some optimizations 
// Two-phase algorithm:
// Phase 1 (sequential): Color high-degree hub vertices optimally.
// Phase 2 (parallel): Speculative coloring with conflict resolution on remaining vertices, using LDF ordering.
ColoringResult color_parallel(const Graph& g, int num_threads);

// Hybrid speculative + Jones-Plassmann coloring (OpenMP).
//
// Three-phase algorithm:
// Phase 1 (sequential): Color high-degree hub vertices optimally.
// Phase 2 (parallel): One round of speculative coloring on remaining vertices 
// Phase 3 (parallel): Jones-Plassmann refinement on the small conflict set ti resolve remaining conflicts
ColoringResult color_hybrid(const Graph& g, int num_threads);

// GPU speculative coloring (CUDA).
// Same Gebremedhin-Manne algorithm mapped to GPU kernels
// Phase 1 (CPU): Sequential hub coloring (same as CPU version).
// Phase 2 (GPU): Iterative tentative-color + conflict-detect kernels using 128-bit register bitmask
ColoringResult color_gpu(const Graph& g);


// Edge-parallel GPU speculative coloring (CUDA).
// Uses the same CPU-side hub preprocessing, but the device phase is 
// edge-parallel forbidden-color accumulation and conflict detection. 
ColoringResult color_gpu_edge(const Graph& g);

// Verify that coloring is valid 
bool verify_coloring(const Graph& g, const std::vector<int>& colors);

#endif // COLORING_H
