// ===========================================================================
// coloring_gpu_wrapper.cpp — C++ wrapper for CUDA graph coloring
//
// This file is compiled with g++ (not nvcc) so it can freely use C++17
// and all STL headers. It handles:
//   - Hub preprocessing (same as CPU version)
//   - Building the initial worklist
//   - Calling the CUDA entry point (extern "C" gpu_color)
//   - Packaging results into ColoringResult
//
// The actual CUDA kernels live in coloring_cuda.cu (compiled by nvcc).
// ===========================================================================

#ifdef CUDA_ENABLED

#include "coloring.h"
#include "timer.h"
#include <algorithm>
#include <vector>
#include <cmath>

// Extern "C" function implemented in coloring_cuda.cu
extern "C" int gpu_color(
    const int* h_row_offsets, int n_plus_1,
    const int* h_col_indices, int num_edges,
    int* h_colors,
    const int* h_worklist, int wsize,
    int* out_total_conflicts,
    int* out_num_rounds);

ColoringResult color_gpu(const Graph& g) {
    Timer timer;
    timer.start();

    int n = g.num_vertices;
    std::vector<int> colors(n, -1);

    // --- CPU-side hub preprocessing (same as color_parallel) ---
    std::vector<int> deg(n);
    int max_deg = 0;
    double sum_deg = 0.0;
    for (int v = 0; v < n; v++) {
        int d = g.degree(v);
        deg[v] = d;
        if (d > max_deg) max_deg = d;
        sum_deg += d;
    }
    int palette_size = max_deg + 2;
    double avg_deg = (n > 0) ? sum_deg / n : 0.0;

    // Identify and color hub vertices on CPU
    double hub_threshold = std::max(avg_deg * 4.0, 1.0);
    std::vector<int> regular_vertices;
    int num_hubs = 0;

    if (n > 1000) {
        std::vector<int> hub_vertices;
        hub_vertices.reserve(n / 100 + 1);
        regular_vertices.reserve(n);
        for (int v = 0; v < n; v++) {
            if (deg[v] > hub_threshold) {
                hub_vertices.push_back(v);
            } else {
                regular_vertices.push_back(v);
            }
        }
        num_hubs = static_cast<int>(hub_vertices.size());

        // Sort hubs by descending degree, color greedily
        std::sort(hub_vertices.begin(), hub_vertices.end(), [&deg](int a, int b) {
            return deg[a] > deg[b];
        });
        std::vector<char> used(palette_size, 0);
        for (int v : hub_vertices) {
            int d = deg[v];
            std::fill(used.begin(), used.begin() + std::min(d + 2, palette_size), (char)0);
            for (const int* nb = g.neighbors_begin(v); nb != g.neighbors_end(v); nb++) {
                int c = colors[*nb];
                if (c >= 0 && c < palette_size) used[c] = 1;
            }
            for (int c = 0; ; c++) {
                if (c >= palette_size || !used[c]) {
                    colors[v] = c;
                    break;
                }
            }
        }
    } else {
        regular_vertices.resize(n);
        for (int v = 0; v < n; v++) regular_vertices[v] = v;
    }

    int wsize = static_cast<int>(regular_vertices.size());

    double init_time = timer.elapsed();

    // --- Call CUDA kernels via C-linkage entry point ---
    int total_conflicts = 0;
    int num_rounds = 0;

    int ret = gpu_color(
        g.row_offsets.data(), n + 1,
        g.col_indices.data(), g.num_edges,
        colors.data(),
        regular_vertices.data(), wsize,
        &total_conflicts, &num_rounds);

    if (ret != 0) {
        // CUDA error — return invalid result
        ColoringResult result;
        result.colors = std::move(colors);
        result.num_colors = 0;
        result.num_conflicts = -1;
        result.num_rounds = 0;
        result.num_hubs = num_hubs;
        result.init_seconds = init_time;
        result.compute_seconds = 0;
        result.elapsed_seconds = timer.elapsed();
        return result;
    }

    double compute_time = timer.elapsed() - init_time;

    int num_colors = *std::max_element(colors.begin(), colors.end()) + 1;

    ColoringResult result;
    result.colors = std::move(colors);
    result.num_colors = num_colors;
    result.num_conflicts = total_conflicts;
    result.num_rounds = num_rounds;
    result.num_hubs = num_hubs;
    result.init_seconds = init_time;
    result.compute_seconds = compute_time;
    result.elapsed_seconds = timer.elapsed();
    return result;
}

#endif // CUDA_ENABLED
