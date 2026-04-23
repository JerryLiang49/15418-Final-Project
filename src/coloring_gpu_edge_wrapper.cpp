// ===========================================================================
// coloring_gpu_edge_wrapper.cpp — C++ wrapper for edge-based CUDA coloring
//
// This implementation keeps the same CPU-side hub preprocessing as the
// existing GPU path but switches the GPU phase to an edge-parallel scheme:
//   1. Accumulate forbidden colors for active vertices over edges
//   2. Tentatively color active vertices from the forbidden masks
//   3. Detect conflicts over edges and reactivate losing vertices
//
// The goal is to preserve the current benchmarking/reporting structure while
// providing a second GPU algorithm that better matches the manycore literature.
// ===========================================================================

#ifdef CUDA_ENABLED

#include "coloring.h"
#include "timer.h"
#include <algorithm>
#include <cmath>
#include <vector>

extern "C" int gpu_warmup();

extern "C" int gpu_color_edge(
    const int* h_row_offsets, int n_plus_1,
    const int* h_col_indices,
    const int* h_edge_src, int num_edges,
    int* h_colors,
    const int* h_active_vertices, int num_active,
    int* out_total_conflicts,
    int* out_num_rounds,
    float* out_transfer_ms,
    float* out_kernel_ms,
    float* out_finalize_ms);

ColoringResult color_gpu_edge(const Graph& g) {
    int n = g.num_vertices;
    std::vector<int> colors(n, -1);

    if (gpu_warmup() != 0) {
        ColoringResult result;
        result.colors = std::move(colors);
        result.num_colors = 0;
        result.num_conflicts = -1;
        result.num_rounds = 0;
        result.num_hubs = 0;
        result.init_seconds = 0;
        result.compute_seconds = 0;
        result.elapsed_seconds = 0;
        return result;
    }

    Timer timer;
    timer.start();

    // CPU-side hub preprocessing intentionally mirrors the existing GPU path so
    // comparisons isolate the GPU coloring core rather than changing two things
    // at once.
    std::vector<int> deg(n);
    int max_deg = 0;
    double sum_deg = 0.0;
    double avg_deg = 0.0;
    double hub_threshold = 1.0;
    std::vector<int> regular_vertices;
    int num_hubs = 0;

    if (n > 1000) {
        std::vector<int> hub_vertices;
        hub_vertices.reserve(n / 100 + 1);
        regular_vertices.reserve(n);
        for (int v = 0; v < n; v++) {
            int d = g.degree(v);
            deg[v] = d;
            if (d > max_deg) max_deg = d;
            sum_deg += d;
        }
        avg_deg = (n > 0) ? sum_deg / n : 0.0;
        hub_threshold = std::max(avg_deg * 8.0, 1.0);
        for (int v = 0; v < n; v++) {
            if (deg[v] > hub_threshold) {
                hub_vertices.push_back(v);
            } else {
                regular_vertices.push_back(v);
            }
        }
        num_hubs = static_cast<int>(hub_vertices.size());
        int palette_size = max_deg + 2;

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
        for (int v = 0; v < n; v++) {
            int d = g.degree(v);
            deg[v] = d;
            if (d > max_deg) max_deg = d;
            sum_deg += d;
        }
        regular_vertices.resize(n);
        for (int v = 0; v < n; v++) regular_vertices[v] = v;
    }

    // Keep the same highest-degree-first work ordering as the baseline GPU
    // path so any performance change is mainly from edge-parallel processing.
    std::sort(regular_vertices.begin(), regular_vertices.end(),
              [&deg](int a, int b) { return deg[a] > deg[b]; });

    // Build an explicit edge-source array so the GPU can process edges
    // directly without converting CSR->COO on device each run.
    std::vector<int> edge_src(g.num_edges);
    for (int v = 0; v < n; v++) {
        for (int e = g.row_offsets[v]; e < g.row_offsets[v + 1]; e++) {
            edge_src[e] = v;
        }
    }

    double cpu_prep_time = timer.elapsed();

    int total_conflicts = 0;
    int num_rounds = 0;
    float gpu_transfer_ms = 0.0f;
    float gpu_kernel_ms = 0.0f;
    float gpu_finalize_ms = 0.0f;

    int ret = gpu_color_edge(
        g.row_offsets.data(), n + 1,
        g.col_indices.data(),
        edge_src.data(), g.num_edges,
        colors.data(),
        regular_vertices.data(), static_cast<int>(regular_vertices.size()),
        &total_conflicts, &num_rounds,
        &gpu_transfer_ms, &gpu_kernel_ms, &gpu_finalize_ms);

    if (ret != 0) {
        ColoringResult result;
        result.colors = std::move(colors);
        result.num_colors = 0;
        result.num_conflicts = -1;
        result.num_rounds = 0;
        result.num_hubs = num_hubs;
        result.init_seconds = cpu_prep_time;
        result.compute_seconds = 0;
        result.elapsed_seconds = timer.elapsed();
        return result;
    }

    int num_colors = *std::max_element(colors.begin(), colors.end()) + 1;

    ColoringResult result;
    result.colors = std::move(colors);
    result.num_colors = num_colors;
    result.num_conflicts = total_conflicts;
    result.num_rounds = num_rounds;
    result.num_hubs = num_hubs;
    result.init_seconds = cpu_prep_time + (double)(gpu_transfer_ms + gpu_finalize_ms) / 1000.0;
    result.compute_seconds = (double)gpu_kernel_ms / 1000.0;
    result.elapsed_seconds = timer.elapsed();
    return result;
}

#endif // CUDA_ENABLED
