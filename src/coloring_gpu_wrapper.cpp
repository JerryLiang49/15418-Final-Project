// C++ host code for CUDA graph coloring
// Handles 
// - Hub preprocessing (same as CPU version)
// - Building the initial worklist
// - Calling the CUDA entry point
// - Packaging results into ColoringResult

#ifdef CUDA_ENABLED

#include "coloring.h"
#include "timer.h"
#include <algorithm>
#include <vector>
#include <cmath>

// Implemented in coloring_cuda.cu.
extern "C" int gpu_warmup();


// out_transfer_ms: cudaMalloc + H2D transfers (one-time setup cost)
// out_kernel_ms: iteration loop only 
// out_finalize_ms: Device to host + cudaFree
extern "C" int gpu_color(
    const int* h_row_offsets, int n_plus_1,
    const int* h_col_indices, int num_edges,
    int* h_colors,
    const int* h_worklist, int wsize,
    int* out_total_conflicts,
    int* out_num_rounds,
    float* out_transfer_ms,
    float* out_kernel_ms,
    float* out_finalize_ms);

ColoringResult color_gpu(const Graph& g) {
    int n = g.num_vertices;
    std::vector<int> colors(n, -1);

    // Warm up before starting the wall-clock timer so total_time reflects
    // a warm end-to-end run rather than first-use context initialization
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

    // CPU-side hub preprocessing (same as color_parallel)
    std::vector<int> deg(n);
    int max_deg = 0;
    double sum_deg = 0.0;
    double avg_deg = 0.0;
    double hub_threshold = 1.0;
    // Identify and color hub vertices on CPU
    std::vector<int> regular_vertices;
    int num_hubs = 0;

    // only preprocess if enough vertices 
    if (n > 1000) {
        std::vector<int> hub_vertices;
        hub_vertices.reserve(n / 100 + 1);
        regular_vertices.reserve(n);
        for (int v = 0; v < n; v++) {
            int d = g.degree(v);
            deg[v] = d;
            if (d > max_deg) {
                max_deg = d;
            } 
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
        for (int v = 0; v < n; v++) {
            int d = g.degree(v);
            deg[v] = d;
            if (d > max_deg) max_deg = d;
            sum_deg += d;
        }
        regular_vertices.resize(n);
        for (int v = 0; v < n; v++) regular_vertices[v] = v;
    }

    // Sort regular vertices by degree (descending) to reduce warp divergence.
    // This way all threads in warp will have similar work 
    std::sort(regular_vertices.begin(), regular_vertices.end(),
              [&deg](int a, int b) { return deg[a] > deg[b]; });

    int wsize = static_cast<int>(regular_vertices.size());

    double cpu_prep_time = timer.elapsed();

    int total_conflicts = 0;
    int num_rounds = 0;
    float gpu_transfer_ms = 0.0f;
    float gpu_kernel_ms = 0.0f;
    float gpu_finalize_ms = 0.0f;

    // Call CUDA kernel 
    int ret = gpu_color(
        g.row_offsets.data(), n + 1,
        g.col_indices.data(), g.num_edges,
        colors.data(),
        regular_vertices.data(), wsize,
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

    // Update Metrics 
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
