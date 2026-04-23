// ===========================================================================
// coloring_cuda_edge.cu — edge-parallel CUDA graph coloring kernels
//
// This variant uses edge-parallel passes for the two irregular pieces of the
// speculative algorithm:
//   1. accumulate forbidden colors for active vertices over edges
//   2. detect conflicts over edges and reactivate losing vertices
//
// Color selection itself remains vertex-parallel. This keeps the code close to
// the existing implementation while exposing more parallelism on irregular
// graphs than the pure vertex-based GPU path.
// ===========================================================================

#include <cuda_runtime.h>
#include <cstdio>

#define BLOCK_SIZE 256

__global__ void init_active_from_list_kernel(
    const int* __restrict__ active_vertices,
    int num_active,
    int* __restrict__ active)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_active) return;
    active[active_vertices[tid]] = 1;
}

__global__ void accumulate_forbidden_edge_kernel(
    const int* __restrict__ edge_src,
    const int* __restrict__ edge_dst,
    const int* __restrict__ colors,
    const int* __restrict__ active,
    unsigned int* __restrict__ forbidden,
    int num_edges)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_edges) return;

    int u = edge_src[tid];
    int v = edge_dst[tid];
    if (!active[v]) return;

    int c = colors[u];
    if (c >= 0 && c < 128) {
        int word = c >> 5;
        unsigned int bit = 1u << (c & 31);
        atomicOr(&forbidden[(size_t)v * 4 + word], bit);
    }
}

__global__ void choose_colors_from_forbidden_kernel(
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    int* __restrict__ colors,
    const int* __restrict__ active,
    const unsigned int* __restrict__ forbidden,
    int n)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n || !active[v]) return;

    const unsigned int* mask = forbidden + (size_t)v * 4;
    int c = -1;
    int pos = __ffs(~mask[0]);
    if (pos) { c = pos - 1; }
    else {
        pos = __ffs(~mask[1]);
        if (pos) { c = 32 + pos - 1; }
        else {
            pos = __ffs(~mask[2]);
            if (pos) { c = 64 + pos - 1; }
            else {
                pos = __ffs(~mask[3]);
                if (pos) { c = 96 + pos - 1; }
            }
        }
    }

    if (c < 0) {
        int start = row_offsets[v];
        int end = row_offsets[v + 1];
        for (int try_c = 128; ; try_c++) {
            bool conflict = false;
            for (int i = start; i < end; i++) {
                if (colors[col_indices[i]] == try_c) {
                    conflict = true;
                    break;
                }
            }
            if (!conflict) {
                c = try_c;
                break;
            }
        }
    }

    colors[v] = c;
}

__global__ void detect_conflicts_edge_kernel(
    const int* __restrict__ edge_src,
    const int* __restrict__ edge_dst,
    int* __restrict__ colors,
    const int* __restrict__ active,
    int* __restrict__ next_active,
    int* __restrict__ next_count,
    int num_edges)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_edges) return;

    int u = edge_src[tid];
    int v = edge_dst[tid];

    // The graph stores both directions; only process each undirected edge once.
    if (u >= v) return;

    int cu = colors[u];
    int cv = colors[v];
    if (cu < 0 || cv < 0 || cu != cv) return;

    int au = active[u];
    int av = active[v];
    int loser = -1;
    if (au && av) {
        loser = v;   // tie-break like the CPU/GPU baseline: larger ID loses
    } else if (au) {
        loser = u;
    } else if (av) {
        loser = v;
    }

    if (loser >= 0) {
        colors[loser] = -1;
        if (atomicCAS(&next_active[loser], 0, 1) == 0) {
            atomicAdd(next_count, 1);
        }
    }
}

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
    float* out_finalize_ms)
{
    int n = n_plus_1 - 1;

    cudaEvent_t ev_start, ev_transfer_done, ev_kernel_done, ev_finalize_done;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_transfer_done);
    cudaEventCreate(&ev_kernel_done);
    cudaEventCreate(&ev_finalize_done);

    int *d_row_offsets, *d_col_indices, *d_edge_src, *d_colors;
    int *d_active_vertices = nullptr, *d_active, *d_next_active, *d_next_count;
    unsigned int* d_forbidden;

    int* h_next_count_pinned = nullptr;
    cudaError_t pinned_err = cudaMallocHost(&h_next_count_pinned, sizeof(int));
    if (pinned_err != cudaSuccess || h_next_count_pinned == nullptr) {
        fprintf(stderr, "cudaMallocHost failed: %s\n", cudaGetErrorString(pinned_err));
        return -1;
    }

    cudaEventRecord(ev_start);

    cudaMalloc(&d_row_offsets, n_plus_1 * sizeof(int));
    cudaMalloc(&d_col_indices, num_edges * sizeof(int));
    cudaMalloc(&d_edge_src, num_edges * sizeof(int));
    cudaMalloc(&d_colors, n * sizeof(int));
    if (num_active > 0) {
        cudaMalloc(&d_active_vertices, num_active * sizeof(int));
    }
    cudaMalloc(&d_active, n * sizeof(int));
    cudaMalloc(&d_next_active, n * sizeof(int));
    cudaMalloc(&d_next_count, sizeof(int));
    cudaMalloc(&d_forbidden, (size_t)n * 4 * sizeof(unsigned int));

    cudaMemcpy(d_row_offsets, h_row_offsets, n_plus_1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, h_col_indices, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_src, h_edge_src, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colors, h_colors, n * sizeof(int), cudaMemcpyHostToDevice);
    if (num_active > 0) {
        cudaMemcpy(d_active_vertices, h_active_vertices, num_active * sizeof(int), cudaMemcpyHostToDevice);
    }

    cudaMemset(d_active, 0, n * sizeof(int));
    cudaMemset(d_next_active, 0, n * sizeof(int));
    if (num_active > 0) {
        int init_grid = (num_active + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_active_from_list_kernel<<<init_grid, BLOCK_SIZE>>>(d_active_vertices, num_active, d_active);
    }

    cudaEventRecord(ev_transfer_done);

    int total_conflicts = 0;
    int num_rounds = 0;
    int active_count = num_active;

    int vertex_grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int edge_grid = (num_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;

    while (active_count > 0) {
        cudaMemsetAsync(d_forbidden, 0, (size_t)n * 4 * sizeof(unsigned int));

        accumulate_forbidden_edge_kernel<<<edge_grid, BLOCK_SIZE>>>(
            d_edge_src, d_col_indices, d_colors, d_active, d_forbidden, num_edges);

        choose_colors_from_forbidden_kernel<<<vertex_grid, BLOCK_SIZE>>>(
            d_row_offsets, d_col_indices, d_colors, d_active, d_forbidden, n);

        cudaMemsetAsync(d_next_active, 0, n * sizeof(int));
        cudaMemsetAsync(d_next_count, 0, sizeof(int));

        detect_conflicts_edge_kernel<<<edge_grid, BLOCK_SIZE>>>(
            d_edge_src, d_col_indices, d_colors, d_active, d_next_active,
            d_next_count, num_edges);

        cudaMemcpy(h_next_count_pinned, d_next_count, sizeof(int), cudaMemcpyDeviceToHost);
        int next_count = *h_next_count_pinned;

        num_rounds++;
        total_conflicts += next_count;

        int* tmp = d_active;
        d_active = d_next_active;
        d_next_active = tmp;
        active_count = next_count;
    }

    cudaEventRecord(ev_kernel_done);

    cudaMemcpy(h_colors, d_colors, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_edge_src);
    cudaFree(d_colors);
    if (d_active_vertices) cudaFree(d_active_vertices);
    cudaFree(d_active);
    cudaFree(d_next_active);
    cudaFree(d_next_count);
    cudaFree(d_forbidden);
    cudaFreeHost(h_next_count_pinned);

    cudaEventRecord(ev_finalize_done);
    cudaEventSynchronize(ev_finalize_done);

    if (out_transfer_ms) cudaEventElapsedTime(out_transfer_ms, ev_start, ev_transfer_done);
    if (out_kernel_ms)   cudaEventElapsedTime(out_kernel_ms, ev_transfer_done, ev_kernel_done);
    if (out_finalize_ms) cudaEventElapsedTime(out_finalize_ms, ev_kernel_done, ev_finalize_done);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_transfer_done);
    cudaEventDestroy(ev_kernel_done);
    cudaEventDestroy(ev_finalize_done);

    *out_total_conflicts = total_conflicts;
    *out_num_rounds = num_rounds;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA edge-coloring error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}
