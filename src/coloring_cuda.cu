// ===========================================================================
// coloring_cuda.cu — GPU graph coloring kernels
//
// Compiled by nvcc with -ccbin g++-11 (CUDA 11.7 requires GCC <= 11).
// Contains two CUDA kernels and a C-linkage entry point called from
// the C++ wrapper (coloring_gpu_wrapper.cpp).
// ===========================================================================

#include <cuda_runtime.h>
#include <cstdio>

#define BLOCK_SIZE 256

// ---------------------------------------------------------------------------
// Kernel 1: Tentative greedy coloring
// One thread per vertex, 128-bit register bitmask for used colors.
// ---------------------------------------------------------------------------
__global__ void tentative_color_kernel(
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    int* __restrict__ colors,
    const int* __restrict__ worklist,
    int wsize)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= wsize) return;

    int v = worklist[tid];
    int start = row_offsets[v];
    int end = row_offsets[v + 1];

    unsigned int used0 = 0, used1 = 0, used2 = 0, used3 = 0;

    for (int i = start; i < end; i++) {
        int c = colors[col_indices[i]];
        if (c >= 0 && c < 128) {
            unsigned int bit = 1u << (c & 31);
            int word = c >> 5;
            if      (word == 0) used0 |= bit;
            else if (word == 1) used1 |= bit;
            else if (word == 2) used2 |= bit;
            else                used3 |= bit;
        }
    }

    int c = -1;
    int pos;
    pos = __ffs(~used0);
    if (pos) { c = pos - 1; }
    else {
        pos = __ffs(~used1);
        if (pos) { c = 32 + pos - 1; }
        else {
            pos = __ffs(~used2);
            if (pos) { c = 64 + pos - 1; }
            else {
                pos = __ffs(~used3);
                if (pos) { c = 96 + pos - 1; }
            }
        }
    }

    // Fallback for colors >= 128
    if (c < 0) {
        for (int try_c = 128; ; try_c++) {
            bool conflict = false;
            for (int i = start; i < end; i++) {
                if (colors[col_indices[i]] == try_c) {
                    conflict = true;
                    break;
                }
            }
            if (!conflict) { c = try_c; break; }
        }
    }

    colors[v] = c;
}

// ---------------------------------------------------------------------------
// Kernel 2: Conflict detection + parallel worklist compaction
// ---------------------------------------------------------------------------
__global__ void detect_conflicts_kernel(
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    int* __restrict__ colors,
    const int* __restrict__ worklist,
    int wsize,
    int* __restrict__ next_worklist,
    int* __restrict__ d_next_count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= wsize) return;

    int v = worklist[tid];
    int cv = colors[v];
    int start = row_offsets[v];
    int end = row_offsets[v + 1];

    for (int i = start; i < end; i++) {
        int u = col_indices[i];
        if (cv == colors[u] && v > u) {
            colors[v] = -1;
            int idx = atomicAdd(d_next_count, 1);
            next_worklist[idx] = v;
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// C-linkage entry point called from coloring_gpu_wrapper.cpp
// ---------------------------------------------------------------------------
extern "C" int gpu_color(
    const int* h_row_offsets, int n_plus_1,
    const int* h_col_indices, int num_edges,
    int* h_colors,
    const int* h_worklist, int wsize,
    int* out_total_conflicts,
    int* out_num_rounds)
{
    int n = n_plus_1 - 1;

    int *d_row_offsets, *d_col_indices, *d_colors;
    int *d_worklist, *d_next_worklist, *d_next_count;

    cudaMalloc(&d_row_offsets, n_plus_1 * sizeof(int));
    cudaMalloc(&d_col_indices, num_edges * sizeof(int));
    cudaMalloc(&d_colors, n * sizeof(int));
    cudaMalloc(&d_worklist, n * sizeof(int));
    cudaMalloc(&d_next_worklist, n * sizeof(int));
    cudaMalloc(&d_next_count, sizeof(int));

    cudaMemcpy(d_row_offsets, h_row_offsets, n_plus_1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, h_col_indices, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colors, h_colors, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_worklist, h_worklist, wsize * sizeof(int), cudaMemcpyHostToDevice);

    int total_conflicts = 0;
    int num_rounds = 0;

    while (wsize > 0) {
        int grid = (wsize + BLOCK_SIZE - 1) / BLOCK_SIZE;

        tentative_color_kernel<<<grid, BLOCK_SIZE>>>(
            d_row_offsets, d_col_indices, d_colors, d_worklist, wsize);

        cudaMemset(d_next_count, 0, sizeof(int));

        detect_conflicts_kernel<<<grid, BLOCK_SIZE>>>(
            d_row_offsets, d_col_indices, d_colors, d_worklist, wsize,
            d_next_worklist, d_next_count);

        int next_count = 0;
        cudaMemcpy(&next_count, d_next_count, sizeof(int), cudaMemcpyDeviceToHost);

        num_rounds++;
        total_conflicts += next_count;

        int* tmp = d_worklist;
        d_worklist = d_next_worklist;
        d_next_worklist = tmp;
        wsize = next_count;
    }

    cudaMemcpy(h_colors, d_colors, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_colors);
    cudaFree(d_worklist);
    cudaFree(d_next_worklist);
    cudaFree(d_next_count);

    *out_total_conflicts = total_conflicts;
    *out_num_rounds = num_rounds;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}
