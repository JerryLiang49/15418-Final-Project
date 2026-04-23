// GPU graph coloring kernels

#include <cuda_runtime.h>
#include <cstdio>

#define BLOCK_SIZE 256


// Tentative greedy coloring for one speculative round 
// One thread per vertex, 128-bit register bitmask for used colors.
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
    // neighbors for this thread 
    int start = row_offsets[v];
    int end = row_offsets[v + 1];

    // 128 bit mask for used colors, fits in 4 registers 
    // If more than 128 colors needed, we go to fallback path
    unsigned int used0 = 0, used1 = 0, used2 = 0, used3 = 0;

    // scan neighbors and mark their colors 
    for (int i = start; i < end; i++) {
        int c = colors[col_indices[i]];
        if (c >= 0 && c < 128) {
            unsigned int bit = 1u << (c & 31);
            int word = c >> 5;
            if (word == 0) used0 |= bit;
            else if (word == 1) used1 |= bit;
            else if (word == 2) used2 |= bit;
            else used3 |= bit;
        }
    }

    // find first unused color 
    int c = -1;
    int pos;
    pos = __ffs(~used0);
    if (pos) { 
        c = pos - 1; // found in first 0-31 block
    }
    else {
        pos = __ffs(~used1);
        if (pos) { 
            c = 32 + pos - 1; // found in second block
        }
        else {
            pos = __ffs(~used2);
            if (pos) { 
                c = 64 + pos - 1; // found in third block
            }
            else {
                pos = __ffs(~used3);
                if (pos) { 
                    c = 96 + pos - 1; 
                }
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

// Conflict detection + parallel worklist compaction kernel 
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
// Force CUDA context initialization before timing so first-use startup overhead is excluded.
extern "C" int gpu_warmup() {
    cudaError_t err = cudaFree(0);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA warmup failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

extern "C" int gpu_color(
    const int* h_row_offsets, int n_plus_1,
    const int* h_col_indices, int num_edges,
    int* h_colors,
    const int* h_worklist, int wsize,
    int* out_total_conflicts,
    int* out_num_rounds,
    float* out_transfer_ms,
    float* out_kernel_ms,
    float* out_finalize_ms)
{
    int n = n_plus_1 - 1;

    // Measure transfer, kernel-loop, and finalize phases separately.
    cudaEvent_t ev_start, ev_transfer_done, ev_kernel_done, ev_finalize_done;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_transfer_done);
    cudaEventCreate(&ev_kernel_done);
    cudaEventCreate(&ev_finalize_done);

    int *d_row_offsets, *d_col_indices, *d_colors;
    int *d_worklist, *d_next_worklist, *d_next_count;

    // Keep the per-round next_count readback in pinned memory.
    int* h_next_count_pinned = nullptr;
    cudaError_t pinned_err = cudaMallocHost(&h_next_count_pinned, sizeof(int));
    if (pinned_err != cudaSuccess || h_next_count_pinned == nullptr) {
        fprintf(stderr, "cudaMallocHost failed: %s\n", cudaGetErrorString(pinned_err));
        return -1;
    }

    cudaEventRecord(ev_start);

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

    cudaEventRecord(ev_transfer_done);

    int total_conflicts = 0;
    int num_rounds = 0;

    while (wsize > 0) {
        int grid = (wsize + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Tentatively color every active vertex in the current worklist.
        tentative_color_kernel<<<grid, BLOCK_SIZE>>>(
            d_row_offsets, d_col_indices, d_colors, d_worklist, wsize);

        cudaMemsetAsync(d_next_count, 0, sizeof(int));

        // Conflicting losers are compacted into d_next_worklist.
        detect_conflicts_kernel<<<grid, BLOCK_SIZE>>>(
            d_row_offsets, d_col_indices, d_colors, d_worklist, wsize,
            d_next_worklist, d_next_count);

        // Read back the size of the next round's worklist.
        cudaMemcpy(h_next_count_pinned, d_next_count, sizeof(int), cudaMemcpyDeviceToHost);
        int next_count = *h_next_count_pinned;

        num_rounds++;
        total_conflicts += next_count;

        // Reuse the two device buffers by swapping their roles each round.
        int* tmp = d_worklist;
        d_worklist = d_next_worklist;
        d_next_worklist = tmp;
        wsize = next_count;
    }

    cudaEventRecord(ev_kernel_done);

    cudaMemcpy(h_colors, d_colors, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_colors);
    cudaFree(d_worklist);
    cudaFree(d_next_worklist);
    cudaFree(d_next_count);
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
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}
