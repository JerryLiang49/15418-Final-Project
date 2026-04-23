// ===========================================================================
// coloring.cpp — Parallel graph coloring algorithms
//
// Implements three coloring strategies:
//   1. Sequential greedy    — optimal baseline, O(V+E)
//   2. Parallel speculative — Gebremedhin-Manne with hub preprocessing
//   3. Hybrid spec + JP     — speculative bulk + Jones-Plassmann refinement
//
// All parallel algorithms use OpenMP and target the GHC 8-core machines.
// The CSR graph representation (graph.h) provides O(1) degree lookups
// and cache-friendly neighbor iteration.
// ===========================================================================

#include "coloring.h"
#include "timer.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <numeric>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// ===========================================================================
// Counting sort for LDF ordering — O(n + max_deg) instead of O(n log n).
//
// Sorts vertex IDs by descending degree using a bucket/counting approach.
// Since degrees are integers in [0, max_deg], we can bin vertices by degree
// and then iterate buckets from highest to lowest. This is much faster than
// comparison sort for large n, and the memory access pattern is sequential
// (cache-friendly).
// ===========================================================================
static void counting_sort_by_degree_desc(
    std::vector<int>& vertices, const std::vector<int>& deg, int max_deg)
{
    int n = static_cast<int>(vertices.size());
    if (n <= 1) return;

    std::vector<std::vector<int>> buckets(max_deg + 1);
    for (int i = 0; i < n; i++) {
        buckets[deg[vertices[i]]].push_back(vertices[i]);
    }
    int idx = 0;
    for (int d = max_deg; d >= 0; d--) {
        for (int v : buckets[d]) {
            vertices[idx++] = v;
        }
    }
}

// ===========================================================================
// Hash function for Jones-Plassmann priorities.
//
// Used by the hybrid algorithm's JP refinement phase. Assigns each vertex
// a deterministic pseudo-random 32-bit priority. Uses a multiply-shift hash
// (Murmur-style finalizer) with excellent avalanche properties.
// ===========================================================================
static inline uint32_t vertex_hash(int v) {
    uint32_t h = static_cast<uint32_t>(v);
    h = ((h >> 16) ^ h) * 0x45d9f3b;
    h = ((h >> 16) ^ h) * 0x45d9f3b;
    h = (h >> 16) ^ h;
    return h;
}

// ===========================================================================
// Sequential greedy coloring
//
// Visits vertices in order 0..n-1, assigns each the smallest color not
// used by any already-colored neighbor. Optimal greedy baseline.
//
// Note: uses vector<char> instead of vector<bool>. In C++, vector<bool>
// is bit-packed — every read/write does shift+mask operations. vector<char>
// uses one byte per entry, making each access a single load/store. This is
// ~2-4x faster for the tight fill/scan loops in greedy coloring.
// ===========================================================================
ColoringResult color_sequential(const Graph& g) {
    Timer timer;
    timer.start();

    int n = g.num_vertices;
    std::vector<int> colors(n, -1);

    int max_deg = 0;
    for (int v = 0; v < n; v++) max_deg = std::max(max_deg, g.degree(v));
    int palette_size = max_deg + 2;
    // vector<char> instead of vector<bool> — avoids bit-packing overhead
    std::vector<char> used(palette_size, 0);

    double init_time = timer.elapsed();

    for (int v = 0; v < n; v++) {
        int deg = g.degree(v);
        std::fill(used.begin(), used.begin() + deg + 2, (char)0);
        for (const int* nb = g.neighbors_begin(v); nb != g.neighbors_end(v); nb++) {
            int c = colors[*nb];
            if (c >= 0 && c < palette_size) {
                used[c] = 1;
            }
        }
        for (int c = 0; ; c++) {
            if (c >= palette_size || !used[c]) {
                colors[v] = c;
                break;
            }
        }
    }

    int num_colors = *std::max_element(colors.begin(), colors.end()) + 1;

    ColoringResult result;
    result.colors = std::move(colors);
    result.num_colors = num_colors;
    result.num_conflicts = 0;
    result.num_rounds = 1;
    result.num_hubs = 0;
    result.init_seconds = init_time;
    result.compute_seconds = timer.elapsed() - init_time;
    result.elapsed_seconds = timer.elapsed();
    return result;
}

// ===========================================================================
// Parallel speculative coloring with multi-phase hub preprocessing
//
// Key optimizations over baseline:
//   - Fused init: degree + sum + sum_deg2 + partitioning in ONE parallel pass
//     (eliminates 2 extra fork/joins that cost ~20-50us each)
//   - Small graph bypass: n < 10000 skips all parallel overhead in init
//   - vector<char> instead of vector<bool> for used[] arrays
//   - Counting sort O(n) instead of comparison sort O(n log n)
//   - Sort skip for regular graphs (CV < 0.3)
//   - Merged barrier: counter reset inside finalize omp single block
//     (eliminates one barrier per round in the compute loop)
//   - Persistent parallel region, atomic worklist compaction, adaptive schedule
// ===========================================================================
ColoringResult color_parallel(const Graph& g, int num_threads) {
    Timer timer;
    timer.start();

#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#else
    (void)num_threads;
#endif

    int n = g.num_vertices;
    std::vector<int> colors(n, -1);

    // -----------------------------------------------------------------------
    // FUSED PARALLEL INIT: degree + stats + CV + partitioning in one pass.
    //
    // Previously this was 3 separate parallel passes (3 fork/joins).
    // Now we fuse degree computation, sum, sum_deg2, and hub classification
    // into a single parallel region. For small graphs (n < 10000), we skip
    // parallel overhead entirely since fork/join (~50us) exceeds the work.
    // -----------------------------------------------------------------------
    std::vector<int> deg(n);
    int max_deg = 0;
    double sum_deg = 0.0;
    double sum_deg2 = 0.0;
    double cv_deg = 0.0;

    std::vector<int> hub_vertices;
    std::vector<int> regular_vertices;

    // Threshold for hub detection: will be set after avg_deg is known.
    // For the fused pass, we do a 2-step: first compute stats, then partition.
    // This is still just 2 passes over deg[] (which is L1-hot after pass 1).

#ifdef _OPENMP
    if (n >= 10000) {
        // --- Large graph: parallel init ---
        // Pass 1: fused degree + max + sum + sum_deg2
        #pragma omp parallel for reduction(max:max_deg) reduction(+:sum_deg,sum_deg2) schedule(static)
        for (int v = 0; v < n; v++) {
            int d = g.degree(v);
            deg[v] = d;
            if (d > max_deg) max_deg = d;
            sum_deg += d;
            sum_deg2 += (double)d * d;
        }

        double avg_deg = (n > 0) ? sum_deg / n : 0.0;
        double var_deg = (n > 0) ? sum_deg2 / n - avg_deg * avg_deg : 0.0;
        cv_deg = (avg_deg > 0) ? std::sqrt(var_deg) / avg_deg : 0.0;

        // Set adaptive schedule based on CV
        if (cv_deg < 0.3) {
            omp_set_schedule(omp_sched_static, 0);
        } else {
            omp_set_schedule(omp_sched_dynamic, 256);
        }

        // Pass 2: parallel hub/regular partitioning (deg[] is L1-hot)
        double hub_threshold = std::max(avg_deg * 8.0, 1.0);
        int nt = omp_get_max_threads();
        std::vector<std::vector<int>> local_hubs(nt);
        std::vector<std::vector<int>> local_regulars(nt);
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            local_hubs[tid].reserve(n / (100 * nt) + 1);
            local_regulars[tid].reserve(n / nt + 1);
            #pragma omp for schedule(static)
            for (int v = 0; v < n; v++) {
                if (deg[v] > hub_threshold) {
                    local_hubs[tid].push_back(v);
                } else {
                    local_regulars[tid].push_back(v);
                }
            }
        }
        for (int t = 0; t < nt; t++) {
            hub_vertices.insert(hub_vertices.end(), local_hubs[t].begin(), local_hubs[t].end());
            regular_vertices.insert(regular_vertices.end(), local_regulars[t].begin(), local_regulars[t].end());
        }
    } else {
        // --- Small graph: sequential init (avoids fork/join overhead) ---
        for (int v = 0; v < n; v++) {
            int d = g.degree(v);
            deg[v] = d;
            if (d > max_deg) max_deg = d;
            sum_deg += d;
        }
        regular_vertices.resize(n);
        for (int v = 0; v < n; v++) regular_vertices[v] = v;
    }
#else
    for (int v = 0; v < n; v++) {
        int d = g.degree(v);
        deg[v] = d;
        if (d > max_deg) max_deg = d;
        sum_deg += d;
        sum_deg2 += (double)d * d;
    }
    double avg_deg_seq = (n > 0) ? sum_deg / n : 0.0;
    double hub_threshold_seq = std::max(avg_deg_seq * 8.0, 1.0);
    double var_deg_seq = (n > 0) ? sum_deg2 / n - avg_deg_seq * avg_deg_seq : 0.0;
    cv_deg = (avg_deg_seq > 0) ? std::sqrt(var_deg_seq) / avg_deg_seq : 0.0;
    hub_vertices.reserve(n / 100 + 1);
    regular_vertices.reserve(n);
    for (int v = 0; v < n; v++) {
        if (deg[v] > hub_threshold_seq && n > 1000) {
            hub_vertices.push_back(v);
        } else {
            regular_vertices.push_back(v);
        }
    }
#endif

    int palette_size = max_deg + 2;
    int num_hubs = static_cast<int>(hub_vertices.size());

    // -----------------------------------------------------------------------
    // PHASE 1: Sequential hub coloring.
    //
    // Hubs must be colored sequentially to guarantee optimal color quality.
    // Parallel hub coloring was tested but caused 70-80% color inflation on
    // power-law graphs (rmat_1m: 133 colors vs 77) because concurrent hub
    // coloring produces suboptimal assignments when hubs share neighborhoods.
    // The sequential cost is acceptable (Amdahl's serial fraction).
    // -----------------------------------------------------------------------
    std::sort(hub_vertices.begin(), hub_vertices.end(), [&deg](int a, int b) {
        return deg[a] > deg[b];
    });

    {
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
    }

    // -----------------------------------------------------------------------
    // LDF ordering: skip for regular graphs, counting sort for irregular.
    // -----------------------------------------------------------------------
    if (cv_deg >= 0.3) {
        counting_sort_by_degree_desc(regular_vertices, deg, max_deg);
    }

    int wsize = static_cast<int>(regular_vertices.size());
    std::vector<int> worklist(std::move(regular_vertices));
    std::vector<int> next_worklist(wsize);

    int total_conflicts = 0;
    int num_rounds = 0;
    std::atomic<int> next_wsize{0};

    double init_time = timer.elapsed();

    // -----------------------------------------------------------------------
    // PHASE 2: Parallel speculative coloring with conflict resolution.
    //
    // Two sub-phases per round:
    //   2a. Tentative greedy coloring (parallel)
    //   2b. Conflict detection + worklist compaction (parallel)
    // A barrier between 2a and 2b ensures all colors are visible before
    // conflict checking. Counter reset is merged into the finalize block.
    // -----------------------------------------------------------------------
    #pragma omp parallel
    {
        std::vector<char> used(palette_size, 0);

        while (wsize > 0) {
            // --- Tentative greedy coloring ---
            #pragma omp for schedule(runtime)
            for (int i = 0; i < wsize; i++) {
                int v = worklist[i];
                int d = deg[v];
                int reset_size = std::min(d + 2, palette_size);
                std::fill(used.begin(), used.begin() + reset_size, (char)0);
                for (const int* nb = g.neighbors_begin(v); nb != g.neighbors_end(v); nb++) {
                    int c = colors[*nb];
                    if (c >= 0 && c < palette_size) {
                        used[c] = 1;
                    }
                }
                for (int c = 0; ; c++) {
                    if (c >= palette_size || !used[c]) {
                        colors[v] = c;
                        break;
                    }
                }
            }
            // implicit barrier — all tentative colors visible

            // --- Conflict detection + parallel worklist compaction ---
            #pragma omp for schedule(runtime)
            for (int i = 0; i < wsize; i++) {
                int v = worklist[i];
                for (const int* nb = g.neighbors_begin(v); nb != g.neighbors_end(v); nb++) {
                    int u = *nb;
                    if (colors[v] == colors[u] && v > u) {
                        colors[v] = -1;
                        int idx = next_wsize.fetch_add(1, std::memory_order_relaxed);
                        next_worklist[idx] = v;
                        break;
                    }
                }
            }
            // implicit barrier — all conflicts detected

            #pragma omp single
            {
                num_rounds++;
                total_conflicts += next_wsize.load(std::memory_order_relaxed);
                std::swap(worklist, next_worklist);
                wsize = next_wsize.load(std::memory_order_relaxed);
                next_wsize.store(0, std::memory_order_relaxed);
            }
            // implicit barrier — all threads see updated wsize
        }
    }

    int num_colors = *std::max_element(colors.begin(), colors.end()) + 1;

    ColoringResult result;
    result.colors = std::move(colors);
    result.num_colors = num_colors;
    result.num_conflicts = total_conflicts;
    result.num_rounds = num_rounds;
    result.num_hubs = num_hubs;
    result.init_seconds = init_time;
    result.compute_seconds = timer.elapsed() - init_time;
    result.elapsed_seconds = timer.elapsed();
    return result;
}

// ===========================================================================
// Hybrid speculative + Jones-Plassmann coloring
//
// Phase 1: Sequential hub coloring
// Phase 2: ONE round of speculative coloring (colors ~99%+)
// Phase 3: JP refinement on tiny conflict set (conflict-free)
//
// Same init optimizations as color_parallel (fused pass, small graph bypass,
// vector<char>, counting sort, sort skip).
// ===========================================================================
ColoringResult color_hybrid(const Graph& g, int num_threads) {
    Timer timer;
    timer.start();

#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#else
    (void)num_threads;
#endif

    int n = g.num_vertices;
    std::vector<int> colors(n, -1);

    // -----------------------------------------------------------------------
    // FUSED PARALLEL INIT (same as color_parallel)
    // -----------------------------------------------------------------------
    std::vector<int> deg(n);
    int max_deg = 0;
    double sum_deg = 0.0;
    double sum_deg2 = 0.0;
    double cv_deg = 0.0;

    std::vector<int> hub_vertices;
    std::vector<int> regular_vertices;

#ifdef _OPENMP
    if (n >= 10000) {
        #pragma omp parallel for reduction(max:max_deg) reduction(+:sum_deg,sum_deg2) schedule(static)
        for (int v = 0; v < n; v++) {
            int d = g.degree(v);
            deg[v] = d;
            if (d > max_deg) max_deg = d;
            sum_deg += d;
            sum_deg2 += (double)d * d;
        }

        double avg_deg = (n > 0) ? sum_deg / n : 0.0;
        double var_deg = (n > 0) ? sum_deg2 / n - avg_deg * avg_deg : 0.0;
        cv_deg = (avg_deg > 0) ? std::sqrt(var_deg) / avg_deg : 0.0;

        double hub_threshold = std::max(avg_deg * 8.0, 1.0);
        int nt = omp_get_max_threads();
        std::vector<std::vector<int>> local_hubs(nt);
        std::vector<std::vector<int>> local_regulars(nt);
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            local_hubs[tid].reserve(n / (100 * nt) + 1);
            local_regulars[tid].reserve(n / nt + 1);
            #pragma omp for schedule(static)
            for (int v = 0; v < n; v++) {
                if (deg[v] > hub_threshold) {
                    local_hubs[tid].push_back(v);
                } else {
                    local_regulars[tid].push_back(v);
                }
            }
        }
        for (int t = 0; t < nt; t++) {
            hub_vertices.insert(hub_vertices.end(), local_hubs[t].begin(), local_hubs[t].end());
            regular_vertices.insert(regular_vertices.end(), local_regulars[t].begin(), local_regulars[t].end());
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
#else
    for (int v = 0; v < n; v++) {
        int d = g.degree(v);
        deg[v] = d;
        if (d > max_deg) max_deg = d;
        sum_deg += d;
        sum_deg2 += (double)d * d;
    }
    double avg_deg_seq = (n > 0) ? sum_deg / n : 0.0;
    double hub_threshold_seq = std::max(avg_deg_seq * 8.0, 1.0);
    double var_deg_seq = (n > 0) ? sum_deg2 / n - avg_deg_seq * avg_deg_seq : 0.0;
    cv_deg = (avg_deg_seq > 0) ? std::sqrt(var_deg_seq) / avg_deg_seq : 0.0;
    hub_vertices.reserve(n / 100 + 1);
    regular_vertices.reserve(n);
    for (int v = 0; v < n; v++) {
        if (deg[v] > hub_threshold_seq && n > 1000) {
            hub_vertices.push_back(v);
        } else {
            regular_vertices.push_back(v);
        }
    }
#endif

    int palette_size = max_deg + 2;
    int num_hubs = static_cast<int>(hub_vertices.size());

    // PHASE 1: Sequential hub coloring
    std::sort(hub_vertices.begin(), hub_vertices.end(), [&deg](int a, int b) {
        return deg[a] > deg[b];
    });

    {
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
    }

    // LDF ordering
    if (cv_deg >= 0.3) {
        counting_sort_by_degree_desc(regular_vertices, deg, max_deg);
    }

    int wsize = static_cast<int>(regular_vertices.size());

    // JP priorities for Phase 3 refinement
    std::vector<uint32_t> priority(n);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int v = 0; v < n; v++) {
        priority[v] = vertex_hash(v);
    }

    std::vector<int> worklist(std::move(regular_vertices));
    std::vector<int> conflict_list(wsize);
    std::atomic<int> conflict_count{0};

    int total_conflicts = 0;
    int num_rounds = 0;

    double init_time = timer.elapsed();

    // PHASE 2: One round of speculative coloring
    #pragma omp parallel
    {
        std::vector<char> used(palette_size, 0);

        #pragma omp for schedule(dynamic, 256)
        for (int i = 0; i < wsize; i++) {
            int v = worklist[i];
            int d = deg[v];
            int reset_size = std::min(d + 2, palette_size);
            std::fill(used.begin(), used.begin() + reset_size, (char)0);
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
        // implicit barrier

        #pragma omp for schedule(dynamic, 256)
        for (int i = 0; i < wsize; i++) {
            int v = worklist[i];
            for (const int* nb = g.neighbors_begin(v); nb != g.neighbors_end(v); nb++) {
                int u = *nb;
                if (colors[v] == colors[u] && v > u) {
                    colors[v] = -1;
                    int idx = conflict_count.fetch_add(1, std::memory_order_relaxed);
                    conflict_list[idx] = v;
                    break;
                }
            }
        }
    }

    num_rounds = 1;
    int num_conflicts_spec = conflict_count.load(std::memory_order_relaxed);
    total_conflicts = num_conflicts_spec;

    // PHASE 3: JP refinement on conflicting vertices
    int jp_wsize = num_conflicts_spec;
    if (jp_wsize > 0) {
        std::vector<int> jp_worklist(conflict_list.begin(), conflict_list.begin() + jp_wsize);
        std::vector<int> jp_next(jp_wsize);
        std::atomic<int> jp_next_wsize{0};

        #pragma omp parallel
        {
            std::vector<char> used(palette_size, 0);

            while (jp_wsize > 0) {
                #pragma omp for schedule(dynamic, 256)
                for (int i = 0; i < jp_wsize; i++) {
                    int v = jp_worklist[i];
                    uint32_t pv = priority[v];

                    bool is_local_max = true;
                    for (const int* nb = g.neighbors_begin(v); nb != g.neighbors_end(v); nb++) {
                        int u = *nb;
                        if (colors[u] >= 0) continue;
                        uint32_t pu = priority[u];
                        if (pu > pv || (pu == pv && u > v)) {
                            is_local_max = false;
                            break;
                        }
                    }

                    if (is_local_max) {
                        int d = deg[v];
                        int reset_size = std::min(d + 2, palette_size);
                        std::fill(used.begin(), used.begin() + reset_size, (char)0);
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
                    } else {
                        int idx = jp_next_wsize.fetch_add(1, std::memory_order_relaxed);
                        jp_next[idx] = v;
                    }
                }
                // implicit barrier

                #pragma omp single
                {
                    num_rounds++;
                    std::swap(jp_worklist, jp_next);
                    jp_wsize = jp_next_wsize.load(std::memory_order_relaxed);
                    jp_next_wsize.store(0, std::memory_order_relaxed);
                }
                // implicit barrier
            }
        }
    }

    int num_colors = *std::max_element(colors.begin(), colors.end()) + 1;

    ColoringResult result;
    result.colors = std::move(colors);
    result.num_colors = num_colors;
    result.num_conflicts = total_conflicts;
    result.num_rounds = num_rounds;
    result.num_hubs = num_hubs;
    result.init_seconds = init_time;
    result.compute_seconds = timer.elapsed() - init_time;
    result.elapsed_seconds = timer.elapsed();
    return result;
}

// ===========================================================================
// Coloring verification — O(V + E)
// ===========================================================================
bool verify_coloring(const Graph& g, const std::vector<int>& colors) {
    for (int v = 0; v < g.num_vertices; v++) {
        if (colors[v] < 0) return false;
        for (const int* nb = g.neighbors_begin(v); nb != g.neighbors_end(v); nb++) {
            if (colors[v] == colors[*nb]) return false;
        }
    }
    return true;
}
