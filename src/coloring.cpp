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
// GCC provides __gnu_parallel::sort — a parallel multiway mergesort.
// On GHC (GCC), this parallelizes the LDF sort across all threads.
// On macOS (Clang), we fall back to std::sort.
#if defined(__GNUC__) && !defined(__clang__)
#include <parallel/algorithm>
#define PARALLEL_SORT __gnu_parallel::sort
#else
#define PARALLEL_SORT std::sort
#endif
#else
#define PARALLEL_SORT std::sort
#endif

// ===========================================================================
// Hash function for Jones-Plassmann priorities.
//
// Used by the hybrid algorithm's JP refinement phase. Assigns each vertex
// a deterministic pseudo-random 32-bit priority. The hash must be:
//   - Deterministic: same vertex always gets same priority
//   - Well-distributed: minimizes priority collisions among neighbors
//   - Fast: called once per vertex during init
//
// We use a multiply-shift hash (Murmur-style finalizer) which has
// excellent avalanche properties — flipping any input bit changes ~50%
// of output bits.
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
// The simplest correct algorithm. Visits vertices in order 0..n-1 and
// assigns each the smallest color not used by any already-colored neighbor.
//
// Properties:
//   - Deterministic, reproducible results
//   - Uses at most max_degree + 1 colors (greedy bound)
//   - Zero conflicts by construction (sequential)
//   - Serves as the quality baseline for parallel algorithms
//
// Implementation notes:
//   - Single shared `used[]` array sized to max_degree + 2, reset per vertex
//   - Only resets used[0..deg+1], not the whole array (cache-friendly)
//   - Iterates neighbors via CSR pointers for spatial locality
// ===========================================================================
ColoringResult color_sequential(const Graph& g) {
    Timer timer;
    timer.start();

    int n = g.num_vertices;
    std::vector<int> colors(n, -1);

    // Find maximum degree to size the palette (used[] array).
    // Greedy coloring never needs more than max_degree + 1 colors.
    int max_deg = 0;
    for (int v = 0; v < n; v++) max_deg = std::max(max_deg, g.degree(v));
    int palette_size = max_deg + 2;  // +2 for safety margin
    std::vector<bool> used(palette_size, false);

    double init_time = timer.elapsed();

    for (int v = 0; v < n; v++) {
        int deg = g.degree(v);

        // Reset only the portion of used[] we'll need (deg+2 entries).
        // This avoids clearing the entire palette for low-degree vertices,
        // which is critical on power-law graphs where most vertices have
        // small degree but a few hubs have degree in the thousands.
        std::fill(used.begin(), used.begin() + deg + 2, false);

        // Mark colors of already-colored neighbors as used
        for (const int* nb = g.neighbors_begin(v); nb != g.neighbors_end(v); nb++) {
            int c = colors[*nb];
            if (c >= 0 && c < palette_size) {
                used[c] = true;
            }
        }

        // Assign the smallest available color
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
// Algorithm overview (Gebremedhin-Manne with extensions):
//
//   INIT (parallel):
//     1. Precompute degree array (parallel fill + reduction for max/sum)
//     2. Compute degree CV to choose static vs dynamic scheduling
//     3. Partition vertices into hubs (degree > 4x average) and regulars
//     4. Sort regulars by descending degree (LDF ordering)
//
//   PHASE 1 — Hub coloring (sequential):
//     Color the ~1-5% highest-degree vertices sequentially. Since these
//     "hub" vertices are connected to many others, coloring them first
//     eliminates the main source of parallel conflicts. This is a small
//     amount of work (few vertices) but has outsized impact on conflict
//     rates for power-law graphs.
//
//   PHASE 2 — Speculative coloring (parallel, iterative):
//     Round loop (converges in 1-2 rounds empirically):
//       2a. TENTATIVE COLORING: Each thread greedily colors its assigned
//           vertices using a thread-local used[] array. Reads neighbor
//           colors (may be stale from concurrent writes → conflicts).
//       2b. CONFLICT DETECTION + COMPACTION: Each thread checks its
//           vertices for same-color neighbors. Conflicts are resolved by
//           the higher-indexed vertex (v > u → v loses). Conflicting
//           vertices are added to the next worklist via atomic fetch_add.
//       2c. FINALIZE: Single thread swaps worklists and updates size.
//
// Key optimizations:
//   - Precomputed degree array: sorting uses contiguous int reads instead
//     of indirect CSR lookups (row_offsets[v+1] - row_offsets[v]).
//     This dramatically improves sort performance (cache-friendly).
//   - Persistent parallel region: one #pragma omp parallel wraps the
//     entire iterative loop, avoiding fork/join overhead per round.
//   - Per-thread used[] allocated once inside the parallel region,
//     reused across all rounds (zero heap allocation in the hot loop).
//   - Parallel worklist compaction: conflicting vertices write directly
//     to next_worklist via atomic<int> fetch_add, eliminating the
//     sequential scan that was previously in an omp single block.
//   - Adaptive scheduling: regular graphs (CV < 0.3) use static schedule
//     for low overhead; irregular graphs use dynamic(256) for load balance.
//   - Parallel init: degree stats via omp reduction, hub partitioning
//     via per-thread local lists, LDF sort via __gnu_parallel::sort.
//
// Conflict resolution rule: when vertices u and v are adjacent and both
// colored c in the same round, the vertex with the higher index (v > u)
// is uncolored and retried. This is a deterministic, symmetric rule that
// doesn't require atomic operations on the colors[] array.
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
    // PARALLEL INIT: precompute degrees, statistics, partitioning, sorting
    // -----------------------------------------------------------------------

    // Precompute degree array into contiguous memory.
    // This avoids repeated CSR lookups (row_offsets[v+1] - row_offsets[v])
    // during sorting and the coloring loop. The contiguous layout enables
    // hardware prefetching and SIMD-friendly access patterns.
    std::vector<int> deg(n);
    int max_deg = 0;
    double sum_deg = 0.0;
#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp for reduction(max:max_deg) reduction(+:sum_deg) schedule(static)
        for (int v = 0; v < n; v++) {
            int d = g.degree(v);
            deg[v] = d;
            if (d > max_deg) max_deg = d;
            sum_deg += d;
        }
    }
#else
    for (int v = 0; v < n; v++) {
        int d = g.degree(v);
        deg[v] = d;
        if (d > max_deg) max_deg = d;
        sum_deg += d;
    }
#endif
    int palette_size = max_deg + 2;
    double avg_deg = (n > 0) ? sum_deg / n : 0.0;

    // Adaptive scheduling decision based on degree coefficient of variation.
    // CV = std_dev(degree) / mean(degree).
    //   CV < 0.3 → uniform graph (grid, ER) → static scheduling is optimal
    //              because all vertices have similar work, and static has
    //              zero runtime overhead.
    //   CV >= 0.3 → skewed graph (RMAT, social) → dynamic scheduling needed
    //               because hub vertices take 100-1000x longer than leaves.
    //               Chunk size 256 balances scheduling granularity vs overhead.
#ifdef _OPENMP
    {
        double sum_deg2 = 0.0;
        #pragma omp parallel for reduction(+:sum_deg2) schedule(static)
        for (int v = 0; v < n; v++) {
            double d = deg[v];
            sum_deg2 += d * d;
        }
        double var_deg = (n > 0) ? sum_deg2 / n - avg_deg * avg_deg : 0.0;
        double cv_deg = (avg_deg > 0) ? std::sqrt(var_deg) / avg_deg : 0.0;
        if (cv_deg < 0.3) {
            omp_set_schedule(omp_sched_static, 0);
        } else {
            omp_set_schedule(omp_sched_dynamic, 256);
        }
    }
#endif

    // -----------------------------------------------------------------------
    // Hub/regular vertex partitioning.
    //
    // Hubs are vertices with degree > 4x the average. On power-law graphs
    // (RMAT, social networks), these represent ~1-5% of vertices but are
    // responsible for the vast majority of conflicts when colored in parallel.
    //
    // Each thread builds its own local hub and regular lists, then we merge.
    // This avoids synchronization during the partitioning scan.
    // -----------------------------------------------------------------------
    double hub_threshold = std::max(avg_deg * 4.0, 1.0);

    std::vector<int> hub_vertices;
    std::vector<int> regular_vertices;

#ifdef _OPENMP
    if (n > 1000) {
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
        // Merge per-thread lists sequentially (cheap: just pointer copies)
        for (int t = 0; t < nt; t++) {
            hub_vertices.insert(hub_vertices.end(), local_hubs[t].begin(), local_hubs[t].end());
            regular_vertices.insert(regular_vertices.end(), local_regulars[t].begin(), local_regulars[t].end());
        }
    } else {
        // Small graphs: no hubs, all vertices are regular
        regular_vertices.resize(n);
        for (int v = 0; v < n; v++) regular_vertices[v] = v;
    }
#else
    hub_vertices.reserve(n / 100 + 1);
    regular_vertices.reserve(n);
    for (int v = 0; v < n; v++) {
        if (deg[v] > hub_threshold && n > 1000) {
            hub_vertices.push_back(v);
        } else {
            regular_vertices.push_back(v);
        }
    }
#endif

    int num_hubs = static_cast<int>(hub_vertices.size());

    // -----------------------------------------------------------------------
    // PHASE 1: Sequential hub coloring.
    //
    // Sort hubs by descending degree so the highest-degree vertices get
    // first pick of colors. This produces optimal greedy coloring for hubs.
    // Since hubs are few (<5% of vertices), this sequential phase is fast.
    // -----------------------------------------------------------------------
    std::sort(hub_vertices.begin(), hub_vertices.end(), [&deg](int a, int b) {
        return deg[a] > deg[b];
    });

    {
        std::vector<bool> used(palette_size, false);
        for (int v : hub_vertices) {
            int d = deg[v];
            std::fill(used.begin(), used.begin() + std::min(d + 2, palette_size), false);
            for (const int* nb = g.neighbors_begin(v); nb != g.neighbors_end(v); nb++) {
                int c = colors[*nb];
                if (c >= 0 && c < palette_size) used[c] = true;
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
    // LDF (Largest Degree First) sort on regular vertices.
    //
    // Processing high-degree vertices first reduces conflicts because they
    // get access to the full color palette before low-degree vertices
    // constrain it. The sort comparator reads from the precomputed deg[]
    // array (contiguous ints) rather than the CSR structure, which improves
    // cache hit rates during the O(n log n) sort.
    //
    // On GCC (GHC machines), __gnu_parallel::sort distributes the sort
    // across all available threads.
    // -----------------------------------------------------------------------
    PARALLEL_SORT(regular_vertices.begin(), regular_vertices.end(), [&deg](int a, int b) {
        return deg[a] > deg[b];
    });

    // Pre-allocate worklist buffers — no heap allocation during coloring
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
    // Single persistent parallel region — threads are created once and
    // reused across all rounds. The while loop checks the shared `wsize`
    // variable; all threads see the updated value after each round's
    // implicit barrier.
    //
    // Memory layout per thread:
    //   - used[]: thread-local boolean array, allocated once, reused per round
    //   - Reads from shared colors[] and worklist[] (read-heavy, few writes)
    //   - Writes to shared colors[] (one write per vertex per round)
    //   - Atomic writes to next_worklist[] (only for conflicting vertices)
    // -----------------------------------------------------------------------
    #pragma omp parallel
    {
        std::vector<bool> used(palette_size, false);

        while (wsize > 0) {
            // --- Phase 2a: Tentative greedy coloring ---
            // Each thread colors its portion of the worklist. Neighbor color
            // reads may be stale (concurrent writes), which can cause conflicts.
            // These are detected and resolved in Phase 2b.
            #pragma omp for schedule(runtime)
            for (int i = 0; i < wsize; i++) {
                int v = worklist[i];
                int d = deg[v];
                int reset_size = std::min(d + 2, palette_size);
                std::fill(used.begin(), used.begin() + reset_size, false);
                for (const int* nb = g.neighbors_begin(v); nb != g.neighbors_end(v); nb++) {
                    int c = colors[*nb];
                    if (c >= 0 && c < palette_size) {
                        used[c] = true;
                    }
                }
                for (int c = 0; ; c++) {
                    if (c >= palette_size || !used[c]) {
                        colors[v] = c;
                        break;
                    }
                }
            }
            // implicit barrier — all tentative colors are visible

            // --- Phase 2b: Conflict detection + parallel worklist compaction ---
            // Reset the atomic counter (single thread, others wait at barrier)
            #pragma omp single
            {
                next_wsize.store(0, std::memory_order_relaxed);
            }
            // implicit barrier — counter is 0 before detection starts

            // Each thread scans its vertices for conflicts. When adjacent
            // vertices v and u have the same color, the higher-index vertex
            // (v > u) loses and is added to the next worklist.
            //
            // The atomic fetch_add gives each conflicting vertex a unique
            // index into next_worklist[], enabling fully parallel compaction
            // without locks or critical sections.
            #pragma omp for schedule(runtime)
            for (int i = 0; i < wsize; i++) {
                int v = worklist[i];
                bool has_conflict = false;
                for (const int* nb = g.neighbors_begin(v); nb != g.neighbors_end(v); nb++) {
                    int u = *nb;
                    if (colors[v] == colors[u] && v > u) {
                        has_conflict = true;
                        break;
                    }
                }
                if (has_conflict) {
                    colors[v] = -1;  // mark for recoloring
                    int idx = next_wsize.fetch_add(1, std::memory_order_relaxed);
                    next_worklist[idx] = v;
                }
            }
            // implicit barrier — all conflicts detected

            // --- Phase 2c: Finalize round ---
            // Single thread swaps worklists and updates the shared loop variable.
            // After the implicit barrier, all threads see the new wsize.
            #pragma omp single
            {
                num_rounds++;
                total_conflicts += next_wsize.load(std::memory_order_relaxed);
                std::swap(worklist, next_worklist);
                wsize = next_wsize.load(std::memory_order_relaxed);
            }
            // implicit barrier — all threads see updated wsize for while check
        }
    }
    // Single join point — parallel region ends

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
// Motivation: Speculative coloring is fast but may require multiple rounds
// of conflict resolution. Jones-Plassmann (JP) is conflict-free but slow
// due to many rounds with small independent sets. The hybrid combines both:
//
//   Phase 1: Sequential hub coloring (same as color_parallel)
//   Phase 2: ONE round of speculative coloring (fast, colors ~99%+ correctly)
//   Phase 3: JP refinement on the tiny conflict set (conflict-free resolution)
//
// JP works well here because the conflict set is tiny (typically <0.1% of
// vertices), so the round overhead that cripples standalone JP is negligible.
//
// Jones-Plassmann algorithm (Phase 3):
//   Each vertex has a deterministic hash-based priority. In each round,
//   a vertex is eligible for coloring only if it has the highest priority
//   among all its uncolored neighbors (forming an independent set). Since
//   independent set members have no edges between them, coloring them
//   simultaneously is conflict-free by construction.
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
    // PARALLEL INIT: same as color_parallel
    // -----------------------------------------------------------------------
    std::vector<int> deg(n);
    int max_deg = 0;
    double sum_deg = 0.0;
#ifdef _OPENMP
    #pragma omp parallel for reduction(max:max_deg) reduction(+:sum_deg) schedule(static)
#endif
    for (int v = 0; v < n; v++) {
        int d = g.degree(v);
        deg[v] = d;
        if (d > max_deg) max_deg = d;
        sum_deg += d;
    }
    int palette_size = max_deg + 2;
    double avg_deg = (n > 0) ? sum_deg / n : 0.0;

    // Hub partitioning (parallel, per-thread local lists)
    double hub_threshold = std::max(avg_deg * 4.0, 1.0);
    std::vector<int> hub_vertices;
    std::vector<int> regular_vertices;

#ifdef _OPENMP
    if (n > 1000) {
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
        regular_vertices.resize(n);
        for (int v = 0; v < n; v++) regular_vertices[v] = v;
    }
#else
    hub_vertices.reserve(n / 100 + 1);
    regular_vertices.reserve(n);
    for (int v = 0; v < n; v++) {
        if (deg[v] > hub_threshold && n > 1000) {
            hub_vertices.push_back(v);
        } else {
            regular_vertices.push_back(v);
        }
    }
#endif

    int num_hubs = static_cast<int>(hub_vertices.size());

    // -----------------------------------------------------------------------
    // PHASE 1: Sequential hub coloring (identical to color_parallel)
    // -----------------------------------------------------------------------
    std::sort(hub_vertices.begin(), hub_vertices.end(), [&deg](int a, int b) {
        return deg[a] > deg[b];
    });

    {
        std::vector<bool> used(palette_size, false);
        for (int v : hub_vertices) {
            int d = deg[v];
            std::fill(used.begin(), used.begin() + std::min(d + 2, palette_size), false);
            for (const int* nb = g.neighbors_begin(v); nb != g.neighbors_end(v); nb++) {
                int c = colors[*nb];
                if (c >= 0 && c < palette_size) used[c] = true;
            }
            for (int c = 0; ; c++) {
                if (c >= palette_size || !used[c]) {
                    colors[v] = c;
                    break;
                }
            }
        }
    }

    // LDF sort on regular vertices using precomputed degrees
    PARALLEL_SORT(regular_vertices.begin(), regular_vertices.end(), [&deg](int a, int b) {
        return deg[a] > deg[b];
    });

    int wsize = static_cast<int>(regular_vertices.size());

    // JP priorities for Phase 3 refinement (parallel hash computation)
    std::vector<uint32_t> priority(n);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int v = 0; v < n; v++) {
        priority[v] = vertex_hash(v);
    }

    // Buffers for speculative coloring and conflict collection
    std::vector<int> worklist(std::move(regular_vertices));
    std::vector<int> conflict_list(wsize);
    std::atomic<int> conflict_count{0};

    int total_conflicts = 0;
    int num_rounds = 0;

    double init_time = timer.elapsed();

    // -----------------------------------------------------------------------
    // PHASE 2: One round of speculative coloring.
    //
    // Colors all regular vertices in parallel, then detects conflicts.
    // The conflict set is typically <0.1% of vertices thanks to hub
    // preprocessing + LDF ordering.
    // -----------------------------------------------------------------------
    #pragma omp parallel
    {
        std::vector<bool> used(palette_size, false);

        // Tentative greedy coloring
        #pragma omp for schedule(dynamic, 256)
        for (int i = 0; i < wsize; i++) {
            int v = worklist[i];
            int d = deg[v];
            int reset_size = std::min(d + 2, palette_size);
            std::fill(used.begin(), used.begin() + reset_size, false);
            for (const int* nb = g.neighbors_begin(v); nb != g.neighbors_end(v); nb++) {
                int c = colors[*nb];
                if (c >= 0 && c < palette_size) used[c] = true;
            }
            for (int c = 0; ; c++) {
                if (c >= palette_size || !used[c]) {
                    colors[v] = c;
                    break;
                }
            }
        }
        // implicit barrier

        // Conflict detection + parallel compaction into conflict_list
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

    // -----------------------------------------------------------------------
    // PHASE 3: Jones-Plassmann refinement on conflicting vertices.
    //
    // Only runs if there are conflicts (typically 0-300 vertices out of
    // hundreds of thousands). JP guarantees conflict-free resolution:
    //   - Each round identifies vertices with highest priority among
    //     their uncolored neighbors (independent set)
    //   - Colors them greedily (no conflicts possible within the set)
    //   - Remaining uncolored vertices go to next round
    //
    // Since the conflict set is tiny, the per-round barrier overhead
    // (which cripples standalone JP on large graphs) is negligible here.
    // -----------------------------------------------------------------------
    int jp_wsize = num_conflicts_spec;
    if (jp_wsize > 0) {
        std::vector<int> jp_worklist(conflict_list.begin(), conflict_list.begin() + jp_wsize);
        std::vector<int> jp_next(jp_wsize);
        std::atomic<int> jp_next_wsize{0};

        #pragma omp parallel
        {
            std::vector<bool> used(palette_size, false);

            while (jp_wsize > 0) {
                #pragma omp single
                {
                    jp_next_wsize.store(0, std::memory_order_relaxed);
                }
                // implicit barrier

                #pragma omp for schedule(dynamic, 256)
                for (int i = 0; i < jp_wsize; i++) {
                    int v = jp_worklist[i];
                    uint32_t pv = priority[v];

                    // Check if v has highest priority among uncolored neighbors.
                    // Ties broken by vertex index (higher index wins).
                    bool is_local_max = true;
                    for (const int* nb = g.neighbors_begin(v); nb != g.neighbors_end(v); nb++) {
                        int u = *nb;
                        if (colors[u] >= 0) continue;  // already colored
                        uint32_t pu = priority[u];
                        if (pu > pv || (pu == pv && u > v)) {
                            is_local_max = false;
                            break;
                        }
                    }

                    if (is_local_max) {
                        // Color greedily — safe because no adjacent uncolored
                        // vertex is also being colored this round
                        int d = deg[v];
                        int reset_size = std::min(d + 2, palette_size);
                        std::fill(used.begin(), used.begin() + reset_size, false);
                        for (const int* nb = g.neighbors_begin(v); nb != g.neighbors_end(v); nb++) {
                            int c = colors[*nb];
                            if (c >= 0 && c < palette_size) used[c] = true;
                        }
                        for (int c = 0; ; c++) {
                            if (c >= palette_size || !used[c]) {
                                colors[v] = c;
                                break;
                            }
                        }
                    } else {
                        // Not in this round's independent set — defer
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
// Coloring verification.
//
// Checks two properties:
//   1. Every vertex has been assigned a color (>= 0)
//   2. No two adjacent vertices share the same color
//
// Runs in O(V + E) time. Used after every benchmark run to validate results.
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
