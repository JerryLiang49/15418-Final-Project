// Parallel graph coloring algorithms
//
// Implements three coloring strategies:
// 1. Sequential greedy — baseline
// 2. Parallel speculative — Gebremedhin-Manne with hub preprocessing
// 3. Hybrid spec + JP — speculative + Jones-Plassmann refinement

#include "coloring.h"
#include "timer.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <numeric>
#include <vector>
#include <omp.h>

// Largest Degree First sorting 
// sorts vertices by descending degree using bucket approach
// better than O(log n) comparison sort and better memory access 
static void counting_sort_by_degree_desc(std::vector<int>& vertices, 
                                         const std::vector<int>& deg, 
                                         int max_deg){
    int n = static_cast<int>(vertices.size());
    if (n <= 1) return;

    // Create max_deg + 1 buckets 
    // Bucket d holds vertices whose degree is d 
    std::vector<std::vector<int>> buckets(max_deg + 1);
    for (int i = 0; i < n; i++) {
        buckets[deg[vertices[i]]].push_back(vertices[i]);
    }
    // rebuild vertices by iterating through buckets from highest d to lowest 
    int idx = 0;
    for (int d = max_deg; d >= 0; d--) {
        for (int v : buckets[d]) {
            vertices[idx++] = v;
        }
    }
}

// Hash function for Jones-Plassmann priorities.
// Assigns each vertex a deterministic pseudo-random 32-bit priority. 
// Uses multiply-shift hash
static inline uint32_t vertex_hash(int v) {
    uint32_t h = static_cast<uint32_t>(v);
    h = ((h >> 16) ^ h) * 0x45d9f3b;
    h = ((h >> 16) ^ h) * 0x45d9f3b;
    h = (h >> 16) ^ h;
    return h;
}

// Sequential greedy coloring
// Visits vertices in order 0..n-1, assigns each the smallest color not
// used by any already-colored neighbor. Optimal greedy baseline.
ColoringResult color_sequential(const Graph& g) {
    Timer timer;
    timer.start();

    int n = g.num_vertices;
    std::vector<int> colors(n, -1);

    int max_deg = 0;
    for (int v = 0; v < n; v++) {
        max_deg = std::max(max_deg, g.degree(v));
    } 
    // size of temporary array used to track which colors are currently unavailable
    int palette_size = max_deg + 2; 
    std::vector<char> used(palette_size, 0); // used (unavailable colors)

    double init_time = timer.elapsed();
    
    // greedily color the vertices in order 
    for (int v = 0; v < n; v++) { // loop thru all vertices in order 
        int deg = g.degree(v);
        // reset the unavailable colors from the prev vertex's iteration 
        std::fill(used.begin(), used.begin() + deg + 2, (char)0);
        // scan neighbors 
        for (const int* neighbor = g.neighbors_begin(v); neighbor != g.neighbors_end(v); neighbor++) {
            int c = colors[*neighbor];
            if (c >= 0 && c < palette_size) {
                used[c] = 1; // mark neighbors color as used 
            }
        }
        // look through available colors 
        for (int c = 0; ; c++) {
            if (c >= palette_size || !used[c]) {
                colors[v] = c;
                break;
            }
        }
    }

    // update metrics 
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

// Parallel speculative coloring with multi-phase hub preprocessing
// Adapts from Gebremedhin-Manne algorithm
ColoringResult color_parallel(const Graph& g, int num_threads) {
    Timer timer;
    timer.start();
    omp_set_num_threads(num_threads);

    int n = g.num_vertices;
    std::vector<int> colors(n, -1);

    // Fused parallel init: degree + stats + CV + partitioning in one pass.
    // Previously this was 3 separate parallel passes (3 fork/joins).
    // Now we combine all the init stuff into single parallel region. 

    std::vector<int> deg(n);
    int max_deg = 0;
    double sum_deg = 0.0;
    double sum_deg2 = 0.0;
    double cv_deg = 0.0;

    std::vector<int> hub_vertices;
    std::vector<int> regular_vertices;

    // Threshold for hub detection will be set after avg_deg is known
    // Hub is any vertex with a high degree (higher than threshold)

    // For small graphs (n < 10000), we skip parallel overhead entirely since fork/join exceeds the work.
    if (n >= 10000) {
        // Pass: fused degree + max + sum + sum_deg2
        #pragma omp parallel for reduction(max:max_deg) reduction(+:sum_deg,sum_deg2) schedule(static)
        for (int v = 0; v < n; v++) {
            int d = g.degree(v);
            deg[v] = d;
            if (d > max_deg) {
                max_deg = d;
            } 
            sum_deg += d;
            sum_deg2 += (double)d * d;
        }

        double avg_deg = (n > 0) ? sum_deg / n : 0.0;
        double var_deg = (n > 0) ? sum_deg2 / n - avg_deg * avg_deg : 0.0;
        // coefficient of variation 
        cv_deg = (avg_deg > 0) ? std::sqrt(var_deg) / avg_deg : 0.0;

        // Set adaptive schedule based on CV
        if (cv_deg < 0.3) {
            omp_set_schedule(omp_sched_static, 0);
        } else {
            omp_set_schedule(omp_sched_dynamic, 256);
        }

        // Pass 2: parallel hub/regular partitioning (deg[] should already be in cache)
        double hub_threshold = std::max(avg_deg * 8.0, 1.0);
        int num_threads = omp_get_max_threads();
        // each thread gets local arrays 
        std::vector<std::vector<int>> local_hubs(num_threads);
        std::vector<std::vector<int>> local_regulars(num_threads);
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            local_hubs[tid].reserve(n / (100 * num_threads) + 1);
            local_regulars[tid].reserve(n / num_threads + 1);
            #pragma omp for schedule(static)
            for (int v = 0; v < n; v++) {
                if (deg[v] > hub_threshold) {
                    local_hubs[tid].push_back(v);
                } else {
                    local_regulars[tid].push_back(v);
                }
            }
        }
        // merge thread local lists to global list 
        for (int t = 0; t < num_threads; t++) {
            hub_vertices.insert(hub_vertices.end(), local_hubs[t].begin(), local_hubs[t].end());
            regular_vertices.insert(regular_vertices.end(), local_regulars[t].begin(), local_regulars[t].end());
        }

    } else {
        // Small graph: sequential init
        for (int v = 0; v < n; v++) {
            int d = g.degree(v);
            deg[v] = d;
            if (d > max_deg) max_deg = d;
            sum_deg += d;
        }
        regular_vertices.resize(n);
        for (int v = 0; v < n; v++) regular_vertices[v] = v;
    }

    int palette_size = max_deg + 2;
    int num_hubs = static_cast<int>(hub_vertices.size());

    // PHASE 1: Sequential hub coloring.
    // Hubs colored sequentially to guarantee optimal color quality.
    // Parallel hub coloring was tested but caused worse coloring results

    // sort hub vertices by degree 
    std::sort(hub_vertices.begin(), hub_vertices.end(), [&deg](int a, int b) {
        return deg[a] > deg[b];
    });
    // sequential 
    {
        std::vector<char> used(palette_size, 0);
        for (int v : hub_vertices) {
            int d = deg[v];
            std::fill(used.begin(), used.begin() + std::min(d + 2, palette_size), (char)0);
            for (const int* neighbor = g.neighbors_begin(v); neighbor != g.neighbors_end(v); neighbor++) {
                int c = colors[*neighbor];
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

    // If graph is irregular, then order non hub vertices by LDF 
    if (cv_deg >= 0.3) {
        counting_sort_by_degree_desc(regular_vertices, deg, max_deg);
    }

    int wsize = static_cast<int>(regular_vertices.size());

    // work list for speculative rounds 

    std::vector<int> worklist(std::move(regular_vertices)); // vertices to be colored this round 
    std::vector<int> next_worklist(wsize); // vertices that lose conflict and need to color next round 

    int total_conflicts = 0;
    int num_rounds = 0;
    std::atomic<int> next_wsize{0};

    double init_time = timer.elapsed();

    // PHASE 2: Parallel speculative coloring with conflict resolution.
    // Two sub-phases per round:
    // 2a: Tentative greedy coloring (parallel)
    // 2b: Conflict detection + worklist compaction (parallel)
    // Barrier between 2a and 2 b 
    #pragma omp parallel // persistent parallel section 
    {
        std::vector<char> used(palette_size, 0); // each thread gets its own used array 

        while (wsize > 0) { // loop until no more vertices in work list 
            // Phase 2a: Tentative greedy coloring
            // Split worklist across threads 
            #pragma omp for schedule(runtime)
            for (int i = 0; i < wsize; i++) {
                int v = worklist[i];
                int d = deg[v];
                int reset_size = std::min(d + 2, palette_size);
                std::fill(used.begin(), used.begin() + reset_size, (char)0);
                // mark neighbor colors as unavailable 
                for (const int* neighbor = g.neighbors_begin(v); neighbor != g.neighbors_end(v); neighbor++) {
                    int c = colors[*neighbor];
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
            // implicit barrier so that all tentative colors visible

            // Phase 2b: Conflict detection + parallel worklist compaction
            #pragma omp for schedule(runtime)
            for (int i = 0; i < wsize; i++) {
                int v = worklist[i];
                // detect same color neighbor and add to next worklist 
                for (const int* neighbor = g.neighbors_begin(v); neighbor != g.neighbors_end(v); neighbor++) {
                    int u = *neighbor;
                    if (colors[v] == colors[u] && v > u) {
                        colors[v] = -1;
                        int idx = next_wsize.fetch_add(1, std::memory_order_relaxed);
                        next_worklist[idx] = v;
                        break;
                    }
                }
            }
            // implicit barrier: all conflicts detected

            #pragma omp single
            {
                num_rounds++;
                total_conflicts += next_wsize.load(std::memory_order_relaxed);
                std::swap(worklist, next_worklist); // next worklist is worklist for next round 
                wsize = next_wsize.load(std::memory_order_relaxed);
                next_wsize.store(0, std::memory_order_relaxed);
            }
            // implicit barrier
        }
    }

    // update metrics 
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

// Hybrid speculative + Jones-Plassmann coloring
// Phase 1: Sequential hub coloring
// Phase 2: ONE round of speculative coloring (colors ~99%+)
// Phase 3: JP refinement on tiny conflict set (conflict-free)
// Same init optimizations as color_parallel
ColoringResult color_hybrid(const Graph& g, int num_threads) {
    Timer timer;
    timer.start();

    omp_set_num_threads(num_threads);

    int n = g.num_vertices;
    std::vector<int> colors(n, -1);

    // Same init as color parallel 
    std::vector<int> deg(n);
    int max_deg = 0;
    double sum_deg = 0.0;
    double sum_deg2 = 0.0;
    double cv_deg = 0.0;

    std::vector<int> hub_vertices;
    std::vector<int> regular_vertices;

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
        int num_threads = omp_get_max_threads();
        std::vector<std::vector<int>> local_hubs(num_threads);
        std::vector<std::vector<int>> local_regulars(num_threads);
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            local_hubs[tid].reserve(n / (100 * num_threads) + 1);
            local_regulars[tid].reserve(n / num_threads + 1);
            #pragma omp for schedule(static)
            for (int v = 0; v < n; v++) {
                if (deg[v] > hub_threshold) {
                    local_hubs[tid].push_back(v);
                } else {
                    local_regulars[tid].push_back(v);
                }
            }
        }
        for (int t = 0; t < num_threads; t++) {
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
            for (const int* neighbor = g.neighbors_begin(v); neighbor != g.neighbors_end(v); neighbor++) {
                int c = colors[*neighbor];
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

    // LDF ordering for irregular non hub vertices 
    if (cv_deg >= 0.3) {
        counting_sort_by_degree_desc(regular_vertices, deg, max_deg);
    }

    int wsize = static_cast<int>(regular_vertices.size());

    // JP priorities for Phase 3 refinement
    std::vector<uint32_t> priority(n);
    #pragma omp parallel for schedule(static)
    for (int v = 0; v < n; v++) {
        priority[v] = vertex_hash(v);
    }

    std::vector<int> worklist(std::move(regular_vertices));
    std::vector<int> conflict_list(wsize);
    std::atomic<int> conflict_count{0};

    int total_conflicts = 0;
    int num_rounds = 0;

    double init_time = timer.elapsed();

    // PHASE 2: Only one round of speculative coloring
    #pragma omp parallel
    {
        std::vector<char> used(palette_size, 0);

        #pragma omp for schedule(dynamic, 256)
        for (int i = 0; i < wsize; i++) {
            int v = worklist[i];
            int d = deg[v];
            int reset_size = std::min(d + 2, palette_size);
            std::fill(used.begin(), used.begin() + reset_size, (char)0);
            for (const int* neighbor = g.neighbors_begin(v); neighbor != g.neighbors_end(v); neighbor++) {
                int c = colors[*neighbor];
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
            for (const int* neighbor = g.neighbors_begin(v); neighbor != g.neighbors_end(v); neighbor++) {
                int u = *neighbor;
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

    // PHASE 3: JP refinement on the small set of conflicting vertices left 
    int jp_wsize = num_conflicts_spec;
    if (jp_wsize > 0) {
        std::vector<int> jp_worklist(conflict_list.begin(), conflict_list.begin() + jp_wsize);
        std::vector<int> jp_next(jp_wsize);
        std::atomic<int> jp_next_wsize{0};

        // Parallel JP refinement 
        #pragma omp parallel
        {
            std::vector<char> used(palette_size, 0);

            while (jp_wsize > 0) { // repeat until no more conflicting 
                #pragma omp for schedule(dynamic, 256)
                // Check if a vertex is a local maximum
                for (int i = 0; i < jp_wsize; i++) {
                    int v = jp_worklist[i];
                    uint32_t pv = priority[v];

                    bool is_local_max = true;
                    for (const int* neighbor = g.neighbors_begin(v); neighbor != g.neighbors_end(v); neighbor++) {
                        int u = *neighbor;
                        if (colors[u] >= 0) continue;
                        uint32_t pu = priority[u];
                        if (pu > pv || (pu == pv && u > v)) {
                            is_local_max = false;
                            break;
                        }
                    }
                    // if vertex is local max, color it greedily 
                    if (is_local_max) {
                        int d = deg[v];
                        int reset_size = std::min(d + 2, palette_size);
                        std::fill(used.begin(), used.begin() + reset_size, (char)0);
                        for (const int* neighbor = g.neighbors_begin(v); neighbor != g.neighbors_end(v); neighbor++) {
                            int c = colors[*neighbor];
                            if (c >= 0 && c < palette_size) used[c] = 1;
                        }
                        for (int c = 0; ; c++) {
                            if (c >= palette_size || !used[c]) {
                                colors[v] = c;
                                break;
                            }
                        }
                    // if not local max, go to next JP round 
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

    // update metrics 
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

// Coloring verification
bool verify_coloring(const Graph& g, const std::vector<int>& colors) {
    for (int v = 0; v < g.num_vertices; v++) {
        if (colors[v] < 0) return false;
        for (const int* neighbor = g.neighbors_begin(v); neighbor != g.neighbors_end(v); neighbor++) {
            if (colors[v] == colors[*neighbor]) return false;
        }
    }
    return true;
}
