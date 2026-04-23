// ===========================================================================
// main.cpp — Graph coloring driver
//
// Loads a graph, runs the selected parallel coloring algorithm, and outputs
// results in either human-readable or CSV format (for benchmarking).
//
// Usage: ./graphcolor <graph_file> [num_threads] [--csv]
//                    [--algorithm spec|hybrid|gpu|gpu-edge]
// ===========================================================================

#include "coloring.h"
#include "graph.h"
#include "timer.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

static void usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <graph_file> [num_threads] [--csv] [--algorithm spec|hybrid|gpu|gpu-edge]\n"
              << "  graph_file   : path to graph file (edge-list or METIS)\n"
              << "  num_threads  : threads for parallel coloring (default: all available)\n"
              << "  --csv        : output results as CSV row (for benchmarking)\n"
              << "  --algorithm  : spec (default), hybrid (spec + JP), gpu, or gpu-edge (CUDA)\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    std::string graph_file = argv[1];
    int num_threads = 0;
    bool csv_mode = false;
    std::string algorithm = "spec";

    for (int i = 2; i < argc; i++) {
        if (std::strcmp(argv[i], "--csv") == 0) {
            csv_mode = true;
        } else if (std::strcmp(argv[i], "--algorithm") == 0 && i + 1 < argc) {
            algorithm = argv[++i];
        } else {
            num_threads = std::atoi(argv[i]);
        }
    }

    // Load graph from file (auto-detects edge-list vs METIS format)
    Timer load_timer;
    load_timer.start();
    Graph g = load_graph(graph_file);
    double load_time = load_timer.elapsed();

    // Compute graph statistics for CSV output and analysis
    int max_deg = 0;
    double sum_deg = 0.0, sum_deg2 = 0.0;
    for (int v = 0; v < g.num_vertices; v++) {
        int d = g.degree(v);
        if (d > max_deg) max_deg = d;
        sum_deg += d;
        sum_deg2 += (double)d * d;
    }
    double avg_deg = (g.num_vertices > 0) ? sum_deg / g.num_vertices : 0.0;
    double var_deg = (g.num_vertices > 0) ? sum_deg2 / g.num_vertices - avg_deg * avg_deg : 0.0;
    double cv_deg = (avg_deg > 0) ? std::sqrt(var_deg) / avg_deg : 0.0;

    // Determine thread count (default: all available cores)
    if (num_threads <= 0) {
#ifdef _OPENMP
        #pragma omp parallel
        #pragma omp single
        num_threads = omp_get_num_threads();
#else
        num_threads = 1;
#endif
    }

    // Run selected algorithm
    ColoringResult par;
    std::string algo_name;
    if (algorithm == "gpu" || algorithm == "gpu-edge") {
#ifdef CUDA_ENABLED
        if (algorithm == "gpu-edge") {
            par = color_gpu_edge(g);
            algo_name = "GPU Edge";
        } else {
            par = color_gpu(g);
            algo_name = "GPU";
        }
        num_threads = 0;  // GPU: no CPU thread count
#else
        std::cerr << "Error: GPU support not compiled. Rebuild with CUDA (nvcc required).\n";
        return 1;
#endif
    } else if (algorithm == "hybrid") {
        par = color_hybrid(g, num_threads);
        algo_name = "Hybrid";
    } else {
        par = color_parallel(g, num_threads);
        algo_name = "Speculative";
    }
    bool par_valid = verify_coloring(g, par.colors);

    // Derived metrics for analysis.
    // Note: GPU counts conflict *events* summed across rounds (a vertex may
    // appear as a conflict in multiple rounds), so we normalize by rounds to
    // report an average per-round rate comparable to the CPU algorithms.
    double conflict_divisor = (double)g.num_vertices;
    if ((algorithm == "gpu" || algorithm == "gpu-edge") && par.num_rounds > 0) {
        conflict_divisor *= (double)par.num_rounds;
    }
    double conflict_rate = (conflict_divisor > 0) ? (double)par.num_conflicts / conflict_divisor * 100.0 : 0.0;

    if (csv_mode) {
        // CSV header (printed by benchmark script):
        // graph,vertices,edges,threads,algorithm,max_deg,avg_deg,cv_deg,
        // colors,conflicts,conflict_rate,rounds,hubs,
        // init_time,compute_time,total_time,valid
        std::cout << graph_file << ","
                  << g.num_vertices << ","
                  << g.num_edges / 2 << ","
                  << num_threads << ","
                  << algorithm << ","
                  << max_deg << ","
                  << std::fixed << std::setprecision(2) << avg_deg << ","
                  << std::fixed << std::setprecision(3) << cv_deg << ","
                  << par.num_colors << ","
                  << par.num_conflicts << ","
                  << std::fixed << std::setprecision(4) << conflict_rate << ","
                  << par.num_rounds << ","
                  << par.num_hubs << ","
                  << std::fixed << std::setprecision(6) << par.init_seconds << ","
                  << std::fixed << std::setprecision(6) << par.compute_seconds << ","
                  << std::fixed << std::setprecision(6) << par.elapsed_seconds << ","
                  << (par_valid ? "yes" : "NO") << "\n";
    } else {
        std::cout << "Graph       : " << graph_file << "\n"
                  << "Vertices    : " << g.num_vertices
                  << "  Edges: " << g.num_edges / 2 << " (undirected)\n"
                  << "Max degree  : " << max_deg
                  << "  Avg degree: " << std::fixed << std::setprecision(1) << avg_deg
                  << "  CV: " << std::setprecision(3) << cv_deg << "\n"
                  << "Load time   : " << load_time << " s\n\n"
                  << "--- " << algo_name << " Coloring (" << ((algorithm == "gpu" || algorithm == "gpu-edge") ? "GPU" : std::to_string(num_threads) + " threads") << ") ---\n"
                  << "Colors used : " << par.num_colors << "\n"
                  << "Conflicts   : " << par.num_conflicts
                  << " (" << std::setprecision(2) << conflict_rate << "% of vertices)\n"
                  << "Rounds      : " << par.num_rounds << "\n"
                  << "Hubs colored: " << par.num_hubs << "\n"
                  << "Init time   : " << std::setprecision(6) << par.init_seconds << " s\n"
                  << "Compute time: " << par.compute_seconds << " s\n"
                  << "Total time  : " << par.elapsed_seconds << " s\n"
                  << "Valid       : " << (par_valid ? "yes" : "NO") << "\n";
    }

    return par_valid ? 0 : 1;
}
