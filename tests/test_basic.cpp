// ===========================================================================
// test_basic.cpp — Unit tests for graph coloring algorithms
//
// Tests correctness of sequential, speculative, and hybrid coloring on
// small hand-built graphs (triangle, path) and loaded graph files.
// Also tests the verification function and graph loading routines.
// ===========================================================================

#include "coloring.h"
#include "graph.h"
#include <cassert>
#include <iostream>

// Build a small test graph (triangle: 0-1-2-0, complete K3)
static Graph make_triangle() {
    Graph g;
    g.num_vertices = 3;
    g.row_offsets = {0, 2, 4, 6};
    g.col_indices = {1, 2, 0, 2, 0, 1};
    g.num_edges = 6;
    return g;
}

// Build a path graph: 0 - 1 - 2 - ... - (n-1)
static Graph make_path(int n) {
    Graph g;
    g.num_vertices = n;
    g.row_offsets.resize(n + 1, 0);
    std::vector<int> cols;
    for (int i = 0; i < n; i++) {
        int deg = 0;
        if (i > 0)     { cols.push_back(i - 1); deg++; }
        if (i < n - 1) { cols.push_back(i + 1); deg++; }
        g.row_offsets[i + 1] = g.row_offsets[i] + deg;
    }
    g.col_indices = cols;
    g.num_edges = static_cast<int>(cols.size());
    return g;
}

// --- Sequential coloring tests ---

static void test_sequential_triangle() {
    std::cout << "test_sequential_triangle ... ";
    Graph g = make_triangle();
    ColoringResult r = color_sequential(g);
    assert(verify_coloring(g, r.colors));
    assert(r.num_colors == 3);  // K3 needs exactly 3 colors
    std::cout << "PASSED\n";
}

static void test_sequential_path() {
    std::cout << "test_sequential_path ... ";
    Graph g = make_path(100);
    ColoringResult r = color_sequential(g);
    assert(verify_coloring(g, r.colors));
    assert(r.num_colors <= 2);  // path graph is bipartite
    std::cout << "PASSED\n";
}

// --- Parallel speculative coloring tests ---

static void test_parallel_triangle() {
    std::cout << "test_parallel_triangle ... ";
    Graph g = make_triangle();
    ColoringResult r = color_parallel(g, 2);
    assert(verify_coloring(g, r.colors));
    assert(r.num_colors == 3);
    std::cout << "PASSED\n";
}

static void test_parallel_path() {
    std::cout << "test_parallel_path ... ";
    Graph g = make_path(1000);
    ColoringResult r = color_parallel(g, 4);
    assert(verify_coloring(g, r.colors));
    assert(r.num_colors <= 3);  // parallel may use 1 extra color
    std::cout << "PASSED\n";
}

// --- Hybrid coloring tests ---

static void test_hybrid_triangle() {
    std::cout << "test_hybrid_triangle ... ";
    Graph g = make_triangle();
    ColoringResult r = color_hybrid(g, 2);
    assert(verify_coloring(g, r.colors));
    assert(r.num_colors == 3);
    std::cout << "PASSED\n";
}

static void test_hybrid_path() {
    std::cout << "test_hybrid_path ... ";
    Graph g = make_path(1000);
    ColoringResult r = color_hybrid(g, 4);
    assert(verify_coloring(g, r.colors));
    assert(r.num_colors <= 3);
    std::cout << "PASSED\n";
}

// --- GPU coloring tests ---

#ifdef CUDA_ENABLED
static void test_gpu_triangle() {
    std::cout << "test_gpu_triangle ... ";
    Graph g = make_triangle();
    ColoringResult r = color_gpu(g);
    assert(verify_coloring(g, r.colors));
    assert(r.num_colors == 3);
    std::cout << "PASSED\n";
}

static void test_gpu_path() {
    std::cout << "test_gpu_path ... ";
    Graph g = make_path(1000);
    ColoringResult r = color_gpu(g);
    assert(verify_coloring(g, r.colors));
    assert(r.num_colors <= 3);
    std::cout << "PASSED\n";
}
#endif

// --- Verification tests ---

static void test_verify_detects_bad_coloring() {
    std::cout << "test_verify_detects_bad_coloring ... ";
    Graph g = make_triangle();
    std::vector<int> bad_colors = {0, 0, 1};  // vertices 0 and 1 are adjacent, both color 0
    assert(!verify_coloring(g, bad_colors));
    std::cout << "PASSED\n";
}

// --- Graph loading tests ---

static void test_load_edge_list() {
    std::cout << "test_load_edge_list ... ";
    Graph g = load_edge_list("graphs/test_100.txt");
    assert(g.num_vertices == 100);
    assert(g.num_edges == 1000);  // 500 undirected edges -> 1000 directed
    for (int i = 0; i < g.num_vertices; i++) {
        assert(g.row_offsets[i] <= g.row_offsets[i + 1]);
    }
    assert(g.row_offsets[g.num_vertices] == g.num_edges);
    std::cout << "PASSED\n";
}

static void test_load_metis() {
    std::cout << "test_load_metis ... ";
    Graph g = load_metis("graphs/triangle.metis");
    assert(g.num_vertices == 3);
    assert(g.num_edges == 6);
    Graph expected = make_triangle();
    assert(g.row_offsets == expected.row_offsets);
    assert(g.col_indices == expected.col_indices);
    ColoringResult r = color_sequential(g);
    assert(verify_coloring(g, r.colors));
    assert(r.num_colors == 3);
    std::cout << "PASSED\n";
}

static void test_load_graph_autodetect() {
    std::cout << "test_load_graph_autodetect ... ";
    Graph g1 = load_graph("graphs/triangle.metis");
    assert(g1.num_vertices == 3 && g1.num_edges == 6);
    Graph g2 = load_graph("graphs/test_100.txt");
    assert(g2.num_vertices == 100);
    std::cout << "PASSED\n";
}

int main() {
    std::cout << "=== Graph Coloring Tests ===\n";
    test_sequential_triangle();
    test_sequential_path();
    test_parallel_triangle();
    test_parallel_path();
    test_hybrid_triangle();
    test_hybrid_path();
#ifdef CUDA_ENABLED
    test_gpu_triangle();
    test_gpu_path();
#endif
    test_verify_detects_bad_coloring();
    test_load_edge_list();
    test_load_metis();
    test_load_graph_autodetect();
    std::cout << "\nAll tests passed!\n";
    return 0;
}
