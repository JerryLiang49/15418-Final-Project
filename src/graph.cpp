#include "graph.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

// Helper: build CSR from an edge pair list (already contains both directions)
static Graph build_csr(int n, std::vector<std::pair<int, int>>& edges) {
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    Graph g;
    g.num_vertices = n;
    g.num_edges = static_cast<int>(edges.size());
    g.row_offsets.resize(n + 1, 0);
    g.col_indices.resize(g.num_edges);

    for (auto& [u, v] : edges) {
        g.row_offsets[u + 1]++;
    }
    for (int i = 1; i <= n; i++) {
        g.row_offsets[i] += g.row_offsets[i - 1];
    }

    std::vector<int> pos(g.row_offsets.begin(), g.row_offsets.end());
    for (auto& [u, v] : edges) {
        g.col_indices[pos[u]++] = v;
    }

    return g;
}

Graph load_edge_list(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open graph file: " + filename);
    }

    int n, m;
    in >> n >> m;

    std::vector<std::pair<int, int>> edges;
    edges.reserve(2 * m);
    for (int i = 0; i < m; i++) {
        int u, v;
        in >> u >> v;
        edges.emplace_back(u, v);
        edges.emplace_back(v, u);
    }

    return build_csr(n, edges);
}

Graph load_metis(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open METIS file: " + filename);
    }

    std::string line;

    // Skip comment lines (start with '%')
    int n = 0, m = 0, fmt = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '%') continue;
        std::istringstream iss(line);
        iss >> n >> m;
        if (iss >> fmt) { /* optional fmt field */ }
        break;
    }

    if (n <= 0) {
        throw std::runtime_error("Invalid METIS header in: " + filename);
    }

    // fmt flags: 0 = no weights, 1 = edge weights, 10 = vertex weights, 11 = both
    bool has_edge_weights = (fmt == 1 || fmt == 11);
    bool has_vertex_weights = (fmt == 10 || fmt == 11);

    std::vector<std::pair<int, int>> edges;
    edges.reserve(2 * m);

    for (int v = 0; v < n; v++) {
        if (!std::getline(in, line)) break;
        if (line.empty() || line[0] == '%') { v--; continue; }

        std::istringstream iss(line);

        // Skip vertex weight if present
        if (has_vertex_weights) {
            int vw;
            iss >> vw;
        }

        int neighbor;
        while (iss >> neighbor) {
            int u = neighbor - 1; // METIS uses 1-indexed vertices
            if (has_edge_weights) {
                int ew;
                iss >> ew; // skip edge weight
            }
            edges.emplace_back(v, u);
            edges.emplace_back(u, v);
        }
    }

    return build_csr(n, edges);
}

// Returns true if filename ends with suffix (case-insensitive)
static bool ends_with(const std::string& s, const std::string& suffix) {
    if (suffix.size() > s.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin(),
                      [](char a, char b) {
                          return std::tolower(static_cast<unsigned char>(a)) ==
                                 std::tolower(static_cast<unsigned char>(b));
                      });
}

Graph load_graph(const std::string& filename) {
    if (ends_with(filename, ".graph") || ends_with(filename, ".metis")) {
        return load_metis(filename);
    }
    return load_edge_list(filename);
}
