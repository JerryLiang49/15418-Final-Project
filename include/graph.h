#ifndef GRAPH_H
#define GRAPH_H

#include <string>
#include <vector>

// CSR (Compressed Sparse Row) graph representation
struct Graph {
    int num_vertices;
    int num_edges;                   // total directed edges stored
    std::vector<int> row_offsets;    // size num_vertices + 1
    std::vector<int> col_indices;    // size num_edges

    // Return the degree of vertex v
    int degree(int v) const { return row_offsets[v + 1] - row_offsets[v]; }

    // Iterators over neighbors of v
    const int* neighbors_begin(int v) const { return col_indices.data() + row_offsets[v]; }
    const int* neighbors_end(int v) const   { return col_indices.data() + row_offsets[v + 1]; }
};

// Load a graph from an edge-list file.
// Expected format: first line "num_vertices num_edges", then one "u v" per line.
// Edges are treated as undirected (both directions stored).
Graph load_edge_list(const std::string& filename);

// Load a graph from a METIS file.
// Comment lines start with '%'. First data line: "num_vertices num_edges [fmt [ncon]]".
// Then one line per vertex listing 1-indexed neighbors (edge/vertex weights ignored).
Graph load_metis(const std::string& filename);

// Auto-detect format and load: .graph/.metis -> METIS, otherwise edge-list.
Graph load_graph(const std::string& filename);

#endif // GRAPH_H
