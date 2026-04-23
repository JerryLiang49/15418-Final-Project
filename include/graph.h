#ifndef GRAPH_H
#define GRAPH_H

#include <string>
#include <vector>

// CSR (Compressed Sparse Row) graph representation
// row_offsets[v] tells us where vertex v's neighbor list starts in col_indices
// row_offset[v+1] tells us where it ends 
struct Graph {
    int num_vertices;
    int num_edges;                   
    std::vector<int> row_offsets; // size num_vertices + 1
    std::vector<int> col_indices; // size num_edges

    // Return the degree of vertex v
    int degree(int v) const { 
        return row_offsets[v + 1] - row_offsets[v]; 
    }

    // Iterators over neighbors of v
    const int* neighbors_begin(int v) const { 
        return col_indices.data() + row_offsets[v]; 
    }
    const int* neighbors_end(int v) const { 
        return col_indices.data() + row_offsets[v + 1]; 
    }
};
// Graph loaders

// Load a graph from an edge-list file.
// Expected format: first line "num_vertices num_edges", then one "u v" per line.
Graph load_edge_list(const std::string& filename);

// Load a graph from a METIS file.
Graph load_metis(const std::string& filename);

// Auto-detect format and load based on file extension
Graph load_graph(const std::string& filename);

#endif // GRAPH_H
