#!/usr/bin/env python3
"""Convert SNAP edge-list files to our edge-list format.

SNAP files have comment lines starting with '#', then tab-separated "u\tv" edges.
Our format: first line "num_vertices num_edges", then one "u v" per line (0-indexed).
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="Convert SNAP graph to edge-list format")
    parser.add_argument("input", help="Input SNAP file")
    parser.add_argument("-o", "--output", required=True, help="Output file")
    args = parser.parse_args()

    edges = set()
    max_vertex = -1

    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('%'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            if u == v:
                continue
            edges.add((min(u, v), max(u, v)))
            max_vertex = max(max_vertex, u, v)

    # Remap vertices to 0-indexed contiguous range
    all_vertices = set()
    for u, v in edges:
        all_vertices.add(u)
        all_vertices.add(v)
    vertex_map = {v: i for i, v in enumerate(sorted(all_vertices))}
    n = len(vertex_map)

    remapped = set()
    for u, v in edges:
        nu, nv = vertex_map[u], vertex_map[v]
        remapped.add((min(nu, nv), max(nu, nv)))

    with open(args.output, 'w') as f:
        f.write(f"{n} {len(remapped)}\n")
        for u, v in sorted(remapped):
            f.write(f"{u} {v}\n")

    print(f"Converted: {n} vertices, {len(remapped)} edges -> {args.output}")


if __name__ == "__main__":
    main()
