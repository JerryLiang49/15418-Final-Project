#!/usr/bin/env python3
"""Generate synthetic graphs in edge-list format for benchmarking."""

import argparse
import random
import sys

# Generate an Erdos-Renyi random graph with n vertices and m edges.
def gen_erdos_renyi(n, m, rng):
    edges = set()
    while len(edges) < m:
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u != v:
            edges.add((min(u, v), max(u, v)))
    return edges

# Generate an R-MAT power-law graph (Kronecker-style) with n vertices and m edges.
def gen_rmat(n, m, rng, a=0.57, b=0.19, c=0.19):

    d = 1.0 - a - b - c
    edges = set()
    # n must be power of 2 for pure R-MAT
    bits = max(1, (n - 1).bit_length())
    actual_n = 1 << bits
    while len(edges) < m:
        u, v = 0, 0
        for depth in range(bits):
            r = rng.random()
            if r < a:
                pass # quadrant (0,0)
            elif r < a + b:
                v += 1 << (bits - 1 - depth)
            elif r < a + b + c:
                u += 1 << (bits - 1 - depth)
            else:
                u += 1 << (bits - 1 - depth)
                v += 1 << (bits - 1 - depth)
        # Map back to [0, n)
        u = u % n
        v = v % n
        if u != v:
            edges.add((min(u, v), max(u, v)))
    return edges

# Generate a 2D grid graph of side x side vertices.
def gen_2d_grid(side): 
    n = side * side
    edges = set()
    for r in range(side):
        for c in range(side):
            v = r * side + c
            if c + 1 < side:
                edges.add((v, v + 1))
            if r + 1 < side:
                edges.add((v, v + side))
    return n, edges


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic graphs in edge-list format")
    parser.add_argument("--type", "-t", choices=["erdos-renyi", "rmat", "grid"],
                        default="erdos-renyi", help="Graph type")
    parser.add_argument("--vertices", "-n", type=int, help="Number of vertices (for erdos-renyi, rmat)")
    parser.add_argument("--edges", "-m", type=int, help="Number of undirected edges (for erdos-renyi, rmat)")
    parser.add_argument("--side", type=int, help="Grid side length (for grid type)")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file (default: stdout)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.type == "grid":
        if not args.side:
            parser.error("--side required for grid type")
        n, edges = gen_2d_grid(args.side)
    elif args.type == "rmat":
        if not args.vertices or not args.edges:
            parser.error("--vertices and --edges required for rmat type")
        n = args.vertices
        edges = gen_rmat(n, args.edges, rng)
    else:  # erdos-renyi
        if not args.vertices or not args.edges:
            parser.error("--vertices and --edges required for erdos-renyi type")
        n = args.vertices
        edges = gen_erdos_renyi(n, args.edges, rng)

    out = open(args.output, "w") if args.output else sys.stdout
    out.write(f"{n} {len(edges)}\n")
    for u, v in sorted(edges):
        out.write(f"{u} {v}\n")
    if args.output:
        out.close()

if __name__ == "__main__":
    main()
