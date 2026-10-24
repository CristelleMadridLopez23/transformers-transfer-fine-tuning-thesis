"""Graph utilities for exact periodic labelled-multigraph checks in SLICES."""
from __future__ import annotations

from collections import Counter
from itertools import permutations, product
from typing import Any

import networkx as nx


def decoded_graph(backend: Any, sequence: str, strategy: int = 4) -> nx.MultiDiGraph:
    """Decode a SLICES string to a graph with species and periodic edge shifts.

    Args:
        backend: Initialized ``slices.core.SLICES`` instance.
        sequence: Whitespace-separated SLICES string.
        strategy: SLICES grammar strategy.
    Returns:
        Bidirected periodic multigraph. Each undirected bond is represented by
        both directed orientations with opposite image vectors, so atom
        reindexing may reverse an edge without losing its translation label.
    """
    backend.from_SLICES(sequence, strategy=strategy, fix_duplicate_edge=False)
    return graph_from_arrays(backend.atom_types, backend.edge_indices, backend.to_jimages)


def graph_from_arrays(atom_types: Any, edge_indices: Any, to_jimages: Any) -> nx.MultiDiGraph:
    """Build a periodic labelled graph from atomic numbers and directed edges.

    Args:
        atom_types: Atomic numbers in the source unit cell.
        edge_indices: Edge endpoint array with shape ``(n_edges, 2)``.
        to_jimages: Integer periodic-image vectors with shape ``(n_edges, 3)``.
    Returns:
        Bidirected periodic multigraph retaining all translation labels.
    """
    graph = nx.MultiDiGraph()
    for node, atomic_number in enumerate(atom_types.tolist()):
        graph.add_node(node, atomic_number=int(atomic_number))
    for edge, shift in zip(edge_indices.tolist(), to_jimages.tolist()):
        i, j = (int(edge[0]), int(edge[1]))
        image = tuple(int(value) for value in shift)
        graph.add_edge(i, j, image=image)
        graph.add_edge(j, i, image=tuple(-value for value in image))
    return graph


def _node_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Compare atom species for graph isomorphism."""
    return left["atomic_number"] == right["atomic_number"]


def _edge_match(left: dict[int, dict[str, Any]], right: dict[int, dict[str, Any]]) -> bool:
    """Compare the multiset of periodic image labels on parallel edges."""
    return Counter(data["image"] for data in left.values()) == Counter(data["image"] for data in right.values())


def periodic_graphs_isomorphic(left: nx.MultiDiGraph, right: nx.MultiDiGraph) -> bool:
    """Check species-labelled periodic multigraph isomorphism.

    Args:
        left: First decoded graph.
        right: Second decoded graph.
    Returns:
        True only if atom labels and all periodic edge shifts match under a
        bijective node permutation.
    """
    if left.number_of_nodes() != right.number_of_nodes() or left.number_of_edges() != right.number_of_edges():
        return False
    matcher = nx.algorithms.isomorphism.MultiDiGraphMatcher(left, right, node_match=_node_match, edge_match=_edge_match)
    return bool(matcher.is_isomorphic())


def periodic_graphs_equivalent(left: nx.MultiDiGraph, right: nx.MultiDiGraph) -> tuple[bool, list[list[int]] | None]:
    """Compare periodic graphs allowing a signed permutation of lattice axes.

    Args:
        left: First decoded graph.
        right: Second decoded graph.
    Returns:
        Pair of equivalence status and the signed permutation matrix applied to
        the left image vectors. This accounts for unit-cell axis reordering and
        sign changes while retaining exact periodic-image labels.
    """
    for order in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            transform = [[0, 0, 0] for _ in range(3)]
            for row, column in enumerate(order):
                transform[row][column] = signs[row]
            mapped = nx.MultiDiGraph()
            mapped.add_nodes_from((node, dict(attrs)) for node, attrs in left.nodes(data=True))
            for source, target, attrs in left.edges(data=True):
                vector = attrs["image"]
                shifted = tuple(sum(vector[row] * transform[row][column] for row in range(3)) for column in range(3))
                mapped.add_edge(source, target, image=shifted)
            if periodic_graphs_isomorphic(mapped, right):
                return True, transform
    return False, None

