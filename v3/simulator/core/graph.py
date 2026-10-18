"""graph.py — a netlist of modules (docs/simulator.md §4.5).

A data edge is (producer "instance.port", consumer "instance.port") or (..., offset): the producer's channel c lands
on the consumer's channel c + offset. `credits` are the credit edges (giver "instance.port", taker "instance.port")
the RTL wires elsewhere than back to the producer. `ports` maps an external name to an inner "instance.port", so a
graph can sit inside another graph. The engine runs the flattened graph, with hierarchical instance names.
"""
from __future__ import annotations

from simulator.core.module import Module


class Graph:
    def __init__(self, name: str, modules: list, edges: list, ports: dict | None = None, credits: list | None = None):
        self.name = name
        self.modules = list(modules)
        self.edges = list(edges)
        self.externalPorts = dict(ports or {})
        self.creditEdges = list(credits or [])
        names = [m.baseName if isinstance(m, Module) else m.name for m in self.modules]
        if len(set(names)) != len(names):
            raise ValueError(f"graph {name}: duplicate instance names")

    def flatten(self, prefix: str = "") -> tuple:
        """(leaf modules with hierarchical names, data edges (a, b, offset), credit edges (a, b),
        external name -> leaf port)."""
        leaves, edges, credits, resolve = [], [], [], {}
        for m in self.modules:
            if isinstance(m, Module):
                m.instance = prefix + m.baseName
                leaves.append(m)
                for port in m.ports():
                    resolve[f"{m.baseName}.{port}"] = f"{m.instance}.{port}"
            else:
                subLeaves, subEdges, subCredits, subExternal = m.flatten(prefix + m.name + ".")
                leaves += subLeaves
                edges += subEdges
                credits += subCredits
                for external, leaf in subExternal.items():
                    resolve[f"{m.name}.{external}"] = leaf

        def resolved(kind: str, a: str, b: str) -> tuple:
            if a not in resolve or b not in resolve:
                raise ValueError(f"graph {self.name}: {kind} {a} -> {b} names an unknown port")
            return resolve[a], resolve[b]
        for e in self.edges:
            a, b = resolved("edge", e[0], e[1])
            edges.append((a, b, e[2] if len(e) > 2 else 0))
        for a, b in self.creditEdges:
            credits.append(resolved("credit edge", a, b))
        external = {}
        for name, inner in self.externalPorts.items():
            if inner not in resolve:
                raise ValueError(f"graph {self.name}: external port {name} -> unknown {inner}")
            external[name] = resolve[inner]
        return leaves, edges, credits, external
