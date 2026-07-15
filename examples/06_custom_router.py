"""
06 — Custom router: plug in your own path-selection function.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qiskit import QuantumCircuit
from dqc_simulator import DQCCircuit, DQCQPU, QPUManager

# 6-qubit GHZ-fanout circuit (CX from qubit 0 to all others).
# Fanout produces multi-hop CXs (e.g. CX(0,2), CX(0,4)) so my_router actually fires —
# the simulator short-circuits routing for directly-adjacent QPU pairs.
N = 6
qc0 = QuantumCircuit(N, N)
qc0.h(0)
for i in range(1, N):
    qc0.cx(0, i)
qc0.measure(range(N), range(N))

# 6-QPU ring + one shortcut, 1 qubit per QPU
qpus = QPUManager()
for i in range(N):
    qpus.add_qpu(DQCQPU(i, "FakeLagosV2", available_comm_qubits=2))
for i in range(N):
    qpus.add_coonnection(i, (i + 1) % N, distance=2)
qpus.add_coonnection(0, 3, distance=1)


_stats = {"success": 0, "fail": 0}


def my_router(start, end, graph, link_params):
    """
    Custom router. Signature: (start, end, graph, link_params) -> path | None
      graph        : {qpu_id: [(neighbor_id, distance), ...]}
      link_params  : {(qpu_a, qpu_b): (success_prob, attempt_time)}
      return       : list of QPU ids forming a path, or None to fall back.

    This example: minimise summed per-link `attempt_time` (raw network delay).
    """
    # 1. Enumerate every simple path from start to end.
    paths, stack = [], [(start, [start])]
    while stack:
        node, path = stack.pop()
        if node == end:
            paths.append(path)
            continue
        for nb, _ in graph.get(node, []):
            if nb not in path:
                stack.append((nb, path + [nb]))
    if not paths:
        _stats["fail"] += 1
        return None                     # None → falls back to qpus.fallback_algo

    # 2. Score each path by total attempt_time (sum across hops, from link_params).
    def cost(path):
        total = 0.0
        for a, b in zip(path[:-1], path[1:]):
            _p, t = link_params.get((a, b)) or link_params.get((b, a)) or (None, None)
            if t is None:
                return float("inf")
            total += t
        return total

    _stats["success"] += 1
    return min(paths, key=cost)


qpus.routing_algo = my_router           # callable, or one of: BFS, Throughput, Fidelity, MinRisk, SecurityConstrained, JointSecurity
qpus.fallback_algo = "BFS"              # used when my_router returns None

qc = DQCCircuit(qc0)
qc.purification = {}
qc.Execution([1] * N, qpus, skip_aer=True, output_level="verbose")

if _stats["fail"] == 0:
    print(f"\nCustom router my_router succeeded without any failures ({_stats['success']} requests)")
else:
    print(f"\nCustom router my_router: {_stats['fail']} failure(s) detected, falling back to {qpus.fallback_algo}")
