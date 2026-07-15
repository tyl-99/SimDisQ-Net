# SimDisQ-Net

A network-aware distributed quantum circuit simulator. SimDisQ-Net extends
SimDisQ, the distributed quantum circuit simulator by Y. Zhang *et al.*, with
first-class network primitives — routing, scheduling, rerouting, purification,
and security-aware path selection — that operate alongside circuit execution.

## What's distinct

- **Configurable routing**: `BFS` (min-hop), `Throughput` (min expected latency),
  `Fidelity` (max path fidelity), plus three security-aware variants.
- **Scheduling under contention**: `FCFS` / `SPF` / `SJF` policies arbitrate
  comm-qubit demand across concurrent requests.
- **Security-aware routing**: assign per-node risk scores; `MinRisk`,
  `SecurityConstrained`, and `JointSecurity` routers trade latency for trust.
- **Purification**: BBPSSW Bell-pair purification, configurable per-link.
- **Network metrics**: per-request wall-clock, end-to-end fidelity, safety
  score, exposed on `result.network_metrics`.

## Install

From the repository root:

```bash
pip install -e .
```

This installs the `dqc_simulator` package and its dependencies
(qiskit, qiskit-aer, qiskit-ibm-runtime, matplotlib, pylatexenc).
Python 3.13+ recommended.

## Hello world

```python
from qiskit import QuantumCircuit
from dqc_simulator import DQCCircuit, DQCQPU, QPUManager

qc0 = QuantumCircuit(8, 8)
qc0.h(0)
for i in range(7):
    qc0.cx(i, i + 1)
qc0.measure(range(8), range(8))

qpus = QPUManager()
for i in range(4):
    qpus.add_qpu(DQCQPU(i, "FakeLagosV2", available_comm_qubits=2))
for i in range(3):
    qpus.add_coonnection(i, i + 1, distance=2)

qc = DQCCircuit(qc0)
qc.purification = {}
qc.Execution([2, 2, 2, 2], qpus, skip_aer=True)
```

Output:

```
═══ SimDisQ-Net ═══
  Network   : 4 QPUs · 3 edges · Throughput routing · FCFS sched
  Workload  : 3 req · 1 batch · reroute=on · purif=off
  Timing    : 0.06 ms wall
  Fidelity  : F_e2e=0.91 (0.88-0.94 over 3 req)
```

## End-to-end example

`Example_GHZ.py` — 12-qubit GHZ across 4 QPUs, including Aer execution,
Hellinger fidelity score, and histogram + circuit diagram PDFs.

```bash
python Example_GHZ.py
```

## Feature walkthroughs

See `examples/` — each script is self-contained and runs in under 2 seconds:

| File | What it shows |
|---|---|
| `01_quickstart.py` | Minimal run, default summary output |
| `02_routing.py` | `BFS` vs `Throughput` vs `Fidelity` on a heterogeneous ring |
| `03_scheduling.py` | `FCFS` vs `SPF` vs `SJF` on a contended workload |
| `04_security.py` | `MinRisk` routing avoids untrusted repeaters |
| `05_purification.py` | BBPSSW Bell-pair purification trade-off |
| `06_custom_router.py` | Plug in your own routing function |

## API surface

`qc.Execution(partition, qpugroup, ...)` — runs the full pipeline. Key kwargs:

| arg | purpose |
|---|---|
| `comm_noise=True` | Inject depolarizing noise on remote gates |
| `scheduling_algorithm="FCFS"` | `FCFS` / `SPF` / `SJF` |
| `allow_reroute=True` | Allow path swap when comm qubits contested |
| `skip_aer=False` | If `True`, return after heralded sim (fast, network-only) |
| `output_level="summary"` | `"quiet"` / `"summary"` (default) / `"verbose"` |

Network metrics on the result: `result.network_metrics` — keys include
`wall_clock`, `mean_path_risk`, `safety_score`, `request_path_risk`,
`request_safety`, `link_metrics`, and ~10 more.

## Topology builders

`QPUManager` builds the network: `.add_qpu(DQCQPU(id, backend))` adds a node,
`.add_coonnection(a, b, distance=km)` adds a link. Set `qpus.routing_algo`
to one of `BFS`, `Throughput`, `Fidelity`, `MinRisk`, `SecurityConstrained`,
`JointSecurity` to switch routing strategy.

For security: assign `security_risk` to `DQCQPU` (range `[0, 1]`), tune
`qpus.high_risk_node_threshold` (default `0.5`) and
`qpus.risk_include_endpoints` (default `False`).

## License

See [LICENSE](LICENSE).
