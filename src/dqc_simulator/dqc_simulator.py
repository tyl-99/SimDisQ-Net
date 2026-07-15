from __future__ import annotations
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.circuit import Instruction, CircuitInstruction
from qiskit_aer import AerSimulator
from qiskit.circuit.library.standard_gates import HGate, XGate, ZGate, CXGate
from qiskit.circuit import Reset, Measure, ClassicalRegister
from qiskit.visualization import plot_histogram
from qiskit import QuantumRegister
from qiskit.quantum_info import Kraus
from qiskit.converters import circuit_to_dag
from qiskit_ibm_runtime.fake_provider import (
    FakeVigoV2,         # 5
    FakeLagosV2,        # 7
    FakeCasablancaV2,
    FakeYorktownV2,
    FakeManilaV2,
    FakeNairobiV2,
    FakeMumbaiV2,
    FakeKolkataV2,
    FakeGuadalupeV2,
    FakeAlmadenV2,
    FakeAthensV2,       # 5
    FakeCambridgeV2                
)

# Backend IonQ
from .backend import IonQ
from .backends import IonQAria, IonQForte, IonQForteEnterprise, IBMVigo, IBMMelbourne, IBMCambridge, IBMAlmaden, IBMHeron, QuantinuumH2

from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error, pauli_error,
    amplitude_damping_error, phase_amplitude_damping_error,
    phase_damping_error, thermal_relaxation_error
)

import math
import numpy as np
import copy
from collections import defaultdict
from pathlib import Path
import heapq
import matplotlib.pyplot as plt

from .backend_params import BackendParams, calculate_link_success_probability

# Routing options for swap path selection (same as plot_link_graph.ROUTING_ALGOS)
ROUTING_ALGOS = [
    "BFS", "Throughput", "Fidelity",
    "MinRisk", "SecurityConstrained", "JointSecurity",
]

class RemoteGate(Instruction):
    """ Remote Gate"""
    def __init__(self, index: int, target: int):
        super().__init__("R", 1, 0, [])
        self.index = index
        self.target = target

class MX(Instruction):
    """MX for Control Side"""
    def __init__(self, index: int, target: int):
        super().__init__("MX", 2, 0, [])
        self.index = index
        self.target = target

class MZ(Instruction):
    """MZ for Target Side"""
    def __init__(self, index: int, target: int):
        super().__init__("MZ", 2, 0, [])
        self.index = index
        self.target = target

class AnsM(Instruction):
    """AnsM Measurement"""
    def __init__(self, mea: int):
        super().__init__("ANS_M", 1, 0, [])
        self.mea = mea

class S_CX(Instruction):
    """Custom CNOT Gate"""
    def __init__(self, control: int, target: int, path):
        super().__init__("S_CX", 2, 0, [])
        self.control = control
        self.target = target
        self.path = path

class MS(Instruction):
    """Multi-qubit Swap Gate"""
    def __init__(self, index: int, target: int):
        super().__init__("MS", 2, 0, [])
        self.index = index
        self.target = target

class IF_Z(Instruction):
    def __init__(self, index: int, target: int):
        super().__init__("IF_Z", 1, 0, [])
        self.index = index
        self.target = target

class IF_X(Instruction):
    """Conditional X Gate"""
    def __init__(self, index: int, target: int):
        super().__init__("IF_X", 1, 0, [])
        self.index = index
        self.target = target

# Mapping backend names to their classes
FAKE_BACKENDS = {
    "FakeVigoV2": FakeVigoV2,
    "FakeLagosV2": FakeLagosV2,
    "FakeCasablancaV2": FakeCasablancaV2,
    "FakeYorktownV2": FakeYorktownV2,
    "FakeManilaV2": FakeManilaV2,
    "FakeNairobiV2": FakeNairobiV2,
    "FakeMumbaiV2": FakeMumbaiV2,
    "FakeKolkataV2": FakeKolkataV2,
    "FakeGuadalupeV2": FakeGuadalupeV2,
    "FakeAlmadenV2": FakeAlmadenV2,
    "FakeAthensV2": FakeAthensV2,
    "FakeCambridgeV2": FakeCambridgeV2,
    "IonQ": IonQ,
    "IonQAria": IonQAria,
    "IonQForte": IonQForte,
    "IonQForteEnterprise": IonQForteEnterprise,
    "IBMVigo": IBMVigo,
    "IBMMelbourne": IBMMelbourne,
    "IBMCambridge": IBMCambridge,
    "IBMAlmaden": IBMAlmaden,
    "IBMHeron": IBMHeron,
    "QuantinuumH2": QuantinuumH2,
}

class QPUManager:
    def __init__(self):
        """
        Initialize QPU manager to handle multiple QPUs and their connections.

        Attributes:
            qpus: List of QPU instances
            map: Adjacency list representing QPU network topology
            size: Total number of QPUs in the manager
        """
        self.qpus = []
        # Adjacency list: {qpu_id: [(neighbor_id, distance), ...]}
        self.map = {}
        # Link params for routing: (qpu_id1, qpu_id2) -> (success_prob, attempt_time)
        self.link_params = {}
        # Routing algorithm: "BFS" | "Throughput" | "Fidelity" |
        # "MinRisk" | "SecurityConstrained" | "JointSecurity" or a callable
        self.routing_algo = "Throughput"
        # When routing_algo is a callable and returns None, use this built-in router.
        self.fallback_algo = "BFS"
        self.security_threshold = 1.0
        self.security_weights = {
            "latency": 1.0,
            "fidelity_loss": 1.0,
            "risk": 1.0,
        }
        # When True, source/target QPUs count toward path risk and the
        # untrusted-path-fraction metric. Default False — endpoints are the
        # user's deliberate workload-placement choice; the security metric
        # should reflect routing-layer exposure (intermediate hops), which is
        # what security-aware routers can actually optimize.
        self.risk_include_endpoints = False
        # Threshold above which a node is considered "high-risk" for the
        # untrusted-path-fraction metric.
        self.high_risk_node_threshold = 0.5
        self.size = 0

    def add_qpu(self, qpu):
        """Add a QPU to the manager"""
        self.qpus.append(qpu)
        qpu_id = qpu.qpu_id
        self.size += 1
        # Initialize adjacency list entry if not exists
        if qpu_id not in self.map:
            self.map[qpu_id] = []

    def get_qpu(self, qpu_id):
        """Retrieve QPU instance by ID"""
        for qpu in self.qpus:
            if qpu.qpu_id == qpu_id:
                return qpu
        return None
    
    def add_coonnection(self, qpu_id1, qpu_id2, distance: float = 0):
        """Add bidirectional connection between two QPUs"""
        qpu1 = self.get_qpu(qpu_id1)
        qpu2 = self.get_qpu(qpu_id2)
        if qpu1 is None or qpu2 is None:
            raise ValueError(f"QPU {qpu_id1} or {qpu_id2} not found.")

        # Add to adjacency list if not already present
        if not any(n == qpu_id2 for n, _ in self.map[qpu_id1]):
            self.map[qpu_id1].append((qpu_id2, distance))
        if not any(n == qpu_id1 for n, _ in self.map[qpu_id2]):
            self.map[qpu_id2].append((qpu_id1, distance))

        # Compute and store link success params for heralded entanglement / throughput routing
        try:
            success_prob, attempt_time = calculate_link_success_probability(
                qpu1.backend, qpu2.backend, distance
            )
            self.link_params[(qpu_id1, qpu_id2)] = (success_prob, attempt_time)
            self.link_params[(qpu_id2, qpu_id1)] = (success_prob, attempt_time)
        except Exception:
            pass  # keep link_params empty for this link; routing falls back to BFS

    def get_link_success_params(self, qpu_id1, qpu_id2):
        """Get (success_probability, attempt_time) for a link. Returns (None, None) if not available."""
        return self.link_params.get((qpu_id1, qpu_id2), (None, None))

    @staticmethod
    def build_depolarizing_noise_for_fidelity(F_link):
        """
        Return a single-qubit depolarizing Kraus instruction calibrated so that
        applying it to one qubit of a perfect Bell pair produces a Werner state
        with Bell fidelity ``F_link``. That is: p = 1 - F_link.
        """
        p = max(0.0, min(1.0, 1.0 - float(F_link)))
        sqrt_1mp = np.sqrt(1.0 - p)
        sqrt_p3 = np.sqrt(p / 3.0) if p > 0 else 0.0
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        K = [sqrt_1mp * I, sqrt_p3 * X, sqrt_p3 * Y, sqrt_p3 * Z]
        return Kraus(K).to_instruction()

    def check_connection(self, qpu_id1, qpu_id2) -> int:
        """Check if two QPUs are connected (returns 1 if connected, 0 otherwise)"""
        if qpu_id1 not in self.map:
            return 0
        return 1 if any(n == qpu_id2 for n, _ in self.map[qpu_id1]) else 0


class DQCQPU:
    def __init__(
        self,
        qpu_id: int,
        backend_name: str = None,
        backend=None,
        memory_coherence_time_s: float | None = None,
        available_comm_qubits: int | None = None,
        security_risk: float = 0.0,
    ):
        """
        Initialize a QPU instance with automatic custom instruction registration.

        :param qpu_id: QPU identifier
        :param backend_name: Name of the backend (e.g., "FakeLagosV2", "IonQ"). Ignored if backend is provided.
        :param backend: Optional backend instance. If provided, used directly; else backend_name is required.
        :param memory_coherence_time_s: Optional memory coherence time in seconds (T2 / storage budget proxy).
        :param available_comm_qubits: Optional override for available communication qubits on this QPU.
        :param security_risk: Trust-risk score in [0, 1], where 0 is trusted and 1 is untrusted.
        """
        if not 0.0 <= float(security_risk) <= 1.0:
            raise ValueError("security_risk must be in [0, 1].")
        self.qpu_id = qpu_id
        self.memory_coherence_time_s = memory_coherence_time_s
        self.available_comm_qubits = available_comm_qubits
        self.security_risk = float(security_risk)

        if backend is None:
            if backend_name is None:
                raise ValueError("Either backend_name or backend must be provided.")
            if backend_name not in FAKE_BACKENDS:
                raise ValueError(
                    f"Unknown backend name '{backend_name}'. "
                    f"Available options are: {list(FAKE_BACKENDS.keys())}"
                )
            backend_cls = FAKE_BACKENDS[backend_name]
            backend = backend_cls()

        # Register custom instructions to target
        target = backend.target
        target.add_instruction(RemoteGate, name="R")
        target.add_instruction(MX, name="MX")
        target.add_instruction(MZ, name="MZ")
        target.add_instruction(AnsM, name="ANS_M")
        target.add_instruction(AnsM, name="IF_Z")
        target.add_instruction(AnsM, name="IF_X")
        target.add_instruction(S_CX, name="MS")

        self.backend = backend
        self.target = backend.target

    def compile_x_gate(self):
        """Compile X gate for this QPU's backend"""
        qc = QuantumCircuit(1)
        qc.x(0)
        compiled = transpile(qc, self.backend)
        return compiled.data

    def compile_z_gate(self):
        """Compile Z gate for this QPU's backend"""
        qc = QuantumCircuit(1)
        qc.z(0)
        compiled = transpile(qc, self.backend)
        return compiled.data

    def __repr__(self):
        name = getattr(self.backend, "name", str(self.backend))
        if callable(name):
            name = name()
        return (
            f"<QPU id={self.qpu_id}, "
            f"backend={name}, num_qubits={self.backend.num_qubits}>"
        )

# -------------------------------------------------------------
class DQCCircuit(QuantumCircuit):
    def __init__(self, *args, **kwargs):
        #  用已有 QuantumCircuit 初始化
        if len(args) == 1 and isinstance(args[0], QuantumCircuit):
            qc = args[0]
            # 保留原寄存器结构
            super().__init__(*qc.qregs, *qc.cregs,
                             name=qc.name,
                             global_phase=qc.global_phase,
                             metadata=copy.deepcopy(qc.metadata) if qc.metadata else None)

            # 复制电路数据
            self.data = copy.deepcopy(qc.data)
        else:
            super().__init__(*args, **kwargs)

        self.step = []                              # circuit snapshots per step
        self.step.append(copy.deepcopy(self))

        self.sub_circuit = []                       # sub-circuits
        self.sub_circuit_trans = []                 # transpiled sub-circuits
        self.result_circuit = None                  # final merged circuit

        self.partition = []                         # partition (groups)
        self.qubit_group = [-1] * self.num_qubits   # qubit index -> group id
        self.Entanglement_swapping = []             # per-group swapping flag
        self.swap_routes = []                       # swap routes (QPU paths)
        self.remote_interactions = []               # (control_qpu, target_qpu, path) per remote CX
        self.request_list = []                      # built by _build_request_list_and_dependencies
        self.depends_on = []                        # depends_on[r] = set of request indices that must complete before r

        self.qubit_tele = []                        # qubit -> its comm-qubit index (teleport)
        self.num_comm_qubits = []                   # num_comm_qubits[group] = available comm qubits
        self.group_comm_slots = []                  # physically allocated comm qubits per group in step[3]/step[4]
        self.group_comm_indices = {}                # group -> allocated comm-qubit indices

        self.qpus = []
        self.merged_qubits_map = {}                 # merged/global qubit -> (sub, local) map
        self.merged_qubits_map_reverse = {}         # (sub, local) -> merged/global qubit map

        self.qpugroup = None

        self.Num_Entanglement_swapping = 0          # total entanglement-swapping hops
        self.Num_RemoteGate = 0
        self.throughput_metrics = None               # set after merge_trans_circuits
        self.schedule = {}                           # set by resource_manager_schedule
        self.scheduled_batches = []                  # final per-batch order after scheduling
        self.network_metrics = {}                    # set by simulate_heralded_entanglement
        self.metrics_request_snapshot = []           # stable pre-rearrange request snapshot for timing samples
        # User-specified per-link purification rounds, e.g.:
        #   qc.purification = {(0, 1): {"rounds": 1}, (2, 3): {"rounds": 2}}
        # N rounds requires 2^N comm qubits per endpoint.
        self.purification = {}
        self.purification_plan = {}                  # set after scheduling
        self.purification_metrics = {}               # summarized purification metrics

    # ------------------------------------------------------------------
    # Purification validation — normalize user dict and check comm qubits
    # ------------------------------------------------------------------
    def _validate_purification(self):
        """
        Normalize ``self.purification`` keys to sorted tuples and verify each
        endpoint has ``2^rounds`` comm qubits available. Raises ValueError if
        not satisfiable by the current architecture.
        """
        if not self.purification:
            self.purification = {}
            return

        normalized = {}
        for link, cfg in self.purification.items():
            key = tuple(sorted(link))
            rounds = int((cfg or {}).get("rounds", 0))
            if rounds <= 0:
                continue
            a, b = key
            need = 2 ** rounds
            for qpu_id in (a, b):
                if qpu_id >= len(self.num_comm_qubits):
                    raise ValueError(
                        f"Purification link {key}: QPU {qpu_id} not in partition"
                    )
                have = self.num_comm_qubits[qpu_id]
                if have < need:
                    raise ValueError(
                        f"Purification link {key}: rounds={rounds} needs "
                        f"{need} comm qubits on QPU {qpu_id}, but only {have} available"
                    )
            normalized[key] = {"rounds": rounds}
        self.purification = normalized

    def _qpu_coherence_times(self, qpu_id):
        """Return (T1, T2) in seconds for a QPU, reading real calibration.

        Custom backends (IonQ*, QuantinuumH2) expose scalar ``t1_time`` /
        ``t2_time``; qiskit fake backends carry per-qubit values in
        ``target.qubit_properties[i].t1/.t2`` — we average those. Returns
        (None, None) when neither form is available. The caller clamps
        T2 <= 2*T1 (required by thermal_relaxation_error).
        """
        q = self.qpugroup.get_qpu(qpu_id) if self.qpugroup else None
        if q is None:
            return (None, None)
        be = getattr(q, "backend", None)
        t1 = getattr(be, "t1_time", None)
        t2 = getattr(be, "t2_time", None)
        if t1 is not None and t2 is not None:
            return (float(t1), float(t2))
        try:
            tgt = getattr(be, "target", None)
            qp = getattr(tgt, "qubit_properties", None) or []
            t1s = [p.t1 for p in qp if getattr(p, "t1", None)]
            t2s = [p.t2 for p in qp if getattr(p, "t2", None)]
            t1 = sum(t1s) / len(t1s) if t1s else None
            t2 = sum(t2s) / len(t2s) if t2s else None
            return (t1, t2)
        except Exception:
            return (None, None)

    def Execution(self, config, qpugroup, comm_noise=False, scheduling_algorithm="FCFS",
                  allow_reroute=True, verbose=False, skip_aer=False, output_level=None,
                  idle_decoherence=False, wait_decoherence=False):
        """Run the full DQC pipeline: partition → batch → route → schedule → herald → noise → execute.

        Args:
            config: Partition spec — list of qubit-index lists, one per QPU group.
                Example: ``[[0, 1], [2, 3]]`` puts qubits 0,1 on QPU 0 and qubits 2,3 on QPU 1.
                You may also pass a list of ints (group sizes), e.g. ``[2, 2]``.
            qpugroup: ``QPUManager`` with QPUs, topology, routing/security policy.
            comm_noise: If True, inject depolarizing noise on remote gates using each
                link's estimated end-to-end fidelity. Default False.
            scheduling_algorithm: ``"FCFS"`` (first-come-first-served), ``"SPF"`` (shortest
                path first), or ``"SJF"`` (shortest expected job first). Default ``"FCFS"``.
            allow_reroute: If True, conflicted requests can take an alternative path
                when their primary path's comm-qubit capacity is exceeded. Default True.
            skip_aer: If True, return early after the heralded entanglement simulation
                — skips Aer transpile + circuit execution. Use for fast network-only sweeps
                where ``result.network_metrics`` is all that's needed. Default False.
            output_level: ``"quiet"`` | ``"summary"`` | ``"verbose"``. ``quiet`` prints
                nothing. ``summary`` (default) prints a compact 4-5 line block with network
                shape, timing, fidelity, and security score (when security routing is active).
                ``verbose`` adds structured Batching/Routing/Heralded/Purification blocks.
            verbose: Deprecated. ``verbose=True`` maps to ``output_level="verbose"``,
                ``verbose=False`` to ``"summary"``. ``output_level`` always wins if set.

        Returns:
            The transpiled+merged ``QuantumCircuit`` ready to run on AerSimulator, with
            ``.network_metrics``, ``.purification_metrics``, and ``.schedule`` attributes
            attached. If ``skip_aer=True``, returns ``self`` (the ``DQCCircuit``) instead;
            ``self.network_metrics`` is still populated.
        """
        import copy as _copy

        if output_level is None:
            output_level = "verbose" if verbose else "summary"
        if output_level not in ("quiet", "summary", "verbose"):
            raise ValueError(
                f"output_level must be 'quiet', 'summary', or 'verbose'; got {output_level!r}"
            )
        self._output_level = output_level
        self.verbose = False
        self.qpugroup = qpugroup
        self.idle_decoherence = idle_decoherence
        self._idle_charge = None  # reset per run; rebuilt once lazily in merge_trans_circuits
        # Optional: add T2 dephasing for the time a request's two data
        # qubits spend waiting in the scheduler queue. Off by default.
        self.wait_decoherence = wait_decoherence
        qpus = self.qpugroup.qpus
        self.purification_plan = {}
        self.purification_metrics = {}
        self.allow_reroute_override = allow_reroute

        # --- Stage 1: partition, comm qubit allocation ---
        self.split(config)
        self.valid_trans()
        self.check_swap_entanglement()
        self.rearrange_with_partition()
        # step[3] now contains the clean cx circuit (before rewrite)

        # --- Compute available comm qubits per group from real backend capacity ---
        # Formula: backend.num_qubits - partition_size, floored at the number of
        # comm qubits physically allocated by rearrange_with_partition (1 or 2).
        _sorted_qpus = sorted(qpus, key=lambda x: x.qpu_id)
        self.num_comm_qubits = []
        for gid, group in enumerate(self.partition):
            allocated = 1 + (1 if self.Entanglement_swapping[gid] == 1 else 0)
            if gid < len(_sorted_qpus):
                qpu_obj = _sorted_qpus[gid]
                override = getattr(qpu_obj, "available_comm_qubits", None)
                if override is not None:
                    available = max(1, int(override))
                else:
                    available = qpu_obj.backend.num_qubits - len(group)
                self.num_comm_qubits.append(max(allocated, available))
            else:
                self.num_comm_qubits.append(allocated)

        # --- Validate user-specified purification against comm qubit capacity ---
        self._validate_purification()

        # --- Compile-time pipeline on step[3] (clean cx gates) ---
        self._build_request_list_and_dependencies()
        self.metrics_request_snapshot = _copy.deepcopy(self.request_list)

        self.rearrange_for_parallel()

        self.resource_manager_schedule(scheduling_algorithm)
        self._sync_scheduled_paths_into_step3()
        self._plan_purification()

        self.simulate_heralded_entanglement()

        # Render tiered output — fires for both skip_aer=True and full Aer runs
        self._render_output(qpugroup, scheduling_algorithm)

        if skip_aer:
            return self

        # --- Rewrite step[3] → step[4] (RemoteGate / MX / MZ) ---
        self.rewrite_cross_group_cnots()

        # --- Per-QPU transpilation ---
        self.physic_split()
        self.transpile_subcircuits(qpus)

        # --- Merge + Aer execution ---
        result_qc = self.merge_trans_circuits(comm_noise)
        result_qc.network_metrics = getattr(self, "network_metrics", {})
        result_qc.purification_metrics = getattr(self, "purification_metrics", {})
        result_qc.schedule = getattr(self, "schedule", {})

        return result_qc

    # ──────────────────────────────────────────────────────────────────────
    # Tiered terminal output: quiet / summary (default) / verbose
    # ──────────────────────────────────────────────────────────────────────

    def _security_active(self):
        """True if any QPU has risk > 0 or a security-aware router is in use."""
        if any(float(getattr(q, "security_risk", 0.0)) > 0 for q in self.qpugroup.qpus):
            return True
        algo = getattr(self.qpugroup, "routing_algo", "")
        return str(algo) in ("MinRisk", "SecurityConstrained", "JointSecurity")

    def _render_output(self, qpugroup, scheduling_algorithm):
        if self._output_level == "quiet":
            return
        self._render_summary_block(qpugroup, scheduling_algorithm)
        if self._output_level == "verbose":
            self._render_verbose_blocks(qpugroup)

    def _render_summary_block(self, qpugroup, scheduling_algorithm):
        nm = getattr(self, "network_metrics", {}) or {}
        pm = getattr(self, "purification_metrics", {}) or {}
        n_qpus = len(qpugroup.qpus)
        n_edges = sum(len(v) for v in qpugroup.map.values()) // 2
        n_remote = len(getattr(self, "request_list", []) or [])
        n_batches = len(getattr(self, "parallel_batches", []) or [])
        wall_ms = nm.get("wall_clock", 0.0) * 1e3
        reroute_on = "on" if getattr(self, "allow_reroute_override", True) else "off"
        purif_on = "on" if getattr(self, "purification", {}) else "off"

        algo = qpugroup.routing_algo
        algo_name = getattr(algo, "__name__", str(algo))
        print("═══ SimDisQ-Net ═══")
        print(f"  Network   : {n_qpus} QPUs · {n_edges} edges · {algo_name} routing · {scheduling_algorithm} sched")
        print(f"  Workload  : {n_remote} req · {n_batches} batch{'es' if n_batches != 1 else ''} · reroute={reroute_on} · purif={purif_on}")
        print(f"  Timing    : {wall_ms:.2f} ms wall")

        e2e_map = pm.get("request_e2e_fidelity", {}) or {}
        if e2e_map:
            vals = list(e2e_map.values())
            mean_e2e = sum(vals) / len(vals)
            print(f"  Fidelity  : F_e2e={mean_e2e:.2f} ({min(vals):.2f}-{max(vals):.2f} over {len(vals)} req)")

        if n_remote:
            R_bar = nm.get("mean_path_risk", 0.0)
            E = nm.get("untrusted_path_fraction", 0.0)
            S = nm.get("safety_score", 1.0)
            n_untrusted = int(round(E * n_remote))
            print(f"  Security  : avg path risk     {R_bar:.2f}  (0=safe, 1=compromised)")
            print(f"              high-risk paths   {E*100:.0f}%   ({n_untrusted} of {n_remote} requests)")
            print(f"              safety score      {S:.2f}/1.0")

    def _render_verbose_blocks(self, qpugroup):
        nm = getattr(self, "network_metrics", {}) or {}
        pm = getattr(self, "purification_metrics", {}) or {}
        schedule = getattr(self, "schedule", {}) or {}
        n_remote = len(getattr(self, "request_list", []) or [])
        n_batches = len(getattr(self, "parallel_batches", []) or [])
        reroutes = nm.get("reroute_count", 0)

        # ── Batching ──
        print("")
        print("──── Batching ────")
        print(f"  {n_remote} reqs → {n_batches} batch{'es' if n_batches != 1 else ''} · {reroutes} reroute{'s' if reroutes != 1 else ''}")

        # ── Routing ──
        algo = qpugroup.routing_algo
        algo_name = getattr(algo, "__name__", str(algo))
        print("")
        print(f"──── Routing ({algo_name}) ────")
        print(f"  {'Req':<5} {'Path':<30} {'Hops':<6} {'safety':<8}")
        safety = nm.get("request_safety", {}) or {}
        path_len = nm.get("request_path_length", {}) or {}
        shown = 0
        max_show = 12
        for rid in sorted(schedule):
            if shown >= max_show:
                remaining = len(schedule) - shown
                if remaining > 0:
                    print(f"  ...   ({remaining} more)")
                break
            path = schedule[rid].get("path", []) or []
            path_str = "→".join(f"QPU{q}" for q in path) if path else "(none)"
            if len(path_str) > 28:
                path_str = path_str[:27] + "…"
            s = safety.get(rid, 1.0)
            print(f"  {rid:<5} {path_str:<30} {path_len.get(rid, 0):<6} {s:<8.2f}")
            shown += 1
        theta = nm.get("high_risk_node_threshold", 0.5)
        inc = nm.get("risk_include_endpoints", False)
        ep_note = "source/target QPUs counted" if inc else "source/target QPUs ignored"
        print(f"  high-risk threshold: r ≥ {theta} · {ep_note}")

        # ── Heralded entanglement ──
        link_metrics = nm.get("link_metrics", {}) or {}
        if link_metrics:
            print("")
            print("──── Heralded entanglement ────")
            print(f"  {'Link':<7} {'theor_p':<10} {'actual_p':<10} {'attempts':<10} {'duration (μs)'}")
            for lk in sorted(link_metrics):
                a, b = lk
                m = link_metrics[lk]
                tot_att = m.get("total_attempts", 0)
                suc = m.get("successes", 0)
                t_s = m.get("time", 0.0)
                actual_p = (suc / tot_att) if tot_att else 0.0
                p_cfg, _ = qpugroup.get_link_success_params(a, b)
                theor_str = f"{p_cfg:.3f}" if p_cfg is not None else "N/A"
                dur_us = t_s * 1e6
                link_str = f"{a}-{b}"
                print(f"  {link_str:<7} {theor_str:<10} {actual_p:<10.3f} {tot_att:<10} {dur_us:>5.0f}")

        # ── Purification ──
        print("")
        print("──── Purification ────")
        user_cfg = getattr(self, "purification", {}) or {}
        if not user_cfg:
            print("  (none configured)")
        else:
            total_rounds = pm.get("total_rounds", 0)
            total_raw = pm.get("total_raw_pairs", 0)
            print(f"  configured links : {len(user_cfg)}")
            print(f"  total rounds     : {total_rounds}")
            print(f"  raw Bell pairs   : {total_raw}")
        print("")

    def _swap_fidelity(self, f_left, f_right):
        """
        End-to-end fidelity after one entanglement swap of two Werner-state Bell pairs.
        Formula: F_out = F_left * F_right + (1 - F_left)(1 - F_right) / 9
        """
        return f_left * f_right + (1.0 - f_left) * (1.0 - f_right) / 9.0

    def _e2e_fidelity(self, link_fidelities):
        """
        Propagate per-link fidelities through entanglement swaps to get
        the end-to-end Bell-pair fidelity delivered to the endpoints.
        For n links there are n-1 swaps; each swap applies _swap_fidelity.
        """
        if not link_fidelities:
            return 1.0
        f = link_fidelities[0]
        for f_next in link_fidelities[1:]:
            f = self._swap_fidelity(f, f_next)
        return f

    def _bbpssw_next_fidelity(self, fidelity):
        """One-round BBPSSW purification update for a Werner-state Bell pair."""
        err = (1.0 - fidelity) / 3.0
        numerator = fidelity**2 + err**2
        denominator = fidelity**2 + (2.0 * fidelity * (1.0 - fidelity) / 3.0) + 5.0 * err**2
        return numerator / denominator

    def _estimate_link_fidelity(self, qpu_id1, qpu_id2):
        """
        Raw Bell-pair fidelity from endpoint gate quality and fiber amplitude
        damping (Werner-state AD formula):

            F_raw = 0.5*(F_2q_A + F_2q_B) * ((1 + exp(-alpha*d/2))/2)^2

        where alpha = 0.2/4.343 (km^-1) is the fiber attenuation coefficient.
        """
        qpu1 = self.qpugroup.get_qpu(qpu_id1) if self.qpugroup else None
        qpu2 = self.qpugroup.get_qpu(qpu_id2) if self.qpugroup else None
        if qpu1 is not None and qpu2 is not None:
            try:
                p1 = BackendParams.from_backend(qpu1.backend)
                p2 = BackendParams.from_backend(qpu2.backend)
                distance = 0.0
                for nb, dist in self.qpugroup.map.get(qpu_id1, []):
                    if nb == qpu_id2:
                        distance = float(dist)
                        break
                alpha = 0.2 / 4.343  # dB/km -> natural
                survival_amp = math.exp(-alpha * distance / 2.0)
                bell_ad_factor = ((1.0 + survival_amp) / 2.0) ** 2
                local_quality = 0.5 * (p1.fidelity_2q + p2.fidelity_2q)
                return min(0.999999, max(0.0, float(local_quality * bell_ad_factor)))
            except Exception:
                pass

        success_prob, _ = self.qpugroup.get_link_success_params(qpu_id1, qpu_id2)
        if success_prob is None:
            return 1.0
        return min(0.999999, max(0.0, 0.5 + 0.5 * math.sqrt(max(float(success_prob), 0.0))))

    def _link_fidelity_with_purification(self, qpu_id1, qpu_id2, rounds):
        """Return F_link after applying ``rounds`` BBPSSW iterations."""
        F = self._estimate_link_fidelity(qpu_id1, qpu_id2)
        for _ in range(int(rounds)):
            F = self._bbpssw_next_fidelity(F)
        return F

    def _link_fidelity_for_request(self, req_id, qpu_id1, qpu_id2):
        """F_link for a given request, respecting per-link user purification."""
        rounds = self._get_request_link_purification_rounds(req_id, qpu_id1, qpu_id2)
        return self._link_fidelity_with_purification(qpu_id1, qpu_id2, rounds)

    def _build_purification_plan_for_path(self, path, num_comm_qubits=None):
        """
        Build a per-path purification plan from user-specified rounds in
        ``self.purification``. No automatic eligibility / target-fidelity logic.

        ``self.purification`` is a dict: {(qpu_a, qpu_b): {"rounds": N}, ...}
        """
        user_cfg = getattr(self, "purification", {}) or {}
        plan = {
            "enabled": bool(user_cfg),
            "needs_purification": False,
            "total_rounds": 0,
            "total_raw_pairs": 0,
            "links": [],
        }
        if len(path) < 2:
            return plan

        for a, b in zip(path[:-1], path[1:]):
            link_key = tuple(sorted((a, b)))
            raw_fidelity = self._estimate_link_fidelity(a, b)
            user_entry = user_cfg.get(link_key)
            rounds = int(user_entry.get("rounds", 0)) if user_entry else 0
            rounds = max(0, rounds)

            post_fidelity = raw_fidelity
            for _ in range(rounds):
                post_fidelity = self._bbpssw_next_fidelity(post_fidelity)
            raw_pairs = 2 ** rounds if rounds > 0 else 1

            entry = {
                "link": (a, b),
                "raw_fidelity": raw_fidelity,
                "post_fidelity": post_fidelity,
                "needs_purification": rounds > 0,
                "eligible": rounds > 0,
                "rounds": rounds,
                "raw_pairs": raw_pairs,
            }
            plan["links"].append(entry)
            if rounds > 0:
                plan["needs_purification"] = True
                plan["total_rounds"] += rounds
                plan["total_raw_pairs"] += raw_pairs
            else:
                plan["total_raw_pairs"] += 1

        # E2E fidelity = fold swap_fidelity across all per-link post-fidelities
        if plan["links"]:
            link_fidelities = [e["post_fidelity"] for e in plan["links"]]
            plan["e2e_fidelity"] = self._e2e_fidelity(link_fidelities)

        return plan

    def _sync_scheduled_paths_into_step3(self):
        """
        Propagate scheduled paths back into step[3] so later rewrite uses the
        same path that scheduling and purification planned on.
        """
        if not self.schedule or len(self.step) < 4 or not self.request_list:
            return

        circ = self.step[3]
        data = list(circ.data)
        new_remote_interactions = list(self.remote_interactions)
        changed = False

        for req in self.request_list:
            req_id = req["id"]
            sched_path = list(self.schedule.get(req_id, {}).get("path", []) or [])
            if not sched_path:
                continue
            req["scheduled_path"] = sched_path
            if "logical_path" not in req:
                req["logical_path"] = list(req.get("path", []))
            if req.get("path") == sched_path:
                continue

            req["path"] = sched_path
            ctrl_qpu = req["ctrl_qpu"]
            tgt_qpu = req["tgt_qpu"]
            new_remote_interactions[req_id] = (ctrl_qpu, tgt_qpu, sched_path)
            pos = req["start_pos"]
            inst_obj = data[pos]
            if inst_obj.operation.name == "S_CX":
                data[pos] = CircuitInstruction(
                    S_CX(ctrl_qpu, tgt_qpu, sched_path),
                    inst_obj.qubits,
                    inst_obj.clbits,
                )
                changed = True

        self.remote_interactions = new_remote_interactions
        if not changed:
            return

        new_circ = DQCCircuit(circ.qubits)
        new_circ.qubit_group = circ.qubit_group
        new_circ.qubit_tele = circ.qubit_tele
        for creg in getattr(circ, "cregs", []):
            new_circ.add_register(ClassicalRegister(len(creg), creg.name))
        for inst_obj in data:
            new_circ.append(inst_obj.operation, inst_obj.qubits, inst_obj.clbits)
        self.step[3] = new_circ

    def _plan_purification(self):
        """
        Compute per-request purification plans (analytical fidelity only) from
        user-specified ``self.purification`` dict. Always runs — produces e2e
        fidelity for every request regardless of whether purification is used.
        """
        user_cfg = getattr(self, "purification", {}) or {}

        plans = {}
        purified_requests = []
        total_rounds = 0
        total_raw_pairs = 0
        request_e2e_fidelity = {}

        for req_id, info in sorted(self.schedule.items()):
            path = list(info.get("path", []) or [])
            plan = self._build_purification_plan_for_path(path, self.num_comm_qubits)
            plan["request_id"] = req_id
            plans[req_id] = plan
            e2e = plan.get("e2e_fidelity")
            if e2e is not None:
                request_e2e_fidelity[req_id] = e2e
            info["purification"] = {
                "needed": plan["needs_purification"],
                "total_rounds": plan["total_rounds"],
                "total_raw_pairs": plan["total_raw_pairs"],
                "e2e_fidelity": e2e,
            }
            if plan["needs_purification"]:
                purified_requests.append(req_id)
            total_rounds += plan["total_rounds"]
            total_raw_pairs += plan["total_raw_pairs"]

        self.purification_plan = plans
        self.purification_metrics = {
            "enabled": bool(user_cfg),
            "user_config": dict(user_cfg),
            "purified_requests": purified_requests,
            "total_rounds": total_rounds,
            "total_raw_pairs": total_raw_pairs,
            "request_plans": plans,
            "request_e2e_fidelity": request_e2e_fidelity,
        }

    def _get_request_link_purification_rounds(self, req_id, qpu_id1, qpu_id2):
        plan = getattr(self, "purification_plan", {}).get(req_id, {})
        if not plan:
            return 0
        want = tuple(sorted((qpu_id1, qpu_id2)))
        for entry in plan.get("links", []):
            if tuple(sorted(entry["link"])) == want and entry.get("eligible", False):
                return int(entry.get("rounds", 0))
        return 0

    # Get the index of a qubit in the circuit
    def get_index(self, q):
        return int(self.find_bit(q).index)

    # Split the circuit based on the provided configuration
    def split(self, config):
        partition = []

        if all(isinstance(x, int) for x in config):
            # 按长度生成索引列表
            start = 0
            for s in config:
                indices = list(range(start, start + s))
                partition.append(indices)
                start += s
        elif all(isinstance(x, (list, tuple)) for x in config):
            # 按指定索引组合, 并排序
            for sub in config:
                partition.append(sorted(sub))
        else:
            raise ValueError("Input must be a list of ints or a list of lists of ints")
        
        self.partition = partition

        for gid, group in enumerate(partition):
            self.Entanglement_swapping.append(0)
            for q in group:
                self.qubit_group[q] = gid
        return partition

    # Preprocessing to validate and decompose cross-group multi-qubit gates to single-qubit gates and CX gates
    # step[1]
    def valid_trans(self):
        """
        遍历 old_circ 的指令，检查是否跨组多比特门。
        如果跨组且不是 CX，则分解后 append 到 new_circ；
        如果同组或是 CX，则直接 append。
        
        参数:
            new_circ: DQCCircuit，目标电路
            old_circ: DQCCircuit，源电路（step[0]）
        """
        old_circ = self

         # === 1. 构建新电路（仅复制量子比特）===
        new_circ = DQCCircuit(old_circ.qubits)
        new_circ.qubit_group = old_circ.qubit_group
        new_circ.qubit_tele = old_circ.qubit_tele
        # === 2. 继承所有经典寄存器 ===
        # 复制通信寄存器（如果有）
        comm_cregs = [creg for creg in getattr(old_circ, "cregs", []) if "Tele" in creg.name]
        for comm_creg in comm_cregs:
            new_circ.add_register(ClassicalRegister(len(comm_creg), comm_creg.name))

        # 复制原始寄存器（如果有）
        orig_cregs = [creg for creg in getattr(old_circ, "cregs", []) if "Tele" not in creg.name]
        for orig_creg in orig_cregs:
            new_circ.add_register(ClassicalRegister(len(orig_creg), orig_creg.name))

        for instri in old_circ.data:
            instr = instri.operation   # 量子门或操作对象
            qargs = instri.qubits      # 作用的量子比特列表
            cargs = instri.clbits      # 作用的经典比特列表
            
            # 先处理cx
            if instr.name in ["cx"]:
                new_circ.append(instr, qargs, cargs)
                continue

            # 单比特门直接 append
            if len(qargs) <= 1:
                new_circ.append(instr, qargs, cargs)
                continue
            
            # 多比特门，判断是否跨组
            groups = [new_circ.qubit_group[old_circ.get_index(q)] for q in qargs]
            if len(set(groups)) > 1:
                # 跨组且不是 CX，分解
                ci = CircuitInstruction(instr, qargs, cargs)
                decomposed_instrs = self.decompose_and_get_data(ci)
                for di in decomposed_instrs:
                    mapped_qubits = [new_circ.qubits[old_circ.get_index(q)] for q in di.qubits]
                    mapped_clbits = [new_circ.clbits[old_circ.get_index(c)] for c in di.clbits]
                    new_circ.append(di.operation, mapped_qubits, mapped_clbits)
            else:
                # 同组，多比特门直接 append
                new_circ.append(instr, qargs, cargs)
        
        self.step.append(new_circ)
        return new_circ
    
    # step[2]
    def check_swap_entanglement(self):
        """Check cross-QPU entanglement; if no direct link, find a SWAP route."""
        swap_routes = []
        remote_interactions = []
        qpu_map = self.qpugroup.map

        old_circ = self.step[1]

        # === 1. Build new circuit (copy qubits only) ===
        new_circ = DQCCircuit(old_circ.qubits)

        for creg in getattr(old_circ, "cregs", []):
            new_circ.add_register(ClassicalRegister(len(creg), creg.name))

        for instr in old_circ.data:
            if instr.operation.name == "cx":
                ctrl, tgt = instr.qubits
                g_ctrl = self.qubit_group[self.get_index(ctrl)]
                g_tgt = self.qubit_group[self.get_index(tgt)]

                if g_ctrl != g_tgt:
                    if not self.qpugroup.check_connection(g_ctrl, g_tgt):
                        # --- Not directly connected: find route (BFS / Throughput / Fidelity) ---
                        path = self._find_swap_path(g_ctrl, g_tgt)
                        if path:
                            swap_routes.append(path)
                            remote_interactions.append((g_ctrl, g_tgt, path))
                            new_circ.append(S_CX(g_ctrl, g_tgt, path), instr.qubits, instr.clbits)
                            self.Num_Entanglement_swapping += len(path) - 2
                        else:
                            print(f"[Warning] No route found between QPU {g_ctrl} and {g_tgt}")
                    else:
                        # Directly connected (adjacent): keep CX; add one entry per remote CX for request_list
                        direct_path = [g_ctrl, g_tgt]
                        if getattr(self.qpugroup, "routing_algo", None) == "SecurityConstrained":
                            threshold = float(getattr(self.qpugroup, "security_threshold", 1.0))
                            if self._path_risk(direct_path) > threshold:
                                # Let the security router either find an acceptable
                                # alternate path or fail closed with ValueError.
                                direct_path = self._find_swap_path(g_ctrl, g_tgt)
                                if not direct_path:
                                    raise ValueError(
                                        f"No path from QPU {g_ctrl} to QPU {g_tgt} satisfies "
                                        f"security_threshold={threshold}."
                                    )
                        remote_interactions.append((g_ctrl, g_tgt, direct_path))
                        if len(direct_path) > 2:
                            swap_routes.append(direct_path)
                            new_circ.append(S_CX(g_ctrl, g_tgt, direct_path), instr.qubits, instr.clbits)
                            self.Num_Entanglement_swapping += len(direct_path) - 2
                        else:
                            new_circ.append(instr.operation, instr.qubits, instr.clbits)
                else:
                    # Same group: keep CX
                    new_circ.append(instr.operation, instr.qubits, instr.clbits)
            else:
                new_circ.append(instr.operation, instr.qubits, instr.clbits)

        for path in swap_routes:
            # Mark intermediate QPUs along the route
            for node in path[1:-1]:
                self.Entanglement_swapping[node] = 1

        self.swap_routes = swap_routes
        self.remote_interactions = remote_interactions
        self.step.append(new_circ)

    def _find_swap_path(self, start, end):
        """
        Find swap route according to qpugroup.routing_algo:
        - If callable: (start, end, graph, link_params) -> path or None; None -> use qpugroup.fallback_algo.
        - "BFS" -> min hops; "Throughput" -> min attempt_time/P; "Fidelity" -> max path F_link (-log F).
        - "MinRisk" -> min path risk; "SecurityConstrained" -> min latency under risk threshold.
        - "JointSecurity" -> weighted min of normalized latency, fidelity loss, and risk.
        """
        algo = getattr(self.qpugroup, "routing_algo", "Throughput")
        graph = self.qpugroup.map
        link_params = self.qpugroup.link_params

        def builtin(name):
            if name == "BFS":
                return self._find_shortest_path(graph, start, end)
            if name == "Throughput":
                return self._find_shortest_path_by_throughput(start, end)
            if name == "Fidelity":
                return self._find_shortest_path_by_fidelity(start, end)
            if name == "MinRisk":
                return self._find_path_by_min_risk(start, end)
            if name == "SecurityConstrained":
                return self._find_path_by_security_constrained_latency(start, end)
            if name == "JointSecurity":
                return self._find_path_by_joint_security(start, end)
            return None

        if callable(algo):
            path = algo(start, end, graph, link_params)
            if path is not None:
                return path
            fallback = getattr(self.qpugroup, "fallback_algo", "BFS")
            fallback_path = builtin(fallback)
            if fallback_path is not None:
                return fallback_path
            return self._find_shortest_path(graph, start, end)

        path = builtin(algo)
        if path is not None:
            return path
        return self._find_shortest_path_by_throughput(start, end)

    def _enumerate_simple_paths(self, start, end, graph=None):
        """Enumerate all simple QPU paths from start to end."""
        graph = graph or self.qpugroup.map
        paths = []

        def dfs(node, visited, path):
            if node == end:
                paths.append(list(path))
                return
            for nb, _ in graph.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    dfs(nb, visited, path)
                    path.pop()
                    visited.remove(nb)

        dfs(start, {start}, [start])
        return paths

    def _path_expected_latency(self, path):
        """Expected path latency under the same concurrent-hop model as Throughput routing."""
        if len(path) < 2:
            return 0.0

        def link_exp_time(u, v):
            p, t = self.qpugroup.get_link_success_params(u, v)
            if p and p > 0:
                return t / p
            return 1.0

        def swap_time(qpu_id):
            try:
                qpu = self.qpugroup.get_qpu(qpu_id) or self.qpugroup.qpus[qpu_id]
                bp = BackendParams.from_backend(qpu.backend)
                return bp.gate_time_2q
            except Exception:
                return 0.0

        link_exp = [link_exp_time(path[i], path[i + 1]) for i in range(len(path) - 1)]
        if len(link_exp) == 1:
            return link_exp[0]
        swap_exp = [swap_time(path[i]) for i in range(1, len(path) - 1)]
        pair_ready = max(link_exp[0], link_exp[1]) + swap_exp[0]
        for i in range(1, len(swap_exp)):
            pair_ready = max(pair_ready, link_exp[i + 1]) + swap_exp[i]
        return pair_ready

    def _path_e2e_fidelity(self, path):
        """Analytical end-to-end path fidelity using raw per-link fidelities."""
        if len(path) < 2:
            return 1.0
        link_fidelities = [
            self._estimate_link_fidelity(a, b)
            for a, b in zip(path[:-1], path[1:])
        ]
        return self._e2e_fidelity(link_fidelities)

    def _path_risk(self, path):
        """
        Probability that at least one included QPU/repeater on the path is risky:
        R(p) = 1 - product(1 - r_v).
        """
        if not path:
            return 0.0
        include_endpoints = bool(getattr(self.qpugroup, "risk_include_endpoints", False))
        nodes = path if include_endpoints else path[1:-1]
        safe_prob = 1.0
        for qpu_id in nodes:
            qpu = self.qpugroup.get_qpu(qpu_id) if self.qpugroup else None
            risk = float(getattr(qpu, "security_risk", 0.0)) if qpu is not None else 0.0
            risk = max(0.0, min(1.0, risk))
            safe_prob *= (1.0 - risk)
        return 1.0 - safe_prob

    def _security_weights(self):
        weights = getattr(self.qpugroup, "security_weights", {}) or {}
        return {
            "latency": float(weights.get("latency", 1.0)),
            "fidelity_loss": float(weights.get("fidelity_loss", 1.0)),
            "risk": float(weights.get("risk", 1.0)),
        }

    def _find_path_by_min_risk(self, start, end):
        paths = self._enumerate_simple_paths(start, end)
        if not paths:
            return None
        return min(paths, key=lambda p: (self._path_risk(p), self._path_expected_latency(p), len(p)))

    def _find_path_by_security_constrained_latency(self, start, end):
        paths = self._enumerate_simple_paths(start, end)
        if not paths:
            return None
        threshold = float(getattr(self.qpugroup, "security_threshold", 1.0))
        feasible = [p for p in paths if self._path_risk(p) <= threshold]
        if not feasible:
            raise ValueError(
                f"No path from QPU {start} to QPU {end} satisfies "
                f"security_threshold={threshold}."
            )
        return min(feasible, key=lambda p: (self._path_expected_latency(p), self._path_risk(p), len(p)))

    def _find_path_by_joint_security(self, start, end):
        paths = self._enumerate_simple_paths(start, end)
        if not paths:
            return None
        weights = self._security_weights()
        latencies = {tuple(p): self._path_expected_latency(p) for p in paths}
        max_latency = max(latencies.values()) if latencies else 0.0
        if max_latency <= 0.0:
            max_latency = 1.0

        def cost(path):
            latency_norm = latencies[tuple(path)] / max_latency
            fidelity_loss = 1.0 - self._path_e2e_fidelity(path)
            risk = self._path_risk(path)
            return (
                weights["latency"] * latency_norm
                + weights["fidelity_loss"] * fidelity_loss
                + weights["risk"] * risk
            )

        return min(paths, key=lambda p: (cost(p), self._path_risk(p), self._path_expected_latency(p), len(p)))

    def _find_shortest_path_by_throughput(self, start, end):
        """
        Find path that minimizes expected entanglement generation time under the
        CONCURRENT HOP model (matches SimDisQ's actual wall clock calculation).

        Per-link expected duration: E[link_k] = t_k / p_k.
        Multi-hop fold:
            pair_ready = max(E[link_0], E[link_1]) + swap_0
            pair_ready = max(pair_ready, E[link_2]) + swap_1
            ...

        Enumerates all simple paths and picks the one with minimum pair_ready.
        Dijkstra cannot be used here because the objective is max-based
        (non-additive), not sum-based.
        """
        paths = self._enumerate_simple_paths(start, end)
        if not paths:
            return None
        return min(paths, key=lambda p: (self._path_expected_latency(p), len(p)))

    def _find_shortest_path_by_fidelity(self, start, end):
        """
        Find path that maximizes per-link Bell pair fidelity.
        Uses Dijkstra with link cost = -log(F_link), where F_link is the raw
        analytical link fidelity (no purification).
        """
        graph = self.qpugroup.map

        def edge_cost(u, v):
            F = self._estimate_link_fidelity(u, v)
            if F is None or F <= 0:
                return 1e9
            return -math.log(max(F, 1e-9))

        return self._dijkstra_path(graph, start, end, edge_cost)

    def _dijkstra_path(self, graph, start, end, edge_cost_fn):
        """Dijkstra: (cost, node, path). Returns path list or None."""
        heap = [(0.0, start, [start])]
        seen = {start: 0.0}
        while heap:
            cost, node, path = heapq.heappop(heap)
            if node == end:
                return path
            if cost > seen.get(node, float("inf")):
                continue
            for neighbor, _ in graph.get(node, []):
                c = edge_cost_fn(node, neighbor)
                new_cost = cost + c
                if new_cost < seen.get(neighbor, float("inf")):
                    seen[neighbor] = new_cost
                    heapq.heappush(heap, (new_cost, neighbor, path + [neighbor]))
        return None

    def _find_shortest_path(self, graph, start, end):
        """Find the shortest path by hop count (BFS). Used as fallback if needed."""
        from collections import deque

        visited = set()
        queue = deque([[start]])

        while queue:
            path = queue.popleft()
            node = path[-1]
            if node == end:
                return path

            if node not in visited:
                visited.add(node)
                for neighbor, _ in graph.get(node, []):
                    if neighbor not in visited:
                        queue.append(path + [neighbor])

        return None

    # step[3]
    def rearrange_with_partition(self):
        """
        根据 partition 重新排列 qubits, 每组加一个通信比特,
        并为 group 之间的通信分配 classical bits。
        """
        if not self.partition:
            raise ValueError("请先设置 self.partition")

        partition = self.partition
        num_groups = len(partition)
        _sorted_qpus = sorted(getattr(self.qpugroup, "qpus", []), key=lambda x: x.qpu_id)
        group_comm_slots = []
        for gid, group in enumerate(partition):
            slots = 1 + (1 if self.Entanglement_swapping[gid] == 1 else 0)
            if gid < len(_sorted_qpus):
                # Keep the physical comm-qubit allocation aligned with the capacity
                # the scheduler will later assume for this QPU.
                available = _sorted_qpus[gid].backend.num_qubits - len(group)
                if available > 0:
                    slots = max(slots, available)
            group_comm_slots.append(slots)
        self.group_comm_slots = group_comm_slots

        # classical bit 数量 = g * (g-1)
        comm_creg = ClassicalRegister(num_groups * (num_groups - 1), "Tele")

        # 每组 qubits 数量 = 原本 + 1(comm)

        # 先统计 partition 中普通 qubit 数量
        total_qubits = sum(len(group) for group in partition)
        # Add the physically allocated communication qubits for each group.
        total_qubits += sum(group_comm_slots)

        new_qreg = QuantumRegister(total_qubits, "q")

        new_circ = DQCCircuit(new_qreg, comm_creg)

        old_circ = self.step[2]
        # === 4. 保留原始经典寄存器信息 ===
        # 如果原始电路 self 中有经典寄存器
        if hasattr( old_circ, "clbits") and old_circ.clbits:
            # 尝试从 self.cregs 中找到原寄存器名
            if hasattr( old_circ, "cregs") and old_circ.cregs:
                # 取第一个 ClassicalRegister 的名字
                orig_name =  old_circ.cregs[0].name
            else:
                orig_name = "c"  # 如果没有记录，则默认命名为 "c"

            # 使用原寄存器名创建新的 ClassicalRegister
            orig_creg = ClassicalRegister(len(old_circ.clbits), orig_name)
            new_circ.add_register(orig_creg)

            # 保存引用方便之后访问
            new_circ.orig_creg = orig_creg
        else:
            new_circ.orig_creg = None

        # ===== 建立旧 qubit -> 新 qubit 映射 =====
        old2new = {}
        group_comm_qubits = {}
        new_index = 0

        for gid, indices in enumerate(partition):
            # 字典映射量子寄存器
            # group 内 qubit
            for qi in indices:
                old2new[qi] = new_qreg[new_index]   
                new_index += 1
            group_comm_qubits[gid] = []
            for _ in range(group_comm_slots[gid]):
                comm_q = new_qreg[new_index]
                group_comm_qubits[gid].append(comm_q)
                new_index += 1

        # ===== 遍历原电路, 重映射到新电路 =====
        for instr in old_circ.data:
            qubit_indices = [old_circ.get_index(q) for q in instr.qubits]

            if all(qi in old2new for qi in qubit_indices):
                new_qargs = [old2new[qi] for qi in qubit_indices]
                new_circ.append(instr.operation, new_qargs, instr.clbits)

        # ==== 给 new_circ.qubit_group 填值 =====
        new_circ.qubit_group = []
        for gid, group in enumerate(partition):
            new_circ.qubit_group.extend([gid] * (len(group) + group_comm_slots[gid]))

        # ===== 给 new_circ.qubit_tele 填值 =====
        qubit_tele = [-1] * len(new_qreg) 

        self.group_comm_indices = {}
        for gid, group in enumerate(partition):
            comm_q = group_comm_qubits[gid][0]
            comm_idx = new_circ.get_index(comm_q)
            self.group_comm_indices[gid] = [new_circ.get_index(q) for q in group_comm_qubits[gid]]

            # 普通 qubit 指向该组通信 qubit
            for qi in group:
                qubit_tele[new_circ.get_index(old2new[qi])] = comm_idx

            # 通信 qubit 自己设为 -1
            for comm_index in self.group_comm_indices[gid]:
                qubit_tele[comm_index] = -1

        # 赋值
        new_circ.qubit_tele = qubit_tele

        # Build remapped-index → original qubit label for readable circuit plots.
        # Data qubits get their original register label (e.g. "q[2]" → "q2").
        # Comm / swap-comm qubits are labelled "comm<gid>" / "swap<gid>".
        new2orig = {}
        swap_count = {}
        for gid, indices in enumerate(partition):
            for qi in indices:
                new_idx = new_circ.get_index(old2new[qi])
                orig_label = f"q{qi}"
                new2orig[new_idx] = orig_label
        # Comm qubits: iterate new_circ qubits and label anything not yet mapped
        comm_gid_counter = {}
        for new_idx in range(new_circ.num_qubits):
            if new_idx not in new2orig:
                gid = new_circ.qubit_group[new_idx]
                comm_gid_counter.setdefault(gid, 0)
                count = comm_gid_counter[gid]
                if count == 0:
                    new2orig[new_idx] = f"comm{gid}"
                else:
                    new2orig[new_idx] = f"aux{gid}_{count}"
                comm_gid_counter[gid] += 1
        self.qubit_label_map = new2orig  # new_idx → human-readable label

        self.step.append(new_circ)
        return new_circ

    # Rewrite cross-group CNOTs into RemoteGate, Measurement, If_X, If_Z
    # step[4]
    def rewrite_cross_group_cnots(self):
        
        old_circ = self.step[3]

        # === 1. 构建新电路（仅复制量子比特）===
        new_circ = DQCCircuit(old_circ.qubits)
        new_circ.qubit_group = old_circ.qubit_group
        new_circ.qubit_tele = old_circ.qubit_tele
        group_tele = {gid: self.step[3].qubit_tele[self.step[3].qubit_group.index(gid)] for gid in set(self.step[3].qubit_group)}

        # === 2. 继承所有经典寄存器 ===
        # 复制通信寄存器（如果有）
        comm_cregs = [creg for creg in getattr(old_circ, "cregs", []) if "Tele" in creg.name]
        for comm_creg in comm_cregs:
            new_circ.add_register(ClassicalRegister(len(comm_creg), comm_creg.name))

        # 复制原始寄存器（如果有）
        orig_cregs = [creg for creg in getattr(old_circ, "cregs", []) if "Tele" not in creg.name]
        for orig_creg in orig_cregs:
            new_circ.add_register(ClassicalRegister(len(orig_creg), orig_creg.name))


        idx_counter = 0  # 全局 index 计数器

        request_by_pos = {r["start_pos"]: r["id"] for r in self.request_list}

        # === 4. 改写电路 ===
        for inst_pos, inst_obj in enumerate(old_circ.data):
            instr = inst_obj.operation
            qargs = inst_obj.qubits
            cargs = inst_obj.clbits
            req_id = request_by_pos.get(inst_pos)
            if instr.name == "cx": 
                ctrl, tgt = qargs
                g_ctrl = new_circ.qubit_group[old_circ.get_index(ctrl)]
                g_tgt = new_circ.qubit_group[old_circ.get_index(tgt)]

                if g_ctrl != g_tgt:
                    # ====== 跨组 CNOT，改写 ======
                    ctrl_index = old_circ.get_index(ctrl)
                    tgt_index = old_circ.get_index(tgt)
                    ctrl_comm_index = old_circ.qubit_tele[ctrl_index]
                    tgt_comm_index = old_circ.qubit_tele[tgt_index]

                    ctrl_comm = new_circ.qubits[ctrl_comm_index]
                    tgt_comm = new_circ.qubits[tgt_comm_index]
                    ctrl_q = new_circ.qubits[ctrl_index]
                    tgt_q = new_circ.qubits[tgt_index]

                    # Reset + RemoteGate (purification handled analytically in merge)
                    new_circ.append(Reset(), [ctrl_comm])
                    new_circ.append(Reset(), [tgt_comm])
                    new_circ.append(RemoteGate(idx_counter, g_tgt), [ctrl_comm])
                    new_circ.append(RemoteGate(idx_counter, g_ctrl), [tgt_comm])
                    idx_counter += 1

                    # CNOT: ctrl->ctrl_comm, tgt_comm->tgt
                    new_circ.append(CXGate(), [ctrl_q, ctrl_comm])
                    new_circ.append(CXGate(), [tgt_comm, tgt_q])

                    # H on tgt_comm
                    new_circ.append(HGate(), [tgt_comm])

                    # Measurement + If_Z
                    # Measurement + If_X
                    mz_inst = MZ(index=idx_counter, target=g_tgt)
                    mx_inst = MX(index=idx_counter, target=g_ctrl)
                    new_circ.append(mz_inst, [ctrl_q, ctrl_comm])
                    new_circ.append(mx_inst, [tgt_comm, tgt_q])
                    idx_counter += 1

                    self.Num_RemoteGate += 1
                else:
                    # ====== 同组 CNOT，保持原样 ======
                    new_circ.append(instr, qargs, cargs)
            elif instr.name == "measure":
                # ====== 测量指令替换为 AnsM ======
                qarg = qargs[0]
                carg = cargs[0] if cargs else None

                # 获取经典比特在其寄存器中的索引
                if carg is not None:
                    mea_index = old_circ.find_bit(carg).index
                else:
                    mea_index = -1  # 没有关联经典比特

                # 创建 AnsM 占位指令
                ansm_gate = AnsM(mea_index)

                # 添加到新电路
                new_circ.append(ansm_gate, [qarg])
            elif instr.name == "S_CX":
                path = instr.path
                ctrl, tgt = qargs
                g_ctrl = new_circ.qubit_group[old_circ.get_index(ctrl)]
                g_tgt = new_circ.qubit_group[old_circ.get_index(tgt)]

                ctrl_index = old_circ.get_index(ctrl)
                tgt_index = old_circ.get_index(tgt)
                ctrl_comm_index = old_circ.qubit_tele[ctrl_index]
                tgt_comm_index = old_circ.qubit_tele[tgt_index]

                ctrl_q = new_circ.qubits[ctrl_index]
                tgt_q = new_circ.qubits[tgt_index]
                ctrl_comm = new_circ.qubits[ctrl_comm_index]
                tgt_comm = new_circ.qubits[tgt_comm_index]

                new_circ.append(Reset(), [ctrl_comm_index])

                for i in range(1, len(path) - 1):
                    mid_g = path[i]
                    tgt_g = path[i + 1]

                    mid_comm_index_1 = group_tele[mid_g]
                    mid_comm_index_2 = mid_comm_index_1 + 1
                    tgt_comm_index = group_tele[tgt_g]

                    mid_comm_1 = new_circ.qubits[mid_comm_index_1]
                    mid_comm_2 = new_circ.qubits[mid_comm_index_2]
                    tgt_comm = new_circ.qubits[tgt_comm_index]

                    if i == 1:
                        # Reset
                        new_circ.append(Reset(), [mid_comm_1])
                        new_circ.append(Reset(), [mid_comm_2])
                        new_circ.append(Reset(), [tgt_comm])

                        # RemoteGate
                        new_circ.append(RemoteGate(idx_counter, mid_g), [ctrl_comm])
                        new_circ.append(RemoteGate(idx_counter, g_ctrl), [mid_comm_1])
                        idx_counter += 1

                        new_circ.append(RemoteGate(idx_counter, tgt_g), [mid_comm_2])
                        new_circ.append(RemoteGate(idx_counter, mid_g), [tgt_comm])
                        idx_counter += 1

                        new_circ.append(CXGate(), [mid_comm_1, mid_comm_2])
                        new_circ.append(HGate(), [mid_comm_1])

                        new_circ.append(IF_Z(index=idx_counter, target=mid_g), [ctrl_comm_index])
                        new_circ.append(MS(index=idx_counter, target=tgt_g), [mid_comm_1, mid_comm_2])
                        new_circ.append(IF_X(index=idx_counter, target=g_ctrl), [tgt_comm_index])
                        idx_counter += 1
                    else:
                        new_circ.append(Reset(), [mid_comm_2])
                        new_circ.append(Reset(), [tgt_comm])

                        # RemoteGate
                        new_circ.append(RemoteGate(idx_counter, tgt_g), [mid_comm_2])
                        new_circ.append(RemoteGate(idx_counter, mid_g), [tgt_comm])
                        idx_counter += 1
                        
                        new_circ.append(CXGate(), [mid_comm_1, mid_comm_2])
                        new_circ.append(HGate(), [mid_comm_1])

                        new_circ.append(IF_Z(index=idx_counter, target=mid_g), [ctrl_comm_index])
                        new_circ.append(MS(index=idx_counter, target=tgt_g), [mid_comm_1, mid_comm_2])
                        new_circ.append(IF_X(index=idx_counter, target=g_ctrl), [tgt_comm_index])
                        idx_counter += 1             

                
                # CNOT: ctrl->ctrl_comm, tgt_comm->tgt
                new_circ.append(CXGate(), [ctrl_q, ctrl_comm])
                new_circ.append(CXGate(), [tgt_comm, tgt_q])

                # H on tgt_comm
                new_circ.append(HGate(), [tgt_comm])

                # Measurement + If_Z
                # Measurement + If_X
                mz_inst = MZ(index=idx_counter, target=g_tgt)
                mx_inst = MX(index=idx_counter, target=g_ctrl)
                new_circ.append(mz_inst, [ctrl_q, ctrl_comm])
                new_circ.append(mx_inst, [tgt_comm, tgt_q])
                idx_counter += 1   
            else:
                # 不是 CNOT 和 Mesurement，保持原样
                new_circ.append(instr, qargs, cargs)

        # --- Post-rewrite fixup: fix any cross-group CX/MX/MZ where tgt_comm ---
        # --- was set to the wrong group by the S_CX loop variable leakage.   ---
        fixed_circ = DQCCircuit(new_circ.qubits)
        fixed_circ.qubit_group = new_circ.qubit_group
        fixed_circ.qubit_tele = new_circ.qubit_tele
        for creg in getattr(new_circ, "cregs", []):
            fixed_circ.add_register(ClassicalRegister(len(creg), creg.name))

        needs_fix = False
        for inst_obj in new_circ.data:
            qs = [new_circ.find_bit(q).index for q in inst_obj.qubits]
            if len(qs) == 2 and inst_obj.operation.name in ("cx", "MZ", "MX"):
                g0 = new_circ.qubit_group[qs[0]] if qs[0] < len(new_circ.qubit_group) else -1
                g1 = new_circ.qubit_group[qs[1]] if qs[1] < len(new_circ.qubit_group) else -1
                if g0 != g1 and g0 >= 0 and g1 >= 0:
                    # Fix: replace the wrong comm qubit with the correct one
                    # for the group that the data qubit belongs to.
                    data_q_idx = qs[1]  # tgt_q is always the second arg
                    data_group = g1
                    correct_comm = group_tele.get(data_group)
                    if correct_comm is not None and correct_comm != qs[0]:
                        new_qargs = [new_circ.qubits[correct_comm], new_circ.qubits[data_q_idx]]
                        fixed_circ.append(inst_obj.operation, new_qargs, inst_obj.clbits)
                        needs_fix = True
                        continue
            fixed_circ.append(inst_obj.operation, inst_obj.qubits, inst_obj.clbits)

        if needs_fix:
            self.step.append(fixed_circ)
        else:
            self.step.append(new_circ)
        return self.step[-1]

    def _build_request_list_and_dependencies(self):
        """
        Build request_list and depends_on from step[3] (clean cx circuit, before rewrite).
        Each remote request is a single cross-group cx instruction in step[3].
        """
        self.request_list = []
        self.depends_on = []
        if not self.remote_interactions or len(self.step) < 4:
            return

        circ = self.step[3]
        data = list(circ.data)
        qubit_group = circ.qubit_group
        n_req = len(self.remote_interactions)

        def instr_global_qubits(inst_obj):
            return [circ.find_bit(q).index for q in inst_obj.qubits]

        def is_remote_cx(inst_obj):
            """True for cross-group cx or S_CX (multi-hop) instructions."""
            op = inst_obj.operation
            if op.name == "S_CX":
                return True
            if op.name != "cx":
                return False
            qs = instr_global_qubits(inst_obj)
            return len(qs) == 2 and qubit_group[qs[0]] != qubit_group[qs[1]]

        # Find all remote cx / S_CX positions in step[3]
        remote_cx_positions = [i for i, inst in enumerate(data) if is_remote_cx(inst)]

        if len(remote_cx_positions) != n_req:
            # Fallback: mismatch — skip
            return

        # Build request_list: one entry per remote cx
        for r, pos in enumerate(remote_cx_positions):
            ctrl_qpu, tgt_qpu, path = self.remote_interactions[r]
            qs = instr_global_qubits(data[pos])
            self.request_list.append({
                "id":       r,
                "ctrl_qpu": ctrl_qpu,
                "tgt_qpu":  tgt_qpu,
                "path":     list(path),
                "logical_path": list(path),
                "qubits":   (qs[0], qs[1]),
                "start_pos": pos,
                "end_pos":   pos,
            })

        # Build depends_on with transitive closure.
        # Direct dep: request j depends on i (i < j) if they share a data qubit.
        # Transitive closure: j inherits the deps of each of its direct deps,
        # so chained dependencies (j shares qubit with k, k shares qubit with i,
        # but j and i don't share) are captured correctly.
        for j in range(n_req):
            direct = set()
            q_j = set(self.request_list[j]["qubits"])
            for i in range(j):
                q_i = set(self.request_list[i]["qubits"])
                if q_i & q_j:
                    direct.add(i)
            closure = set(direct)
            for k in direct:
                closure |= self.depends_on[k]
            self.depends_on.append(closure)

    def rearrange_for_parallel(self):
        """
        Compute parallel_batches metadata (groups of conflict-free remote requests
        for joint scheduling) WITHOUT modifying the circuit. The downstream pipeline
        (rewrite_cross_group_cnots, physic_split, transpile, merge_trans_circuits)
        operates on the original gate order in step[3].

        parallel_batches uses the original request IDs from self.request_list, so
        no remapping or rebuild is needed.
        """
        if not self.request_list or len(self.step) < 4:
            return

        n_req = len(self.request_list)

        def req_data_qubits(r):
            q0, q1 = self.request_list[r]["qubits"]
            qs = set()
            if q0 is not None: qs.add(q0)
            if q1 is not None: qs.add(q1)
            return qs

        # --- Parallel batch detection ---
        # Greedy iteration: each unassigned r seeds a new batch; later requests join
        # the current batch if they don't share data qubits with anything already in
        # the batch and don't transitively depend on any batch member.
        parallel_batches = []
        assigned = set()

        for r in range(n_req):
            if r in assigned:
                continue
            batch = [r]
            assigned.add(r)
            batch_qubits = req_data_qubits(r)
            for r2 in range(r + 1, n_req):
                if r2 in assigned:
                    continue
                r2_qs = req_data_qubits(r2)
                r2_depends_on_batch = any(
                    dep in batch for dep in self.depends_on[r2]
                )
                if batch_qubits.isdisjoint(r2_qs) and not r2_depends_on_batch:
                    batch.append(r2)
                    assigned.add(r2)
                    batch_qubits |= r2_qs
            parallel_batches.append(batch)

        self.parallel_batches = parallel_batches

    def resource_manager_schedule(self, algorithm="FCFS"):
        """
        For each parallel batch (in order from rearrange_for_parallel):
          1. Route each request (preferred path → BFS fallback)
          2. Build QPU demand map and detect comm qubit conflicts
          3. Reroute conflicted requests if a less-congested path exists
          4. Sort: free requests first, conflicted after, purification requests last
          5. Assign comm qubits using earliest-free-time table (tiebreak: lowest index)
          6. Compute start_time = max(free_at of all assigned comm qubits on path)
          7. Update comm qubit free times (duration filled later by heralded simulation)
          8. Reorder step[3] within each batch to match final sorted order

        Stores result as self.schedule:
          { req_id: { "path": [...], "start_time": float, "comm": {qpu_id: comm_idx} } }

        num_comm_qubits per QPU group = 1 + Entanglement_swapping[group]
        (1 base comm qubit, +1 if this group does entanglement swapping)
        """
        if not self.request_list:
            self.schedule = {}
            return

        num_groups = len(self.partition)

        # Use pre-computed comm qubit capacity (backend.num_qubits - partition_size)
        # Falls back to 1 + swap_flag if not yet computed (e.g. called standalone)
        if self.num_comm_qubits and len(self.num_comm_qubits) == num_groups:
            num_comm_qubits = self.num_comm_qubits
        else:
            num_comm_qubits = [
                1 + (1 if self.Entanglement_swapping[g] == 1 else 0)
                for g in range(num_groups)
            ]

        # qpu_comm_free[group_id][comm_idx] = time when that comm qubit is free
        qpu_comm_free = {g: [0.0] * num_comm_qubits[g] for g in range(num_groups)}

        parallel_batches = getattr(self, "parallel_batches", None) or [
            [r] for r in range(len(self.request_list))
        ]

        schedule = {}
        scheduled_batches = []
        reroute_count = 0
        # Allow rerouting conflicted requests (purification does not affect this)
        allow_reroute = getattr(self, "allow_reroute_override", None)
        allow_reroute = True if allow_reroute is None else bool(allow_reroute)

        def get_route(src, dst):
            preferred = self._find_swap_path(src, dst)
            if preferred:
                return preferred
            return self._find_shortest_path(self.qpugroup.map, src, dst) or [src, dst]

        def expected_request_duration(path):
            """
            E[duration] using concurrent hop model:
              E[link_k] = t_k / p_k  (geometric mean)
            Single-hop: E[T] = t/p
            Multi-hop:  fold left with max + swap_time, mirroring the sampled model.
              pair_ready = max(E[link0], E[link1]) + swap0
              pair_ready = max(pair_ready, E[link2]) + swap1  ...
            """
            link_exp = []
            for a, b in zip(path[:-1], path[1:]):
                p, t = self.qpugroup.get_link_success_params(a, b)
                if p and p > 0:
                    link_exp.append(t / p)
            if not link_exp:
                return 0.0
            if len(link_exp) == 1:
                return link_exp[0]
            swap_exp = []
            for intermediate_qpu_id in path[1:-1]:
                qpu_obj = self.qpugroup.qpus[intermediate_qpu_id]
                bp = BackendParams.from_backend(qpu_obj.backend)
                swap_exp.append(bp.gate_time_2q)
            pair_ready = max(link_exp[0], link_exp[1]) + swap_exp[0]
            for i in range(1, len(swap_exp)):
                pair_ready = max(pair_ready, link_exp[i + 1]) + swap_exp[i]
            return pair_ready

        def assign_and_schedule(req_id, path):
            """Assign earliest-free comm qubit(s) on each QPU in path, return start_time.

            Endpoint QPUs (first/last in path) need 1 comm qubit.
            Intermediate QPUs (swap nodes) need 2 comm qubits simultaneously —
            one for the left link and one for the right link of the swap.
            assigned[qpu] = [cidx, ...]  — always a list, length 1 or 2.
            """
            assigned = {}
            start_time = 0.0
            for idx, qpu in enumerate(path):
                if qpu >= num_groups:
                    continue
                cap = num_comm_qubits[qpu]
                is_intermediate = 0 < idx < len(path) - 1
                slots_needed = 2 if is_intermediate else 1

                sorted_slots = sorted(range(cap), key=lambda i: (qpu_comm_free[qpu][i], i))
                chosen = sorted_slots[:slots_needed]  # take earliest-free slot(s)

                assigned[qpu] = chosen
                ready = max(qpu_comm_free[qpu][s] for s in chosen)
                start_time = max(start_time, ready)

                # Reserve slots immediately so concurrent same-batch requests
                # see them as occupied and pick different free slots.
                for s in chosen:
                    qpu_comm_free[qpu][s] = math.nextafter(qpu_comm_free[qpu][s], math.inf)
            return start_time, assigned

        def sort_conflicted_requests(req_ids, paths):
            if algorithm == "FCFS":
                return list(req_ids)
            if algorithm == "SPF":
                return sorted(req_ids, key=lambda r: len(paths[r]))
            if algorithm == "SJF":
                return sorted(req_ids, key=lambda r: expected_request_duration(paths[r]))
            return list(req_ids)

        def resolve_conflicted_routes(req_ids, paths, initial_committed_links=None):
            ordered = sort_conflicted_requests(req_ids, paths)
            if not allow_reroute:
                return ordered

            conflict_committed_links = set(initial_committed_links or [])

            def _commit_path(path):
                for a, b in zip(path[:-1], path[1:]):
                    conflict_committed_links.add((min(a, b), max(a, b)))

            start_idx = 0
            if ordered and not initial_committed_links:
                first = ordered[0]
                _commit_path(paths[first])
                start_idx = 1

            for req_id in ordered[start_idx:]:
                req = self.request_list[req_id]
                src = req.get("ctrl_qpu")
                dst = req.get("tgt_qpu")
                if src is None or dst is None:
                    _commit_path(paths[req_id])
                    continue
                orig_path = paths[req_id]
                if not orig_path or len(orig_path) <= 2:
                    _commit_path(orig_path)
                    continue
                alt_map = {
                    node: [
                        (nb, d) for nb, d in neighbors
                        if (min(node, nb), max(node, nb)) not in conflict_committed_links
                    ]
                    for node, neighbors in self.qpugroup.map.items()
                }
                # Use the configured routing_algo (Throughput / Probability / BFS)
                # so rerouting respects the same optimisation criterion as initial routing.
                original_map = self.qpugroup.map
                self.qpugroup.map = alt_map
                try:
                    alt = self._find_swap_path(src, dst)
                except ValueError:
                    alt = None
                finally:
                    self.qpugroup.map = original_map
                if alt and alt != orig_path:
                    paths[req_id] = alt
                    nonlocal reroute_count
                    reroute_count += 1
                else:
                    if alt:
                        pass
                    else:
                        pass
                _commit_path(paths[req_id])
            return ordered

        circ = self.step[3]
        data = list(circ.data)

        for batch_idx, batch in enumerate(parallel_batches):
            # 1. Route all requests in this batch
            paths = {}
            for req_id in batch:
                req = self.request_list[req_id]
                src = req.get("ctrl_qpu")
                dst = req.get("tgt_qpu")
                if src is None or dst is None:
                    paths[req_id] = []
                    continue
                existing_path = req.get("path")
                paths[req_id] = existing_path if existing_path else get_route(src, dst)

            # 2. Per-request comm-qubit demand per QPU (purification-aware)
            user_purif = getattr(self, "purification", {}) or {}

            def _req_demand(path):
                """Return {qpu: qubits_needed} for a request path."""
                demand = {}
                for i, qpu in enumerate(path):
                    base = 2 if (0 < i < len(path) - 1) else 1
                    purif_boost = 0
                    for a, b in zip(path[:-1], path[1:]):
                        link_key = tuple(sorted((a, b)))
                        if link_key in user_purif and qpu in (a, b):
                            rounds = int(user_purif[link_key].get("rounds", 0))
                            if rounds > 0:
                                purif_boost = max(purif_boost, 2 ** rounds)
                    demand[qpu] = max(base, purif_boost)
                return demand

            request_demand = {r: _req_demand(paths[r]) for r in batch}

            # Total demand per QPU across the batch
            total_demand = {qpu: 0 for qpu in range(num_groups)}
            for r in batch:
                for qpu, n in request_demand[r].items():
                    if qpu < num_groups:
                        total_demand[qpu] += n

            preview_plans = {
                req_id: self._build_purification_plan_for_path(paths[req_id], num_comm_qubits)
                for req_id in batch
            }
            purify_reqs = [r for r in batch if preview_plans[r]["needs_purification"]]
            non_purify_batch = [r for r in batch if r not in purify_reqs]

            # 3. Classify: free vs conflicted (demand within this batch vs capacity)
            conflicted_qpus = {
                qpu for qpu, d in total_demand.items() if d > num_comm_qubits[qpu]
            }
            free_reqs = [r for r in non_purify_batch if not any(q in conflicted_qpus for q in paths[r])]
            conflict_reqs = [r for r in non_purify_batch if any(q in conflicted_qpus for q in paths[r])]
            
            # Print conflict detection summary
            if conflicted_qpus:
                conflict_summary = ", ".join(
                    f"QPU{qpu} (demand={total_demand[qpu]}, capacity={num_comm_qubits[qpu]})"
                    for qpu in sorted(conflicted_qpus)
                )
            if free_reqs:
                if not conflicted_qpus:
                    pass
            if conflict_reqs:
                if not conflicted_qpus and not free_reqs:
                    pass
            if purify_reqs:
                if not conflicted_qpus and not free_reqs and not conflict_reqs:
                    pass

            # 4. Within conflicted requests, admit as many highest-priority requests
            # as the current comm-qubit capacity allows. Only overflow requests are
            # candidates for reroute / wait.
            sorted_conflict = sort_conflicted_requests(conflict_reqs, paths)
            remaining_now = {qpu: num_comm_qubits[qpu] for qpu in range(num_groups)}
            for req_id in free_reqs:
                for qpu, n in request_demand[req_id].items():
                    if qpu < num_groups:
                        remaining_now[qpu] -= n

            admitted_conflict = []
            overflow_conflict = []
            for req_id in sorted_conflict:
                demand = request_demand[req_id]
                if all(qpu >= num_groups or remaining_now[qpu] >= n for qpu, n in demand.items()):
                    admitted_conflict.append(req_id)
                    for qpu, n in demand.items():
                        if qpu < num_groups:
                            remaining_now[qpu] -= n
                else:
                    overflow_conflict.append(req_id)

            committed_links = set()
            for req_id in admitted_conflict:
                path = paths[req_id]
                for a, b in zip(path[:-1], path[1:]):
                    committed_links.add((min(a, b), max(a, b)))

            rerouted_overflow = resolve_conflicted_routes(
                overflow_conflict, paths, initial_committed_links=committed_links
            )
            purify_sorted = sort_conflicted_requests(purify_reqs, paths)
            if allow_reroute and purify_sorted:
                purify_conflict = [r for r in purify_sorted if any(q in conflicted_qpus for q in paths[r])]
                resolve_conflicted_routes(purify_conflict, paths)

            if admitted_conflict:
                pass
            if overflow_conflict:
                pass

            final_order = free_reqs + admitted_conflict + rerouted_overflow + purify_sorted
            scheduled_batches.append(list(final_order))

            # Show which requests can start immediately at this batch instant and
            # which must wait because the prioritized earlier requests consume the
            # currently available comm-qubit slots on their paths.
            available_now = {qpu: num_comm_qubits[qpu] for qpu in range(num_groups)}
            ready_now = []
            must_wait = []
            for req_id in final_order:
                path = paths[req_id]
                if all(qpu >= num_groups or available_now[qpu] > 0 for qpu in path):
                    ready_now.append(req_id)
                    for qpu in path:
                        if qpu < num_groups:
                            available_now[qpu] -= 1
                else:
                    must_wait.append(req_id)
            if ready_now:
                pass
            if must_wait:
                pass
            must_wait_set = set(must_wait)

            # 6 & 7. Assign comm qubits, compute start times, update free table.
            # Reserve slots for immediate-start requests before any must-wait
            # requests so parallel-ready work does not get displaced.
            assignment_order = ready_now + must_wait
            for req_id in assignment_order:
                path = paths[req_id]
                if not path:
                    schedule[req_id] = {"path": [], "start_time": 0.0, "comm": {}, "scheduler_wait": False}
                    continue
                exp_dur = expected_request_duration(path)
                start_time, assigned = assign_and_schedule(req_id, path)
                for qpu, cidx_list in assigned.items():
                    # Use expected duration so subsequent requests see this slot as
                    # occupied for the right amount of time, enforcing proper queuing.
                    for cidx in cidx_list:
                        qpu_comm_free[qpu][cidx] = start_time + exp_dur if exp_dur > 0 else math.nextafter(start_time, math.inf)
                schedule[req_id] = {
                    "path":           path,
                    "start_time":     start_time,
                    "comm":           assigned,
                    "scheduler_wait": req_id in must_wait_set,
                    "purify":         preview_plans.get(req_id, {}).get("needs_purification", False),
                    "expected_dur":   exp_dur,
                }

            # 8. Reorder step[3] within this batch to match final_order.
            # NOTE: this step is currently a no-op. Prior versions rewrote the
            # `data` list to reflect SPF/SJF ordering, but the in-place position
            # swap interacted badly with subsequent batches' request_list lookups
            # (and with the S_CX rewrite on paths crossing reordered boundaries),
            # producing mismatched qubit arities in physic_split. The scheduler
            # already records the correct order in `schedule[req_id]["start_time"]`
            # and `scheduled_batches`, which drive every analytical metric
            # (wall clock, per-request times, reroutes). The original circuit
            # order is fine for circuit correctness because each remote request
            # is an independent block in step[3] before rewrite.

        # Rebuild step[3] with any within-batch reordering applied
        new_circ = DQCCircuit(circ.qubits)
        new_circ.qubit_group = circ.qubit_group
        new_circ.qubit_tele = circ.qubit_tele
        for creg in getattr(circ, "cregs", []):
            new_circ.add_register(ClassicalRegister(len(creg), creg.name))
        for inst_obj in data:
            new_circ.append(inst_obj.operation, inst_obj.qubits, inst_obj.clbits)
        self.step[3] = new_circ

        # Refresh request_list positions after reorder (authoritative rebuild)
        self._build_request_list_and_dependencies()

        self.schedule = schedule
        self.scheduled_batches = scheduled_batches
        self.reroute_count = reroute_count

    def _link_coherence_time_s(self, qpu_a: int, qpu_b: int):
        """Min configured memory coherence time across the two endpoint QPUs."""
        vals = []
        for qid in (qpu_a, qpu_b):
            q = self.qpugroup.get_qpu(qid) if self.qpugroup else None
            if q is None:
                continue
            t = getattr(q, "memory_coherence_time_s", None)
            if t is not None:
                vals.append(float(t))
        return min(vals) if vals else None

    def _print_routes_with_sampled_herald_metrics(
        self, scheduled_batches, schedule, sampled_request_metrics
    ):
        """Print scheduled routes once herald samples exist (coherence line + herald line per hop)."""
        for batch_idx, batch in enumerate(scheduled_batches):
            for req_id in batch:
                path = list(schedule.get(req_id, {}).get("path", []) or [])
                if not path or len(path) < 2:
                    continue
                samples = sampled_request_metrics.get(req_id, {}).get("link_samples", [])
                n_edges = len(path) - 1
                dur_by_hi = [None] * n_edges
                sj = 0
                for hi, (xa, xb) in enumerate(zip(path[:-1], path[1:])):
                    px, tx = self.qpugroup.get_link_success_params(xa, xb)
                    if px is None or tx is None or px <= 0:
                        continue
                    if sj >= len(samples):
                        break
                    lk, _na, dur_ij, _rp, _rd = samples[sj]
                    sj += 1
                    exp_lk = (min(xa, xb), max(xa, xb))
                    if lk != exp_lk:
                        continue
                    dur_by_hi[hi] = float(dur_ij)
                coh_edge = [
                    self._link_coherence_time_s(path[j], path[j + 1])
                    for j in range(n_edges)
                ]
                link_decohere = [False] * n_edges
                # Whole request: mark link k if any *other* link's herald duration
                # exceeds link k's coherence (memory on k cannot outlast that step).
                for k in range(n_edges):
                    ck = coh_edge[k]
                    if ck is None:
                        continue
                    for i in range(n_edges):
                        if i == k:
                            continue
                        di = dur_by_hi[i]
                        if di is not None and di > ck:
                            link_decohere[k] = True
                            break
                # Consecutive hops A=k, B=k+1: (dur_B - dur_A) > coherence(link A).
                for k in range(n_edges - 1):
                    ck = coh_edge[k]
                    if ck is None:
                        continue
                    da = dur_by_hi[k]
                    db = dur_by_hi[k + 1]
                    if da is None or db is None:
                        continue
                    if (db - da) > ck:
                        link_decohere[k] = True

                si = 0
                for hi, (a, b) in enumerate(zip(path[:-1], path[1:])):
                    coh = self._link_coherence_time_s(a, b)
                    coh_str = f"{coh:.6f}s" if coh is not None else "N/A"
                    p, t = self.qpugroup.get_link_success_params(a, b)
                    if p is None or t is None or p <= 0:
                        continue
                    if si >= len(samples):
                        break
                    link_key, n_att, dur, raw_pairs, rounds = samples[si]
                    si += 1
                    expected_lk = (min(a, b), max(a, b))
                    if link_key != expected_lk:
                        continue
                    req_p = raw_pairs / n_att if n_att else 0.0
                    extra = ""
                    if rounds > 0:
                        extra = f", purification_rounds={rounds}, raw_pairs={raw_pairs}"
                    flag = "  [decohere]" if link_decohere[hi] else ""

    def simulate_heralded_entanglement(self):
        """
        For each request in self.schedule (in start_time order):
          - Sample number of attempts per link: n ~ Geometric(p)
          - Duration per link: n * attempt_time
          - finish_time = start_time + sum(durations along path)
          - Update comm qubit free times for accurate per-QPU wait tracking
          - Collect per-link and per-QPU metrics

        Stores result as self.network_metrics.
        """
        schedule = getattr(self, "schedule", {})
        if not schedule:
            self.network_metrics = {}
            return

        num_groups = len(self.partition)

        # Use pre-computed comm qubit capacity (backend.num_qubits - partition_size)
        # Falls back to 1 + swap_flag if not yet computed (e.g. called standalone)
        if self.num_comm_qubits and len(self.num_comm_qubits) == num_groups:
            num_comm_qubits = self.num_comm_qubits
        else:
            num_comm_qubits = [
                1 + (1 if self.Entanglement_swapping[g] == 1 else 0)
                for g in range(num_groups)
            ]
        qpu_comm_free = {g: [0.0] * num_comm_qubits[g] for g in range(num_groups)}
        qpu_comm_owner = {g: [None] * num_comm_qubits[g] for g in range(num_groups)}
        purification_plan = getattr(self, "purification_plan", {})
        request_snapshot = getattr(self, "metrics_request_snapshot", None) or []

        # link_metrics[link_key] = {"total_attempts": int, "successes": int, "time": float,
        #                            "requests": [(req_id, n_att, dur), ...]}
        link_metrics = {}
        qpu_busy = [0.0] * num_groups
        qpu_wait = [0.0] * num_groups
        request_wait_time = {}
        request_batch_wait_time = {}
        qpu_wait_details = {g: [] for g in range(num_groups)}
        finish_times = []

        sampled_request_metrics = {}
        for req in sorted(request_snapshot, key=lambda item: item.get("id", 0)):
            req_id = req.get("id")
            if req_id is None:
                continue
            path = list(schedule.get(req_id, {}).get("path", []) or [])
            if not path:
                path = list(req.get("path", []) or [])
            req_plan = purification_plan.get(req_id, {})
            req_link_plan = {
                tuple(sorted(entry["link"])): entry for entry in req_plan.get("links", [])
            }

            link_samples = []
            link_durations = []   # sampled herald duration per hop, in path order
            swap_times = []       # gate_time_2q at each intermediate QPU, in path order

            for a, b in zip(path[:-1], path[1:]):
                link_key = (min(a, b), max(a, b))
                p, t = self.qpugroup.get_link_success_params(a, b)
                if p is None or t is None or p <= 0:
                    continue
                link_plan = req_link_plan.get(link_key, {})
                raw_pairs = 1
                if link_plan.get("eligible", False) and link_plan.get("rounds", 0) > 0:
                    raw_pairs = int(link_plan.get("raw_pairs", 1))
                n_att = 0
                dur = 0.0
                for _ in range(raw_pairs):
                    one_att = int(np.random.geometric(p))
                    n_att += one_att
                    dur += one_att * t
                link_durations.append(dur)
                link_samples.append(
                    (link_key, n_att, dur, raw_pairs, int(link_plan.get("rounds", 0)))
                )

            # Collect per-intermediate-node BSM times in path order.
            # Full BSM = CX (gate_time_2q) + H (gate_time_1q) + measurement (t_readout).
            for intermediate_qpu_id in path[1:-1]:
                qpu_obj = self.qpugroup.qpus[intermediate_qpu_id]
                bp = BackendParams.from_backend(qpu_obj.backend)
                t_1q = getattr(qpu_obj.backend, "t_1q", 0.0)
                t_readout = getattr(qpu_obj.backend, "t_readout", 0.0)
                swap_times.append(bp.gate_time_2q + t_1q + t_readout)

            # Concurrent hop model: all links are attempted simultaneously.
            # A swap at node k can only happen once both the left pair (hops 0..k)
            # and the right link (hop k+1) are ready. Fold left:
            #   pair_ready = max(link[0], link[1]) + swap[0]
            #   pair_ready = max(pair_ready, link[2]) + swap[1]  ...
            # Single-hop: no swaps, total = link duration only.
            if not link_durations:
                total_dur = 0.0
                swap_overhead = 0.0
            elif len(link_durations) == 1:
                total_dur = link_durations[0]
                swap_overhead = 0.0
            else:
                pair_ready = max(link_durations[0], link_durations[1]) + swap_times[0]
                for i in range(1, len(swap_times)):
                    pair_ready = max(pair_ready, link_durations[i + 1]) + swap_times[i]
                total_dur = pair_ready
                swap_overhead = sum(swap_times)

            sampled_request_metrics[req_id] = {
                "total_dur": total_dur,
                "link_samples": link_samples,
                "swap_overhead": swap_overhead,
            }

        scheduled_batches = getattr(self, "scheduled_batches", None) or []
        self._print_routes_with_sampled_herald_metrics(
            scheduled_batches, schedule, sampled_request_metrics
        )

        # Process in scheduled order, but derive actual start_time from sampled
        # comm-qubit free times rather than the scheduler's expected-duration estimate.
        for req_id in sorted(schedule, key=lambda r: schedule[r]["start_time"]):
            info = schedule[req_id]
            assigned = info["comm"]
            sampled = sampled_request_metrics.get(req_id, {"total_dur": 0.0, "link_samples": []})
            total_dur = sampled["total_dur"]
            for link_key, n_att, dur, raw_pairs, rounds in sampled["link_samples"]:
                if link_key not in link_metrics:
                    link_metrics[link_key] = {
                        "total_attempts": 0, "successes": 0,
                        "time": 0.0, "requests": []
                    }
                link_metrics[link_key]["total_attempts"] += n_att
                link_metrics[link_key]["successes"]      += raw_pairs
                link_metrics[link_key]["time"]           += dur
                link_metrics[link_key]["requests"].append(
                    (req_id, n_att, dur, raw_pairs, rounds)
                )

            # Actual start = when all assigned comm qubits are free (sampled chain).
            actual_start = max(
                (qpu_comm_free[qpu][cidx]
                 for qpu, cidx_list in assigned.items() if qpu < num_groups
                 for cidx in cidx_list),
                default=0.0,
            )
            finish_time = actual_start + total_dur
            finish_times.append(finish_time)

            # Update comm qubit free times and per-QPU metrics.
            # qpu_busy is counted once per QPU (not once per slot).
            req_wait = 0.0
            for qpu, cidx_list in assigned.items():
                if qpu >= num_groups:
                    continue
                for cidx in cidx_list:
                    wait = max(0.0, qpu_comm_free[qpu][cidx] - actual_start)
                    req_wait = max(req_wait, wait)
                    qpu_wait[qpu] += wait
                    if wait > 0.0:
                        prev_req = qpu_comm_owner[qpu][cidx]
                        qpu_wait_details[qpu].append((req_id, prev_req, wait))
                    qpu_comm_free[qpu][cidx] = finish_time
                    qpu_comm_owner[qpu][cidx] = req_id
                qpu_busy[qpu] += total_dur  # once per QPU
            request_wait_time[req_id] = req_wait

        # Compute per-request wait considering only requests within the same batch.
        for batch in getattr(self, "scheduled_batches", []):
            batch_qpu_comm_free = {g: [0.0] * num_comm_qubits[g] for g in range(num_groups)}
            for req_id in batch:
                info = schedule.get(req_id, {})
                assigned = info.get("comm", {})
                sampled = sampled_request_metrics.get(req_id, {"total_dur": 0.0})
                total_dur = sampled.get("total_dur", 0.0)
                batch_wait = 0.0
                for qpu, cidx_list in assigned.items():
                    if qpu >= num_groups:
                        continue
                    for cidx in cidx_list:
                        batch_wait = max(batch_wait, batch_qpu_comm_free[qpu][cidx])
                finish_time = batch_wait + total_dur
                for qpu, cidx_list in assigned.items():
                    if qpu >= num_groups:
                        continue
                    for cidx in cidx_list:
                        batch_qpu_comm_free[qpu][cidx] = finish_time
                request_batch_wait_time[req_id] = batch_wait

        # Final-simulator wait metric: only requests that waited before being served
        # in their current batch contribute to per-QPU wait, attributed to the QPUs
        # involved in that request's path.
        qpu_wait = [0.0] * num_groups
        qpu_wait_details = {g: [] for g in range(num_groups)}
        for req_id, batch_wait in request_batch_wait_time.items():
            if batch_wait <= 0.0:
                continue
            path = schedule.get(req_id, {}).get("path", [])
            for qpu in path:
                if qpu >= num_groups:
                    continue
                qpu_wait[qpu] += batch_wait
                qpu_wait_details[qpu].append((req_id, None, batch_wait))

        wall_clock = max(finish_times) if finish_times else 0.0
        for a, nbrs in sorted(self.qpugroup.map.items()):
            for b, _ in nbrs:
                if a < b:
                    p_cfg, t_cfg = self.qpugroup.get_link_success_params(a, b)
                    p_str = f"{p_cfg:.4f}" if p_cfg is not None else "N/A"
                    t_str = f"{t_cfg:.6f}s" if t_cfg is not None else "N/A"
        for lk in sorted(link_metrics):
            a, b = lk
            m = link_metrics[lk]
            total_att = m["total_attempts"]
            suc       = m["successes"]
            t_s       = m["time"]
            actual_p  = suc / total_att if total_att else 0.0
            for (rid, n_att, dur, raw_pairs, rounds) in m["requests"]:
                req_p = raw_pairs / n_att if n_att else 0.0
                extra = ""
                if rounds > 0:
                    extra = f", purification_rounds={rounds}, raw_pairs={raw_pairs}"
        for g in range(num_groups):
            for waited_req, prev_req, wait in qpu_wait_details[g]:
                if prev_req is not None:
                    pass
                else:
                    pass
        request_expected_time = {}
        for req_id in sorted(schedule):
            total_dur = sampled_request_metrics.get(req_id, {}).get("total_dur", 0.0)
            path = schedule.get(req_id, {}).get("path", [])
            qpus = ", ".join(f"QPU{q}" for q in path)
            expected_dur = 0.0
            for a, b in zip(path[:-1], path[1:]):
                p, t = self.qpugroup.get_link_success_params(a, b)
                if p and p > 0:
                    expected_dur += t / p
            request_expected_time[req_id] = expected_dur
            swap_overhead = sampled_request_metrics.get(req_id, {}).get("swap_overhead", 0.0)
            swap_str = f"  swap_overhead={swap_overhead:.6f}s" if swap_overhead > 0 else ""
        pmetrics = getattr(self, "purification_metrics", {}) or {}
        if pmetrics:
            user_cfg = pmetrics.get("user_config", {})
            if user_cfg:
                pass
            e2e_map = pmetrics.get("request_e2e_fidelity", {})
            if e2e_map:
                for rid, f_e2e in sorted(e2e_map.items()):
                    pass

        request_paths = {
            req_id: (schedule.get(req_id, {}).get("path", []) or [])
            for req_id in schedule
        }
        request_path_risk = {
            req_id: self._path_risk(p) for req_id, p in request_paths.items()
        }
        risk_vals = list(request_path_risk.values())

        # Security metrics (paper Eq 8, 10)
        theta = float(getattr(self.qpugroup, "high_risk_node_threshold", 0.5))
        include_endpoints = bool(getattr(self.qpugroup, "risk_include_endpoints", False))

        def _path_uses_high_risk(path):
            if not path:
                return False
            nodes = path if include_endpoints else path[1:-1]
            for v in nodes:
                qpu = self.qpugroup.get_qpu(v) if self.qpugroup else None
                r = float(getattr(qpu, "security_risk", 0.0)) if qpu is not None else 0.0
                if r >= theta:
                    return True
            return False

        request_path_length = {rid: max(0, len(p) - 1) for rid, p in request_paths.items()}
        request_uses_high_risk = {rid: _path_uses_high_risk(p) for rid, p in request_paths.items()}
        request_safety = {rid: 1.0 - request_path_risk[rid] for rid in schedule}

        n_paths = len(schedule)
        n_untrusted = sum(1 for v in request_uses_high_risk.values() if v)
        untrusted_path_fraction = (n_untrusted / n_paths) if n_paths else 0.0
        mean_path_risk = (sum(risk_vals) / len(risk_vals)) if risk_vals else 0.0
        safety_score = 1.0 - 0.5 * (mean_path_risk + untrusted_path_fraction)
        mean_path_length = (sum(request_path_length.values()) / n_paths) if n_paths else 0.0

        self.network_metrics = {
            "link_metrics":   link_metrics,
            "qpu_busy_time":  qpu_busy,
            "qpu_wait_time":  qpu_wait,
            "qpu_wait_details": qpu_wait_details,
            "request_wait_time": request_wait_time,
            "request_batch_wait_time": request_batch_wait_time,
            "request_expected_time": request_expected_time,
            "request_time_used": {req_id: sampled_request_metrics.get(req_id, {}).get("total_dur", 0.0) for req_id in schedule},
            "request_swap_overhead": {req_id: sampled_request_metrics.get(req_id, {}).get("swap_overhead", 0.0) for req_id in schedule},
            "wall_clock":     wall_clock,
            "reroute_count":  getattr(self, "reroute_count", 0),
            "purification":   getattr(self, "purification_metrics", {}),
            "request_path_risk": request_path_risk,
            "request_safety": request_safety,
            "request_path_length": request_path_length,
            "request_uses_high_risk_repeater": request_uses_high_risk,
            "mean_path_risk": mean_path_risk,
            "max_path_risk": max(risk_vals) if risk_vals else 0.0,
            "mean_path_length": mean_path_length,
            "untrusted_path_fraction": untrusted_path_fraction,
            "safety_score": safety_score,
            "high_risk_node_threshold": theta,
            "risk_include_endpoints": include_endpoints,
        }

    # Physically split the circuit into sub-circuits based on partition
    def physic_split(self):
        if self.group_comm_slots and len(self.group_comm_slots) == len(self.partition):
            config = [len(group) + self.group_comm_slots[gid] for gid, group in enumerate(self.partition)]
        else:
            config = [len(group) + 1 for group in self.partition]
            for gid, flag in enumerate(self.Entanglement_swapping):
                if flag == 1:
                    config[gid] += 1

        temp_circuit = self.step[4]
        sub_circuits = []

        if all(isinstance(x, int) for x in config):
            # 按数量顺序分割, 例如 [2,2,3]
            start = 0
            for s in config:
                sub = DQCCircuit(s)
                sub.qubit_group = temp_circuit.qubit_group[start:start+s]

                # 给子电路加 classical bits
                num_clbits = (len(config) - 1) * 2
                if num_clbits > 0:
                    creg = ClassicalRegister(num_clbits, "c")
                    sub.add_register(creg)

                # Copy operations that act on these qubits
                for instr in temp_circuit.data:
                    qubit_indices = [temp_circuit.get_index(q) for q in instr.qubits]
                    # Keep only qubits inside this sub-circuit
                    indices_in_sub = [i for i, qi in enumerate(qubit_indices) if start <= qi < start+s]
                    if indices_in_sub:
                        new_qargs = [sub.qubits[qi - start] for i, qi in enumerate(qubit_indices) if i in indices_in_sub]
                        new_cargs = []
                        # For barriers, create a fresh barrier on the subset of qubits so that
                        # the barrier's arity matches the number of qubits in this subcircuit.
                        if instr.operation.name == "barrier":
                            sub.barrier(*new_qargs)
                        else:
                            sub.append(instr.operation, new_qargs, new_cargs)

                sub_circuits.append(sub)
                start += s

        elif all(isinstance(x, (list, tuple)) for x in config):
            # Split by explicit indices, e.g. [[1,3],[0,2,4]]
            for indices in config:
                sub = DQCCircuit(len(indices))
                sub.qubit_type = [self.qubit_type[i] for i in indices]

                for instr in self.data:
                    qubit_indices = [self.get_index(q) for q in instr.qubits]
                    # Keep only qubits whose indices are in `indices`
                    indices_in_sub = [i for i, qi in enumerate(qubit_indices) if qi in indices]
                    if indices_in_sub:
                        new_qargs = [sub.qubits[indices.index(qi)] for i, qi in enumerate(qubit_indices) if i in indices_in_sub]
                        new_cargs = []
                        if instr.operation.name == "barrier":
                            sub.barrier(*new_qargs)
                        else:
                            sub.append(instr.operation, new_qargs, new_cargs)

                sub_circuits.append(sub)

        else:
            raise ValueError("Config must be a list of ints or a list of lists of ints")

        self.sub_circuit = sub_circuits
        return sub_circuits

    # Protect custom instructions with barriers
    def protect_custom_instructions(self, subcirc):
        """
        为 RemoteGate、Measurement、If_X、If_Z、AnsM 添加 barrier 保护。
        """
        new_circ = DQCCircuit(*subcirc.qregs, *subcirc.cregs)
        
        for inst_obj in subcirc.data:
            inst = inst_obj.operation
            qargs = inst_obj.qubits
            cargs = inst_obj.clbits

            if isinstance(inst, (MX, MZ, AnsM, IF_Z, IF_X, MS, S_CX)):               # 在同样的 qubits 上加 barrier
                new_circ.barrier(*qargs)
                new_circ.append(inst, qargs, cargs)
                new_circ.barrier(*qargs)
                # print(f"[Info] Protecting instruction: {inst.name} on qubits {[q for q in qargs]}")
            elif isinstance(inst, (RemoteGate)):
                new_circ.barrier()
                new_circ.append(inst, qargs, cargs)
                new_circ.barrier()
            else:
                new_circ.append(inst, qargs, cargs)

        return new_circ

    # Restore custom instructions by removing barriers
    def restore_custom_instructions(self, subcirc):
        """
        还原 RemoteGate、Measurement、If_X、If_Z、AnsM 的 barrier 保护。
        """
        new_circ = DQCCircuit(*subcirc.qregs, *subcirc.cregs)
        
        for inst_obj in subcirc.data:
            inst = inst_obj.operation
            qargs = inst_obj.qubits
            cargs = inst_obj.clbits

            if inst.name != "barrier":
                new_circ.append(inst, qargs, cargs)
            # if isinstance(inst, (RemoteGate, MX, MZ, AnsM)):
            #     print(f"[Info] Restoring instruction: {inst.name} on qubits {[q for q in qargs]}")
        return new_circ

    def transpile_subcircuits(self, qpus, layout_out=None):
        """
        Transpile each sub-circuit using the corresponding QPU backend target,
        and optional layout mapping per subcircuit.

        :param qpus: list of QPU instances (each having .backend and .target)
        :param layout_out: list of layouts; each layout is a list of physical qubit indices
                        corresponding to the sub-circuit's logical qubits
                        e.g. [[0,1,2,3], [5,6,7,8]]
        """
        # ---- 按 qpu_id 排序 ----
        self.qpus = sorted(qpus, key=lambda x: x.qpu_id)

        if not self.sub_circuit:
            raise ValueError("No sub-circuits found. Please populate self.sub_circuit first.")
        if len(qpus) != len(self.sub_circuit):
            raise ValueError(
                f"Number of QPUs ({len(qpus)}) does not match number of sub-circuits ({len(self.sub_circuit)})."
            )

        # ---- 检查 layout_out ----
        if layout_out is not None:
            if len(layout_out) != len(self.sub_circuit):
                raise ValueError("layout_out length must match number of sub-circuits.")

        # ---- 检查子电路规模是否符合后端限制 ----
        for idx, (sub_circ, qpu) in enumerate(zip(self.sub_circuit, qpus)):
            backend = qpu.backend
            size = sub_circ.num_qubits
            qpu_id = getattr(qpu, "qpu_id", idx)
            backend_name = getattr(backend, "name", "unknown")

            if size > backend.num_qubits:
                raise ValueError(
                    f"QPU {qpu_id}: requested size={size} exceeds "
                    f"the maximum number of qubits ({backend.num_qubits}) "
                    f"supported by backend {backend_name}."
                )

        self.sub_circuit_trans = []

        # ---- 对每个子电路执行 transpile ----
        for idx, (sub, qpu) in enumerate(zip(self.sub_circuit, qpus)):
            try:
                # Step 1: 保护自定义指令
                sub = self.protect_custom_instructions(sub)

                layout = None
                if layout_out is not None and idx < len(layout_out):
                    layout = layout_out[idx]

                # Step 3: 调用 transpile
                if layout is not None:
                    sub = transpile(
                        sub,
                        backend=qpu.backend,
                        initial_layout=layout,
                        optimization_level=3,
                    )
                else:
                    sub = transpile(
                        sub,
                        backend=qpu.backend,
                        optimization_level=3,
                    )

                # Step 4: 还原自定义指令
                sub = self.restore_custom_instructions(sub)

                self.sub_circuit_trans.append(sub)
                bname = getattr(qpu.backend, "name", str(qpu.backend))
                bname = bname() if callable(bname) else bname
                print(f"[Info] Sub-circuit {idx} transpiled successfully on {bname}.")

            except Exception as e:
                bname = getattr(qpu.backend, "name", str(qpu.backend))
                bname = bname() if callable(bname) else bname
                print(f"[Error] Failed to transpile sub-circuit {idx} on {bname}: {e}")
                self.sub_circuit_trans.append(None)

        return self.sub_circuit_trans
    
    # Merge the transpiled sub-circuits into a complete circuit
    def merge_trans_circuits(self, comm_noise = False):
        """
        Merge `self.sub_circuit_trans` into one full circuit.

        Supports cross-subcircuit pairing (R, M+IF_X/IF_Z) using a stack to jump
        between subcircuits. Pairing is bidirectional: encountering either side
        (M or IF_X/IF_Z) can trigger the match.
        """
        # === 构建 qubit / cbit 映射 ===
        qubits_map = {}
        merged_qubits_map = {}
        cbits_map = {}
        global_q_index = 0
        global_c_index = 0
        for i, sub in enumerate(self.sub_circuit_trans):
            if sub is None:
                continue
            local_indices = {sub.find_bit(q).index for instr in sub.data for q in instr.qubits}
            for local_index in sorted(local_indices):
                qubits_map[(i, local_index)] = global_q_index
                merged_qubits_map[global_q_index] = (i, local_index)
                global_q_index += 1

        # Record index for noise model
        self.merged_qubits_map = merged_qubits_map
        self.merged_qubits_map_reverse = qubits_map

        num_sub = len(self.sub_circuit_trans)
        for i in range(num_sub):
            for j in range(num_sub):
                if i != j:
                    cbits_map[(i, j)] = global_c_index
                    global_c_index += 1

        # === 初始化变量 ===
        # 创建新电路（只初始化量子比特）
        new_circ = DQCCircuit(len(qubits_map))
        new_circ.qubit_group = self.step[1].qubit_group

        # === 1. 添加通信寄存器 ===
        tele_creg = ClassicalRegister(len(cbits_map), "Tele")
        new_circ.add_register(tele_creg)

        # === 2. 保留原始经典寄存器 ===
        if hasattr(self, "clbits") and self.clbits:
            if hasattr(self, "cregs") and self.cregs:
                orig_name = self.cregs[0].name
            else:
                orig_name = "c"
            orig_creg = ClassicalRegister(len(self.clbits), orig_name)
            new_circ.add_register(orig_creg)

        indices = [0] * len(self.sub_circuit_trans)
        paired_op = {}      # {index: (sub_index, instr)}
        paired_op2 = {}
        paired_done = set() # already paired
        call_stack = []
        now = 0
        step = 0

        # Throughput metrics for heralded entanglement (attempts until success per link)
        throughput_metrics = {"link_metrics": {}, "total_attempts": 0, "total_time": 0.0,
                              "idle_wait_time": 0.0}

        # wait_decoherence: collect each remote gate's two DATA-qubit merged indices,
        # keyed by the MZ/MX pairing index (idx_counter). The endpoint CX of every
        # remote gate (single- or multi-hop) lands as an MZ on the control group
        # (operands [ctrl_q, ctrl_comm]) and an MX on the target group (operands
        # [tgt_comm, tgt_q]); the partner-counterpart conditional gates target the
        # OTHER endpoint's data qubit. So the four data-qubit slots touched by a pair
        # are first_qs / target_qs at the [0]/[1] positions selected by gate name.
        # Filled only when self.wait_decoherence is on. idx -> set(merged data qubits).
        wait_pair_data_qubits = {}

        # Helper for conditional ops
        def apply_conditional_gate(gate_list, qubit, clbit):
            with new_circ.if_test((clbit, 1)):
                for instr in gate_list:
                    new_circ.append(instr.operation, [qubit], instr.clbits)

        while True:
            done = all(sub is None or indices[i] >= len(sub.data) for i, sub in enumerate(self.sub_circuit_trans))
            if done:
                break

            sub = self.sub_circuit_trans[now]
            if sub is None or indices[now] >= len(sub.data):
                if call_stack:
                    now = call_stack.pop()
                    continue
                else:
                    now = (now + 1) % len(self.sub_circuit_trans)
                    continue

            instr = sub.data[indices[now]]
            inst = instr.operation
            op_name = inst.name.upper()
            local_indices = [sub.find_bit(q).index for q in instr.qubits]
            global_qs = [qubits_map[(now, idx)] for idx in local_indices]
            target = getattr(inst, "target", None)
            idx = getattr(inst, "index", None)
            mea = getattr(inst,"mea", None)

            # print(f"\n[STEP] now={now}, target={target},idx={idx}, inst={inst.name}, indices={indices}")

            # === R 门配对生成 Bell 态 ===
            if op_name == "R" and idx not in paired_done:
                target_sub = self.sub_circuit_trans[target] if target is not None else None

                if idx not in paired_op:
                    paired_op[idx] = now
                    if target_sub:
                        call_stack.append(now)
                        now = target
                        continue
                else:
                    first_now = paired_op.pop(idx)
                    first_sub = self.sub_circuit_trans[first_now]
                    first_instr = first_sub.data[indices[first_now]]
                    first_qs = [qubits_map[(first_now, first_sub.find_bit(q).index)] for q in first_instr.qubits]
                    target_qs = global_qs

                    # Heralded entanglement: simulate number of attempts until success
                    success_prob, attempt_time = self.qpugroup.get_link_success_params(first_now, now)
                    n_attempts = 1
                    if success_prob is not None and attempt_time is not None and success_prob > 0:
                        n_attempts = int(np.random.geometric(success_prob))
                        link_key = (min(first_now, now), max(first_now, now))
                        if link_key not in throughput_metrics["link_metrics"]:
                            throughput_metrics["link_metrics"][link_key] = {"attempts": 0, "successes": 0, "time": 0.0}
                        throughput_metrics["link_metrics"][link_key]["attempts"] += n_attempts
                        throughput_metrics["link_metrics"][link_key]["successes"] += 1
                        throughput_metrics["link_metrics"][link_key]["time"] += n_attempts * attempt_time
                        throughput_metrics["total_attempts"] += n_attempts
                        throughput_metrics["total_time"] += n_attempts * attempt_time
                        # Idle wait the data qubits sit through while THIS pair is
                        # distributed. r rounds of purification need 2^r raw pairs, each
                        # distributed, so the wait scales ~2^r (serial, an upper bound).
                        # Used only by the idle_decoherence channel applied to the data
                        # qubits at measurement time; does not change total_time.
                        _up = getattr(self, "purification", {}) or {}
                        _r = int(_up.get(tuple(sorted((first_now, now))), {}).get("rounds", 0))
                        throughput_metrics["idle_wait_time"] += (2 ** _r) * n_attempts * attempt_time

                    # Initialize the Bell state (one successful entanglement)
                    new_circ.initialize([1/np.sqrt(2),0,0,1/np.sqrt(2)], first_qs + target_qs)
                    if comm_noise:
                        # Unified noise model: inject depolarizing(1 - F_link) where
                        # F_link is the analytical per-link fidelity (AD + purification).
                        user_purif = getattr(self, "purification", {}) or {}
                        link_key = tuple(sorted((first_now, now)))
                        rounds = 0
                        if link_key in user_purif:
                            rounds = int(user_purif[link_key].get("rounds", 0))
                        F_link = self._link_fidelity_with_purification(first_now, now, rounds)
                        noise_instr = QPUManager.build_depolarizing_noise_for_fidelity(F_link)

                        if isinstance(target_qs, list):
                            noise_qargs = [new_circ.qubits[i] for i in target_qs]
                        else:
                            noise_qargs = [new_circ.qubits[target_qs]]
                        if noise_instr is not None:
                            new_circ.append(noise_instr, noise_qargs)

                    if self.idle_decoherence:
                        # Optional: add T2 dephasing to every qubit for the batch's longest
                        # entanglement setup time. Off by default.
                        if self._idle_charge is None:
                            _rtu = (getattr(self, "network_metrics", {}) or {}).get("request_time_used", {})
                            self._idle_charge = {}
                            for _b in (getattr(self, "parallel_batches", []) or []):
                                if _b:
                                    self._idle_charge[min(_b)] = max(_rtu.get(r, 0.0) for r in _b)
                        t_gate = self._idle_charge.get(idx, 0.0)
                        if t_gate and t_gate > 0:
                            for gi in range(len(new_circ.qubits)):
                                sub_i = self.merged_qubits_map.get(gi, (None,))[0]
                                if sub_i is None:
                                    continue
                                T1, T2 = self._qpu_coherence_times(sub_i)
                                if not (T1 and T2 and T1 > 0 and T2 > 0):
                                    continue
                                T2_eff = min(T2, 2.0 * T1)
                                new_circ.append(
                                    thermal_relaxation_error(T1, T2_eff, t_gate).to_instruction(),
                                    [new_circ.qubits[gi]])

                    paired_done.add(idx)
                    indices[now] += 1
                    indices[first_now] += 1
                    now = call_stack.pop() if call_stack else now
                    continue

            # === M / IF_X / IF_Z 双向配对 ===
            elif op_name in ("MX", "MZ") and idx not in paired_done:
                if idx not in paired_op:
                    paired_op[idx] = now
                    if target_sub:
                        call_stack.append(now)
                        now = target
                    continue

                # 已经配对
                first_now = paired_op.pop(idx)
                first_sub = self.sub_circuit_trans[first_now]
                first_instr = first_sub.data[indices[first_now]]

                first_qs = [qubits_map[(first_now, first_sub.find_bit(q).index)] 
                            for q in first_instr.qubits]
                target_qs = global_qs

                # classical bit 对应
                cbit_idx1 = cbits_map[(now, first_now)]
                cbit_idx2 = cbits_map[(first_now, now)]
                clbit_obj1 = new_circ.clbits[cbit_idx1]
                clbit_obj2 = new_circ.clbits[cbit_idx2]


                if first_instr.name == "MX":
                    new_circ.measure(first_qs[0], clbit_obj1)
                    apply_conditional_gate(self.qpus[now].compile_z_gate(), target_qs[0], clbit_obj1)
                    new_circ.measure(target_qs[1], clbit_obj2)
                    apply_conditional_gate(self.qpus[first_now].compile_x_gate(), first_qs[1], clbit_obj2)

                else:
                    new_circ.measure(first_qs[1], clbit_obj1)
                    apply_conditional_gate(self.qpus[now].compile_x_gate(), target_qs[1], clbit_obj1)
                    new_circ.measure(target_qs[0], clbit_obj2)
                    apply_conditional_gate(self.qpus[first_now].compile_z_gate(), first_qs[0], clbit_obj2)
                    

                # wait_decoherence bookkeeping: record the two DATA qubits of this
                # remote gate (MZ data = operand[0] on ctrl group; MX data = operand[1]
                # on tgt group). first_qs / target_qs hold whichever side was seen first.
                if getattr(self, "wait_decoherence", False):
                    if first_instr.name == "MX":
                        data_qs = {first_qs[1], target_qs[0]}
                    else:  # first is MZ
                        data_qs = {first_qs[0], target_qs[1]}
                    wait_pair_data_qubits.setdefault(idx, set()).update(data_qs)

                # 更新状态
                paired_done.add(idx)
                indices[now] += 1
                indices[first_now] += 1
                now = call_stack.pop() if call_stack else now
                continue
            elif op_name == "ANS_M":
                # global_qs[0] 是量子比特索引，mea 是存储的经典比特索引
                q_idx = global_qs[0]
                c_idx = mea  # mea 已经是全局经典比特索引

                # 在 new_circ 中找到对应的 Clbit 对象
                clbit_obj = None
                temp_idx = c_idx
                for creg in new_circ.cregs:
                    if temp_idx < len(creg):
                        clbit_obj = creg[temp_idx]
                        break
                    else:
                        temp_idx -= len(creg)

                if clbit_obj is None:
                    raise ValueError(f"Cannot find the classical bit index {c_idx} corresponding to a Clbit")

                # 添加测量操作
                new_circ.measure(new_circ.qubits[q_idx], clbit_obj)
                indices[now] += 1
                continue
            elif op_name in ("IF_X", "IF_Z", "MS") and idx not in paired_done:
                target_sub = self.sub_circuit_trans[target] if target is not None else None
                if idx not in paired_op:
                    paired_op[idx] = now
                    if target_sub:
                        call_stack.append(now)
                        now = target
                    continue
                if idx not in paired_op2:
                    paired_op2[idx] = now
                    if target_sub:
                        call_stack.append(now)
                        now = target
                    continue

                # 已经配对
                first_now = paired_op.pop(idx)
                first_sub = self.sub_circuit_trans[first_now]
                first_instr = first_sub.data[indices[first_now]]

                second_now = paired_op2.pop(idx)
                second_sub = self.sub_circuit_trans[second_now]
                second_instr = second_sub.data[indices[second_now]]

                first_qs = [qubits_map[(first_now, first_sub.find_bit(q).index)] 
                            for q in first_instr.qubits]
                
                second_qs = [qubits_map[(second_now, second_sub.find_bit(q).index)] 
                            for q in second_instr.qubits]
                
                # instr global_qs
                third_qs = [qubits_map[(now, self.sub_circuit_trans[now].find_bit(q).index)] 
                            for q in instr.qubits]

                # classical bit 对应
                cbit_idx1 = cbits_map[(first_now, second_now)]  
                cbit_idx2 = cbits_map[(second_now, first_now)]  
                cbit_idx3 = cbits_map[(second_now, now)]        
                cbit_idx4 = cbits_map[(now, second_now)]                
                cbit_idx5 = cbits_map[(now, first_now)]         
                cbit_idx6 = cbits_map[(first_now, now)]         

                clbit_obj1 = new_circ.clbits[cbit_idx1]
                clbit_obj2 = new_circ.clbits[cbit_idx2] 
                clbit_obj3 = new_circ.clbits[cbit_idx3]
                clbit_obj4 = new_circ.clbits[cbit_idx4] 
                clbit_obj5 = new_circ.clbits[cbit_idx5]
                clbit_obj6 = new_circ.clbits[cbit_idx6] 
                # print("Entanglement Swapping", first_instr.name,second_instr.name, instr.name)
                if first_instr.name == "IF_Z":
                    new_circ.measure(second_qs[0], clbit_obj2)
                    apply_conditional_gate(self.qpus[first_now].compile_z_gate(), first_qs, clbit_obj2)
                    new_circ.measure(second_qs[1], clbit_obj3)
                    apply_conditional_gate(self.qpus[now].compile_x_gate(), third_qs, clbit_obj3)
                elif first_instr.name == "MS":
                    new_circ.measure(first_qs[0], clbit_obj6)
                    apply_conditional_gate(self.qpus[now].compile_z_gate(), third_qs, clbit_obj6)
                    new_circ.measure(first_qs[1], clbit_obj1)
                    apply_conditional_gate(self.qpus[second_now].compile_x_gate(), second_qs, clbit_obj1)
                elif first_instr.name == "IF_X":
                    new_circ.measure(third_qs[0], clbit_obj4)
                    apply_conditional_gate(self.qpus[second_now].compile_z_gate(), second_qs, clbit_obj4)
                    new_circ.measure(third_qs[1], clbit_obj5)
                    apply_conditional_gate(self.qpus[first_now].compile_x_gate(), first_qs, clbit_obj5)
                # 更新状态
                paired_done.add(idx)
                indices[now] += 1
                indices[first_now] += 1
                indices[second_now] += 1 
                now = call_stack.pop() if call_stack else now
                now = call_stack.pop() if call_stack else now
                continue
            # === 普通门 ===
            else:
                new_circ.append(inst, global_qs, [])
                indices[now] += 1
                continue

            step += 1
            if step > 50000:
                raise RuntimeError(f"[Error] Possible deadlock. indices={indices}, now={now}, stack={call_stack}")

        # Optional wait_decoherence: charge each request's queue wait as
        # T2 dephasing on its two data qubits. Off by default.
        if getattr(self, "wait_decoherence", False) and wait_pair_data_qubits:
            nm = getattr(self, "network_metrics", {}) or {}
            batch_wait = nm.get("request_batch_wait_time", {}) or {}
            # Map MZ/MX pairing index -> req_id. The endpoint MZ/MX pairs are emitted
            # in request order in step[4] (one pair per remote request), so the r-th
            # distinct pairing index seen there is request r.
            idx_to_req = {}
            if len(self.step) > 4:
                seen = []
                for inst_obj in self.step[4].data:
                    iname = inst_obj.operation.name
                    if iname in ("MZ", "MX"):
                        pidx = getattr(inst_obj.operation, "index", None)
                        if pidx is not None and pidx not in seen:
                            seen.append(pidx)
                idx_to_req = {pidx: r for r, pidx in enumerate(seen)}

            for pidx, data_qs in wait_pair_data_qubits.items():
                req_id = idx_to_req.get(pidx)
                if req_id is None:
                    continue
                wait = batch_wait.get(req_id, 0.0)
                if not wait or wait <= 0:
                    continue
                # Verify the data qubits land on the request's own endpoints before
                # charging them — a wrong target would silently corrupt the result.
                req = self.request_list[req_id] if req_id < len(self.request_list) else None
                ctrl_grp = req.get("ctrl_qpu") if req else None
                tgt_grp = req.get("tgt_qpu") if req else None
                for gi in data_qs:
                    sub_i = self.merged_qubits_map.get(gi, (None,))[0]
                    if sub_i is None:
                        continue
                    if req is not None and sub_i not in (ctrl_grp, tgt_grp):
                        # Mapping mismatch: skip rather than dephase the wrong qubit.
                        continue
                    T1, T2 = self._qpu_coherence_times(sub_i)
                    if not (T1 and T2 and T1 > 0 and T2 > 0):
                        continue
                    T2_eff = min(T2, 2.0 * T1)
                    new_circ.append(
                        thermal_relaxation_error(T1, T2_eff, wait).to_instruction(),
                        [new_circ.qubits[gi]])

        self.result_circuit = new_circ
        self.result_circuit.qubit_group = self.step[4].qubit_group
        self.throughput_metrics = throughput_metrics
        new_circ.throughput_metrics = throughput_metrics
        return new_circ

    def decompose_and_get_data(self, instr: CircuitInstruction, basis_gates=None):
        """
        Recursively decompose one instruction into a basic gate set.

        Args:
            instr: CircuitInstruction to decompose
            basis_gates: allowed basis gate names (default: {'cx'} + 1q gates)

        Returns:
            A list of CircuitInstruction objects that keep the original qubit refs
            (can be appended directly into `circuit.data`).
        """
        if basis_gates is None:
            # 定义基础门集合
            basis_gates = {'cx', 'u3', 'u2', 'u1', 'id', 'x', 'y', 'z', 
                        'h', 's', 't', 'rx', 'ry', 'rz', 'sx', 'p'}
        
        # If it's already a basis op (or has no definition), return as-is
        if (not hasattr(instr.operation, "definition") or 
            instr.operation.name in basis_gates):
            return [instr]
        
        # Decompose via a temporary circuit
        n_qubits = len(instr.qubits)
        qc_temp = QuantumCircuit(n_qubits)
        qc_temp.append(instr.operation, list(range(n_qubits)))
        
        # Map temp qubits back to the original qubit objects
        temp_to_orig = {qc_temp.qubits[i]: instr.qubits[i] for i in range(n_qubits)}
        
        # Decompose once. Guard against cases where decompose() either fails or
        # returns the same instruction (which would cause infinite recursion).
        try:
            decomposed = qc_temp.decompose()
        except RecursionError:
            # If Qiskit's high-level synthesis recurses internally, just
            # treat this instruction as atomic for our purposes.
            return [instr]
        
        # If decomposition produced exactly one instruction with the same
        # operation type/name, assume further decomposition will not help.
        if (
            len(decomposed.data) == 1
            and decomposed.data[0].operation.name == instr.operation.name
        ):
            return [instr]
        
        result = []
        for sub_instr in decomposed.data:
            # Map temp qubits back to the original qubits
            mapped_qubits = [temp_to_orig[q] for q in sub_instr.qubits]
            
            # Create a new instruction
            new_instr = CircuitInstruction(
                sub_instr.operation, 
                mapped_qubits, 
                sub_instr.clbits
            )
            
            # Recurse if still not in basis
            result.extend(self.decompose_and_get_data(new_instr, basis_gates))
        
        return result

    def reduce_noise_model(self, subset_qubits, noise_model=None, coupling_map=None):
        """
        Crop a NoiseModel to only keep noise on `subset_qubits`.

        Keeps basis_gates, description, and optionally a cropped coupling map.

        Args:
            subset_qubits: list[int] qubits to keep
            noise_model: NoiseModel (optional). Defaults to self.backend_noise_model
            coupling_map: list[tuple] (optional). If provided, crop it too.

        Returns:
            NoiseModel
        """

        if noise_model is None:
            if getattr(self, "backend_noise_model", None) is None:
                raise ValueError("No noise model provided or set in self.backend_noise_model.")
            noise_model = self.backend_noise_model

        # Prefer built-in reduce() if available
        if hasattr(noise_model, "reduce"):
            return noise_model.reduce(subset_qubits)

        sub_model = NoiseModel()

        # --- 1) Keep quantum gate noise ---
        for instr_name, qerrors in noise_model._local_quantum_errors.items():
            for qubits, error in qerrors.items():
                if all(q in subset_qubits for q in qubits):
                    sub_model.add_quantum_error(error, instr_name, qubits)

        # --- 2) Keep readout noise (measure) ---
        for qubit, error in noise_model._local_readout_errors.items():
            if qubit[0] in subset_qubits:  # qubit is a tuple, e.g. (0,)
                sub_model.add_readout_error(error, [qubit[0]])

        # --- 3) Keep basis_gates and description ---
        if hasattr(noise_model, "basis_gates"):
            sub_model._basis_gates = list(noise_model.basis_gates)
        if hasattr(noise_model, "description"):
            sub_model._description = noise_model.description

        # --- 4) Optionally crop coupling map ---
        target_coupling_map = coupling_map if coupling_map is not None else getattr(self, "coupling_map", None)
        if target_coupling_map is not None:
            sub_model._coupling_map = [edge for edge in target_coupling_map if all(q in subset_qubits for q in edge)]
        else:
            sub_model._coupling_map = None

        return sub_model

    # Get a combined noise model for the distributed circuit based on QPU backends
    def get_noise_model(self):
        """
        Build a combined noise model by cropping each QPU noise model and merging
        them into one model on globally indexed qubits.
        """
        # === Step 1: Parameter checks ===
        qpus = self.qpugroup.qpus
        qpus = sorted(qpus, key=lambda x: x.qpu_id)
        if not self.sub_circuit:
            raise ValueError("No sub-circuits found.")
        if len(qpus) != len(self.sub_circuit):
            raise ValueError(
                f"Number of QPUs ({len(qpus)}) does not match sub-circuits ({len(self.sub_circuit)})."
            )

        # === Step 2: Collect local qubits for each sub-circuit ===
        sub_qubit_maps = []
        for idx, _ in enumerate(self.sub_circuit):
            local_qubits = [
                local_index
                for global_q, (sub_index, local_index) in self.merged_qubits_map.items()
                if sub_index == idx
            ]
            sub_qubit_maps.append(local_qubits)

        # === Step 3: Global qubit mapping: (sub_idx, local_qubit) -> global_qubit ===
        qubit_mapping = self.merged_qubits_map_reverse

        # === Step 4: Init global NoiseModel ===
        combined_noise_model = NoiseModel()
        combined_basis_gates = set()

        # === Step 5: Crop and merge per-QPU noise ===
        for idx, qpu in enumerate(qpus):
            backend_noise = NoiseModel.from_backend(qpu.backend)
            local_qubits = sub_qubit_maps[idx]

            # --- 5a. Crop local noise ---
            reduced_noise = self.reduce_noise_model(
                subset_qubits=local_qubits,
                noise_model=backend_noise,
                coupling_map=getattr(qpu.backend, "coupling_map", None),
            )

            # --- 5b. Merge quantum gate noise ---
            for instr_name, qerrors in reduced_noise._local_quantum_errors.items():
                for qubits_tuple, error in qerrors.items():
                    # Unpack qubit indices
                    qubit_indices = [q if isinstance(q, int) else q[0] for q in qubits_tuple]
                    global_qubits = tuple(qubit_mapping[(idx, q)] for q in qubit_indices)
                    combined_noise_model.add_quantum_error(error, instr_name, global_qubits)

            # --- 5c. Merge readout noise ---
            for qubit, error in reduced_noise._local_readout_errors.items():
                q_local = qubit[0] if isinstance(qubit, tuple) else qubit
                if (idx, q_local) not in qubit_mapping:
                    print(f"Warning: qubit mapping missing for {(idx, q_local)}")
                    continue
                global_qubit = qubit_mapping[(idx, q_local)]
                combined_noise_model.add_readout_error(error, [global_qubit])

            # --- 5d. Merge basis_gates ---
            if hasattr(reduced_noise, "basis_gates"):
                combined_basis_gates.update(reduced_noise.basis_gates)

        # === Step 6: Finalize basis_gates/description ===
        combined_noise_model._basis_gates = list(combined_basis_gates)
        combined_noise_model._description = "Combined noise model from multiple QPUs"

        return combined_noise_model
    
def detect_swap_pattern(circuit):
    swap_count = 0
    instructions = list(circuit.data)
    
    i = 0
    while i < len(instructions) - 2:
        inst1 = instructions[i]
        inst2 = instructions[i + 1]
        inst3 = instructions[i + 2]
        
        # 检查是否都是 CX 门
        if (inst1.operation.name == 'cx' and 
            inst2.operation.name == 'cx' and 
            inst3.operation.name == 'cx'):
            
            # 获取量子比特索引
            q1_0 = circuit.find_bit(inst1.qubits[0]).index
            q1_1 = circuit.find_bit(inst1.qubits[1]).index
            
            q2_0 = circuit.find_bit(inst2.qubits[0]).index
            q2_1 = circuit.find_bit(inst2.qubits[1]).index
            
            q3_0 = circuit.find_bit(inst3.qubits[0]).index
            q3_1 = circuit.find_bit(inst3.qubits[1]).index
            
            # 检查 SWAP 模式: CX(a,b), CX(b,a), CX(a,b)
            if (q1_0 == q3_0 and q1_1 == q3_1 and  # 第1和第3个相同
                q2_0 == q1_1 and q2_1 == q1_0):     # 第2个是反向的
                swap_count += 1
                print(f"  检测到 SWAP 模式在位置 {i}: q{q1_0} ↔ q{q1_1}")
                i += 3  # 跳过这3个门
                continue
        
        i += 1
    
    print("\n" + "=" * 70)
    print("检测 CNOT 模式")
    print("=" * 70)
    print(f"检测到的 SWAP 数量: {swap_count}")
    
    return swap_count   
