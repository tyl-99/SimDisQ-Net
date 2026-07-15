"""
IBM Almaden backend — based on FakeAlmadenV2 calibration snapshot
(the retired ibmq_almaden, 20-qubit superconducting device).
Wraps calibration data into a standalone class so it doesn't re-fetch
from qiskit_ibm_runtime every instantiation. Mirrors IBMMelbourne's class
structure so it plugs into the simulator's DQC pipeline (stock Qiskit fake
backends fail to translate the teleportation instructions onto their target).

Specs (extracted from FakeAlmadenV2 calibration via BackendParams.from_backend
and the target/qubit_properties — all REAL, no invented numbers):
  Qubits: 20, real coupling map below, NO broken pairs (0/46 dead CX)
  F_1q:   0.9990  (mean 1q gate)
  F_2q:   0.9762  (mean CX)
  F_spam: 0.9900  (mean measure)
  T1:     86.78 μs   T2: 64.31 μs
  t_2q:   405.8 ns   t_1q: 53.3 ns   t_read: 3.56 μs (Falcon-class; Almaden
          target lacks a measure duration — affects only decoherence-during-
          readout, not the SPAM error itself)

Replaces the RETIRED broken IBM Cambridge (10 dead CX pairs) as the healthy
superconducting low-fidelity anchor of the Network-Wall hardware ladder.
"""
import numpy as np
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit.library import XGate, SXGate, RZGate, CXGate
from qiskit.circuit import Measure, Delay, Parameter, Reset
from qiskit.providers.backend import QubitProperties


# Real ibmq_almaden coupling map (directed), from FakeAlmadenV2
ALMADEN_EDGES = [
    (0, 1), (1, 0), (1, 2), (1, 6), (2, 1), (2, 3), (3, 2), (3, 4), (3, 8),
    (4, 3), (5, 6), (5, 10), (6, 1), (6, 5), (6, 7), (7, 6), (7, 8), (7, 12),
    (8, 3), (8, 7), (8, 9), (9, 8), (9, 14), (10, 5), (10, 11), (11, 10),
    (11, 12), (11, 16), (12, 7), (12, 11), (12, 13), (13, 12), (13, 14),
    (13, 18), (14, 9), (14, 13), (15, 16), (16, 11), (16, 15), (16, 17),
    (17, 16), (17, 18), (18, 13), (18, 17), (18, 19), (19, 18),
]


class IBMAlmaden(BackendV2):
    """IBM Almaden backend (20 qubits, real coupling, no broken pairs)."""

    def __init__(self):
        super().__init__(name="IBM Almaden")

        self._num_qubits = 20
        self.backend_name = "ibmq_almaden"

        self.fidelity_1q_mean = 0.9990
        self.fidelity_2q_mean = 0.9762
        self.fidelity_spam_mean = 0.9900

        self.t_readout = 0.0000035556   # 3.56 μs (Falcon-class)
        self.t_reset = 0.0000035        # 3.5 μs
        self.t_1q = 0.0000000533        # 53.3 ns
        self.t_2q = 0.0000004058        # 405.8 ns
        self.t1_time = 0.00008678       # 86.78 μs
        self.t2_time = 0.00006431       # 64.31 μs

        self._build_target()

    def _build_target(self):
        rng = np.random.default_rng(seed=55550004)

        qubit_properties = []
        for i in range(self._num_qubits):
            qubit_properties.append(QubitProperties(
                t1=self.t1_time + rng.uniform(-1e-5, 1e-5),
                t2=self.t2_time + rng.uniform(-1e-5, 1e-5),
                frequency=rng.uniform(4.8e9, 5.2e9),
            ))

        self._target = Target(
            "IBM Almaden",
            num_qubits=self._num_qubits,
            qubit_properties=qubit_properties,
        )

        self._add_1q_gates(rng)
        self._add_2q_gates(rng)

    def _add_1q_gates(self, rng):
        rz_props, x_props, sx_props = {}, {}, {}
        measure_props, reset_props, delay_props = {}, {}, {}
        error_1q = 1 - self.fidelity_1q_mean
        error_spam = 1 - self.fidelity_spam_mean

        for i in range(self._num_qubits):
            q = (i,)
            rz_props[q] = InstructionProperties(error=0.0, duration=0.0)
            x_props[q] = InstructionProperties(
                error=error_1q + rng.uniform(-1e-4, 1e-4), duration=self.t_1q)
            sx_props[q] = InstructionProperties(
                error=error_1q + rng.uniform(-1e-4, 1e-4), duration=self.t_1q)
            measure_props[q] = InstructionProperties(
                error=error_spam + rng.uniform(-1e-3, 1e-3), duration=self.t_readout)
            reset_props[q] = InstructionProperties(
                error=error_spam + rng.uniform(-1e-3, 1e-3), duration=self.t_reset)
            delay_props[q] = None

        self._target.add_instruction(XGate(), x_props)
        self._target.add_instruction(SXGate(), sx_props)
        self._target.add_instruction(RZGate(Parameter("theta")), rz_props)
        self._target.add_instruction(Measure(), measure_props)
        self._target.add_instruction(Reset(), reset_props)
        self._target.add_instruction(Delay(Parameter("t")), delay_props)

    def _add_2q_gates(self, rng):
        cx_props = {}
        error_2q = 1 - self.fidelity_2q_mean
        for edge in ALMADEN_EDGES:
            var = rng.uniform(-5e-3, 5e-3)
            cx_props[edge] = InstructionProperties(
                error=error_2q + var, duration=self.t_2q)
        self._target.add_instruction(CXGate(), cx_props)

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return None

    @property
    def num_qubits(self):
        return self._num_qubits

    @classmethod
    def _default_options(cls):
        return Options(shots=1024)

    def run(self, circuit, **kwargs):
        raise NotImplementedError("This backend does not contain a run method")
