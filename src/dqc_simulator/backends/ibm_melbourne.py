"""
IBM Melbourne backend — based on FakeMelbourneV2 calibration snapshot
(the retired ibmq_16_melbourne, 15-qubit superconducting device).
Wraps calibration data into a standalone class so it doesn't re-fetch
from qiskit_ibm_runtime every instantiation.

Specs (extracted from FakeMelbourneV2 calibration):
  Qubits: 15, sparse coupling (real coupling map below)
  F_1q:   0.998598 (mean 1q gate)
  F_2q:   0.96831  (mean CX)
  F_spam: 0.94186  (mean measure)
  T1:     54.85 μs   T2: 54.81 μs
  t_2q:   0.929 μs   t_1q: 89 ns   t_read: 3.56 μs

Fills the mid-fidelity gap (between IBM Cambridge ~0.806 and the trapped-ion
cluster ~0.99) in the Network-Wall hardware ladder, and has enough qubits to
host a 6-per-QPU balanced bipartition (unlike 5-qubit Vigo).
"""
import numpy as np
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit.library import XGate, SXGate, RZGate, CXGate
from qiskit.circuit import Measure, Delay, Parameter, Reset
from qiskit.providers.backend import QubitProperties


# Real ibmq_16_melbourne coupling map (directed), from FakeMelbourneV2
MELBOURNE_EDGES = [
    (0, 1), (0, 14), (1, 0), (1, 2), (1, 13), (2, 1), (2, 3), (2, 12),
    (3, 2), (3, 4), (3, 11), (4, 3), (4, 5), (4, 10), (5, 4), (5, 6),
    (5, 9), (6, 5), (6, 8), (7, 8), (8, 6), (8, 7), (8, 9), (9, 5),
    (9, 8), (9, 10), (10, 4), (10, 9), (10, 11), (11, 3), (11, 10),
    (11, 12), (12, 2), (12, 11), (12, 13), (13, 1), (13, 12), (13, 14),
    (14, 0), (14, 13),
]


class IBMMelbourne(BackendV2):
    """IBM Melbourne backend (15 qubits, sparse coupling)."""

    def __init__(self):
        super().__init__(name="IBM Melbourne")

        self._num_qubits = 15
        self.backend_name = "ibmq_16_melbourne"

        self.fidelity_1q_mean = 0.998598
        self.fidelity_2q_mean = 0.96831
        self.fidelity_spam_mean = 0.94186

        self.t_readout = 0.0000035556   # 3.56 μs
        self.t_reset = 0.0000035        # 3.5 μs
        self.t_1q = 0.0000000889        # 89 ns
        self.t_2q = 0.0000009287        # 929 ns
        self.t1_time = 0.00005485       # 54.85 μs
        self.t2_time = 0.00005481       # 54.81 μs

        self._build_target()

    def _build_target(self):
        rng = np.random.default_rng(seed=55550003)

        qubit_properties = []
        for i in range(self._num_qubits):
            qubit_properties.append(QubitProperties(
                t1=self.t1_time + rng.uniform(-1e-5, 1e-5),
                t2=self.t2_time + rng.uniform(-1e-5, 1e-5),
                frequency=rng.uniform(4.8e9, 5.2e9),
            ))

        self._target = Target(
            "IBM Melbourne",
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
        for edge in MELBOURNE_EDGES:
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
