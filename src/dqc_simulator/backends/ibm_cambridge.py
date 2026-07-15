"""
IBM Cambridge backend — based on FakeCambridgeV2 calibration snapshot.
28-qubit heavy-hex coupling with known broken qubit pairs (F_2q = 0.0).

This backend reproduces the real ibm_cambridge chip including its defects,
as used in the SimDisQ paper for Arch-A baseline.

Specs (from FakeCambridgeV2 calibration):
  Qubits: 28, heavy-hex coupling
  F_1q:   0.9990
  F_2q:   0.8058 (mean — dragged down by 5 broken CX pairs with F=0.0)
  F_2q:   0.9720 (median — healthy pairs only)
  F_spam: 0.9900
  t_2q:   0.6 μs

Broken CX pairs: (10,11), (14,15), (15,18), (20,21), (26,27)
"""
import numpy as np
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit.library import XGate, SXGate, RZGate, CXGate
from qiskit.circuit import Measure, Delay, Parameter, Reset
from qiskit.providers.backend import QubitProperties


CAMBRIDGE_EDGES = [
    (0, 1), (0, 5), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 4),
    (4, 3), (4, 6), (5, 0), (5, 9), (6, 4), (6, 13), (7, 8), (7, 16),
    (8, 7), (8, 9), (9, 5), (9, 8), (9, 10), (10, 9), (10, 11), (11, 10),
    (11, 12), (11, 17), (12, 11), (12, 13), (13, 6), (13, 12), (13, 14),
    (14, 13), (14, 15), (15, 14), (15, 18), (16, 7), (16, 19), (17, 11),
    (17, 23), (18, 15), (18, 27), (19, 16), (19, 20), (20, 19), (20, 21),
    (21, 20), (21, 22), (22, 21), (22, 23), (23, 17), (23, 22), (23, 24),
    (24, 23), (24, 25), (25, 24), (25, 26), (26, 25), (26, 27), (27, 18),
    (27, 26),
]

# Per-pair F_2q from FakeCambridgeV2 calibration
CAMBRIDGE_CX_FIDELITY = {
    (0, 1): 0.9682, (1, 0): 0.9682, (0, 5): 0.9764, (5, 0): 0.9764,
    (1, 2): 0.9794, (2, 1): 0.9794, (2, 3): 0.9811, (3, 2): 0.9811,
    (3, 4): 0.9769, (4, 3): 0.9769, (4, 6): 0.9775, (6, 4): 0.9775,
    (5, 9): 0.9750, (9, 5): 0.9750, (6, 13): 0.9649, (13, 6): 0.9649,
    (7, 8): 0.9770, (8, 7): 0.9770, (7, 16): 0.9800, (16, 7): 0.9800,
    (8, 9): 0.9748, (9, 8): 0.9748, (9, 10): 0.9715, (10, 9): 0.9715,
    (10, 11): 0.0000, (11, 10): 0.0000,  # BROKEN
    (11, 12): 0.9654, (12, 11): 0.9654,
    (11, 17): 0.9683, (17, 11): 0.9683,
    (12, 13): 0.9857, (13, 12): 0.9857,
    (13, 14): 0.9720, (14, 13): 0.9720,
    (14, 15): 0.0000, (15, 14): 0.0000,  # BROKEN
    (15, 18): 0.0000, (18, 15): 0.0000,  # BROKEN
    (16, 19): 0.9552, (19, 16): 0.9552,
    (17, 23): 0.9684, (23, 17): 0.9684,
    (18, 27): 0.9713, (27, 18): 0.9713,
    (19, 20): 0.9221, (20, 19): 0.9221,
    (20, 21): 0.0000, (21, 20): 0.0000,  # BROKEN
    (21, 22): 0.9143, (22, 21): 0.9143,
    (22, 23): 0.9280, (23, 22): 0.9280,
    (23, 24): 0.9723, (24, 23): 0.9723,
    (24, 25): 0.9861, (25, 24): 0.9861,
    (25, 26): 0.9629, (26, 25): 0.9629,
    (26, 27): 0.0000, (27, 26): 0.0000,  # BROKEN
}


class IBMCambridge(BackendV2):
    """IBM Cambridge backend (28 qubits, heavy-hex, with broken pairs)."""

    def __init__(self):
        super().__init__(name="IBM Cambridge")

        self._num_qubits = 28
        self.backend_name = "ibm_cambridge"

        self.fidelity_1q_mean = 0.9990
        self.fidelity_2q_mean = 0.8058   # mean including broken pairs
        self.fidelity_spam_mean = 0.9900

        self.t_readout = 0.0000036   # 3.6 μs
        self.t_reset = 0.0000035     # 3.5 μs
        self.t_1q = 0.000000036      # 36 ns
        self.t_2q = 0.0000006        # 600 ns (0.6 μs)
        self.t1_time = 0.000070      # 70 μs
        self.t2_time = 0.000050      # 50 μs

        self._build_target()

    def _build_target(self):
        rng = np.random.default_rng(seed=55550002)

        qubit_properties = []
        for i in range(self._num_qubits):
            qubit_properties.append(QubitProperties(
                t1=self.t1_time + rng.uniform(-1e-5, 1e-5),
                t2=self.t2_time + rng.uniform(-1e-5, 1e-5),
                frequency=rng.uniform(4.8e9, 5.2e9),
            ))

        self._target = Target(
            "IBM Cambridge",
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
                error=error_spam + rng.uniform(-1e-4, 1e-4), duration=self.t_readout)
            reset_props[q] = InstructionProperties(
                error=error_spam + rng.uniform(-1e-4, 1e-4), duration=self.t_reset)
            delay_props[q] = None

        self._target.add_instruction(XGate(), x_props)
        self._target.add_instruction(SXGate(), sx_props)
        self._target.add_instruction(RZGate(Parameter("theta")), rz_props)
        self._target.add_instruction(Measure(), measure_props)
        self._target.add_instruction(Reset(), reset_props)
        self._target.add_instruction(Delay(Parameter("t")), delay_props)

    def _add_2q_gates(self, rng):
        cx_props = {}
        for edge in CAMBRIDGE_EDGES:
            fid = CAMBRIDGE_CX_FIDELITY.get(edge, 0.97)
            error = max(0.0, 1 - fid)
            cx_props[edge] = InstructionProperties(
                error=error, duration=self.t_2q)
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
