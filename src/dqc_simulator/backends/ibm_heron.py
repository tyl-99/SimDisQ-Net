"""
IBM Heron backend — based on FakeFez (ibm_fez, Heron r2, 156-qubit
current-generation superconducting device). Wraps calibration data into a
standalone class so it plugs into the simulator's DQC pipeline (stock Qiskit
fake backends fail to translate the teleportation instructions onto their
target). Coupling map (real, healthy edges only) loaded from
ibm_heron_fez_coupling.json.

Specs (extracted from FakeFez calibration via the target/qubit_properties —
all REAL, no invented numbers):
  Qubits: 154, real heavy-hex coupling (338 healthy edges; disabled edges
          excluded so they can't contaminate the fidelity, same fix applied
          after the broken-Cambridge lesson; 2 qubits left isolated by edge
          removal are dropped so the coupling graph stays connected for sabre)
  F_2q:   0.9962  (HEALTHY-pair median CZ; the raw mean 0.955 is dragged down
          by disabled edges that real circuits route around)
  F_1q:   0.99971 (mean SX/X)
  F_spam: 0.9868  (mean measure)
  T1:     145 μs   T2: 90.5 μs
  t_2q:   84.2 ns  t_1q: 24 ns   t_read: 1.56 μs

Represents CURRENT IBM hardware at the high-fidelity end of the Network-Wall
ladder (alongside the trapped-ion devices), so the ladder is not stale
old-IBM-vs-new-ion. Heron's native 2q gate is CZ; modeled here with CXGate (the
simulator's CX basis) carrying Heron's real 2q error magnitude.
"""
import json
import numpy as np
from pathlib import Path
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit.library import XGate, SXGate, RZGate, CXGate
from qiskit.circuit import Measure, Delay, Parameter, Reset
from qiskit.providers.backend import QubitProperties

_COUPLING = json.load(open(Path(__file__).resolve().parent / "ibm_heron_fez_coupling.json"))
HERON_EDGES = [tuple(e) for e in _COUPLING["edges"]]


class IBMHeron(BackendV2):
    """IBM Heron r2 backend (ibm_fez, 156 qubits, real heavy-hex coupling)."""

    def __init__(self):
        super().__init__(name="IBM Heron")

        self._num_qubits = _COUPLING["num_qubits"]
        self.backend_name = _COUPLING["backend_name"]

        self.fidelity_1q_mean = 0.99971
        self.fidelity_2q_mean = 0.9962
        self.fidelity_spam_mean = 0.9868

        self.t_readout = 0.00000156     # 1.56 μs
        self.t_reset = 0.00000156       # 1.56 μs
        self.t_1q = 0.000000024         # 24 ns
        self.t_2q = 0.0000000842        # 84.2 ns
        self.t1_time = 0.000145         # 145 μs
        self.t2_time = 0.0000905        # 90.5 μs

        self._build_target()

    def _build_target(self):
        rng = np.random.default_rng(seed=55550005)

        qubit_properties = []
        for i in range(self._num_qubits):
            qubit_properties.append(QubitProperties(
                t1=self.t1_time + rng.uniform(-1e-5, 1e-5),
                t2=self.t2_time + rng.uniform(-1e-5, 1e-5),
                frequency=rng.uniform(4.8e9, 5.2e9),
            ))

        self._target = Target(
            "IBM Heron",
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
        for edge in HERON_EDGES:
            var = rng.uniform(-5e-4, 5e-4)
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
