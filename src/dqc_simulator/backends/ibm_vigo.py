"""
IBM Vigo backend — based on FakeVigoV2 calibration snapshot.
Wraps calibration data into a standalone class so it doesn't
re-fetch from qiskit_ibm_runtime every instantiation.

Specs (from FakeVigoV2 calibration):
  Qubits: 5, T-shape coupling
  F_1q:   0.9995
  F_2q:   0.9912 (mean across all CX pairs)
  F_spam: 0.9666
  t_2q:   0.4 μs
"""
import numpy as np
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit.library import XGate, SXGate, RZGate, CXGate
from qiskit.circuit import Measure, Delay, Parameter, Reset
from qiskit.providers.backend import QubitProperties


# Coupling map: T-shape
#     2
#     |
# 0 - 1 - 3 - 4
VIGO_EDGES = [(0, 1), (1, 0), (1, 2), (2, 1), (1, 3), (3, 1), (3, 4), (4, 3)]


class IBMVigo(BackendV2):
    """IBM Vigo backend (5 qubits, T-shape coupling)."""

    def __init__(self):
        super().__init__(name="IBM Vigo")

        self._num_qubits = 5
        self.backend_name = "ibm_vigo"

        self.fidelity_1q_mean = 0.9995
        self.fidelity_2q_mean = 0.9912
        self.fidelity_spam_mean = 0.9666

        self.t_readout = 0.0000036   # 3.6 μs
        self.t_reset = 0.0000035     # 3.5 μs
        self.t_1q = 0.000000036      # 36 ns
        self.t_2q = 0.0000004        # 400 ns (0.4 μs)
        self.t1_time = 0.000080      # 80 μs
        self.t2_time = 0.000060      # 60 μs

        self._build_target()

    def _build_target(self):
        rng = np.random.default_rng(seed=55550001)

        qubit_properties = []
        for i in range(self._num_qubits):
            qubit_properties.append(QubitProperties(
                t1=self.t1_time + rng.uniform(-1e-5, 1e-5),
                t2=self.t2_time + rng.uniform(-1e-5, 1e-5),
                frequency=rng.uniform(4.8e9, 5.2e9),
            ))

        self._target = Target(
            "IBM Vigo",
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
                error=error_1q + rng.uniform(-1e-5, 1e-5), duration=self.t_1q)
            sx_props[q] = InstructionProperties(
                error=error_1q + rng.uniform(-1e-5, 1e-5), duration=self.t_1q)
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
        error_2q = 1 - self.fidelity_2q_mean
        for edge in VIGO_EDGES:
            var = rng.uniform(-2e-3, 2e-3)
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
