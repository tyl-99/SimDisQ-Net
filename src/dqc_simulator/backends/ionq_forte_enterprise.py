"""
IonQ Forte Enterprise 1 backend — based on real calibration data.
Source: IonQ calibration JSON (qpu.forte-enterprise-1), date 2025-08-03.

Specs:
  Qubits: 36, fully connected
  F_1q:   0.9999 (mean)
  F_2q:   0.9899 (mean), 0.9943 (median)
  F_spam: 0.9957 (mean)
  t_1q:   63 μs
  t_2q:   650 μs
  t_read: 250 μs
  t_reset: 150 μs
  T1:     188 s
  T2:     0.95 s
"""
import numpy as np
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit.library import XGate, SXGate, RZGate, CZGate
from qiskit.circuit import Measure, Delay, Parameter, Reset
from qiskit.providers.backend import QubitProperties


class IonQForteEnterprise(BackendV2):
    """IonQ Forte Enterprise 1 backend (36 qubits, fully connected)."""

    def __init__(self):
        super().__init__(name="IonQ Forte Enterprise 1")

        self._num_qubits = 36
        self.backend_name = "qpu.forte-enterprise-1"

        # From calibration JSON
        self.fidelity_1q_mean = 0.9999
        self.fidelity_2q_mean = 0.9899
        self.fidelity_spam_mean = 0.9957

        self.t_readout = 0.00025    # 250 μs
        self.t_reset = 0.00015      # 150 μs
        self.t_1q = 0.000063        # 63 μs
        self.t_2q = 0.00065         # 650 μs
        self.t1_time = 188          # 188 s
        self.t2_time = 0.95         # 0.95 s

        self._build_target()

    def _build_target(self):
        rng = np.random.default_rng(seed=20250803002)

        qubit_properties = []
        for i in range(self._num_qubits):
            qubit_properties.append(QubitProperties(
                t1=self.t1_time + rng.uniform(-10.0, 10.0),
                t2=self.t2_time + rng.uniform(-0.05, 0.05),
                frequency=rng.uniform(4.5e9, 5.5e9),
            ))

        self._target = Target(
            "IonQ Forte Enterprise 1",
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
        cz_props = {}
        error_2q = 1 - self.fidelity_2q_mean
        for i in range(self._num_qubits):
            for j in range(i + 1, self._num_qubits):
                var = rng.uniform(-3e-3, 3e-3)
                cz_props[(i, j)] = InstructionProperties(
                    error=error_2q + var, duration=self.t_2q)
                cz_props[(j, i)] = InstructionProperties(
                    error=error_2q + var, duration=self.t_2q)
        self._target.add_instruction(CZGate(), cz_props)

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
