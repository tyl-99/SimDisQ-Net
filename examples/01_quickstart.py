"""
01 — Quickstart: minimal SimDisQ-Net run.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qiskit import QuantumCircuit
from dqc_simulator import DQCCircuit, DQCQPU, QPUManager

# 6-qubit chain circuit
N = 6
qc0 = QuantumCircuit(N, N)
qc0.h(0)
for i in range(N - 1):
    qc0.cx(i, i + 1)
qc0.measure(range(N), range(N))

# 6-QPU ring + one shortcut, 1 qubit per QPU
qpus = QPUManager()
for i in range(N):
    qpus.add_qpu(DQCQPU(i, "FakeLagosV2", available_comm_qubits=2))
for i in range(N):
    qpus.add_coonnection(i, (i + 1) % N, distance=2)
qpus.add_coonnection(0, 3, distance=1)

qc = DQCCircuit(qc0)
qc.purification = {}
qc.Execution([1] * N, qpus, skip_aer=True, output_level="verbose")  # quiet, summary, verbose
