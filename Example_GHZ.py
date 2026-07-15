"""
SimDisQ-Net quickstart: 12-qubit GHZ distributed across 4 QPUs.

End-to-end smoke test — partitions a GHZ circuit, runs the full network +
circuit pipeline, executes on AerSimulator, scores the result, and saves
the histogram and circuit diagram as PDFs.

For focused feature walkthroughs (routing, scheduling, security, purification),
see the `examples/` folder.

    python Example_GHZ.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

from dqc_simulator import DQCCircuit, DQCQPU, QPUManager

SEED = 7
np.random.seed(SEED)


# ============================================================
# Fidelity metric: Hellinger fidelity for GHZ state
# ============================================================
def score(exp_probs: dict) -> float:
    """
    Compute Hellinger fidelity for an n-qubit GHZ state.

    Args:
        exp_probs: Experimental probability distribution
                   e.g., {'000...0': p0, '111...1': p1}

    Returns:
        fidelity in [0, 1]
    """
    n = len(next(iter(exp_probs)))

    # Ideal GHZ distribution
    ideal_probs = {
        '0' * n: 0.5,
        '1' * n: 0.5
    }

    # Hellinger fidelity
    fidelity = sum(
        np.sqrt(ideal_probs.get(k, 0)) * np.sqrt(p)
        for k, p in exp_probs.items()
    )
    return fidelity


# 1. Build the 12-qubit GHZ circuit
N = 12
qc0 = QuantumCircuit(N, N)
qc0.h(0)
for i in range(N - 1):
    qc0.cx(i, i + 1)
qc0.measure(range(N), range(N))

# 2. Configure 4 QPUs in a linear chain
qpus = QPUManager()
for i in range(4):
    qpus.add_qpu(DQCQPU(i, "FakeVigoV2"))
for i in range(3):
    qpus.add_coonnection(i, i + 1, distance=5)

# 3. Run the full pipeline.
#    Default output_level="summary" prints the network summary block to stdout.
qc = DQCCircuit(qc0)
qc.purification = {}
result_qc = qc.Execution(
    [3, 3, 3, 3],
    qpus,
    comm_noise=True,
    scheduling_algorithm="SJF",
)

# 4. Run on AerSimulator with the combined local + communication noise model
sim = AerSimulator(noise_model=qc.get_noise_model(), seed_simulator=SEED)
compiled = transpile(result_qc, sim, seed_transpiler=SEED)
counts = sim.run(compiled, shots=1000, seed_simulator=SEED).result().get_counts()

# 5. Post-processing measurement results
counts_res = {}
for bitstring, cnt in counts.items():
    bits = bitstring[:N]
    counts_res[bits] = counts_res.get(bits, 0) + cnt

# Compute and report Hellinger fidelity vs ideal GHZ distribution
total = sum(counts_res.values())
probs = {k: v / total for k, v in counts_res.items()}
hellinger = score(probs)
print(f"\nHellinger fidelity: {hellinger:.2f}")

# 6. Save histogram + circuit diagram as PDFs
plot_histogram(counts_res).savefig("GHZ_12qubit_DQC_histogram.pdf")
result_qc.draw("mpl", scale=0.7, fold=100).savefig("GHZ_12qubit_DQC_circuit.pdf")
print("Saved: GHZ_12qubit_DQC_histogram.pdf, GHZ_12qubit_DQC_circuit.pdf")
