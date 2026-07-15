"""
Backend parameter extraction for heralded entanglement success-rate calculation.
Unifies IonQ (attribute-based) and IBM fake backends (target-based).
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class BackendParams:
    """Parameters extracted from a backend for link success-rate calculation."""
    fidelity_1q: float
    fidelity_2q: float
    fidelity_spam: float
    gate_time_2q: float  # seconds

    @classmethod
    def from_backend(cls, backend) -> "BackendParams":
        """Extract parameters from IonQ (hasattr) or IBM-style backend (target)."""
        if hasattr(backend, "fidelity_1q_mean"):
            return cls(
                fidelity_1q=backend.fidelity_1q_mean,
                fidelity_2q=backend.fidelity_2q_mean,
                fidelity_spam=backend.fidelity_spam_mean,
                gate_time_2q=getattr(backend, "t_2q", 1e-6),
            )
        # IBM fake backends: use target
        target = backend.target
        fidelity_1q = _get_1q_fidelity_from_target(backend, target)
        fidelity_2q = _get_2q_fidelity_from_target(backend, target)
        fidelity_spam = _get_spam_fidelity_from_target(backend, target)
        gate_time_2q = _get_2q_time_from_target(target)
        return cls(
            fidelity_1q=fidelity_1q,
            fidelity_2q=fidelity_2q,
            fidelity_spam=fidelity_spam,
            gate_time_2q=gate_time_2q,
        )


def _get_1q_fidelity_from_target(backend, target) -> float:
    """Average 1-qubit gate fidelity from target (X or SX)."""
    errors = []
    for name in ("x", "sx"):
        if name in target:
            for q in range(backend.num_qubits):
                key = (q,)
                if key in target[name]:
                    err = target[name][key].error
                    if err is not None:
                        errors.append(err)
    return float(1 - np.mean(errors)) if errors else 0.999


def _get_2q_fidelity_from_target(backend, target) -> float:
    """Average 2-qubit gate fidelity from target (CX or similar)."""
    errors = []
    for name in ("cx", "cz", "ecr"):
        if name in target:
            for key, props in target[name].items():
                if props.error is not None:
                    errors.append(props.error)
    return float(1 - np.mean(errors)) if errors else 0.99


def _get_spam_fidelity_from_target(backend, target) -> float:
    """Average SPAM (measurement) fidelity from target."""
    errors = []
    if "measure" in target:
        for q in range(backend.num_qubits):
            key = (q,)
            if key in target["measure"]:
                err = target["measure"][key].error
                if err is not None:
                    errors.append(err)
    return float(1 - np.mean(errors)) if errors else 0.99


def _get_2q_time_from_target(target) -> float:
    """Average 2-qubit gate duration in seconds."""
    durs = []
    for name in ("cx", "cz", "ecr"):
        if name in target:
            for props in target[name].values():
                if props.duration is not None:
                    durs.append(props.duration)
    return float(np.mean(durs)) if durs else 1e-6


def calculate_link_success_probability(backend1, backend2, distance: float):
    """
    Compute heralded entanglement success rate and attempt time for a link.

    Args:
        backend1, backend2: Qiskit backends at each end of the link.
        distance: Physical distance (e.g. km).

    Returns:
        tuple: (success_probability, attempt_time_seconds)
    """
    p1 = BackendParams.from_backend(backend1)
    p2 = BackendParams.from_backend(backend2)

    P_emit1 = p1.fidelity_1q * p1.fidelity_2q
    P_emit2 = p2.fidelity_1q * p2.fidelity_2q
    alpha = 0.2 / 4.343  # dB/km -> natural

    # Two photon arms travelling to a heralding station.
    # Currently assumes midpoint heralding (d1 = d2 = distance/2),
    # but split explicitly so asymmetric placement can be added later.
    d1 = distance / 2.0
    d2 = distance / 2.0
    P_transmission = np.exp(-alpha * d1) * np.exp(-alpha * d2)
    P_BSM = 0.5

    success_prob = P_emit1 * P_emit2 * P_transmission * P_BSM
    success_prob = float(np.clip(success_prob, 1e-6, 1.0))

    t_emit = max(p1.gate_time_2q, p2.gate_time_2q)
    # Photon arms travel to the heralding station — we wait for the slower one.
    # For symmetric midpoint heralding (current case), max(d1, d2) = distance/2.
    # Same expression handles asymmetric placement if d1 != d2 in the future.
    c_fiber = 200000.0  # speed of light in fiber, km/s
    t_prop = max(d1, d2) / c_fiber
    t_bsm = 1e-6
    # Classical herald signal returns to both endpoints; the slower path bounds it.
    t_classical = max(d1, d2) / c_fiber
    attempt_time = t_emit + t_prop + t_bsm + t_classical

    return success_prob, attempt_time

