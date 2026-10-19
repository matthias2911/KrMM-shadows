"""
Simulate Werner-state Pauli measurements, grouped as NU unitaries x NM outcomes each.
NU = 3**nqubits * order.

Outcome probabilities are not computed here, they are loaded from a precomputed cache
"""

import argparse
import os
from itertools import product

import numpy as np
from scipy.linalg import eigvalsh


def werner_density_matrix(nqubits: int, t: float) -> np.ndarray:
    """rho_W(t) = (I - t F) / (d^2 - d t), d = 2^(N/2), balanced cut."""
    if nqubits % 2 != 0:
        raise ValueError("Only even nqubits allowed for werner")
    d = 2 ** (nqubits // 2)
    D = d * d
    denom = d**2 - d * t
    if np.isclose(denom, 0.0):
        raise ValueError(f"Unphysical: d^2 - d t = 0 for d={d}, t={t}")

    # swap operator: |a,b> -> |b,a>
    F = np.zeros((D, D), dtype=complex)
    for a in range(d):
        for b in range(d):
            F[a * d + b, b * d + a] = 1.0

    rho = (np.eye(D, dtype=complex) - t * F) / denom
    assert np.isclose(np.trace(rho), 1.0, atol=1e-10)
    assert np.allclose(rho, rho.conj().T, atol=1e-10)
    return rho


def partial_transpose_half(rho: np.ndarray, nqubits: int) -> np.ndarray:
    """Partial transpose on the second half of the qubits."""
    dh = 2 ** (nqubits // 2)
    T = rho.reshape(dh, dh, dh, dh)          # (a, b, a', b')
    T = np.transpose(T, (0, 3, 2, 1))        # swap b <-> b'
    return T.reshape(rho.shape)


def load_probs_cache(probs_dir: str, nqubits: int):
    """Load the precomputed probabilities. Returns (key -> row index dict, probs array)."""
    zero_digit = "0" if nqubits < 10 else ""
    probs_file = f"{probs_dir}/qb_{zero_digit}{nqubits}.npz"
    if not os.path.exists(probs_file):
        raise FileNotFoundError(f"Probability cache not found: {probs_file}")

    data = np.load(probs_file)
    keys = data["keys"]
    probs = data["probs"]

    if probs.ndim != 2 or probs.shape[0] != len(keys) or probs.shape[1] != 2**nqubits:
        raise ValueError(
            f"Cache {probs_file} has probs shape {probs.shape} and {len(keys)} keys, "
            f"expected ({len(keys)}, {2**nqubits}) for {nqubits} qubits"
        )

    key_to_row = {int(k): r for r, k in enumerate(keys)}
    print(f"Loaded probs cache from {probs_file} ({len(keys)} settings)")
    return key_to_row, probs


def simulate(nqubits, NM, order, t, path, seed, random_settings, probs_dir):
    rng = np.random.default_rng(seed)
    N = nqubits
    NU = 3**N * order
    if nqubits == 10 and NM == 1:
        NU = int(10**6)
        random_settings = True
    D = 2**N

    # ---- metadata (moments from exact spectra, for cross-checking in Julia) ----
    rho = werner_density_matrix(N, t)
    ev = eigvalsh(rho)
    ev_pt = eigvalsh(partial_transpose_half(rho, N))
    moments = np.array([np.sum(ev**n) for n in range(1, order + 1)], dtype=np.float64)
    moments_pt = np.array([np.sum(ev_pt**n) for n in range(1, order + 1)], dtype=np.float64)
    transposes = np.array([0] * (N // 2) + [1] * (N // 2), dtype=np.int32)
    del rho

    # ---- precomputed outcome probabilities ----
    key_to_row, probs_all = load_probs_cache(probs_dir, N)
    pow3 = 3 ** np.arange(N)  # little-endian base-3 key, same as the original script

    # ---- measurement settings: (NU, N) ----
    if random_settings:
        inds = rng.integers(0, 3, size=(NU, N))
    else:
        # every one of the 3^N Pauli settings, repeated `order` times
        all_settings = np.array(list(product(range(3), repeat=N)), dtype=np.int64)
        inds = np.tile(all_settings, (order, 1))

    # ---- sample NM outcomes per unitary: (NU, NM, N) ----
    clicks = np.empty((NU, NM, N), dtype=np.int8)
    shifts = N - 1 - np.arange(N)  # big-endian: qubit 0 = most significant bit

    uniq, inverse = np.unique(inds, axis=0, return_inverse=True)
    inverse = inverse.ravel()

    # check that the cache covers every setting we need before sampling anything
    uniq_keys = (uniq * pow3).sum(axis=1)
    missing = [int(k) for k in uniq_keys if int(k) not in key_to_row]
    if missing:
        raise KeyError(
            f"{len(missing)} measurement settings not in probs cache, e.g. key {missing[0]}"
        )

    for k, key in enumerate(uniq_keys):
        probs = probs_all[key_to_row[int(key)]]
        for u in np.where(inverse == k)[0]:
            outcomes = rng.choice(D, size=NM, p=probs)
            clicks[u] = (outcomes[:, None] >> shifts) & 1

    # ---- save ----
    os.makedirs(path, exist_ok=True)
    zero_digit = "0" if N < 10 else ""
    fname = f"{path}/qb_{zero_digit}{N}.npz"
    np.savez(
        fname,
        num_qubits=np.array([N], dtype=np.int32),
        num_unitaries=np.array([NU], dtype=np.int64),
        num_meas=np.array([NM], dtype=np.int64),         # outcomes PER unitary
        max_order=np.array([order], dtype=np.int32),
        pt=np.array([1], dtype=np.int32),
        werner=np.array([1], dtype=np.int32),
        werner_par=np.array([t], dtype=np.float64),
        transposes=transposes,
        moments=moments,
        moments_pt=moments_pt,
        # C-order flattened, see Julia reshape below
        inds=inds.astype(np.int32).flatten(),            # logical shape (NU, N)
        clicks=clicks.astype(np.int32).flatten(),        # logical shape (NU, NM, N)
    )
    print(f"Saved {NU} unitaries x {NM} outcomes ({N} qubits) to {fname}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Simulate Werner-state measurement data")
    p.add_argument("-qb", "--nqubits", type=int, default=4)
    p.add_argument("-m", "--NM", type=int, default=100, help="outcomes per unitary")
    p.add_argument("-o", "--order", type=int, default=16,
                   help="max moment order; NU = 3^nqubits * order")
    p.add_argument("-t", "--t", type=float, default=0.0, help="werner parameter")
    p.add_argument("-p", "--path", type=str, default="data")
    p.add_argument("-pp", "--probs_dir", type=str, default="None",
                   help="directory with precomputed probs (qb_XX.npz); default: <path>/probs")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--random_settings", action="store_true",
                   help="draw NU random Pauli settings instead of all 3^n repeated `order` times")
    a = p.parse_args()

    probs_dir = f"{a.path}/probs" if a.probs_dir == "None" else a.probs_dir
    simulate(a.nqubits, a.NM, a.order, a.t, a.path, a.seed, a.random_settings, probs_dir)