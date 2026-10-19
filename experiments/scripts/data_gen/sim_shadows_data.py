"""
Simulating the system and storing data
"""

import numpy as np
from itertools import product
import functools
import pandas as pd
import argparse
import os
from scipy.linalg import eigvalsh
import time

def multikron(*args):
    """Calculate the Kronecker product of multiple matrices"""
    return functools.reduce(np.kron, args, 1 + 0j)

def ghz_statevector(nqudits:int):
    return sum(multikron(*[k] * nqudits) for k in np.eye(2)) / np.sqrt(2)

def bell():
    bell_state = ghz_statevector(2)
    return bell_state

def bits_from_int(x: int, nbits: int) -> np.ndarray:
    """Return length-nbits array of bits for integer x (big-endian)."""
    return np.array([(x >> (nbits - 1 - i)) & 1 for i in range(nbits)], dtype=np.int8)

def rand_clicks(num_qubits, num_meas=1):
    """draw random clicks for fast maximally mixed state."""
    clicks = np.random.choice([0, 1], size=(num_meas, num_qubits))  # Random outcomes
    if num_meas > 1:
        return clicks
    else:
        return clicks[0] # flatten

# From https://github.com/zhangdajian-sdu/Krylov_Shadow_Tomography/blob/main/my_functions.py
def gen_Z_hamil(n):
    """
    Generate the Hamiltonian for an n-qubit system.
    """
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.eye(2, dtype=complex)

    dim = 2**n
    hamiltonian = np.zeros((dim, dim), dtype=complex)

    # Loop through each term, which is a Kronecker product of Pauli-Z and identity matrices
    for i in range(n):
        # Build the list of matrices: identity everywhere except sigma_z at position i
        matrices = [identity] * i + [sigma_z] + [identity] * (n - i - 1)

        # Compute the Kronecker product for the i-th term
        term_i = matrices[0]
        for mat in matrices[1:]:
            term_i = np.kron(term_i, mat)
        term_i = 0.5 * term_i

        # Add the current term to the total Hamiltonian
        hamiltonian += term_i

    return hamiltonian

# From https://github.com/zhangdajian-sdu/Krylov_Shadow_Tomography/blob/main/my_functions.py
def get_QFI(rho, H):
    """
    Calculating QFI
    From 
    """
    # Get eigenvalues and eigenstates of the density matrix rho
    evals, evecs = np.linalg.eigh(rho)

    # Initialize QFI
    QFI = 0

    # Calculate the QFI using the given formula
    for k in range(len(evals)):
        for l in range(len(evals)):
            if k != l:
                p_k = evals[k]
                p_l = evals[l]
                if p_k + p_l > 1e-10:  # Avoid division by zero
                    # Calculate the inner product of commutator and its adjoint
                    inner_product = evecs[:, k].conj() @ H @ evecs[:, l]
                    inner_product = inner_product * inner_product.conjugate()
                    QFI += 2 * (p_k - p_l)**2 / (p_k + p_l) * inner_product

    return QFI.real

def compute_n_star(rho: np.ndarray, H: np.ndarray, atol: float = 1e-8) -> int:
    """
    n* = N[S] - N[J]
    Wang, Zhang, "Superiority of Krylov shadow tomography..." 2026
    """
    evals, evecs = np.linalg.eigh(rho)
    d = len(evals)

    # Cluster eigenvalues into equivalence classes (transitive, via sorted gaps)
    order = np.argsort(evals)
    sorted_evals = evals[order]
    labels = np.zeros(d, dtype=int)
    label = 0
    for i in range(1, d):
        if sorted_evals[i] - sorted_evals[i - 1] > atol:
            label += 1
        labels[i] = label
    eig_class = np.empty(d, dtype=int)
    eig_class[order] = labels
    n_classes = label + 1
    class_value = np.array([sorted_evals[labels == c].mean() for c in range(n_classes)])

    H_eig = evecs.conj().T @ H @ evecs

    sums_to_matrix_elements = {}
    for ci in range(n_classes):
        for cj in range(n_classes):
            if ci == cj:
                continue
            mu = class_value[ci] + class_value[cj]
            key = round(mu / atol) * atol
            idx_i = np.where(eig_class == ci)[0]
            idx_j = np.where(eig_class == cj)[0]
            elems = [H_eig[i, j] for i in idx_i for j in idx_j]
            sums_to_matrix_elements.setdefault(key, []).extend(elems)

    N_S = len(sums_to_matrix_elements)
    N_J = sum(
        1 for elements in sums_to_matrix_elements.values()
        if all(abs(e) < atol for e in elements)
    )
    return N_S - N_J

class Simulator:
    """
    Simulator class for generating measurement data
    """

    def __init__(self, nqubits:int, shots: int, order: int, name: str = 'ghz', t: float = 0.0, run_id=0, path="data", compute_probs=True, probs_dir="None", rank=2):
        """
        Initialize the simulator
        :param nqubits: Number of qubits
        :param shots: Number of shots
        :param state: Initial state
        :param name: Name of the simulation (for saving)
        :param run_id: Run identifier (for saving)
        :param t: t parameter for werner states
        """
        self.nqubits = nqubits
        self.all_probs = None
        self.shots = shots
        self.order = order
        self.t = t
        self.run_id = run_id
        self.name = name
        self.compute_probs = compute_probs        
        self.probs_dir = probs_dir        
        self.rank = rank        

        # create directory for data

        self.path = path

        os.makedirs(self.path, exist_ok=True)

        if self.run_id != 0:
            assert False, "Support for multiple runs has been removed"

        # define gates
        self.H = np.array([[1, 1],[1, -1]]) / np.sqrt(2) # Hadamard gate
        self.S = np.array([[1, 0],[0, 1j]]) # Phase gate

        # define Pauli rotations for measurement
        self.PAULI_ROTATIONS_LIST = [
            np.eye(2), # Z-measurement
            self.H, # X-measurement
            self.H @ self.S.conj().T # Y-measurement
        ]

        if name in ("ghz", "bell"):
            self.state = ghz_statevector(nqubits) if name == 'ghz' else bell()
            self.rho = None
            if name == "bell" and nqubits != 2:
                raise ValueError("Only 2 qubits allowed for bell state")            
        elif name == "werner":
            if t is None:
                raise ValueError("t parameter must be provided for werner states")
            if nqubits % 2 != 0:
                raise ValueError("Only even nqubits allowed for werner")            
            
            self.state = None
            self.rho = self.werner_density_matrix()
        elif name == "mm":        
            self.state = None
            self.rho = None    
        elif name == "rankX":        
            self.state = None
            self.rho = self.rankX_density_matrix()                

        if name == "ghz" or name == "werner" or name == "rankX":
            # calculate eigenvalues
            if self.rho is None:
                rho = self.state.reshape(-1, 1) @ self.state.conj().T.reshape(1, -1)
            else:
                rho = self.rho

            # compute eigenvalues of rho along half number of qubits
            # store all 0s if num_qubits % 2 != 0
            evs = eigvalsh(rho)
            self.eigenvalues = np.array(evs, dtype=np.float64)

            # partial transpose
            if self.nqubits % 2 == 0:
                # equal split: |A| = |B| = n//2
                dims = [2**(self.nqubits // 2), 2**(self.nqubits // 2)]
                rho_tensor = rho.reshape(dims + dims)
                axes = list(range(2 * len(dims)))
                axes[1], axes[3] = axes[3], axes[1]  # swap B indices
                rho_pt = np.transpose(rho_tensor, axes=axes).reshape(rho.shape)
                assert np.allclose(rho_pt, rho_pt.conj().T, atol=1e-10)
                
                # get eigenvalues of pt
                evs = eigvalsh(rho_pt)
                self.pt = np.array([1], dtype=np.int32)
                self.eigenvalues_pt = np.array(evs, dtype=np.float64)
                self.transposes = np.array([0] * (self.nqubits // 2) + [1] * (self.nqubits // 2), dtype=np.int32)
            else :
                # unequal split: |A| = n//2, |B| = n//2 + 1
                n_a = self.nqubits // 2          
                n_b = self.nqubits - n_a         
                dim_a = 2 ** n_a
                dim_b = 2 ** n_b
                dims = [dim_a, dim_b]
                rho_tensor = rho.reshape(dims + dims)  
                axes = [0, 1, 2, 3]
                axes[1], axes[3] = axes[3], axes[1]  
                rho_pt = np.transpose(rho_tensor, axes=axes).reshape(rho.shape)
                evs = np.linalg.eigvals(rho_pt).real # rho is not necessarily hermitian for unqequal split
                self.pt = np.array([1], dtype=np.int32)
                self.eigenvalues_pt = np.array(evs, dtype=np.float64)
                self.transposes = np.array([0] * n_a + [1] * n_b, dtype=np.int32)   

            if name == "rankX":
                H = gen_Z_hamil(self.nqubits)
                self.QFI = get_QFI(rho, H)

                n_star = compute_n_star(rho, H)
                assert n_star == int(rank * (rank + 1) / 2)
                self.n_star = n_star

        else:
            self.eigenvalues = np.zeros(int(2**self.nqubits), dtype=np.float64)
            self.eigenvalues_pt = np.zeros(int(2**self.nqubits), dtype=np.float64)
            self.pt = np.array([0], dtype=np.int32)
            self.transposes = np.zeros(self.nqubits, dtype=np.int32)   

    def local_dim_from_qubits(self) -> int:
        """Balanced cut N/2|N/2 => local dim d = 2^(N/2)."""
        if self.nqubits < 2:
            raise ValueError("Need at least 2 qubits.")
        if self.nqubits % 2 != 0:
            raise ValueError("Werner requires even N (balanced cut).")
        return 2 ** (self.nqubits // 2)

    def flip_operator_balanced(self) -> np.ndarray:
        """
        Flip (swap) operator F on C^d ⊗ C^d, where d = 2^(N/2),
        acting as F |a,b> = |b,a> on the balanced bipartition.

        Basis is the computational basis with first half = subsystem A, second half = subsystem B.
        """
        d = self.local_dim_from_qubits()
        D = d * d  # = 2^N
        F = np.zeros((D, D), dtype=complex)

        # index mapping: |a>_A |b>_B corresponds to global basis index a*d + b
        for a in range(d):
            for b in range(d):
                ab = a * d + b
                ba = b * d + a
                F[ab, ba] = 1.0
        return F

    def werner_density_matrix(self) -> np.ndarray:
        """
        Werner state on balanced cut:
            rho_W(t) = (I - t F) / (d^2 - d t),
        with d = 2^(N/2). Physical for t in [-1, 1].

        Returns rho as (2^N x 2^N) complex matrix.
        """
        d = self.local_dim_from_qubits()
        D = d * d
        denom = d**2 - d * self.t
        if np.isclose(denom, 0.0):
            raise ValueError(f"Unphysical: d^2 - d t = 0 for d={d}, t={self.t}")

        I = np.eye(D, dtype=complex)
        F = self.flip_operator_balanced()
        rho = (I - self.t * F) / denom
        # Optional sanity checks (cheap at N<=10)
        # trace ~ 1, hermitian
        if not np.isclose(np.trace(rho), 1.0, atol=1e-10):
            raise ValueError("Werner rho has wrong trace (numerical issue).")
        if not np.allclose(rho, rho.conj().T, atol=1e-10):
            raise ValueError("Werner rho not Hermitian (numerical issue).")

        return rho

    def rankX_density_matrix(self, seed=25) -> np.ndarray:
        """
        Random rank-X state.
        Returns rho as (2^N x 2^N) complex matrix.
        """
        D = int(2**self.nqubits)
        rank = self.rank

        if rank < 1 or rank > D:
            raise ValueError(f"Rank must be between 1 and {D} (got {rank}).")

        rng = np.random.default_rng(seed)

        real_part = rng.normal(size=(D, rank))
        imag_part = rng.normal(size=(D, rank))
        A = real_part + 1j * imag_part
        rho = A @ A.conj().T
        rho = rho / np.trace(rho)

        if not np.isclose(np.trace(rho), 1.0, atol=1e-10):
            raise ValueError("Rank-X has wrong trace (numerical issue).")
        if not np.allclose(rho, rho.conj().T, atol=1e-10):
            raise ValueError("RankX not Hermitian (numerical issue).")

        return rho
    
    def apply_local_unitaries_to_rho(self, rotations: list) -> np.ndarray:
        """
        Compute rho' = U rho U† where U = ⊗_q U_list[q], without forming U explicitly.

        Implementation:
        reshape rho to tensor with indices (a0..aN-1, b0..bN-1),
        apply U on each 'a' index and U† on each 'b' index via tensordot.
        """
        n = len(rotations)
        D = 2 ** n
        if self.rho.shape != (D, D):
            raise ValueError(f"rho must have shape {(D, D)} for n={n}.")

        T = self.rho.reshape((2,) * n + (2,) * n)  # (a..., b...)

        for q, U in enumerate(rotations):
            # Left multiply on ket index a_q
            # Contract U_{i, a_q} with T_{..., a_q, ...} -> replaces a_q by i
            T = np.tensordot(U, T, axes=([1], [q]))
            # tensordot puts new index in front; move it back to position q
            T = np.moveaxis(T, 0, q)

            # Right multiply on bra index b_q with U†:
            Ud = U.conj().T
            bq = n + q
            T = np.tensordot(Ud, T, axes=([0], [bq]))
            T = np.moveaxis(T, 0, bq)

        return T.reshape(D, D)
    
    def probs_from_rho_and_setting(self, rotations: list) -> np.ndarray:
        """
        Given density matrix rho and a local rotation list (one 2x2 per qubit),
        return measurement probabilities in computational basis after rotation:
        probs[x] = <x| U rho U† |x>.
        """
        # assert np.isclose(np.trace(self.rho), 1.0, atol=1e-10)
        rho_rot = self.apply_local_unitaries_to_rho(rotations)
        probs = np.real(np.diagonal(rho_rot))
        # assert np.isclose(np.trace(rho_rot), 1.0, atol=1e-10)
        # Numerical cleanup
        probs = np.clip(probs, 0.0, 1.0)
        s = probs.sum()
        if s <= 0:
            raise ValueError("Probabilities sum to 0 (numerical issue).")
        return probs / s
    
    def simulate(self, batchsize: int = 1000):
        """
        Simulate Pauli measurement data and save as CSV in batches.
        """
        rng = np.random.default_rng(np.random.randint(0, 2**32) + self.run_id)
        rng2 = np.random.default_rng(np.random.randint(0, 2**32) + self.run_id + 1) # just to be safe

        nbatches = (self.shots + batchsize - 1) // batchsize
        D = 2 ** self.nqubits

        for b in range(nbatches):
            start = b * batchsize
            stop = min((b + 1) * batchsize, self.shots)
            B = stop - start

            if self.name != "mm":
                # Sample random Pauli settings: each shot gets an nqubits-vector in {0,1,2}
                inds_per_qubit = rng.integers(0, 3, size=(B, self.nqubits), endpoint=False)

                clicks_per_qubit = np.empty((B, self.nqubits), dtype=np.int8)

                # To reduce repeated work: group identical settings inside the batch
                # Convert each setting vector to a base-3 integer key.
                keys = (inds_per_qubit * (3 ** np.arange(self.nqubits))).sum(axis=1)
                uniq_keys, inv = np.unique(keys, return_inverse=True)

                start = time.perf_counter()
                if self.probs_dir == "None":
                    self.probs_dir = f"{self.path}/probs"
                os.makedirs(self.probs_dir, exist_ok=True)

                zero_digit = "0" if self.nqubits < 10 else ""
                probs_file = f"{self.probs_dir}/qb_{zero_digit}{self.nqubits}.npz"

                if self.compute_probs:
                    # Precompute probs for each unique setting in this batch
                    print(f"uniqu_keys shape: {uniq_keys.shape}")
                    print(f"3**n: {int(3**self.nqubits)}")
                    probs_cache = {}
                    for uk in uniq_keys:
                        # Recover per-qubit indices from uk (base-3 digits)
                        inds = []
                        tmp = int(uk)
                        for _ in range(self.nqubits):
                            inds.append(tmp % 3)
                            tmp //= 3
                        inds = inds  # little-endian w.r.t. the 3**np.arange mapping above

                        rotations = tuple(self.PAULI_ROTATIONS_LIST[i] for i in inds)

                        if self.rho is None:
                            # Pure state path: probs = |U |psi>|^2
                            Upsi = multikron(*rotations) @ self.state
                            probs = np.abs(Upsi) ** 2
                            probs = probs / probs.sum()
                        else:
                            # Mixed state path: probs = diag(U rho U†)
                            probs = self.probs_from_rho_and_setting(rotations)

                        probs_cache[int(uk)] = probs

                    # Save probability cache
                    np.savez(
                        probs_file,
                        keys=np.array(list(probs_cache.keys()), dtype=np.int64),
                        probs=np.array(list(probs_cache.values()), dtype=np.float64),
                    )

                    print(f"Saved probs_cache to {probs_file}")

                else:
                    if not os.path.exists(probs_file):
                        raise FileNotFoundError(
                            f"Probability cache not found: {probs_file}"
                        )

                    data = np.load(probs_file)
                    probs_keys = data["keys"]
                    probs_values = data["probs"]

                    probs_cache = {
                        int(k): p for k, p in zip(probs_keys, probs_values)
                    }

                    assert len(uniq_keys) == len(probs_cache), "Len must match"

                    for uk in uniq_keys:
                        if uk not in probs_cache:
                            raise KeyError(
                                f"Probability for measurement setting {int(keys[i])} "
                                f"not found in cache {probs_file}"
                            )                    
                    
                    print(f"Loaded probs_cache from {probs_file}")

                elapsed = time.perf_counter() - start
                print(f"Elapsed time computing/loading probs: {elapsed:.6f} s")       

                start = time.perf_counter()
                # Sample outcomes shot-by-shot using cached probs
                for i in range(B):
                    probs = probs_cache[int(keys[i])]
                    outcome = rng2.choice(D, p=probs)
                    clicks_per_qubit[i, :] = bits_from_int(int(outcome), self.nqubits)
                elapsed = time.perf_counter() - start
                print(f"Elapsed time sampling outcomes: {elapsed:.6f} s")                         

                # Save batch
                ind_df = pd.DataFrame(inds_per_qubit, columns=[f"ind{i}" for i in range(self.nqubits)])
                click_df = pd.DataFrame(clicks_per_qubit, columns=[f"qb{i}" for i in range(self.nqubits)])
                data = pd.concat([ind_df, click_df], axis=1)

                if b != 0:
                    print("WARNING: batch support has been removed!")

                inds_np = ind_df.to_numpy(dtype=np.int32).flatten()
                clicks_np = click_df.to_numpy(dtype=np.int32).flatten()       

            elif self.name == "mm":
                inds_per_qubit = rng.integers(0, 3, size=(B, self.nqubits), endpoint=False)      
                ind_df = pd.DataFrame(inds_per_qubit, columns=[f"ind{i}" for i in range(self.nqubits)])

                clicks_mat = rand_clicks(self.nqubits, B)

                inds_np = ind_df.to_numpy(dtype=np.int32).flatten()
                clicks_np = (clicks_mat.astype(np.int32)).flatten()       

            max_order = self.order
            if self.name == "rankX":
                max_order = self.n_star
            moments = np.zeros(max_order, dtype=np.float64)
            moments_pt = np.zeros(max_order, dtype=np.float64)
            moments[0] = np.sum(self.eigenvalues)
            moments_pt[0] = np.sum(self.eigenvalues_pt)
            QFI = np.array([-1.0], dtype=np.float64)
            if self.name == "ghz" or self.name == "werner" or self.name == "rankX":
                for n in range(2, max_order + 1):
                    moments[n - 1] = np.sum(self.eigenvalues**(n))
                    moments_pt[n - 1] = np.sum(self.eigenvalues_pt**(n))

                if self.name == "rankX":
                    QFI[0] = self.QFI

            elif self.name == "mm":
                for n in range(1, max_order + 1):
                    moments[n - 1] = 2**(self.nqubits*(1 - n))
                    moments_pt[n - 1] = 0

            werner = np.array([1], dtype=np.int32) if self.name == "werner" else np.array([0], dtype=np.int32)

            zero_digit = "0" if self.nqubits < 10 else ""
            np.savez(f"{self.path}/qb_{zero_digit}{self.nqubits}.npz", 
                    num_qubits=np.array([self.nqubits], dtype=np.int32), # number of qubits
                    num_meas=np.array([B], dtype=np.int32), # number of total measurements, WARNING: this risks overflowing! (~2e9)
                    max_order=np.array([max_order], dtype=np.int32), # max moment order that was computed
                    pt=self.pt, # whether we have pt data
                    werner=werner, # whether we are a werner state
                    werner_par=np.array([self.t], dtype=np.float64), # t par of werner state
                    transposes=self.transposes, # tranpositions that were performed 
                    moments=moments, # correct moments based on eigenvalues
                    moments_pt=moments_pt, # correct pt moments of above given tranpositions based on eigenvalues
                    QFI=QFI, # correct QFI
                    inds=inds_np, # unitary indices
                    clicks=clicks_np, # measurements results
            )

        print(f"Data saved in {self.path}")

        
    
    def get_probs(self):
        """
        Returns probabilities for all possible measurement outcomes
        """
        self.all_probs = [
            np.abs(multikron(*np.take(self.PAULI_ROTATIONS_LIST, inds, axis=0)) @ self.state) ** 2
            for inds in product(range(3), repeat=self.nqubits)
        ]

    def draw_inds_clicks(self):
        """"
        Draw measurement outcomes based on the probabilities
        """
        rng = np.random.default_rng(self.run_id)
        all_inds = rng.integers(0, 3 ** self.nqubits, size = self.shots)
        
        clicks_binary = np.array(rng.multinomial(1, np.take(self.all_probs, all_inds, axis=0), size = self.shots), dtype=bool)

        change_to_fock = np.tile(list(product(range(2), repeat = self.nqubits)), (self.shots, 1)).reshape(self.shots, 2 ** self.nqubits, self.nqubits)
        all_clicks_per_qubit = np.hstack(change_to_fock[clicks_binary].reshape(self.shots, self.nqubits)).reshape(self.shots, self.nqubits)
        all_inds_per_qubit = np.take(list(product(range(3), repeat = self.nqubits)), all_inds, axis=0)
        return all_inds_per_qubit, all_clicks_per_qubit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulating measurement data')
    parser.add_argument("-qb", '--nqubits', type=int, help='Number of qubits', default=4)
    parser.add_argument('--shots', type=int, help='Number of shots', default=int(1e4))
    parser.add_argument("-s", '--state', type=str, help='Input state vector', default='ghz', choices=['ghz', 'bell', 'werner', 'mm', 'rankX'])
    parser.add_argument("-t", '--t', type=float, help='t parameter for werner states', default=0.0)
    parser.add_argument("-n", '--num_samples', type=int, help='number of samples', default=1)
    parser.add_argument("-o", '--order', type=int, help='max order to compute moments for', default=16)
    parser.add_argument("-p", '--path', type=str, help='path to store results', default="data")
    parser.add_argument("-c", '--compute_probs', type=int, help='whether to compute or load probs', default=1)
    parser.add_argument("-pp", '--probs_dir', type=str, help='path to store or load probs', default="None")
    parser.add_argument("-r", '--rank', type=int, help='rank of rankX state', default=2)

    args = parser.parse_args()

    if args.state not in ['ghz','bell', 'werner', 'mm', 'rankX']:
        raise ValueError('Invalid state vector')
    
    if args.shots > int(2e9):
        raise ValueError(f"num_meas is currently stored as int32, {args.shots} is too much")
    print()
    print(f"Starting simulation of {args.state} state with {args.nqubits} qubits and {args.shots} shots...")
    for i in range(args.num_samples):
        print(f"Run {i+1}/{args.num_samples}")
        sim = Simulator(args.nqubits, args.shots, args.order, name=args.state, t=args.t, run_id=i, path=args.path, compute_probs=args.compute_probs, probs_dir=args.probs_dir, rank=args.rank)
        sim.simulate(batchsize=args.shots)