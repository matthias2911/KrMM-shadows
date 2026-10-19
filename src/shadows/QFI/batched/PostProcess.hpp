#pragma once

#include "constants.hpp"

#include "KrMV/KrMV.hpp"

#include <cassert>
#include <iostream>

#include <cblas.h>

namespace shadows {

namespace QFI {

namespace batched {

// NOTICE
// this function is copy pasted from moments/batched/PostProcess.hpp
template <std::size_t num_qubits>
void reshape(z_t* __restrict__ const batches, const z_t* __restrict__ const Pmats, const std::size_t order) {

  constexpr std::size_t ss_dim = std::size_t{1} << num_qubits; // 2**num_qubits, state-space dimension

  for (std::size_t midx = 0; midx < order; ++midx) {

    z_t* const batches_midx = batches + ss_dim * ss_dim * midx;
    const z_t* const Pmats_midx = Pmats + ss_dim * midx;
    for (std::size_t row = 0; row < ss_dim; ++row) {

      z_t* const batches_row = batches_midx + ss_dim * row;
      const z_t* const Pmats_row = Pmats_midx + order * ss_dim * row;
      for (std::size_t col = 0; col < ss_dim; ++col) {

        batches_row[col] = Pmats_row[col];
      } // col
    } // row
  } // midx
} // done

// for reshaping output of vectorized version
template <typename VecXd, std::size_t num_qubits>
void reshape(z_t* __restrict__ const batches, const double* __restrict__ const Pmats, const std::size_t order) {

  constexpr std::size_t ss_dim = std::size_t{1} << num_qubits; // 2**num_qubits, state-space dimension
  constexpr std::size_t Vsize = VecXd::size();

  for (std::size_t midx = 0; midx < order; ++midx) {

    z_t* const batches_midx = batches + ss_dim * ss_dim * midx;
    const double* const Pmats_midx = Pmats + 2 * ss_dim * midx;
    for (std::size_t row = 0; row < ss_dim; ++row) {

      z_t* const batches_row = batches_midx + ss_dim * row;
      const double* const Pmats_row = Pmats_midx + 2 * order * ss_dim * row;
      for (std::size_t col = 0; col < ss_dim; ++col) {

        const std::size_t idx = col + Vsize * (col / Vsize);
        batches_row[col] = z_t{Pmats_row[idx], Pmats_row[idx + Vsize]};
      } // col
    } // row
  } // midx
} // done

double binom_stable(std::size_t k, long long l) {
  if (l < 0 || l > static_cast<long long>(k)) {
    return 0.0;
  }
  long long kk = static_cast<long long>(k);
  long long r = (l < kk - l) ? l : (kk - l); // use smaller side for fewer multiplications
  double result = 1.0;
  for (long long i = 0; i < r; ++i) {
    result *= static_cast<double>(kk - i);
    result /= static_cast<double>(i + 1);
  }
  return result;
}

double Tval_fac(const std::size_t kidx, const std::size_t lidx) {
  const long long l = static_cast<long long>(lidx);
  const std::size_t k = kidx;

  const double term = binom_stable(k, l) - 2.0 * binom_stable(k, l - 1) + binom_stable(k, l - 2);
  return term / std::pow(2.0, static_cast<double>(k));
}

void multiply(const z_t* A, const z_t* B, z_t* C, const z_t alpha, const z_t beta, const int ss_dim) {
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              ss_dim,            // M
              ss_dim,            // N
              ss_dim,            // K
              &alpha, A, ss_dim, // A, lda
              B, ss_dim,         // B, ldb
              &beta, C, ss_dim   // C, ldc
  );
}

template <std::size_t num_qubits> std::vector<z_t> get_pauli_Z_hamiltonian() {

  constexpr std::size_t ss_dim = std::size_t{1} << num_qubits;
  constexpr std::size_t matrix_size = ss_dim * ss_dim;

  alignas(ALIGNMENT) std::array<z_t, ss_dim> buffer_arr0 = {z_t{0.0, 0.0}};
  alignas(ALIGNMENT) std::array<z_t, ss_dim> buffer_arr1 = {z_t{0.0, 0.0}};
  alignas(ALIGNMENT) std::array<z_t, ss_dim> buffer_arr2 = {z_t{0.0, 0.0}};
  std::vector<z_t> hamiltonian(matrix_size, z_t{0.0, 0.0});

  z_t* buff0 = buffer_arr0.data();
  z_t* buff1 = buffer_arr1.data();
  z_t* src = buffer_arr2.data();

  for (std::size_t row = 0; row < ss_dim; ++row) {

    z_t* const Hbuff = hamiltonian.data() + ss_dim * row;

    std::fill(src, src + ss_dim, z_t{0.0, 0.0}); // memset is sketchy for std::complex
    src[row] = z_t{1.0, 0.0};
    KrMV_Z<num_qubits, true>(Hbuff, src, 1.0, 1.0, nullptr, nullptr, buff0, buff1);
  }
  return hamiltonian;
}

double chain_multiply_trace(const z_t* batches, const z_t* hamiltonian, z_t* src, z_t* dest, const z_t alpha,
                            const z_t beta, const int ss_dim, const std::size_t m) {

  const int matrix_size = ss_dim * ss_dim;

  const std::size_t kidx = m - 2;

  double Tval = 0.0;

  for (std::size_t lidx = 0; lidx < kidx + 3; ++lidx) {

    z_t trace{0.0, 0.0};

    const std::size_t ridx = std::min(lidx, kidx + 2 - lidx);

    std::size_t bidx = 0;

    if (ridx == 0) {
      std::memcpy(dest, hamiltonian, matrix_size * sizeof(z_t));
      multiply(dest, hamiltonian, src, alpha, beta, static_cast<int>(ss_dim));
    } else {
      std::memcpy(dest, hamiltonian, matrix_size * sizeof(z_t));
      multiply(dest, batches + matrix_size * bidx, src, alpha, beta, static_cast<int>(ss_dim));
      ++bidx;
    }

    for (std::size_t idx = 1; idx <= m; ++idx) {

      if (idx == ridx) {
        multiply(src, hamiltonian, dest, alpha, beta, static_cast<int>(ss_dim));
        std::swap(src, dest);
        continue;
      }

      multiply(src, batches + matrix_size * bidx, dest, alpha, beta, static_cast<int>(ss_dim));
      std::swap(src, dest);
      ++bidx;
    }

    for (std::size_t idx = 0; idx < ss_dim; ++idx)
      trace += src[ss_dim * idx + idx];

    Tval += Tval_fac(kidx, lidx) * trace.real();
  }

  return Tval;
}

void combine_batches(z_t* batches, std::vector<std::size_t>& num_snaps, std::size_t& start, const int ss_dim,
                     const std::size_t m) {

  const int matrix_size = ss_dim * ss_dim;

  // combine batches as equally as possible for lower orders
  if (m > 1) {
    std::size_t end = m - 1;

    z_t* src = batches + matrix_size * end;
    z_t* dest = batches + matrix_size * start;

    const double src_ns = static_cast<double>(num_snaps[end]);
    const double dest_ns = static_cast<double>(num_snaps[start]);
    const double s_ns = static_cast<double>(num_snaps[start] + num_snaps[end]);

    const z_t src_fac{src_ns / s_ns, 0.0};
    const z_t dest_fac{dest_ns / s_ns, 0.0};

    cblas_zscal(static_cast<int>(matrix_size), &dest_fac, dest, 1);        // dest *= dest_fac
    cblas_zaxpy(static_cast<int>(matrix_size), &src_fac, src, 1, dest, 1); // dest += src_fac * src
    num_snaps[start] = num_snaps[start] + num_snaps[end];

    if (start + 1 < end - 1)
      ++start;
    else
      start = 0;
  }
}

template <std::size_t num_qubits>
void post_process(double* __restrict__ const Tvals, z_t* __restrict__ const batches, const std::size_t n_star,
                  const std::size_t num_meas) {
  constexpr std::size_t ss_dim = std::size_t{1} << num_qubits;
  constexpr std::size_t matrix_size = ss_dim * ss_dim;
  constexpr std::size_t total_size = 2 * matrix_size * sizeof(z_t);

  // batches: num_data_points*order*ridx + order*didx + midx
  // Tvals: kidx

  const std::size_t k_order = 2 * n_star;
  const std::size_t order = 2 * n_star + 1;
  const std::size_t num_recs = n_star + 1;

  std::memset(Tvals, 0, k_order * sizeof(double));

  assert(total_size % ALIGNMENT == 0);
  z_t* const mat_buffs = static_cast<z_t*>(std::aligned_alloc(ALIGNMENT, total_size));
  std::fill(mat_buffs, mat_buffs + 2 * ss_dim * ss_dim, z_t{0.0, 0.0}); // memset is sketchy for std::complex
  z_t* src = mat_buffs;
  z_t* dest = mat_buffs + matrix_size;

  std::vector<std::size_t> num_snaps(order, 0);
  for (std::size_t midx = 0; midx < order; ++midx)
    num_snaps[midx] = num_meas / order + (midx == order - 1 ? num_meas % order : 0);

  std::vector<z_t> hamiltonian_vec = get_pauli_Z_hamiltonian<num_qubits>();
  const z_t* const hamiltonian = hamiltonian_vec.data();

  const z_t alpha{1.0, 0.0};
  const z_t beta{0.0, 0.0};

  std::size_t start = 0;

  for (std::size_t m = order; m > 1; --m) {

    Tvals[m - 2] = chain_multiply_trace(batches, hamiltonian, src, dest, alpha, beta, ss_dim, m);
    combine_batches(batches, num_snaps, start, ss_dim, m);
  }

  free(mat_buffs);
}

} // namespace batched

} // namespace QFI

} // namespace shadows
