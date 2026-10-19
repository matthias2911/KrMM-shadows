#pragma once

#include "constants.hpp"

#include <iostream>

#include <cassert>
#include <cblas.h>

namespace shadows {

namespace moments {

namespace batched {

// for reshaping output of scalar version
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

template <std::size_t num_qubits>
void post_process(double* __restrict__ const moments, z_t* __restrict__ const batches, const std::size_t order,
                  const std::size_t num_meas) {
  constexpr std::size_t ss_dim = std::size_t{1} << num_qubits;
  constexpr std::size_t matrix_size = ss_dim * ss_dim;
  constexpr std::size_t total_size = 2 * matrix_size * sizeof(z_t);

  assert(total_size % ALIGNMENT == 0);
  z_t* const mat_buffs = static_cast<z_t*>(std::aligned_alloc(ALIGNMENT, total_size));
  std::fill(mat_buffs, mat_buffs + 2 * ss_dim * ss_dim, z_t{0.0, 0.0}); // memset is sketchy for std::complex

  std::vector<std::size_t> num_snaps(order, 0);
  for (std::size_t midx = 0; midx < order; ++midx)
    num_snaps[midx] = num_meas / order + (midx == order - 1 ? num_meas % order : 0);

  const z_t alpha{1.0, 0.0};
  const z_t beta{0.0, 0.0};

  std::size_t start = 0;

  for (std::size_t m = order; m > 0; --m) {

    z_t trace{0.0, 0.0};

    if (m == 1) {
      for (std::size_t idx = 0; idx < ss_dim; ++idx)
        trace += batches[ss_dim * idx + idx];

    } else {

      z_t* src = mat_buffs;
      z_t* dest = mat_buffs + matrix_size;

      multiply(batches, batches + matrix_size, src, alpha, beta, static_cast<int>(ss_dim));

      for (std::size_t idx = 2; idx < m; ++idx) {
        multiply(src, batches + matrix_size * idx, dest, alpha, beta, static_cast<int>(ss_dim));
        std::swap(src, dest);
      }

      for (std::size_t idx = 0; idx < ss_dim; ++idx)
        trace += src[ss_dim * idx + idx];
    }

    moments[m - 1] = trace.real();

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

  free(mat_buffs);
}

} // namespace batched

} // namespace moments

} // namespace shadows
