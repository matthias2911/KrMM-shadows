#pragma once

#include "KrMV/KrMV.hpp"

namespace shadows {

namespace moments {

namespace batched {

// scalar
// performs full KrMV recurrence (for one row)
template <std::size_t num_qubits>
void KrMV_recurrence(z_t* __restrict__ const Pmats_row, const int32_t* __restrict__ const inds,
                     const int32_t* __restrict__ const clicks, const int32_t* __restrict__ const transposes,
                     const std::size_t row, const std::size_t num_meas, const std::size_t order) {

  constexpr std::size_t ss_dim = std::size_t{1} << num_qubits; // 2**num_qubits, state-space dimension

  alignas(ALIGNMENT) std::array<z_t, ss_dim> buffer_arr0 = {z_t{0.0, 0.0}};
  alignas(ALIGNMENT) std::array<z_t, ss_dim> buffer_arr1 = {z_t{0.0, 0.0}};
  alignas(ALIGNMENT) std::array<z_t, ss_dim> id_arr = {z_t{0.0, 0.0}};
  z_t* const buffer0 = buffer_arr0.data();
  z_t* const buffer1 = buffer_arr1.data();
  const z_t* const id_buffer = id_arr.data();

  // set id_arr buffer to identity matrix, necessary for order 1
  id_arr[row] = z_t{1.0, 0.0};

  z_t* src = buffer0;
  z_t* dest = buffer1;

  // for each batch
  for (std::size_t bidx = 0; bidx < order; ++bidx) {

    const std::size_t num_meas_batch = num_meas / order;
    const std::size_t tidx_start = num_meas_batch * bidx;
    const std::size_t tidx_end = tidx_start + num_meas_batch + (bidx == order - 1 ? num_meas % order : 0);

    z_t* const Pmats_batch = Pmats_row + ss_dim * bidx;

    for (std::size_t tidx = tidx_start; tidx < tidx_end; ++tidx) {

      const int32_t* const inds_tidx = inds + num_qubits * tidx;
      const int32_t* const clicks_tidx = clicks + num_qubits * tidx;

      const double alpha = static_cast<double>(1) / static_cast<double>(tidx - tidx_start + 1);
      const double beta = static_cast<double>(tidx - tidx_start) / static_cast<double>(tidx - tidx_start + 1);

      KrMV<num_qubits>(Pmats_batch, id_buffer, alpha, beta, transposes, inds_tidx, clicks_tidx, dest, src);

    } // tidx
  } // bidx
} // done

// vectorized
// performs full KrMV recurrence (for one row)
template <typename VecXd, std::size_t num_qubits>
void KrMV_recurrence(double* __restrict__ const Pmats_row, const int32_t* __restrict__ const inds,
                     const int32_t* __restrict__ const clicks, const int32_t* __restrict__ const transposes,
                     const std::size_t row, const std::size_t num_meas, const std::size_t order) {

  constexpr std::size_t ss_dim = std::size_t{1} << num_qubits; // 2**num_qubits, state-space dimension
  constexpr std::size_t Vsize = VecXd::size();

  alignas(ALIGNMENT) std::array<double, 2 * ss_dim> buffer_arr0 = {0.0};
  alignas(ALIGNMENT) std::array<double, 2 * ss_dim> buffer_arr1 = {0.0};
  alignas(ALIGNMENT) std::array<double, 2 * ss_dim> id_arr = {0.0};
  double* const buffer0 = buffer_arr0.data();
  double* const buffer1 = buffer_arr1.data();
  const double* const id_buffer = id_arr.data();

  // set id_arr buffer to identity matrix, necessary for order 1
  const std::size_t tr_idx = row + Vsize * (row / Vsize);
  id_arr[tr_idx] = 1.0;

  double* src = buffer0;
  double* dest = buffer1;

  // for each batch
  for (std::size_t bidx = 0; bidx < order; ++bidx) {

    const std::size_t num_meas_batch = num_meas / order;
    const std::size_t tidx_start = num_meas_batch * bidx;
    const std::size_t tidx_end = tidx_start + num_meas_batch + (bidx == order - 1 ? num_meas % order : 0);

    double* const Pmats_batch = Pmats_row + 2 * ss_dim * bidx;

    for (std::size_t tidx = tidx_start; tidx < tidx_end; ++tidx) {

      const int32_t* const inds_tidx = inds + num_qubits * tidx;
      const int32_t* const clicks_tidx = clicks + num_qubits * tidx;

      const double alpha = static_cast<double>(1) / static_cast<double>(tidx - tidx_start + 1);
      const double beta = static_cast<double>(tidx - tidx_start) / static_cast<double>(tidx - tidx_start + 1);

      KrMV<VecXd, num_qubits>(Pmats_batch, id_buffer, alpha, beta, transposes, inds_tidx, clicks_tidx, dest, src);

    } // tidx
  } // bidx
} // done

} // namespace batched

} // namespace moments

} // namespace shadows
