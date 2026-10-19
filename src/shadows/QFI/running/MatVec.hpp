#pragma once

#include "KrMV/KrMV.hpp"
#include <cstdint>

namespace shadows {

namespace QFI {

namespace running {

// scalar
// performs full KrMV recurrence (for one row)
template <std::size_t num_qubits>
void KrMV_recurrence(z_t* __restrict__ const Pmats, const int32_t* __restrict__ const inds,
                     const int32_t* __restrict__ const clicks, const std::size_t row, const std::size_t ridx,
                     const std::size_t num_meas, const std::size_t order, const std::size_t num_data_points,
                     double* __restrict__ const inter_traces, const std::size_t* __restrict__ const inter_tidx) {

  constexpr std::size_t ss_dim = std::size_t{1} << num_qubits; // 2**num_qubits, state-space dimension

  std::fill(Pmats, Pmats + (order * ss_dim), z_t{0.0, 0.0}); // memset is sketchy for std::complex

  std::array<int32_t, num_qubits> transposes_arr = {0}; // to conform with KrMV implementation
  alignas(ALIGNMENT) std::array<z_t, ss_dim> buffer_arr0 = {z_t{0.0, 0.0}};
  alignas(ALIGNMENT) std::array<z_t, ss_dim> buffer_arr1 = {z_t{0.0, 0.0}};
  alignas(ALIGNMENT) std::array<z_t, ss_dim> buffer_arr2 = {z_t{0.0, 0.0}};
  alignas(ALIGNMENT) std::array<z_t, ss_dim> buffer_arr3 = {z_t{0.0, 0.0}};
  const int32_t* const transposes = transposes_arr.data();
  z_t* buff0 = buffer_arr0.data();
  z_t* buff1 = buffer_arr1.data();
  z_t* src = buffer_arr2.data();

  // Hbuff gets 1 row of Pauli-Z Hamiltonian
  src[row] = z_t{1.0, 0.0};
  KrMV_Z<num_qubits, true>(buffer_arr3.data(), src, 1.0, 1.0, inds, clicks, buff0, buff1);
  const z_t* const Hbuff = buffer_arr3.data();

  std::size_t dpt_idx = 0;

  // for each measurement
  for (std::size_t tidx = 0; tidx < num_meas; ++tidx) {

    // intermediate traces
    const bool get_inter_trace = (tidx == inter_tidx[dpt_idx]);
    double* const inter_traces_tidx = inter_traces + order * dpt_idx;
    if (get_inter_trace)
      ++dpt_idx;

    const int32_t* const inds_tidx = inds + num_qubits * tidx;
    const int32_t* const clicks_tidx = clicks + num_qubits * tidx;

    // set src buffer to Hamiltonian row
    std::memcpy(src, Hbuff, ss_dim * sizeof(z_t));

    // for each order
    for (std::size_t midx = 0; midx < order; ++midx) {

      if (tidx + midx + 1 <= num_meas) {

        z_t* const data_midx = Pmats + ss_dim * midx;
        const int32_t* const inds_midx = inds_tidx + num_qubits * midx;
        const int32_t* const clicks_midx = clicks_tidx + num_qubits * midx;

        const double alpha = static_cast<double>(midx + 1) / static_cast<double>(tidx + midx + 1);
        const double beta = static_cast<double>(tidx) / static_cast<double>(tidx + midx + 1);

        if (midx != ridx)
          KrMV<num_qubits>(data_midx, src, alpha, beta, transposes, inds_midx, clicks_midx, buff0);
        else
          KrMV_Z<num_qubits, false>(data_midx, src, alpha, beta, inds_midx, clicks_midx, buff0, buff1);

        if (get_inter_trace) // extract (real) trace of curr. batch of rows, assumes that traces is init. with all 0s
          inter_traces_tidx[midx] += src[row].real();
      }
    } // midx
  } // tidx
} // done

// vectorized
// performs full KrMV recurrence (for one row)
template <typename VecXd, std::size_t num_qubits>
void KrMV_recurrence(double* __restrict__ const Pmats, const int32_t* __restrict__ const inds,
                     const int32_t* __restrict__ const clicks, const std::size_t row, const std::size_t ridx,
                     const std::size_t num_meas, const std::size_t order, const std::size_t num_data_points,
                     double* __restrict__ const inter_traces, const std::size_t* __restrict__ const inter_tidx) {

  constexpr std::size_t ss_dim = std::size_t{1} << num_qubits; // 2**num_qubits, state-space dimension
  constexpr std::size_t Vsize = VecXd::size();

  const std::size_t total_size = 2 * order * ss_dim * sizeof(double);
  std::memset(Pmats, 0, total_size);

  std::array<int32_t, num_qubits> transposes_arr = {0}; // to conform with KrMV implementation
  alignas(ALIGNMENT) std::array<double, 2 * ss_dim> buffer_arr0 = {0.0};
  alignas(ALIGNMENT) std::array<double, 2 * ss_dim> buffer_arr1 = {0.0};
  alignas(ALIGNMENT) std::array<double, 2 * ss_dim> buffer_arr2 = {0.0};
  alignas(ALIGNMENT) std::array<double, 2 * ss_dim> buffer_arr3 = {0.0};
  const int32_t* const transposes = transposes_arr.data();
  double* buff0 = buffer_arr0.data();
  double* buff1 = buffer_arr1.data();
  double* src = buffer_arr2.data();

  const std::size_t tr_idx = row + Vsize * (row / Vsize);

  // Hbuff gets 1 row of Pauli-Z Hamiltonian
  src[tr_idx] = 1.0;
  KrMV_Z<VecXd, num_qubits, true>(buffer_arr3.data(), src, 1.0, 1.0, inds, clicks, buff0, buff1);
  const double* const Hbuff = buffer_arr3.data();

  std::size_t dpt_idx = 0;

  // for each measurement
  for (std::size_t tidx = 0; tidx < num_meas; ++tidx) {

    // intermediate traces
    const bool get_inter_trace = (tidx == inter_tidx[dpt_idx]);
    double* const inter_traces_tidx = inter_traces + order * dpt_idx;
    if (get_inter_trace)
      ++dpt_idx;

    const int32_t* const inds_tidx = inds + num_qubits * tidx;
    const int32_t* const clicks_tidx = clicks + num_qubits * tidx;

    // set src buffer to Hamiltonian row
    std::memcpy(src, Hbuff, 2 * ss_dim * sizeof(double));

    // for each order
    for (std::size_t midx = 0; midx < order; ++midx) {

      if (tidx + midx + 1 <= num_meas) {

        double* const data_midx = Pmats + 2 * ss_dim * midx;
        const int32_t* const inds_midx = inds_tidx + num_qubits * midx;
        const int32_t* const clicks_midx = clicks_tidx + num_qubits * midx;

        const double alpha = static_cast<double>(midx + 1) / static_cast<double>(tidx + midx + 1);
        const double beta = static_cast<double>(tidx) / static_cast<double>(tidx + midx + 1);

        if (midx != ridx)
          KrMV<VecXd, num_qubits>(data_midx, src, alpha, beta, transposes, inds_midx, clicks_midx, buff0);
        else
          KrMV_Z<VecXd, num_qubits, false>(data_midx, src, alpha, beta, inds_midx, clicks_midx, buff0, buff1);

        if (get_inter_trace) // extract (real) trace of curr. batch of rows, assumes that traces is init. with all 0s
          inter_traces_tidx[midx] += src[tr_idx];
      }
    } // midx
  } // tidx
} // done

} // namespace running

} // namespace QFI

} // namespace shadows
