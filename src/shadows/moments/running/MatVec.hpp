#pragma once

#include "KrMV/KrMV.hpp"

namespace shadows {

namespace moments {

namespace running {

// scalar
// performs full KrMV recurrence (for one row)
template <std::size_t num_qubits>
void KrMV_recurrence(z_t* __restrict__ const Pmats, double* __restrict__ const traces,
                     const int32_t* __restrict__ const inds, const int32_t* __restrict__ const clicks,
                     const int32_t* __restrict__ const transposes, const std::size_t row, const std::size_t num_meas,
                     const std::size_t order, const std::size_t tr_start, const std::size_t num_data_points,
                     double* __restrict__ const inter_traces, const std::size_t* __restrict__ const inter_tidx) {

  constexpr std::size_t ss_dim = std::size_t{1} << num_qubits; // 2**num_qubits, state-space dimension

  std::fill(Pmats, Pmats + (order * ss_dim), z_t{0.0, 0.0}); // memset is sketchy for std::complex

  alignas(ALIGNMENT) std::array<z_t, ss_dim> buffer_arr0 = {z_t{0.0, 0.0}};
  alignas(ALIGNMENT) std::array<z_t, ss_dim> buffer_arr1 = {z_t{0.0, 0.0}};
  z_t* const buffer0 = buffer_arr0.data();
  z_t* const buffer1 = buffer_arr1.data();

  z_t* src = buffer0;
  z_t* dest = buffer1;

  std::size_t dpt_idx = 0;

  // for each measurement
  for (std::size_t tidx = 0; tidx < num_meas; ++tidx) {

    // intermediate traces
    const bool get_inter_trace = num_data_points ? (tidx == inter_tidx[dpt_idx]) : false;
    double* const inter_traces_tidx = num_data_points ? inter_traces + order * dpt_idx : nullptr;
    if (get_inter_trace)
      ++dpt_idx;

    const bool get_trace = (tidx + order >= tr_start); // whether to extract results
    double* const traces_tidx = traces + order * (tidx + order - tr_start);
    const int32_t* const inds_tidx = inds + num_qubits * tidx;
    const int32_t* const clicks_tidx = clicks + num_qubits * tidx;

    // set src buffer to identity matrix, necessary for order 1
    std::fill(src, src + ss_dim, z_t{0.0, 0.0}); // memset is sketchy for std::complex
    src[row] = z_t{1.0, 0.0};

    // for each order
    for (std::size_t midx = 0; midx < order; ++midx) {

      if (tidx + midx + 1 <= num_meas) {

        z_t* const data_midx = Pmats + ss_dim * midx;
        const int32_t* const inds_midx = inds_tidx + num_qubits * midx;
        const int32_t* const clicks_midx = clicks_tidx + num_qubits * midx;

        const double alpha = static_cast<double>(midx + 1) / static_cast<double>(tidx + midx + 1);
        const double beta = static_cast<double>(tidx) / static_cast<double>(tidx + midx + 1);

        KrMV<num_qubits>(data_midx, src, alpha, beta, transposes, inds_midx, clicks_midx, dest);

        if (get_trace) // extract (real) trace of curr. batch of rows, assumes that traces is init. with all 0s
          traces_tidx[midx] += src[row].real();

        if (get_inter_trace) // extract (real) trace of curr. batch of rows, assumes that traces is init. with all 0s
          inter_traces_tidx[midx] += src[row].real();
      }
    } // midx
  }   // tidx
} // done

// vectorized
// performs full KrMV recurrence (for one row)
template <typename VecXd, std::size_t num_qubits>
void KrMV_recurrence(double* __restrict__ const Pmats, double* __restrict__ const traces,
                     const int32_t* __restrict__ const inds, const int32_t* __restrict__ const clicks,
                     const int32_t* __restrict__ const transposes, const std::size_t row, const std::size_t num_meas,
                     const std::size_t order, const std::size_t tr_start, const std::size_t num_data_points,
                     double* __restrict__ const inter_traces, const std::size_t* __restrict__ const inter_tidx) {

  constexpr std::size_t ss_dim = std::size_t{1} << num_qubits; // 2**num_qubits, state-space dimension
  constexpr std::size_t Vsize = VecXd::size();

  const std::size_t total_size = 2 * order * ss_dim * sizeof(double);
  std::memset(Pmats, 0, total_size);

  alignas(ALIGNMENT) std::array<double, 2 * ss_dim> buffer_arr0 = {0.0};
  alignas(ALIGNMENT) std::array<double, 2 * ss_dim> buffer_arr1 = {0.0};
  double* const buffer0 = buffer_arr0.data();
  double* const buffer1 = buffer_arr1.data();

  double* src = buffer0;
  double* dest = buffer1;

  std::size_t dpt_idx = 0;

  // for each measurement
  for (std::size_t tidx = 0; tidx < num_meas; ++tidx) {

    // intermediate traces
    const bool get_inter_trace = num_data_points ? (tidx == inter_tidx[dpt_idx]) : false;
    double* const inter_traces_tidx = num_data_points ? inter_traces + order * dpt_idx : nullptr;
    if (get_inter_trace)
      ++dpt_idx;

    // final traces
    const bool get_trace = (tidx + order >= tr_start); // whether to extract results
    double* const traces_tidx = traces + order * (tidx + order - tr_start);

    const std::size_t tr_idx = row + Vsize * (row / Vsize);

    const int32_t* const inds_tidx = inds + num_qubits * tidx;
    const int32_t* const clicks_tidx = clicks + num_qubits * tidx;

    // set src buffer to identity matrix, necessary for order 1
    std::memset(src, 0.0, 2 * ss_dim * sizeof(double));
    src[tr_idx] = 1.0;

    // for each order
    for (std::size_t midx = 0; midx < order; ++midx) {

      if (tidx + midx + 1 <= num_meas) {

        double* const data_midx = Pmats + 2 * ss_dim * midx;
        const int32_t* const inds_midx = inds_tidx + num_qubits * midx;
        const int32_t* const clicks_midx = clicks_tidx + num_qubits * midx;

        const double alpha = static_cast<double>(midx + 1) / static_cast<double>(tidx + midx + 1);
        const double beta = static_cast<double>(tidx) / static_cast<double>(tidx + midx + 1);

        KrMV<VecXd, num_qubits>(data_midx, src, alpha, beta, transposes, inds_midx, clicks_midx, dest);

        if (get_trace) // extract (real) trace of curr. batch of rows, assumes that traces is init. with all 0s
          traces_tidx[midx] += data_midx[tr_idx];

        if (get_inter_trace) // extract (real) trace of curr. batch of rows, assumes that traces is init. with all 0s
          inter_traces_tidx[midx] += data_midx[tr_idx];
      }
    } // midx
  }   // tidx
} // done

} // namespace running

} // namespace moments

} // namespace shadows
