#pragma once

#include "constants.hpp"
#include <iostream>

namespace shadows {

namespace moments {

namespace running {

void reduce(double* __restrict__ const traces_shared, const std::size_t order, const std::size_t len_var_meas,
            const std::size_t num_threads) {

  const std::size_t tvm_per_thread = order * len_var_meas;

  // reduce
  for (std::size_t tid = 1; tid < num_threads; ++tid) {

    const double* const traces_tid = traces_shared + tvm_per_thread * tid;

    for (std::size_t vidx = 0; vidx < len_var_meas; ++vidx) {

      const double* const traces_vidx = traces_tid + order * vidx;
      double* const traces = traces_shared + order * vidx;

      for (std::size_t midx = 0; midx < order; ++midx) {

        traces[midx] += traces_vidx[midx];
      }
    }
  }
}

// performs the post-processing of collected traces to extract std. errs. and moments estimates
void post_process(double* __restrict__ const moments, double* __restrict__ const standard_errs,
                  double* __restrict__ const traces_shared, const std::size_t num_meas, const std::size_t order,
                  const std::size_t len_var_meas, const std::size_t tr_start) {

  // extract moment estimates
  for (std::size_t midx = 0; midx < order; ++midx)
    moments[midx] = traces_shared[order * (len_var_meas - midx - 1) + midx];

  const std::size_t new_len_var_meas = len_var_meas - 1; // because of scan pattern
  std::vector<double> traces(order * new_len_var_meas, 0.0);

  // transform to norm. innovations
  for (std::size_t vidx = 1; vidx < len_var_meas; ++vidx) {

    for (std::size_t midx = 0; midx < order; ++midx) {

      if (vidx + midx < len_var_meas) {

        const std::size_t tidx = tr_start - order + vidx;
        const double norm1 = static_cast<double>(tidx + midx + 1) / static_cast<double>(midx + 1);
        const double norm2 = static_cast<double>(tidx) / static_cast<double>(midx + 1);
        double inov = norm1 * traces_shared[order * vidx + midx] - norm2 * traces_shared[order * (vidx - 1) + midx];
        traces[order * (vidx - 1) + midx] = inov;
      }
    }
  }

  // N = new_len_var_meas
  // m = order

  std::memset(standard_errs, 0.0, order * sizeof(double));
  std::vector<double> means(order, 0.0);

  // welfords
  for (std::size_t vidx = 0; vidx < new_len_var_meas; ++vidx) {

    const double* const traces_vidx = traces.data() + order * vidx;
    const double n = static_cast<double>(vidx + 1);

    for (std::size_t midx = 0; midx < order; ++midx) {

      if (vidx + midx < new_len_var_meas) {

        const double delta = traces_vidx[midx] - means[midx];
        means[midx] += delta / n;
        const double delta2 = traces_vidx[midx] - means[midx];
        standard_errs[midx] += delta * delta2; // M2 accumulator
      }
    }
  }

  for (std::size_t midx = 0; midx < order; ++midx) {
    // one -1 from exclusive scan, the other -1 to obtain the unbiased variance
    const double denom = static_cast<double>(new_len_var_meas - midx - 1);
    standard_errs[midx] /= denom;
  }

  for (std::size_t midx = 0; midx < order; ++midx) {

    // actual order
    const std::size_t nidx = midx + 1;
    const double nidx_d = static_cast<double>(nidx);

    // not each order considers the same num_meas at the end
    const double num_meas_d = static_cast<double>(num_meas);

    // sigma parameter (leading term) of U-statistics variance
    const double sigma = standard_errs[midx];

    // estimated variance
    const double variance = nidx_d * nidx_d * sigma / num_meas_d;

    // estimated standard error
    standard_errs[midx] = std::sqrt(variance);
  }
}

// performs the post-processing of collected traces to extract std. errs. and moments estimates
void post_process_inter(double* __restrict__ const inter_moments, double* __restrict__ const inter_standard_errs,
                        const double* __restrict__ const inter_traces, std::size_t* __restrict__ const inter_tidx,
                        const std::size_t order, const std::size_t var_offset, const std::size_t num_data_points) {

  for (std::size_t didx = 0; didx < num_data_points; ++didx) {

    double* const moments_didx = inter_moments + order * didx;
    double* const std_errs_didx = inter_standard_errs + order * didx;
    const double* const traces_didx = inter_traces + var_offset * order * didx;
    const std::size_t* const tidx_didx = inter_tidx + var_offset * didx;

    // extract moment estimates
    for (std::size_t midx = 0; midx < order; ++midx)
      moments_didx[midx] = traces_didx[order * (var_offset - 1) + midx];

    const std::size_t new_len_var_meas = var_offset - 1; // because of scan pattern
    std::vector<double> traces(order * new_len_var_meas, 0.0);

    // transform to norm. innovations
    for (std::size_t vidx = 1; vidx < var_offset; ++vidx) {

      for (std::size_t midx = 0; midx < order; ++midx) {

        const std::size_t tidx = tidx_didx[vidx];
        const double norm1 = static_cast<double>(tidx + midx + 1) / static_cast<double>(midx + 1);
        const double norm2 = static_cast<double>(tidx) / static_cast<double>(midx + 1);
        double inov = norm1 * traces_didx[order * vidx + midx] - norm2 * traces_didx[order * (vidx - 1) + midx];
        traces[order * (vidx - 1) + midx] = inov;
      }
    }

    // N = new_len_var_meas
    // m = order

    std::memset(std_errs_didx, 0.0, order * sizeof(double));
    std::vector<double> means(order, 0.0);

    // welfords
    for (std::size_t vidx = 0; vidx < new_len_var_meas; ++vidx) {

      const double* const traces_vidx = traces.data() + order * vidx;
      const double n = static_cast<double>(vidx + 1);
      for (std::size_t midx = 0; midx < order; ++midx) {

        const double delta = traces_vidx[midx] - means[midx];
        means[midx] += delta / n;
        const double delta2 = traces_vidx[midx] - means[midx];
        std_errs_didx[midx] += delta * delta2; // M2 accumulator
      }
    }

    for (std::size_t midx = 0; midx < order; ++midx) {
      // one -1 from exclusive scan, the other -1 to obtain the unbiased variance
      const double denom = static_cast<double>(new_len_var_meas - midx - 1);
      std_errs_didx[midx] /= denom;
    }

    const std::size_t num_meas_dpt = tidx_didx[var_offset - 1];

    for (std::size_t midx = 0; midx < order; ++midx) {

      // actual order
      const std::size_t nidx = midx + 1;
      const double nidx_d = static_cast<double>(nidx);

      // not each order considers the same num_meas at the end
      const double num_meas_d = static_cast<double>(num_meas_dpt + midx + 1);

      // sigma parameter (leading term) of U-statistics variance
      const double sigma = std_errs_didx[midx];

      // estimated variance
      const double variance = nidx_d * nidx_d * sigma / num_meas_d;

      // estimated standard error
      std_errs_didx[midx] = std::sqrt(variance);
    }

    inter_tidx[didx] = num_meas_dpt;
  }
}

} // namespace running

} // namespace moments

} // namespace shadows
