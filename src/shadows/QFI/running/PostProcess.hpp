#pragma once

#include "constants.hpp"

#include <cassert>
#include <iostream>

namespace shadows {

namespace QFI {

namespace running {

void reduce(double* __restrict__ const traces_shared, const std::size_t vals_per_thread,
            const std::size_t num_threads) {
  for (std::size_t tid = 1; tid < num_threads; ++tid) {
    const double* const traces_tid = traces_shared + vals_per_thread * tid;
    for (std::size_t idx = 0; idx < vals_per_thread; ++idx) {
      traces_shared[idx] += traces_tid[idx];
    }
  }
}

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

// performs the post-processing of collected traces to extract std. errs. and moments estimates
void post_process(double* __restrict__ const Tvals, double* __restrict__ const inter_Tvals,
                  const double* __restrict__ const inter_traces, const std::size_t n_star,
                  const std::size_t num_data_points) {

  // inter_traces: num_data_points*order*ridx + order*didx + midx
  // inter_Tvals: k_order*didx + kidx
  // Tvals: kidx

  const std::size_t k_order = 2 * n_star;
  const std::size_t order = 2 * n_star + 1;
  const std::size_t num_recs = n_star + 1;

  std::memset(Tvals, 0, k_order * sizeof(double));
  std::memset(inter_Tvals, 0, num_data_points * k_order * sizeof(double));

  for (std::size_t didx = 0; didx < num_data_points; ++didx) {

    bool final = (didx == num_data_points - 1);
    const double* const traces_didx = inter_traces + order * didx;
    double* const inter_Tvals_didx = inter_Tvals + k_order * didx;

    for (std::size_t kidx = 0; kidx < k_order; ++kidx) {

      const std::size_t midx = kidx + 1;
      const double* const traces_midx = traces_didx + midx;

      for (std::size_t lidx = 0; lidx < kidx + 3; ++lidx) {

        const std::size_t ridx = std::min(lidx, kidx + 2 - lidx);
        const std::size_t idx = num_data_points * order * ridx;
        assert(ridx < num_recs);
        inter_Tvals_didx[kidx] += Tval_fac(kidx, lidx) * traces_midx[idx];
        if (final)
          Tvals[kidx] += Tval_fac(kidx, lidx) * traces_midx[idx];
      }
    }
  }
}

} // namespace running

} // namespace QFI

} // namespace shadows
