#pragma once

#include "constants.hpp"

namespace shadows {

namespace QFI {

namespace batched {

std::size_t get_nts(const int log2_num_threads, const std::size_t ss_dim);

std::size_t get_vw(const bool use_simd, const std::size_t num_qubits);

// traces must be initialized with only 0.0s !
void KrMM_recurrence(double* const Tvals, const int32_t* const inds, const int32_t* const clicks,
                     const std::size_t num_qubits, const std::size_t num_meas, const std::size_t n_star,
                     const bool use_simd, const int log2_num_threads, double* const kernel_times);

} // namespace batched

} // namespace QFI

} // namespace shadows
