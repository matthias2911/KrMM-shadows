#pragma once

#include "constants.hpp"

namespace shadows {

namespace moments {

namespace batched {

std::size_t get_nts(const int log2_num_threads, const std::size_t ss_dim);

std::size_t get_vw(const bool use_simd, const std::size_t num_qubits);

// traces must be initialized with only 0.0s !
void KrMM_recurrence(double* const moments, const int32_t* const inds,
                     const int32_t* const clicks, const int32_t* const transposes, const std::size_t num_qubits,
                     const std::size_t num_meas, const std::size_t order,
                     const bool use_simd, const int log2_num_threads);

} // namespace batched

} // namespace moments

} // namespace shadows
