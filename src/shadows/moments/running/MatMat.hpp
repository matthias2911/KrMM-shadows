#pragma once

#include "constants.hpp"

namespace shadows {

namespace moments {

namespace running {

std::size_t get_nts(const int log2_num_threads, const std::size_t ss_dim, const int world_size);

std::size_t get_vw(const bool use_simd, const std::size_t num_qubits);

std::size_t get_nrks(const int world_size, const std::size_t nts, const std::size_t ss_dim);

bool fill_inter_tidx(std::size_t* __restrict__ const inter_tidx, const std::size_t num_meas, const std::size_t order,
                     const std::size_t num_data_points, const std::size_t var_offset);

// traces must be initialized with only 0.0s !
void KrMM_recurrence(double* const moments, double* const standard_errs, const int32_t* const inds,
                     const int32_t* const clicks, const int32_t* const transposes, const std::size_t num_qubits,
                     const std::size_t num_meas, const std::size_t order, const std::size_t var_offset,
                     const bool use_simd, const int log2_num_threads, const std::size_t num_data_points,
                     double* const inter_moments, double* const inter_std_errs, std::size_t* const inter_tidx);

} // namespace running

} // namespace moments

} // namespace shadows
