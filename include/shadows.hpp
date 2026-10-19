#pragma once

#include "constants.hpp"

namespace shadows {

namespace moments {

namespace running {

tuple_dvec_t predict_moments(const int32_t* const inds, const int32_t* const clicks, const int32_t* const transposes,
                             const std::size_t num_qubits, const std::size_t num_meas, const std::size_t order,
                             const std::size_t var_offset, const std::size_t num_data_points = 0,
                             double* const inter_moments = nullptr, double* const inter_std_errs = nullptr,
                             std::size_t* const inter_tidx = nullptr, const int log2_num_threads = -1,
                             const bool use_simd = true);

// determine the number of threads to use
std::size_t get_num_threads(const int log2_num_threads, const std::size_t ss_dim, const int world_size);

// determine the number of ranks to use
std::size_t get_num_ranks(const int world_size, const std::size_t nts, const std::size_t ss_dim);

// this is just for data collection
std::size_t get_vector_width(const bool use_simd, const std::size_t num_qubits);

// required memory in MB
double rmMB(const std::size_t num_qubits, const std::size_t order, const std::size_t num_meas,
            const std::size_t var_offset, const std::size_t num_threads, const std::size_t num_ranks);

// required stack per thread in MB
double rsptMB(const std::size_t num_qubits, const std::size_t order);

// required flops in Mega Flops
double rfMF(const std::size_t num_qubits, const std::size_t order, const std::size_t num_meas,
            const std::size_t var_offset, const std::size_t num_threads, const std::size_t num_data_points);

// required memory traffic in MB
double rmtMB(const std::size_t num_qubits, const std::size_t order, const std::size_t num_meas,
             const std::size_t var_offset, const std::size_t num_threads);

} // namespace running

namespace batched {

std::vector<double> predict_moments(const int32_t* const inds, const int32_t* const clicks,
                                    const int32_t* const transposes, const std::size_t num_qubits,
                                    const std::size_t num_meas, const std::size_t order,
                                    const int log2_num_threads = -1, const bool use_simd = true);

// determine the number of threads to use
std::size_t get_num_threads(const int log2_num_threads, const std::size_t ss_dim);

// this is just for data collection
std::size_t get_vector_width(const bool use_simd, const std::size_t num_qubits);

// required memory in MB
double rmMB(const std::size_t num_qubits, const std::size_t order, const std::size_t num_meas,
            const std::size_t num_threads);

// required stack per thread in MB
double rsptMB(const std::size_t num_qubits);

// required flops in Mega Flops
double rfMF(const std::size_t num_qubits, const std::size_t num_meas);

// required memory traffic in MB
double rmtMB(const std::size_t num_qubits, const std::size_t num_meas, const std::size_t num_threads);

} // namespace batched

} // namespace moments

namespace QFI {

namespace running {

std::vector<double> predict_Tvals(const int32_t* const inds, const int32_t* const clicks, const std::size_t num_qubits,
                                  const std::size_t num_meas, const std::size_t n_star, const std::size_t var_offset,
                                  const std::size_t num_data_points = 0, double* inter_Tvals = nullptr,
                                  std::size_t* inter_tidx = nullptr, double* kernel_times = nullptr,
                                  const int log2_num_threads = -1, const bool use_simd = true);

// determine the number of threads to use
std::size_t get_num_threads(const int log2_num_threads, const std::size_t ss_dim);

// this is just for data collection
std::size_t get_vector_width(const bool use_simd, const std::size_t num_qubits);

// required memory in MB
double rmMB(const std::size_t num_qubits, const std::size_t order, const std::size_t num_meas,
            const std::size_t num_threads);

// required stack per thread in MB
double rsptMB(const std::size_t num_qubits, const std::size_t order);

// required flops in Mega Flops
double rfMF(const std::size_t num_qubits, const std::size_t order, const std::size_t num_meas,
            const std::size_t num_threads, const std::size_t num_data_points);

// required memory traffic in MB
double rmtMB(const std::size_t num_qubits, const std::size_t order, const std::size_t num_meas,
             const std::size_t num_threads);

} // namespace running

namespace batched {

std::vector<double> predict_Tvals(const int32_t* const inds, const int32_t* const clicks, const std::size_t num_qubits,
                                  const std::size_t num_meas, const std::size_t n_star, const std::size_t var_offset,
                                  double* kernel_times, const int log2_num_threads, const bool use_simd);

// determine the number of threads to use
std::size_t get_num_threads(const int log2_num_threads, const std::size_t ss_dim);

// this is just for data collection
std::size_t get_vector_width(const bool use_simd, const std::size_t num_qubits);

// required memory in MB
double rmMB(const std::size_t num_qubits, const std::size_t order, const std::size_t num_meas,
            const std::size_t num_threads);

// required stack per thread in MB
double rsptMB(const std::size_t num_qubits, const std::size_t order);

// required flops in Mega Flops
double rfMF(const std::size_t num_qubits, const std::size_t order, const std::size_t num_meas,
            const std::size_t num_threads);

// required memory traffic in MB
double rmtMB(const std::size_t num_qubits, const std::size_t order, const std::size_t num_meas,
             const std::size_t num_threads);

} // namespace batched

} // namespace QFI

} // namespace shadows
