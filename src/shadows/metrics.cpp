#include "shadows.hpp"

#include "constants.hpp"

#include "shadows/moments/batched/MatMat.hpp"
#include "shadows/moments/running/MatMat.hpp"

#include "shadows/QFI/running/MatMat.hpp"
#include "shadows/QFI/batched/MatMat.hpp"

#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>

#include <omp.h>

// this is just included in main file for gathering fixed computational metrics
// most of these are outdated/ redundant/ incorrect
// sorry about that

namespace shadows {

namespace moments {

namespace running {

// determine the number of threads to use
std::size_t get_num_threads(const int log2_num_threads, const std::size_t ss_dim, const int world_size) {
  return get_nts(log2_num_threads, ss_dim, world_size);
}

// determine the number of ranks to use
std::size_t get_num_ranks(const int world_size, const std::size_t nts, const std::size_t ss_dim) {
  return get_nrks(world_size, nts, ss_dim);
}

// this is just for data collection
std::size_t get_vector_width(const bool use_simd, const std::size_t num_qubits) { return get_vw(use_simd, num_qubits); }

// required memory in MB
double rmMB(const std::size_t num_qubits, const std::size_t order, const std::size_t num_meas,
            const std::size_t var_offset, const std::size_t num_threads, const std::size_t num_ranks) {

  const std::size_t ss_dim = std::size_t{1} << num_qubits;
  const std::size_t len_var_meas = var_offset + order;

  std::size_t req_mem = 0;

  req_mem += num_meas * num_qubits * sizeof(int32_t); // inds
  req_mem += num_meas * num_qubits * sizeof(int32_t); // clicks
  req_mem += num_qubits * sizeof(int32_t);            // transposes

  req_mem += order * sizeof(double); // moments
  req_mem += order * sizeof(double); // standard_errs

  req_mem += num_threads * order * len_var_meas * sizeof(double); // traces_shared

  req_mem += num_threads * 2 * order * ss_dim * sizeof(double); // Pmats
  req_mem += num_threads * 2 * ss_dim * sizeof(double);         // buffer0
  req_mem += num_threads * 2 * ss_dim * sizeof(double);         // buffer1

  req_mem *= num_ranks;

  return static_cast<double>(req_mem) / 1'000'000.0;
}

// required stack per thread in MB
double rsptMB(const std::size_t num_qubits, const std::size_t order) {

  const std::size_t ss_dim = std::size_t{1} << num_qubits;

  std::size_t req_stack = 0;

  req_stack += 2 * order * ss_dim * sizeof(double); // Pmats
  req_stack += 2 * ss_dim * sizeof(double);         // buffer0
  req_stack += 2 * ss_dim * sizeof(double);         // buffer1

  return static_cast<double>(req_stack) / 1'000'000.0;
}

// required flops in Mega Flops
double rfMF(const std::size_t num_qubits, const std::size_t order, const std::size_t num_meas,
            const std::size_t var_offset, const std::size_t num_threads, const std::size_t num_data_points) {

  const std::size_t ss_dim = std::size_t{1} << num_qubits;
  [[maybe_unused]] const std::size_t len_var_meas = var_offset + order;

  std::size_t req_flops = 0;

  req_flops += 14 * num_meas * order * num_qubits * ss_dim * ss_dim; // all gekmm's

  req_flops += 2 * num_meas * order * ss_dim * ss_dim; // all additions

  req_flops += num_threads * num_data_points * order; // reduction

  // req_flops += len_var_meas * order * (ss_dim - 1); // taking traces
  // req_flops += len_var_meas * order * 2; // inovations + normalization
  // req_flops += len_var_meas * order * 6; // welfords

  return static_cast<double>(req_flops) / 1'000'000.0;
}

// required memory traffic in MB
double rmtMB(const std::size_t num_qubits, const std::size_t order, const std::size_t num_meas,
             const std::size_t var_offset, const std::size_t num_threads) {

  const std::size_t ss_dim = std::size_t{1} << num_qubits;
  [[maybe_unused]] const std::size_t len_var_meas = var_offset + order;
  [[maybe_unused]] const std::size_t nts = num_threads;

  std::size_t req_mem_traffic = 0;

  req_mem_traffic += ss_dim * num_meas * num_qubits * order * sizeof(int32_t); // load inds
  req_mem_traffic += ss_dim * num_meas * num_qubits * order * sizeof(int32_t); // load clicks

  req_mem_traffic += 2 * num_meas * order * ss_dim * ss_dim * sizeof(double); // load Pmats
  req_mem_traffic += 2 * num_meas * order * ss_dim * ss_dim * sizeof(double); // write Pmats

  // req_mem_traffic += num_threads * order * len_var_meas * sizeof(double); // write traces
  // req_mem_traffic += num_threads * order * len_var_meas * sizeof(double); // load traces
  // req_mem_traffic += num_threads * order * len_var_meas * sizeof(double); // write traces

  // req_mem_traffic += num_threads * order * sizeof(double); // write moments
  // req_mem_traffic += num_threads * order * sizeof(double); // write standard_errs

  return static_cast<double>(req_mem_traffic) / 1'000'000.0;
}

} // namespace running

namespace batched {

// determine the number of threads to use
std::size_t get_num_threads(const int log2_num_threads, const std::size_t ss_dim) {
  return get_nts(log2_num_threads, ss_dim);
}

// this is just for data collection
std::size_t get_vector_width(const bool use_simd, const std::size_t num_qubits) { return get_vw(use_simd, num_qubits); }

// required memory in MB
double rmMB(const std::size_t num_qubits, const std::size_t order, const std::size_t num_meas,
            const std::size_t num_threads) {

  const std::size_t ss_dim = std::size_t{1} << num_qubits;

  std::size_t req_mem = 0;

  req_mem += num_meas * num_qubits * sizeof(int32_t); // inds
  req_mem += num_meas * num_qubits * sizeof(int32_t); // clicks
  req_mem += num_qubits * sizeof(int32_t);            // transposes

  req_mem += order * sizeof(double); // moments

  req_mem += 2 * order * ss_dim * ss_dim * sizeof(double); // Pmats
  req_mem += 2 * order * ss_dim * ss_dim * sizeof(double); // batches
  req_mem += num_threads * 2 * ss_dim * sizeof(double);    // buffer0
  req_mem += num_threads * 2 * ss_dim * sizeof(double);    // buffer1
  req_mem += num_threads * 2 * ss_dim * sizeof(double);    // id_arr

  return static_cast<double>(req_mem) / 1'000'000.0;
}

// required stack per thread in MB
double rsptMB(const std::size_t num_qubits) {

  const std::size_t ss_dim = std::size_t{1} << num_qubits;

  std::size_t req_stack = 0;

  req_stack += 2 * ss_dim * sizeof(double); // Pmats_row
  req_stack += 2 * ss_dim * sizeof(double); // buffer0
  req_stack += 2 * ss_dim * sizeof(double); // buffer1
  req_stack += 2 * ss_dim * sizeof(double); // id_arr

  return static_cast<double>(req_stack) / 1'000'000.0;
}

// required flops in Mega Flops (w/o post-processing)
double rfMF(const std::size_t num_qubits, const std::size_t num_meas) {

  const std::size_t ss_dim = std::size_t{1} << num_qubits;

  std::size_t req_flops = 0;

  req_flops += 14 * num_meas * num_qubits * ss_dim * ss_dim; // all gekmm's

  req_flops += 2 * num_meas * ss_dim * ss_dim; // all additions

  return static_cast<double>(req_flops) / 1'000'000.0;
}

// required memory traffic in MB
double rmtMB(const std::size_t num_qubits, const std::size_t num_meas, const std::size_t num_threads) {

  const std::size_t ss_dim = std::size_t{1} << num_qubits;
  [[maybe_unused]] const std::size_t nts = num_threads;

  std::size_t req_mem_traffic = 0;

  req_mem_traffic += ss_dim * num_meas * num_qubits * sizeof(int32_t); // load inds
  req_mem_traffic += ss_dim * num_meas * num_qubits * sizeof(int32_t); // load clicks

  req_mem_traffic += 2 * num_meas * ss_dim * ss_dim * sizeof(double); // load Pmats
  req_mem_traffic += 2 * num_meas * ss_dim * ss_dim * sizeof(double); // write Pmats

  // req_mem_traffic += num_threads * order * len_var_meas * sizeof(double); // write traces
  // req_mem_traffic += num_threads * order * len_var_meas * sizeof(double); // load traces
  // req_mem_traffic += num_threads * order * len_var_meas * sizeof(double); // write traces

  // req_mem_traffic += num_threads * order * sizeof(double); // write moments
  // req_mem_traffic += num_threads * order * sizeof(double); // write standard_errs

  return static_cast<double>(req_mem_traffic) / 1'000'000.0;
}

} // namespace batched

} // namespace moments

namespace QFI {

namespace running {

// determine the number of threads to use
std::size_t get_num_threads(const int log2_num_threads, const std::size_t ss_dim) {
  return get_nts(log2_num_threads, ss_dim);
}

// this is just for data collection
std::size_t get_vector_width(const bool use_simd, const std::size_t num_qubits) { return get_vw(use_simd, num_qubits); }

// required memory in MB
double rmMB(const std::size_t num_qubits, const std::size_t order, const std::size_t num_meas,
            const std::size_t num_threads) {

  const std::size_t ss_dim = std::size_t{1} << num_qubits;

  std::size_t req_mem = 0;

  req_mem += num_meas * num_qubits * sizeof(int32_t); // inds
  req_mem += num_meas * num_qubits * sizeof(int32_t); // clicks
  req_mem += num_qubits * sizeof(int32_t);            // transposes

  req_mem += order * sizeof(double); // moments
  req_mem += order * sizeof(double); // standard_errs

  req_mem += num_threads * order * sizeof(double); // traces_shared

  req_mem += num_threads * 2 * order * ss_dim * sizeof(double); // Pmats
  req_mem += num_threads * 2 * ss_dim * sizeof(double);         // buffer0
  req_mem += num_threads * 2 * ss_dim * sizeof(double);         // buffer1

  return static_cast<double>(req_mem) / 1'000'000.0;
}

// required stack per thread in MB
double rsptMB(const std::size_t num_qubits, const std::size_t order) {

  const std::size_t ss_dim = std::size_t{1} << num_qubits;

  std::size_t req_stack = 0;

  req_stack += 2 * order * ss_dim * sizeof(double); // Pmats
  req_stack += 2 * ss_dim * sizeof(double);         // buffer0
  req_stack += 2 * ss_dim * sizeof(double);         // buffer1

  return static_cast<double>(req_stack) / 1'000'000.0;
}

// required flops in Mega Flops
double rfMF(const std::size_t num_qubits, const std::size_t order, const std::size_t num_meas,
            const std::size_t num_threads, const std::size_t num_data_points) {

  const std::size_t ss_dim = std::size_t{1} << num_qubits;

  std::size_t req_flops = 0;

  req_flops += 14 * num_meas * order * num_qubits * ss_dim * ss_dim; // all gekmm's

  req_flops += 2 * num_meas * order * ss_dim * ss_dim; // all additions

  req_flops += num_threads * num_data_points * order; // reduction

  // req_flops += len_var_meas * order * (ss_dim - 1); // taking traces
  // req_flops += len_var_meas * order * 2; // inovations + normalization
  // req_flops += len_var_meas * order * 6; // welfords

  return static_cast<double>(req_flops) / 1'000'000.0;
}

// required memory traffic in MB
double rmtMB(const std::size_t num_qubits, const std::size_t order, const std::size_t num_meas,
             const std::size_t num_threads) {

  const std::size_t ss_dim = std::size_t{1} << num_qubits;
  [[maybe_unused]] const std::size_t nts = num_threads;

  std::size_t req_mem_traffic = 0;

  req_mem_traffic += ss_dim * num_meas * num_qubits * order * sizeof(int32_t); // load inds
  req_mem_traffic += ss_dim * num_meas * num_qubits * order * sizeof(int32_t); // load clicks

  req_mem_traffic += 2 * num_meas * order * ss_dim * ss_dim * sizeof(double); // load Pmats
  req_mem_traffic += 2 * num_meas * order * ss_dim * ss_dim * sizeof(double); // write Pmats

  // req_mem_traffic += num_threads * order * len_var_meas * sizeof(double); // write traces
  // req_mem_traffic += num_threads * order * len_var_meas * sizeof(double); // load traces
  // req_mem_traffic += num_threads * order * len_var_meas * sizeof(double); // write traces

  // req_mem_traffic += num_threads * order * sizeof(double); // write moments
  // req_mem_traffic += num_threads * order * sizeof(double); // write standard_errs

  return static_cast<double>(req_mem_traffic) / 1'000'000.0;
}

} // namespace running

namespace batched {

// determine the number of threads to use
std::size_t get_num_threads(const int log2_num_threads, const std::size_t ss_dim) {
  return get_nts(log2_num_threads, ss_dim);
}

// this is just for data collection
std::size_t get_vector_width(const bool use_simd, const std::size_t num_qubits) { return get_vw(use_simd, num_qubits); }

// required memory in MB
double rmMB(const std::size_t num_qubits, const std::size_t order, const std::size_t num_meas,
            const std::size_t num_threads) {

  const std::size_t ss_dim = std::size_t{1} << num_qubits;

  std::size_t req_mem = 0;

  req_mem += num_meas * num_qubits * sizeof(int32_t); // inds
  req_mem += num_meas * num_qubits * sizeof(int32_t); // clicks
  req_mem += num_qubits * sizeof(int32_t);            // transposes

  req_mem += order * sizeof(double); // moments
  req_mem += order * sizeof(double); // standard_errs

  req_mem += num_threads * order * sizeof(double); // traces_shared

  req_mem += num_threads * 2 * order * ss_dim * sizeof(double); // Pmats
  req_mem += num_threads * 2 * ss_dim * sizeof(double);         // buffer0
  req_mem += num_threads * 2 * ss_dim * sizeof(double);         // buffer1

  return static_cast<double>(req_mem) / 1'000'000.0;
}

// required stack per thread in MB
double rsptMB(const std::size_t num_qubits, const std::size_t order) {

  const std::size_t ss_dim = std::size_t{1} << num_qubits;

  std::size_t req_stack = 0;

  req_stack += 2 * order * ss_dim * sizeof(double); // Pmats
  req_stack += 2 * ss_dim * sizeof(double);         // buffer0
  req_stack += 2 * ss_dim * sizeof(double);         // buffer1

  return static_cast<double>(req_stack) / 1'000'000.0;
}

// required flops in Mega Flops
double rfMF(const std::size_t num_qubits, const std::size_t order, const std::size_t num_meas,
            const std::size_t num_threads) {

  const std::size_t ss_dim = std::size_t{1} << num_qubits;

  std::size_t req_flops = 0;

  req_flops += 14 * num_meas * order * num_qubits * ss_dim * ss_dim; // all gekmm's

  req_flops += 2 * num_meas * order * ss_dim * ss_dim; // all additions

  req_flops += num_threads * order; // reduction

  // req_flops += len_var_meas * order * (ss_dim - 1); // taking traces
  // req_flops += len_var_meas * order * 2; // inovations + normalization
  // req_flops += len_var_meas * order * 6; // welfords

  return static_cast<double>(req_flops) / 1'000'000.0;
}

// required memory traffic in MB
double rmtMB(const std::size_t num_qubits, const std::size_t order, const std::size_t num_meas,
             const std::size_t num_threads) {

  const std::size_t ss_dim = std::size_t{1} << num_qubits;
  [[maybe_unused]] const std::size_t nts = num_threads;

  std::size_t req_mem_traffic = 0;

  req_mem_traffic += ss_dim * num_meas * num_qubits * order * sizeof(int32_t); // load inds
  req_mem_traffic += ss_dim * num_meas * num_qubits * order * sizeof(int32_t); // load clicks

  req_mem_traffic += 2 * num_meas * order * ss_dim * ss_dim * sizeof(double); // load Pmats
  req_mem_traffic += 2 * num_meas * order * ss_dim * ss_dim * sizeof(double); // write Pmats

  // req_mem_traffic += num_threads * order * len_var_meas * sizeof(double); // write traces
  // req_mem_traffic += num_threads * order * len_var_meas * sizeof(double); // load traces
  // req_mem_traffic += num_threads * order * len_var_meas * sizeof(double); // write traces

  // req_mem_traffic += num_threads * order * sizeof(double); // write moments
  // req_mem_traffic += num_threads * order * sizeof(double); // write standard_errs

  return static_cast<double>(req_mem_traffic) / 1'000'000.0;
}

} // namespace running

} // namespace QFI

} // namespace shadows
