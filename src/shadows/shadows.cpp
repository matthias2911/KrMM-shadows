#include "shadows.hpp"

#include "shadows/moments/batched/MatMat.hpp"
#include "shadows/moments/running/MatMat.hpp"

#include "shadows/QFI/running/MatMat.hpp"
#include "shadows/QFI/batched/MatMat.hpp"

#include <cassert>
#include <cstring>

namespace shadows {

namespace moments {

namespace running {

tuple_dvec_t predict_moments(const int32_t* const inds, const int32_t* const clicks, const int32_t* const transposes,
                             const std::size_t num_qubits, const std::size_t num_meas, const std::size_t order,
                             const std::size_t var_offset, const std::size_t num_data_points, double* inter_moments,
                             double* inter_std_errs, std::size_t* inter_tidx, const int log2_num_threads,
                             const bool use_simd) {
  assert(num_qubits > 1);
  assert(num_meas > 2 * order);
  assert(var_offset < num_meas / 2);
  assert(num_data_points * order + order - 1 <= num_meas);

  // stores final moments of each order
  std::vector<double> moments(order, 0.0);

  // stores final estimated standard errors
  std::vector<double> standard_errs(order, 0.0);

  fill_inter_tidx(inter_tidx, num_meas, order, num_data_points, var_offset);

  // assert(false);

  KrMM_recurrence(moments.data(), standard_errs.data(), inds, clicks, transposes, num_qubits, num_meas, order,
                  var_offset, use_simd, log2_num_threads, num_data_points, inter_moments, inter_std_errs, inter_tidx);

  return {moments, standard_errs};
}

} // namespace running

namespace batched {

std::vector<double> predict_moments(const int32_t* const inds, const int32_t* const clicks,
                                    const int32_t* const transposes, const std::size_t num_qubits,
                                    const std::size_t num_meas, const std::size_t order, const int log2_num_threads,
                                    const bool use_simd) {
  assert(num_qubits > 1);
  assert(num_meas > order);

  // stores final moments of each order
  std::vector<double> moments(order, 0.0);

  KrMM_recurrence(moments.data(), inds, clicks, transposes, num_qubits, num_meas, order, use_simd, log2_num_threads);

  return moments;
}

} // namespace batched

} // namespace moments

namespace QFI {

namespace running {

std::vector<double> predict_Tvals(const int32_t* const inds, const int32_t* const clicks, const std::size_t num_qubits,
                                  const std::size_t num_meas, const std::size_t n_star, const std::size_t var_offset,
                                  const std::size_t num_data_points, double* inter_Tvals, std::size_t* inter_tidx,
                                  double* kernel_times, const int log2_num_threads, const bool use_simd) {
  const std::size_t order = 2 * n_star + 1;

  assert(num_qubits > 1);
  assert(num_meas > 2 * order);
  assert(num_data_points * order + order - 1 <= num_meas);
  assert(num_data_points > 0);

  // stores final Tvals of each order
  std::vector<double> Tvals(order, 0.0);

  fill_inter_tidx(inter_tidx, num_meas, order, num_data_points, var_offset);

  assert(inter_tidx[num_data_points - 1] == num_meas - order);

  // assert(false);

  KrMM_recurrence(Tvals.data(), inds, clicks, num_qubits, num_meas, n_star, use_simd, log2_num_threads, num_data_points,
                  inter_Tvals, inter_tidx, kernel_times);

  return Tvals;
}

} // namespace running

namespace batched {

std::vector<double> predict_Tvals(const int32_t* const inds, const int32_t* const clicks, const std::size_t num_qubits,
                                  const std::size_t num_meas, const std::size_t n_star, const std::size_t var_offset,
                                  double* kernel_times, const int log2_num_threads, const bool use_simd) {
  const std::size_t order = 2 * n_star + 1;

  assert(num_qubits > 1);
  assert(num_meas > 2 * order);

  // stores final Tvals of each order
  std::vector<double> Tvals(order, 0.0);

  KrMM_recurrence(Tvals.data(), inds, clicks, num_qubits, num_meas, n_star, use_simd, log2_num_threads, kernel_times);

  return Tvals;
}

} // namespace batched

} // namespace QFI

} // namespace shadows
