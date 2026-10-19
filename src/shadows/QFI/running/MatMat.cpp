#include "shadows/QFI/running/MatMat.hpp"

#include "shadows/QFI/running/MatVec.hpp"
#include "shadows/QFI/running/PostProcess.hpp"

#include <chrono>

#include <omp.h>

namespace shadows {

namespace QFI {

namespace running {

// determine the number of threads to use
// assumes SMT is available
std::size_t get_nts(const int log2_num_threads, const std::size_t ss_dim) {
  std::size_t num_threads;
  const int world_size = 1;
  // std::cerr << "Max available threads: " << omp_get_max_threads() << "\n" << std::endl;
  const double log2_max_threads = std::log2(static_cast<std::size_t>(omp_get_max_threads()));
  const std::size_t max_num_threads_sys = std::size_t{1} << static_cast<std::size_t>(log2_max_threads);
  const std::size_t max_num_threads = (max_num_threads_sys > ss_dim) ? ss_dim : max_num_threads_sys;
  if (log2_num_threads < 0) {

    std::size_t num_ranks = 1;
    if (max_num_threads < ss_dim) {
      num_ranks = ss_dim / max_num_threads;
      if (num_ranks > world_size)
        num_ranks = world_size;
    }

    if (max_num_threads == omp_get_max_threads()) {
      if (num_ranks == world_size)
        num_threads = max_num_threads;
      else
        num_threads = max_num_threads / 2;
    } else {
      num_threads = max_num_threads;
    }

  } else {
    const std::size_t requested_threads = std::size_t{1} << static_cast<std::size_t>(log2_num_threads);
    num_threads = (requested_threads > max_num_threads) ? max_num_threads : requested_threads;
  }
  return num_threads;
}

// determine the vector size (#doubles) to use
// this is just for data collection
std::size_t get_vw(const bool use_simd, const std::size_t num_qubits) {
#if INSTRSET >= 9 // AVX512
  constexpr std::size_t maxVs = 8;
#elif INSTRSET >= 7 // AVX
  constexpr std::size_t maxVs = 4;
#elif INSTRSET >= 2 // SSE
  constexpr std::size_t maxVs = 2;
#else               // nothing
  constexpr std::size_t maxVs = 0;
#endif
  const std::size_t ss_dim = std::size_t{1} << num_qubits; // 2**num_qubits, state-space dimension
  std::size_t Vsize = (maxVs > 0 && ss_dim / maxVs >= 2) ? maxVs
                      : (maxVs >= 4 && ss_dim / 4 >= 2)  ? 4
                      : (maxVs >= 2 && ss_dim / 2 >= 2)  ? 2
                                                         : 1;
  if (!use_simd)
    Vsize = 1;
  return Vsize * 8;
}

// determine the correct inter_tidx to avoid determining this at runtime
// equal log spreads, last one is num_meas - order
bool fill_inter_tidx(std::size_t* __restrict__ const inter_tidx, const std::size_t num_meas, const std::size_t order,
                     const std::size_t num_data_points, const std::size_t var_offset) {
  if (num_data_points == 0)
    return 0;

  assert(order < num_meas);

  const std::size_t max_start = num_meas - order;

  if (num_data_points == 1) {
    inter_tidx[0] = max_start;
  } else {
    const std::size_t lo_start = 0;

    const double lo = std::log(1.0);
    const double hi = std::log(static_cast<double>(max_start - lo_start) + 1.0);

    for (std::size_t i = 0; i < num_data_points; ++i) {
      const double t = static_cast<double>(i) / static_cast<double>(num_data_points - 1);
      const double val = std::exp(lo + t * (hi - lo)) - 1.0;
      long long rounded = static_cast<long long>(std::llround(val));
      if (rounded < 0)
        rounded = 0;
      std::size_t offset = static_cast<std::size_t>(rounded);
      if (offset > max_start - lo_start)
        offset = max_start - lo_start;

      inter_tidx[i] = lo_start + offset;
    }

    for (std::size_t i = 1; i < num_data_points; ++i) {
      if (inter_tidx[i] <= inter_tidx[i - 1])
        inter_tidx[i] = inter_tidx[i - 1] + 1;
    }

    inter_tidx[num_data_points - 1] = max_start;
    for (std::size_t i = num_data_points - 1; i-- > 0;) {
      if (inter_tidx[i] >= inter_tidx[i + 1])
        inter_tidx[i] = inter_tidx[i + 1] - 1;
    }
  }

  assert(inter_tidx[num_data_points - 1] == max_start);
  for (std::size_t i = 1; i < num_data_points; ++i) {
    assert(inter_tidx[i] > inter_tidx[i - 1]);
  }

  inter_tidx[num_data_points] = num_meas;

  return 0;
}

// Kronecker Matrix-Matrix recurrence
// essentially just maps threads and nodes to their respective KrMV recurrences
// and post-processes gathered data
template <std::size_t num_qubits, bool __use_simd>
void KrMM_recurrence(double* __restrict__ const Tvals, const int32_t* __restrict__ const inds,
                     const int32_t* __restrict__ const clicks, const std::size_t num_meas, const std::size_t n_star,
                     const int log2_num_threads, const std::size_t num_data_points,
                     double* __restrict__ const inter_Tvals, std::size_t* __restrict__ const inter_tidx,
                     double* __restrict__ const kernel_times) {
  constexpr std::size_t ss_dim = std::size_t{1} << num_qubits; // 2**num_qubits, state-space dimension

#if INSTRSET >= 9 // AVX512
  constexpr std::size_t maxVs = 8;
#elif INSTRSET >= 7 // AVX
  constexpr std::size_t maxVs = 4;
#elif INSTRSET >= 2 // SSE
  constexpr std::size_t maxVs = 2;
#else               // nothing
  constexpr std::size_t maxVs = 0;
#endif

  constexpr std::size_t Vsize = (maxVs > 0 && ss_dim / maxVs >= 2) ? maxVs
                                : (maxVs >= 4 && ss_dim / 4 >= 2)  ? 4
                                : (maxVs >= 2 && ss_dim / 2 >= 2)  ? 2
                                                                   : 0;

  constexpr bool use_simd = Vsize > 0 && __use_simd;

  using VecXd = tern_t<Vsize == 8, vcl::Vec8d, tern_t<Vsize == 4, vcl::Vec4d, tern_t<Vsize == 2, vcl::Vec2d, void>>>;

  // extract the number of threads
  const std::size_t nts = get_nts(log2_num_threads, ss_dim);

  const std::size_t order = 2 * n_star + 1;

  // intermediate estimates
  // num_data_points*order*(n_star + 1)*tid + num_data_points*order*ridx + order*dpt_idx + midx
  std::vector<double> inter_traces_vec(nts * num_data_points * order * (n_star + 1), 0.0);
  double* const inter_traces_shared = inter_traces_vec.data();

#ifdef DEBUG
  std::cerr << "\n--------------- launch pars --------------- " << std::endl;
  std::cerr << "num_qubits=" << num_qubits << std::endl;
  std::cerr << "order=" << order << std::endl;
  std::cerr << "num_meas=" << num_meas << std::endl;
  std::cerr << "Vsize=" << Vsize << std::endl;
  std::cerr << "use_simd=" << use_simd << std::endl;
  std::cerr << "nts=" << nts << std::endl;
  std::cerr << "num_data_points=" << num_data_points << std::endl;
  std::cerr << "--------------- launch pars --------------- \n" << std::endl;
#endif

  using Clock = std::chrono::high_resolution_clock;
  using Duration = std::chrono::duration<double>;

  auto start_time = Clock::now();

#pragma omp parallel num_threads(nts)
  {
    const std::size_t tid = omp_get_thread_num();
    const std::size_t num_threads = omp_get_num_threads();
    assert(num_threads == nts);
    assert(tid < num_threads);

    const std::size_t rows_per_thread = ss_dim / num_threads;
    const std::size_t row_start = rows_per_thread * tid;
    const std::size_t row_end = row_start + rows_per_thread;

    double* const inter_traces_tid = inter_traces_shared + num_data_points * order * (n_star + 1) * tid;

    if constexpr (use_simd) {

      // allocate aligned Pmats
      const std::size_t total_size = 2 * order * ss_dim * sizeof(double);
      assert(total_size % ALIGNMENT == 0);
      double* const Pmats = static_cast<double*>(std::aligned_alloc(ALIGNMENT, total_size));

      for (std::size_t ridx = 0; ridx < n_star + 1; ++ridx) {

        double* const inter_traces = inter_traces_tid + num_data_points * order * ridx;
        for (std::size_t row = row_start; row < row_end; ++row) {
          KrMV_recurrence<VecXd, num_qubits>(Pmats, inds, clicks, row, ridx, num_meas, order, num_data_points,
                                             inter_traces, inter_tidx);
        }
      }
      free(Pmats);
    } else {

      // allocate aligned Pmats
      const std::size_t total_size = order * ss_dim * sizeof(z_t);
      assert(total_size % ALIGNMENT == 0);
      z_t* const Pmats = static_cast<z_t*>(std::aligned_alloc(ALIGNMENT, total_size));

      for (std::size_t ridx = 0; ridx < n_star + 1; ++ridx) {

        double* const inter_traces = inter_traces_tid + num_data_points * order * ridx;
        for (std::size_t row = row_start; row < row_end; ++row) {
          KrMV_recurrence<num_qubits>(Pmats, inds, clicks, row, ridx, num_meas, order, num_data_points, inter_traces,
                                      inter_tidx);
        }
      }
      free(Pmats);
    }

  } // omp parallel

  auto stop_time = Clock::now();
  double time = Duration(stop_time - start_time).count();

  reduce(inter_traces_shared, num_data_points * order * (n_star + 1), nts);

  post_process(Tvals, inter_Tvals, inter_traces_shared, n_star, num_data_points);

  kernel_times[0] = time;
};

// dispatcher function for fixing num_qubits and use_simd at compile time
void KrMM_recurrence(double* const Tvals, const int32_t* const inds, const int32_t* const clicks,
                     const std::size_t num_qubits, const std::size_t num_meas, const std::size_t n_star,
                     const bool use_simd, const int log2_num_threads, const std::size_t num_data_points,
                     double* const inter_Tvals, std::size_t* const inter_tidx, double* const kernel_times) {
  switch (num_qubits) {
  case 2:
    if (use_simd)
      KrMM_recurrence<2, true>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, num_data_points, inter_Tvals,
                               inter_tidx, kernel_times);
    else
      KrMM_recurrence<2, false>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, num_data_points, inter_Tvals,
                                inter_tidx, kernel_times);
    break;
  case 3:
    if (use_simd)
      KrMM_recurrence<3, true>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, num_data_points, inter_Tvals,
                               inter_tidx, kernel_times);
    else
      KrMM_recurrence<3, false>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, num_data_points, inter_Tvals,
                                inter_tidx, kernel_times);
    break;
  case 4:
    if (use_simd)
      KrMM_recurrence<4, true>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, num_data_points, inter_Tvals,
                               inter_tidx, kernel_times);
    else
      KrMM_recurrence<4, false>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, num_data_points, inter_Tvals,
                                inter_tidx, kernel_times);
    break;
  case 5:
    if (use_simd)
      KrMM_recurrence<5, true>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, num_data_points, inter_Tvals,
                               inter_tidx, kernel_times);
    else
      KrMM_recurrence<5, false>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, num_data_points, inter_Tvals,
                                inter_tidx, kernel_times);
    break;
  case 6:
    if (use_simd)
      KrMM_recurrence<6, true>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, num_data_points, inter_Tvals,
                               inter_tidx, kernel_times);
    else
      KrMM_recurrence<6, false>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, num_data_points, inter_Tvals,
                                inter_tidx, kernel_times);
    break;
  case 7:
    if (use_simd)
      KrMM_recurrence<7, true>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, num_data_points, inter_Tvals,
                               inter_tidx, kernel_times);
    else
      KrMM_recurrence<7, false>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, num_data_points, inter_Tvals,
                                inter_tidx, kernel_times);
    break;
  case 8:
    if (use_simd)
      KrMM_recurrence<8, true>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, num_data_points, inter_Tvals,
                               inter_tidx, kernel_times);
    else
      KrMM_recurrence<8, false>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, num_data_points, inter_Tvals,
                                inter_tidx, kernel_times);
    break;
  case 9:
    if (use_simd)
      KrMM_recurrence<9, true>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, num_data_points, inter_Tvals,
                               inter_tidx, kernel_times);
    else
      KrMM_recurrence<9, false>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, num_data_points, inter_Tvals,
                                inter_tidx, kernel_times);
    break;
  case 10:
    if (use_simd)
      KrMM_recurrence<10, true>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, num_data_points, inter_Tvals,
                                inter_tidx, kernel_times);
    else
      KrMM_recurrence<10, false>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, num_data_points, inter_Tvals,
                                 inter_tidx, kernel_times);
    break;
  case 11:
    if (use_simd)
      KrMM_recurrence<11, true>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, num_data_points, inter_Tvals,
                                inter_tidx, kernel_times);
    else
      KrMM_recurrence<11, false>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, num_data_points, inter_Tvals,
                                 inter_tidx, kernel_times);
    break;
  case 12:
    if (use_simd)
      KrMM_recurrence<12, true>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, num_data_points, inter_Tvals,
                                inter_tidx, kernel_times);
    else
      KrMM_recurrence<12, false>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, num_data_points, inter_Tvals,
                                 inter_tidx, kernel_times);
    break;
  case 13:
    if (use_simd)
      KrMM_recurrence<13, true>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, num_data_points, inter_Tvals,
                                inter_tidx, kernel_times);
    else
      KrMM_recurrence<13, false>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, num_data_points, inter_Tvals,
                                 inter_tidx, kernel_times);
    break;
  case 14:
    if (use_simd)
      KrMM_recurrence<14, true>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, num_data_points, inter_Tvals,
                                inter_tidx, kernel_times);
    else
      KrMM_recurrence<14, false>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, num_data_points, inter_Tvals,
                                 inter_tidx, kernel_times);
    break;
  case 15:
    if (use_simd)
      KrMM_recurrence<15, true>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, num_data_points, inter_Tvals,
                                inter_tidx, kernel_times);
    else
      KrMM_recurrence<15, false>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, num_data_points, inter_Tvals,
                                 inter_tidx, kernel_times);
    break;
  case 16:
    if (use_simd)
      KrMM_recurrence<16, true>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, num_data_points, inter_Tvals,
                                inter_tidx, kernel_times);
    else
      KrMM_recurrence<16, false>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, num_data_points, inter_Tvals,
                                 inter_tidx, kernel_times);
    break;
  default:
    assert(false);
    break;
  }
};

} // namespace running

} // namespace QFI

} // namespace shadows
