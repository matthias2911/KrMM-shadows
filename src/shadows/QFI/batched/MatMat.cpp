#include "shadows/QFI/batched/MatMat.hpp"
#include "shadows/QFI/batched/MatVec.hpp"

#include "shadows/QFI/batched/PostProcess.hpp"

#include <chrono>

#include <omp.h>

namespace shadows {

namespace QFI {

namespace batched {

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

// Kronecker Matrix-Matrix recurrence
// essentially just maps threads and nodes to their respective KrMV recurrences
// and post-processes gathered data
template <std::size_t num_qubits, bool __use_simd>
void KrMM_recurrence(double* __restrict__ const Tvals, const int32_t* __restrict__ const inds,
                     const int32_t* __restrict__ const clicks, const std::size_t num_meas, const std::size_t n_star,
                     const int log2_num_threads, double* __restrict__ const kernel_times) {
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

#ifdef DEBUG
  std::cerr << "\n--------------- launch pars --------------- " << std::endl;
  std::cerr << "num_qubits=" << num_qubits << std::endl;
  std::cerr << "order=" << order << std::endl;
  std::cerr << "num_meas=" << num_meas << std::endl;
  std::cerr << "Vsize=" << Vsize << std::endl;
  std::cerr << "use_simd=" << use_simd << std::endl;
  std::cerr << "nts=" << nts << std::endl;
  std::cerr << "--------------- launch pars --------------- \n" << std::endl;
#endif

  const std::size_t total_size = 2 * order * ss_dim * ss_dim * sizeof(double);
  assert(total_size % ALIGNMENT == 0);
  z_t* const batches = static_cast<z_t*>(std::aligned_alloc(ALIGNMENT, total_size));
  std::fill(batches, batches + order * ss_dim * ss_dim, z_t{0.0, 0.0}); // memset is sketchy for std::complex

  std::array<int32_t, num_qubits> transposes_arr = {0}; // to conform with KrMV implementation
  const int32_t* const transposes = transposes_arr.data();

  double time = 0.0;

  // VECTORIZED
  if constexpr (use_simd) {

    // allocate aligned Pmats
    // access row of batch via: 2 * order * ss_dim * row_idx + 2 * ss_dim * order_idx + col_idx + Vsize * (col_idx //
    // Vsize) + Vsize * imag_idx
    double* const Pmats = static_cast<double*>(std::aligned_alloc(ALIGNMENT, total_size));
    std::memset(Pmats, 0.0, total_size);

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
      const std::size_t row_end = rows_per_thread * tid + rows_per_thread;

      for (std::size_t row = row_start; row < row_end; ++row) {

        double* const Pmats_row = Pmats + 2 * order * ss_dim * row;
        KrMV_recurrence<VecXd, num_qubits>(Pmats_row, inds, clicks, transposes, row, num_meas, order);
      }

    } // omp parallel

    auto stop_time = Clock::now();
    time = Duration(stop_time - start_time).count();

    // reshape Pmats into batches
    reshape<VecXd, num_qubits>(batches, Pmats, order);

    free(Pmats);

  } else { // SCALAR

    // allocate aligned Pmats
    // access row of batch via: 2 * order * ss_dim * row_idx + 2 * ss_dim * order_idx + col_idx + Vsize * (col_idx //
    // Vsize) + Vsize * imag_idx
    z_t* const Pmats = static_cast<z_t*>(std::aligned_alloc(ALIGNMENT, total_size));
    std::fill(Pmats, Pmats + order * ss_dim * ss_dim, z_t{0.0, 0.0}); // memset is sketchy for std::complex

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
      const std::size_t row_end = rows_per_thread * tid + rows_per_thread;

      for (std::size_t row = row_start; row < row_end; ++row) {

        z_t* const Pmats_row = Pmats + order * ss_dim * row;
        KrMV_recurrence<num_qubits>(Pmats_row, inds, clicks, transposes, row, num_meas, order);
      }

    } // omp parallel

    auto stop_time = Clock::now();
    time = Duration(stop_time - start_time).count();

    // reshape Pmats into batches
    reshape<num_qubits>(batches, Pmats, order);

    free(Pmats);
  }

  post_process<num_qubits>(Tvals, batches, n_star, num_meas);

  kernel_times[0] = time;
};

// dispatcher function for fixing num_qubits and use_simd at compile time
void KrMM_recurrence(double* const Tvals, const int32_t* const inds, const int32_t* const clicks,
                     const std::size_t num_qubits, const std::size_t num_meas, const std::size_t n_star,
                     const bool use_simd, const int log2_num_threads, double* const kernel_times) {
  switch (num_qubits) {
  case 2:
    if (use_simd)
      KrMM_recurrence<2, true>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, kernel_times);
    else
      KrMM_recurrence<2, false>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, kernel_times);
    break;
  case 3:
    if (use_simd)
      KrMM_recurrence<3, true>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, kernel_times);
    else
      KrMM_recurrence<3, false>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, kernel_times);
    break;
  case 4:
    if (use_simd)
      KrMM_recurrence<4, true>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, kernel_times);
    else
      KrMM_recurrence<4, false>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, kernel_times);
    break;
  case 5:
    if (use_simd)
      KrMM_recurrence<5, true>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, kernel_times);
    else
      KrMM_recurrence<5, false>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, kernel_times);
    break;
  case 6:
    if (use_simd)
      KrMM_recurrence<6, true>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, kernel_times);
    else
      KrMM_recurrence<6, false>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, kernel_times);
    break;
  case 7:
    if (use_simd)
      KrMM_recurrence<7, true>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, kernel_times);
    else
      KrMM_recurrence<7, false>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, kernel_times);
    break;
  case 8:
    if (use_simd)
      KrMM_recurrence<8, true>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, kernel_times);
    else
      KrMM_recurrence<8, false>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, kernel_times);
    break;
  case 9:
    if (use_simd)
      KrMM_recurrence<9, true>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, kernel_times);
    else
      KrMM_recurrence<9, false>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, kernel_times);
    break;
  case 10:
    if (use_simd)
      KrMM_recurrence<10, true>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, kernel_times);
    else
      KrMM_recurrence<10, false>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, kernel_times);
    break;
  case 11:
    if (use_simd)
      KrMM_recurrence<11, true>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, kernel_times);
    else
      KrMM_recurrence<11, false>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, kernel_times);
    break;
  case 12:
    if (use_simd)
      KrMM_recurrence<12, true>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, kernel_times);
    else
      KrMM_recurrence<12, false>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, kernel_times);
    break;
  case 13:
    if (use_simd)
      KrMM_recurrence<13, true>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, kernel_times);
    else
      KrMM_recurrence<13, false>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, kernel_times);
    break;
  case 14:
    if (use_simd)
      KrMM_recurrence<14, true>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, kernel_times);
    else
      KrMM_recurrence<14, false>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, kernel_times);
    break;
  case 15:
    if (use_simd)
      KrMM_recurrence<15, true>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, kernel_times);
    else
      KrMM_recurrence<15, false>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, kernel_times);
    break;
  case 16:
    if (use_simd)
      KrMM_recurrence<16, true>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, kernel_times);
    else
      KrMM_recurrence<16, false>(Tvals, inds, clicks, num_meas, n_star, log2_num_threads, kernel_times);
    break;
  default:
    assert(false);
    break;
  }
};

} // namespace batched

} // namespace QFI

} // namespace shadows
