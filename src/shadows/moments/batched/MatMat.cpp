#include "shadows/moments/batched/MatMat.hpp"

#include "shadows/moments/batched/MatVec.hpp"
#include "shadows/moments/batched/PostProcess.hpp"

#include <chrono>

#include <mpi.h>
#include <omp.h>

namespace shadows {

namespace moments {

namespace batched {

// determine the number of threads to use
std::size_t get_nts(const int log2_num_threads, const std::size_t ss_dim) {
  std::size_t num_threads;
  const std::size_t max_num_threads_sys = static_cast<std::size_t>(omp_get_max_threads());
  const std::size_t max_num_threads = (max_num_threads_sys > ss_dim) ? ss_dim : max_num_threads_sys;
  if (log2_num_threads < 0) {
    num_threads = max_num_threads;
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
void KrMM_recurrence(double* __restrict__ const moments, const int32_t* __restrict__ const inds,
                     const int32_t* __restrict__ const clicks, const int32_t* __restrict__ const transposes,
                     const std::size_t num_meas, const std::size_t order, const int log2_num_threads) {
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

      const std::size_t base = ss_dim / num_threads; 
      const std::size_t rem = ss_dim % num_threads;  
      const std::size_t row_start = base * tid + std::min<std::size_t>(tid, rem);
      const std::size_t row_end = row_start + base + (tid < rem ? 1 : 0);

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

      const std::size_t base = ss_dim / num_threads; 
      const std::size_t rem = ss_dim % num_threads;  
      const std::size_t row_start = base * tid + std::min<std::size_t>(tid, rem);
      const std::size_t row_end = row_start + base + (tid < rem ? 1 : 0);

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

  post_process<num_qubits>(moments, batches, order, num_meas);

  // this is really ugly; use order 1 moment as storage for kernel runtime for performance metrics e.g. kernel
  // throughput; this runtime should not be used at all in distributed setting (before MPI barrier)
  moments[0] = time; // full runtime of algorithm is obtained in main

  free(batches);
};

// dispatcher function for fixing num_qubits and use_simd at compile time
void KrMM_recurrence(double* const moments, const int32_t* const inds, const int32_t* const clicks,
                     const int32_t* const transposes, const std::size_t num_qubits, const std::size_t num_meas,
                     const std::size_t order, const bool use_simd, const int log2_num_threads) {
  switch (num_qubits) {
  case 2:
    if (use_simd)
      KrMM_recurrence<2, true>(moments, inds, clicks, transposes, num_meas, order, log2_num_threads);
    else
      KrMM_recurrence<2, false>(moments, inds, clicks, transposes, num_meas, order, log2_num_threads);
    break;
  case 3:
    if (use_simd)
      KrMM_recurrence<3, true>(moments, inds, clicks, transposes, num_meas, order, log2_num_threads);
    else
      KrMM_recurrence<3, false>(moments, inds, clicks, transposes, num_meas, order, log2_num_threads);
    break;
  case 4:
    if (use_simd)
      KrMM_recurrence<4, true>(moments, inds, clicks, transposes, num_meas, order, log2_num_threads);
    else
      KrMM_recurrence<4, false>(moments, inds, clicks, transposes, num_meas, order, log2_num_threads);
    break;
  case 5:
    if (use_simd)
      KrMM_recurrence<5, true>(moments, inds, clicks, transposes, num_meas, order, log2_num_threads);
    else
      KrMM_recurrence<5, false>(moments, inds, clicks, transposes, num_meas, order, log2_num_threads);
    break;
  case 6:
    if (use_simd)
      KrMM_recurrence<6, true>(moments, inds, clicks, transposes, num_meas, order, log2_num_threads);
    else
      KrMM_recurrence<6, false>(moments, inds, clicks, transposes, num_meas, order, log2_num_threads);
    break;
  case 7:
    if (use_simd)
      KrMM_recurrence<7, true>(moments, inds, clicks, transposes, num_meas, order, log2_num_threads);
    else
      KrMM_recurrence<7, false>(moments, inds, clicks, transposes, num_meas, order, log2_num_threads);
    break;
  case 8:
    if (use_simd)
      KrMM_recurrence<8, true>(moments, inds, clicks, transposes, num_meas, order, log2_num_threads);
    else
      KrMM_recurrence<8, false>(moments, inds, clicks, transposes, num_meas, order, log2_num_threads);
    break;
  case 9:
    if (use_simd)
      KrMM_recurrence<9, true>(moments, inds, clicks, transposes, num_meas, order, log2_num_threads);
    else
      KrMM_recurrence<9, false>(moments, inds, clicks, transposes, num_meas, order, log2_num_threads);
    break;
  case 10:
    if (use_simd)
      KrMM_recurrence<10, true>(moments, inds, clicks, transposes, num_meas, order, log2_num_threads);
    else
      KrMM_recurrence<10, false>(moments, inds, clicks, transposes, num_meas, order, log2_num_threads);
    break;
  case 11:
    if (use_simd)
      KrMM_recurrence<11, true>(moments, inds, clicks, transposes, num_meas, order, log2_num_threads);
    else
      KrMM_recurrence<11, false>(moments, inds, clicks, transposes, num_meas, order, log2_num_threads);
    break;
  case 12:
    if (use_simd)
      KrMM_recurrence<12, true>(moments, inds, clicks, transposes, num_meas, order, log2_num_threads);
    else
      KrMM_recurrence<12, false>(moments, inds, clicks, transposes, num_meas, order, log2_num_threads);
    break;
  case 13:
    if (use_simd)
      KrMM_recurrence<13, true>(moments, inds, clicks, transposes, num_meas, order, log2_num_threads);
    else
      KrMM_recurrence<13, false>(moments, inds, clicks, transposes, num_meas, order, log2_num_threads);
    break;
  case 14:
    if (use_simd)
      KrMM_recurrence<14, true>(moments, inds, clicks, transposes, num_meas, order, log2_num_threads);
    else
      KrMM_recurrence<14, false>(moments, inds, clicks, transposes, num_meas, order, log2_num_threads);
    break;
  case 15:
    if (use_simd)
      KrMM_recurrence<15, true>(moments, inds, clicks, transposes, num_meas, order, log2_num_threads);
    else
      KrMM_recurrence<15, false>(moments, inds, clicks, transposes, num_meas, order, log2_num_threads);
    break;
  case 16:
    if (use_simd)
      KrMM_recurrence<16, true>(moments, inds, clicks, transposes, num_meas, order, log2_num_threads);
    else
      KrMM_recurrence<16, false>(moments, inds, clicks, transposes, num_meas, order, log2_num_threads);
    break;
  default:
    assert(false);
    break;
  }
};

} // namespace batched

} // namespace moments

} // namespace shadows
