#include "shadows/moments/running/MatMat.hpp"

#include "shadows/moments/running/MatVec.hpp"
#include "shadows/moments/running/PostProcess.hpp"

#include <chrono>

#include <mpi.h>
#include <omp.h>

namespace shadows {

namespace moments {

namespace running {

// determine the number of threads to use
// assumes SMT is available
std::size_t get_nts(const int log2_num_threads, const std::size_t ss_dim, const int world_size) {
  std::size_t num_threads;
  const std::size_t max_num_threads_sys = static_cast<std::size_t>(omp_get_max_threads());
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

// determine the number of ranks to use
std::size_t get_nrks(const int world_size, const std::size_t nts, const std::size_t ss_dim) {
  std::size_t num_ranks = 1;
  if (nts < ss_dim) {
    num_ranks = ss_dim / nts;
    if (num_ranks > world_size)
      num_ranks = world_size;
  }
  return num_ranks;
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
// each data point idx is at least var_offset + order apart from the next
// final data point is at most num_meas - order - 1
bool fill_inter_tidx(std::size_t* __restrict__ const inter_tidx, const std::size_t num_meas, const std::size_t order,
                     const std::size_t num_data_points, const std::size_t var_offset) {
  if (num_data_points == 0)
    return 0;

  assert(num_data_points * order + order - 1 <= num_meas);
  assert((num_data_points - 1) * var_offset + order <= num_meas);
  assert(var_offset > order);
  assert(order < num_meas);

  const std::size_t max_start = num_meas - order;
  const std::size_t min_gap = var_offset + order + 1;

  assert(min_gap <= max_start);
  assert((num_data_points - 1) * min_gap + min_gap <= max_start || num_data_points == 1);

  auto write_block = [&](std::size_t point_idx, std::size_t data_idx) {
    std::size_t* __restrict__ const block = inter_tidx + point_idx * var_offset;
    for (std::size_t k = 0; k < var_offset; ++k)
      block[k] = data_idx - (var_offset - 1) + k;
  };

  if (num_data_points == 1) {
    assert(min_gap <= max_start);
    write_block(0, max_start);
  } else {

    const std::size_t lo_start = min_gap;
    assert((num_data_points - 1) * min_gap + lo_start <= max_start);

    const std::size_t extra_budget = max_start - lo_start - (num_data_points - 1) * min_gap;

    const double lo = std::log(1.0);
    const double hi = std::log(static_cast<double>(extra_budget) + 1.0);

    std::size_t prev_extra = 0;
    for (std::size_t i = 0; i < num_data_points; ++i) {
      const double t = static_cast<double>(i) / static_cast<double>(num_data_points - 1);
      const double val = std::exp(lo + t * (hi - lo)) - 1.0;
      long long rounded = static_cast<long long>(std::llround(val));
      if (rounded < 0)
        rounded = 0;
      std::size_t extra = static_cast<std::size_t>(rounded);

      if (i > 0 && extra < prev_extra)
        extra = prev_extra;
      if (extra > extra_budget)
        extra = extra_budget;

      const std::size_t data_idx = lo_start + i * min_gap + extra;
      write_block(i, data_idx);
      prev_extra = extra;
    }
  }

  // sanity checks
  assert(inter_tidx[(num_data_points - 1) * var_offset + var_offset - 1] == max_start);
  assert(inter_tidx[var_offset - 1] >= min_gap);
  for (std::size_t i = 1; i < num_data_points; ++i) {
    const std::size_t cur = inter_tidx[i * var_offset + var_offset - 1];
    const std::size_t prv = inter_tidx[(i - 1) * var_offset + var_offset - 1];
    assert(cur - prv >= min_gap);
  }

  inter_tidx[num_data_points * var_offset] = num_meas;

  return 0;
}

// Kronecker Matrix-Matrix recurrence
// essentially just maps threads and nodes to their respective KrMV recurrences
// and post-processes gathered data
template <std::size_t num_qubits, bool __use_simd>
void KrMM_recurrence(double* __restrict__ const moments, double* __restrict__ const standard_errs,
                     const int32_t* __restrict__ const inds, const int32_t* __restrict__ const clicks,
                     const int32_t* __restrict__ const transposes, const std::size_t num_meas, const std::size_t order,
                     const std::size_t var_offset, const int log2_num_threads, const std::size_t num_data_points,
                     double* __restrict__ const inter_moments, double* __restrict__ const inter_std_errs,
                     std::size_t* __restrict__ const inter_tidx) {
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

  // get rank and world size
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  assert(world_rank < world_size);

  // extract the number of threads
  const std::size_t nts = get_nts(log2_num_threads, ss_dim, world_size);

  const std::size_t num_ranks = get_nrks(world_size, nts, ss_dim);
  assert(num_ranks > 0 && (num_ranks & (num_ranks - 1)) == 0); // pow of 2
  assert(num_ranks * nts <= ss_dim);

  // create new comm with only world_ranks < num_ranks
  // world_rank is passed as key -> world_rank == active_rank if world_rank < num_ranks
  MPI_Comm active_comm;
  int color = (world_rank < num_ranks) ? 0 : MPI_UNDEFINED;
  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &active_comm);

  if (world_rank < num_ranks) {

    int active_rank, active_size;
    MPI_Comm_rank(active_comm, &active_rank);
    MPI_Comm_size(active_comm, &active_size);

    const std::size_t rid = static_cast<std::size_t>(active_rank);

    assert(rid == static_cast<std::size_t>(world_rank));
    assert(num_ranks == static_cast<std::size_t>(active_size));
    assert(num_ranks <= static_cast<std::size_t>(world_size));

    // final estimates
    const std::size_t tr_start = num_meas - var_offset;
    assert(tr_start < num_meas);
    const std::size_t len_var_meas = num_meas - tr_start + order;
    const std::size_t tvm_per_thread = order * len_var_meas;
    std::vector<double> traces_vec(nts * tvm_per_thread, 0.0); // order*len_var_meas*tid + order*var_meas + midx
    double* const traces_shared = traces_vec.data();

    // intermediate estimates
    // num_data_points*var_offset*order*tid + var_offset*order*dpt_idx + order*vidx + midx
    std::vector<double> inter_traces_vec(nts * num_data_points * order * var_offset, 0.0);
    double* const inter_traces_shared = inter_traces_vec.data();

#ifdef DEBUG
    std::cerr << "\n--------------- launch pars --------------- " << std::endl;
    std::cerr << "num_qubits=" << num_qubits << std::endl;
    std::cerr << "order=" << order << std::endl;
    std::cerr << "num_meas=" << num_meas << std::endl;
    std::cerr << "var_offset=" << var_offset << std::endl;
    std::cerr << "Vsize=" << Vsize << std::endl;
    std::cerr << "use_simd=" << use_simd << std::endl;
    std::cerr << "nts=" << nts << std::endl;
    std::cerr << "world_size=" << world_size << std::endl;
    std::cerr << "tr_start=" << tr_start << std::endl;
    std::cerr << "len_var_meas=" << len_var_meas << std::endl;
    std::cerr << "num_data_points=" << num_data_points << std::endl;
    std::cerr << "--------------- launch pars --------------- \n" << std::endl;
#endif

    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double>;

    // std::cerr << "Hello from rank " << world_rank << " of " << world_size << " total processes.\n" << std::endl;

    auto start_time = Clock::now();

#pragma omp parallel num_threads(nts)
    {
      const std::size_t tid = omp_get_thread_num();
      const std::size_t num_threads = omp_get_num_threads();
      assert(num_threads == nts);
      assert(tid < num_threads);

      const std::size_t rows_per_rank = ss_dim / num_ranks;
      const std::size_t base = rows_per_rank / num_threads;
      const std::size_t rem = rows_per_rank % num_threads;
      const std::size_t row_start = rows_per_rank * rid + base * tid + std::min<std::size_t>(tid, rem);
      const std::size_t row_end = row_start + base + (tid < rem ? 1 : 0);

      double* const traces = traces_shared + tvm_per_thread * tid;
      double* const inter_traces =
          num_data_points ? inter_traces_shared + num_data_points * order * var_offset * tid : nullptr;

      if constexpr (use_simd) {

        // allocate aligned Pmats
        const std::size_t total_size = 2 * order * ss_dim * sizeof(double);
        assert(total_size % ALIGNMENT == 0);
        double* const Pmats = static_cast<double*>(std::aligned_alloc(ALIGNMENT, total_size));

        for (std::size_t row = row_start; row < row_end; ++row) {
          KrMV_recurrence<VecXd, num_qubits>(Pmats, traces, inds, clicks, transposes, row, num_meas, order, tr_start,
                                             num_data_points, inter_traces, inter_tidx);
        }
        free(Pmats);
      } else {

        // allocate aligned Pmats
        const std::size_t total_size = order * ss_dim * sizeof(z_t);
        assert(total_size % ALIGNMENT == 0);
        z_t* const Pmats = static_cast<z_t*>(std::aligned_alloc(ALIGNMENT, total_size));

        for (std::size_t row = row_start; row < row_end; ++row) {
          KrMV_recurrence<num_qubits>(Pmats, traces, inds, clicks, transposes, row, num_meas, order, tr_start,
                                      num_data_points, inter_traces, inter_tidx);
        }
        free(Pmats);
      }

    } // omp parallel

    auto stop_time = Clock::now();
    double time = Duration(stop_time - start_time).count();

    reduce(inter_traces_shared, order * var_offset, num_data_points, nts);
    reduce(traces_shared, order, len_var_meas, nts);

    MPI_Barrier(active_comm);

    // reduce over ranks
    if (num_ranks > 1) {
      if (rid == 0) {
        MPI_Reduce(MPI_IN_PLACE, inter_traces_shared, num_data_points * var_offset * order, MPI_DOUBLE, MPI_SUM, 0,
                   active_comm);
        MPI_Reduce(MPI_IN_PLACE, traces_shared, tvm_per_thread, MPI_DOUBLE, MPI_SUM, 0, active_comm);
      } else {
        MPI_Reduce(inter_traces_shared, NULL, num_data_points * var_offset * order, MPI_DOUBLE, MPI_SUM, 0,
                   active_comm);
        MPI_Reduce(traces_shared, NULL, tvm_per_thread, MPI_DOUBLE, MPI_SUM, 0, active_comm);
      }
      MPI_Barrier(active_comm);
    }

    if (rid == 0) {
      post_process_inter(inter_moments, inter_std_errs, inter_traces_shared, inter_tidx, order, var_offset,
                         num_data_points);
      post_process(moments, standard_errs, traces_shared, num_meas, order, len_var_meas, tr_start);

      // this is really ugly; use order 1 moment as storage for kernel runtime for performance metrics e.g. kernel
      // throughput; this runtime should not be used at all in distributed setting (before MPI barrier)
      moments[0] = time; // full runtime of algorithm is obtained in main
    }

    MPI_Barrier(active_comm);
    MPI_Comm_free(&active_comm);
  }

  MPI_Barrier(MPI_COMM_WORLD);
};

// dispatcher function for fixing num_qubits and use_simd at compile time
void KrMM_recurrence(double* const moments, double* const standard_errs, const int32_t* const inds,
                     const int32_t* const clicks, const int32_t* const transposes, const std::size_t num_qubits,
                     const std::size_t num_meas, const std::size_t order, const std::size_t var_offset,
                     const bool use_simd, const int log2_num_threads, const std::size_t num_data_points,
                     double* const inter_moments, double* const inter_std_errs, std::size_t* const inter_tidx) {
  switch (num_qubits) {
  case 2:
    if (use_simd)
      KrMM_recurrence<2, true>(moments, standard_errs, inds, clicks, transposes, num_meas, order, var_offset,
                               log2_num_threads, num_data_points, inter_moments, inter_std_errs, inter_tidx);
    else
      KrMM_recurrence<2, false>(moments, standard_errs, inds, clicks, transposes, num_meas, order, var_offset,
                                log2_num_threads, num_data_points, inter_moments, inter_std_errs, inter_tidx);
    break;
  case 3:
    if (use_simd)
      KrMM_recurrence<3, true>(moments, standard_errs, inds, clicks, transposes, num_meas, order, var_offset,
                               log2_num_threads, num_data_points, inter_moments, inter_std_errs, inter_tidx);
    else
      KrMM_recurrence<3, false>(moments, standard_errs, inds, clicks, transposes, num_meas, order, var_offset,
                                log2_num_threads, num_data_points, inter_moments, inter_std_errs, inter_tidx);
    break;
  case 4:
    if (use_simd)
      KrMM_recurrence<4, true>(moments, standard_errs, inds, clicks, transposes, num_meas, order, var_offset,
                               log2_num_threads, num_data_points, inter_moments, inter_std_errs, inter_tidx);
    else
      KrMM_recurrence<4, false>(moments, standard_errs, inds, clicks, transposes, num_meas, order, var_offset,
                                log2_num_threads, num_data_points, inter_moments, inter_std_errs, inter_tidx);
    break;
  case 5:
    if (use_simd)
      KrMM_recurrence<5, true>(moments, standard_errs, inds, clicks, transposes, num_meas, order, var_offset,
                               log2_num_threads, num_data_points, inter_moments, inter_std_errs, inter_tidx);
    else
      KrMM_recurrence<5, false>(moments, standard_errs, inds, clicks, transposes, num_meas, order, var_offset,
                                log2_num_threads, num_data_points, inter_moments, inter_std_errs, inter_tidx);
    break;
  case 6:
    if (use_simd)
      KrMM_recurrence<6, true>(moments, standard_errs, inds, clicks, transposes, num_meas, order, var_offset,
                               log2_num_threads, num_data_points, inter_moments, inter_std_errs, inter_tidx);
    else
      KrMM_recurrence<6, false>(moments, standard_errs, inds, clicks, transposes, num_meas, order, var_offset,
                                log2_num_threads, num_data_points, inter_moments, inter_std_errs, inter_tidx);
    break;
  case 7:
    if (use_simd)
      KrMM_recurrence<7, true>(moments, standard_errs, inds, clicks, transposes, num_meas, order, var_offset,
                               log2_num_threads, num_data_points, inter_moments, inter_std_errs, inter_tidx);
    else
      KrMM_recurrence<7, false>(moments, standard_errs, inds, clicks, transposes, num_meas, order, var_offset,
                                log2_num_threads, num_data_points, inter_moments, inter_std_errs, inter_tidx);
    break;
  case 8:
    if (use_simd)
      KrMM_recurrence<8, true>(moments, standard_errs, inds, clicks, transposes, num_meas, order, var_offset,
                               log2_num_threads, num_data_points, inter_moments, inter_std_errs, inter_tidx);
    else
      KrMM_recurrence<8, false>(moments, standard_errs, inds, clicks, transposes, num_meas, order, var_offset,
                                log2_num_threads, num_data_points, inter_moments, inter_std_errs, inter_tidx);
    break;
  case 9:
    if (use_simd)
      KrMM_recurrence<9, true>(moments, standard_errs, inds, clicks, transposes, num_meas, order, var_offset,
                               log2_num_threads, num_data_points, inter_moments, inter_std_errs, inter_tidx);
    else
      KrMM_recurrence<9, false>(moments, standard_errs, inds, clicks, transposes, num_meas, order, var_offset,
                                log2_num_threads, num_data_points, inter_moments, inter_std_errs, inter_tidx);
    break;
  case 10:
    if (use_simd)
      KrMM_recurrence<10, true>(moments, standard_errs, inds, clicks, transposes, num_meas, order, var_offset,
                                log2_num_threads, num_data_points, inter_moments, inter_std_errs, inter_tidx);
    else
      KrMM_recurrence<10, false>(moments, standard_errs, inds, clicks, transposes, num_meas, order, var_offset,
                                 log2_num_threads, num_data_points, inter_moments, inter_std_errs, inter_tidx);
    break;
  case 11:
    if (use_simd)
      KrMM_recurrence<11, true>(moments, standard_errs, inds, clicks, transposes, num_meas, order, var_offset,
                                log2_num_threads, num_data_points, inter_moments, inter_std_errs, inter_tidx);
    else
      KrMM_recurrence<11, false>(moments, standard_errs, inds, clicks, transposes, num_meas, order, var_offset,
                                 log2_num_threads, num_data_points, inter_moments, inter_std_errs, inter_tidx);
    break;
  case 12:
    if (use_simd)
      KrMM_recurrence<12, true>(moments, standard_errs, inds, clicks, transposes, num_meas, order, var_offset,
                                log2_num_threads, num_data_points, inter_moments, inter_std_errs, inter_tidx);
    else
      KrMM_recurrence<12, false>(moments, standard_errs, inds, clicks, transposes, num_meas, order, var_offset,
                                 log2_num_threads, num_data_points, inter_moments, inter_std_errs, inter_tidx);
    break;
  case 13:
    if (use_simd)
      KrMM_recurrence<13, true>(moments, standard_errs, inds, clicks, transposes, num_meas, order, var_offset,
                                log2_num_threads, num_data_points, inter_moments, inter_std_errs, inter_tidx);
    else
      KrMM_recurrence<13, false>(moments, standard_errs, inds, clicks, transposes, num_meas, order, var_offset,
                                 log2_num_threads, num_data_points, inter_moments, inter_std_errs, inter_tidx);
    break;
  case 14:
    if (use_simd)
      KrMM_recurrence<14, true>(moments, standard_errs, inds, clicks, transposes, num_meas, order, var_offset,
                                log2_num_threads, num_data_points, inter_moments, inter_std_errs, inter_tidx);
    else
      KrMM_recurrence<14, false>(moments, standard_errs, inds, clicks, transposes, num_meas, order, var_offset,
                                 log2_num_threads, num_data_points, inter_moments, inter_std_errs, inter_tidx);
    break;
  case 15:
    if (use_simd)
      KrMM_recurrence<15, true>(moments, standard_errs, inds, clicks, transposes, num_meas, order, var_offset,
                                log2_num_threads, num_data_points, inter_moments, inter_std_errs, inter_tidx);
    else
      KrMM_recurrence<15, false>(moments, standard_errs, inds, clicks, transposes, num_meas, order, var_offset,
                                 log2_num_threads, num_data_points, inter_moments, inter_std_errs, inter_tidx);
    break;
  case 16:
    if (use_simd)
      KrMM_recurrence<16, true>(moments, standard_errs, inds, clicks, transposes, num_meas, order, var_offset,
                                log2_num_threads, num_data_points, inter_moments, inter_std_errs, inter_tidx);
    else
      KrMM_recurrence<16, false>(moments, standard_errs, inds, clicks, transposes, num_meas, order, var_offset,
                                 log2_num_threads, num_data_points, inter_moments, inter_std_errs, inter_tidx);
    break;
  default:
    assert(false);
    break;
  }
};

} // namespace running

} // namespace moments

} // namespace shadows
