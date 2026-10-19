#include "shadows.hpp"

#include "utils.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iomanip>

#include <mpi.h>

// processes .npz file in <data_in_file>
// slices input data into iid samples and writes avg. results to <data_out_file>

namespace Mutils = utils::moments;
namespace SMR = shadows::moments::running;
namespace SMB = shadows::moments::batched;

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // std::cerr << "Hello from rank " << world_rank << " of " << world_size << " total processes in main.\n" <<
  // std::endl;

  const utils::Config pars = utils::Config(argc, argv);
  Mutils::npz_parser data = Mutils::npz_parser(pars.in_file);

  const std::size_t num_meas =
      static_cast<std::size_t>(pars.sample_factor * static_cast<double>(data.num_meas / pars.num_runs));
  const std::size_t num_qubits = data.num_qubits;

  data.add_used_pars(num_meas, pars.order, pars.num_runs, pars.num_data_points);

  assert((pars.sample_factor > 0.0) && (pars.sample_factor <= 1.0));
  assert(num_meas * num_qubits * pars.num_runs <= data.num_meas * data.num_qubits);
  // assert((pars.version == "running") || (pars.version == "batched"));
  assert((pars.version.rfind("running", 0) == 0) || (pars.version.rfind("batched", 0) == 0));
  assert(!((pars.PT == 1) && (data.pt == 0)));

  const bool running_ver = (pars.version.rfind("running", 0) == 0);
  const bool batched_ver = !running_ver;

  utils::Timer stopwatch(pars.num_runs);

  std::vector<double> moments(pars.num_runs * pars.order, 0.0);
  std::vector<double> standard_errs(pars.num_runs * pars.order, 0.0);

  std::vector<double> moments_run(pars.num_runs * pars.order, 0.0);
  std::vector<double> standard_errs_run(pars.num_runs * pars.order, 0.0);

  std::vector<double> inter_moments(pars.num_runs * pars.num_data_points * pars.order, 0.0);
  std::vector<double> inter_std_errs(pars.num_runs * pars.num_data_points * pars.order, 0.0);
  std::vector<std::size_t> inter_tidx(pars.num_data_points * pars.var_offset + 1, 0); // pad by 1

  const int32_t* transposes = nullptr;
  const std::vector<int32_t> no_transposes(num_qubits, 0);
  if (pars.PT)
    transposes = data.transposes;
  else
    transposes = no_transposes.data();

  // allocate aligned inds and clicks
  std::size_t total_size = num_meas * num_qubits * sizeof(int32_t);
  std::size_t total_size_padded = ((total_size + shadows::ALIGNMENT - 1) / shadows::ALIGNMENT) * shadows::ALIGNMENT;
  assert(total_size_padded % shadows::ALIGNMENT == 0);
  assert(total_size_padded >= total_size);
  int32_t* const inds = static_cast<int32_t*>(std::aligned_alloc(shadows::ALIGNMENT, total_size_padded));
  int32_t* const clicks = static_cast<int32_t*>(std::aligned_alloc(shadows::ALIGNMENT, total_size_padded));

#ifdef DEBUG
  std::cerr << "--------------- data from " << pars.in_file << " ---------------" << std::endl;
  data.print(2 * data.num_qubits, pars.order, std::cerr);
#endif

  for (std::size_t run = 0; run < pars.num_runs; ++run) {

    std::memcpy(inds, data.inds + num_meas * num_qubits * run, total_size);
    std::memcpy(clicks, data.clicks + num_meas * num_qubits * run, total_size);

    if (running_ver) {
      double* const inter_moments_run =
          pars.num_data_points ? inter_moments.data() + pars.num_data_points * pars.order * run : nullptr;
      double* const inter_std_errs_run =
          pars.num_data_points ? inter_std_errs.data() + pars.num_data_points * pars.order * run : nullptr;
      std::size_t* const inter_tidx_run = pars.num_data_points ? inter_tidx.data() : nullptr;

      stopwatch.start();
      std::tie(moments_run, standard_errs_run) = SMR::predict_moments(
          inds, clicks, transposes, num_qubits, num_meas, pars.order, pars.var_offset, pars.num_data_points,
          inter_moments_run, inter_std_errs_run, inter_tidx_run, pars.log2_num_threads, pars.use_simd);
      stopwatch.stop();
    }

    if (batched_ver) {
      if (world_rank == 0) {
        stopwatch.start();
        moments_run = SMB::predict_moments(inds, clicks, transposes, num_qubits, num_meas, pars.order,
                                           pars.log2_num_threads, pars.use_simd);
        stopwatch.stop();
      }
    }

#ifdef DEBUG // print results
    Mutils::debug_print(num_qubits, num_meas, pars.order, pars.var_offset, pars.num_data_points, moments_run,
                        standard_errs_run, stopwatch[run], data.moments, data.moments_pt, inter_moments.data(),
                        inter_std_errs.data(), inter_tidx.data());
#endif

    for (std::size_t nidx = 1; nidx <= pars.order; ++nidx) {
      std::size_t idx = pars.order * run + (nidx - 1);
      moments[idx] = moments_run[nidx - 1];
      standard_errs[idx] = standard_errs_run[nidx - 1];
    }
  }

  std::size_t num_threads;
  std::size_t num_ranks;
  std::size_t vector_width;
  double rmMB;
  double rsptMB;
  double rfMF;
  double rmtMB;
  if (running_ver) {
    num_threads = SMR::get_num_threads(pars.log2_num_threads, std::size_t{1} << num_qubits, world_size);
    num_ranks = SMR::get_num_ranks(world_size, num_threads, std::size_t{1} << num_qubits);
    vector_width = SMR::get_vector_width(pars.use_simd, num_qubits);
    rmMB = SMR::rmMB(num_qubits, pars.order, num_meas, pars.var_offset, num_threads, num_ranks);
    rsptMB = SMR::rsptMB(num_qubits, pars.order);
    rfMF = SMR::rfMF(num_qubits, pars.order, num_meas, pars.var_offset, num_threads, pars.num_data_points);
    rmtMB = SMR::rmtMB(num_qubits, pars.order, num_meas, pars.var_offset, num_threads);
  }
  if (batched_ver) {
    num_threads = SMB::get_num_threads(pars.log2_num_threads, std::size_t{1} << num_qubits);
    num_ranks = 1;
    vector_width = SMB::get_vector_width(pars.use_simd, num_qubits);
    rmMB = SMB::rmMB(num_qubits, pars.order, num_meas, num_threads);
    rsptMB = SMB::rsptMB(num_qubits);
    rfMF = SMB::rfMF(num_qubits, num_meas);
    rmtMB = SMB::rmtMB(num_qubits, num_meas, num_threads);
  }

  // write results
  if (world_rank == 0)
    data.write(pars.out_file, pars.PT, num_threads, num_ranks, vector_width, pars.var_offset, rmMB, rsptMB, rfMF, rmtMB,
               stopwatch.median(), moments.data(), standard_errs.data(), inter_moments.data(), inter_std_errs.data(),
               inter_tidx.data());

  free(inds);
  free(clicks);

  MPI_Finalize();

  return EXIT_SUCCESS;
}