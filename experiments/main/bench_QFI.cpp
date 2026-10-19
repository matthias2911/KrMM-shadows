#include "shadows.hpp"

#include "utils.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iomanip>

// processes .npz file in <data_in_file>
// slices input data into iid samples and writes avg. results to <data_out_file>

namespace Qutils = utils::QFI;
namespace SQR = shadows::QFI::running;
namespace SQB = shadows::QFI::batched;

int main(int argc, char* argv[]) {

  const utils::Config pars = utils::Config(argc, argv);
  Qutils::npz_parser data = Qutils::npz_parser(pars.in_file);

  const std::size_t num_meas =
      static_cast<std::size_t>(pars.sample_factor * static_cast<double>(data.num_meas / pars.num_runs));
  const std::size_t num_qubits = data.num_qubits;

  const std::size_t n_star = data.n_star;
  const std::size_t k_order = 2 * data.n_star;

  data.add_used_pars(num_meas, k_order, pars.num_runs, pars.num_data_points);

  assert((pars.sample_factor > 0.0) && (pars.sample_factor <= 1.0));
  assert(num_meas * num_qubits * pars.num_runs <= data.num_meas * data.num_qubits);
  assert((pars.version.rfind("running", 0) == 0) || (pars.version.rfind("batched", 0) == 0));
  assert(data.QFI >= 0.0);

  const bool running_ver = (pars.version.rfind("running", 0) == 0);
  const bool batched_ver = !running_ver;

  utils::Timer stopwatch(pars.num_runs);

  std::vector<double> Tvals(pars.num_runs * k_order, 0.0);

  std::vector<double> Tvals_run(pars.num_runs * k_order, 0.0);

  std::vector<double> inter_Tvals(pars.num_runs * pars.num_data_points * k_order, 0.0);
  std::vector<std::size_t> inter_tidx(pars.num_data_points + 1, 0);

  std::vector<double> kernel_times(pars.num_runs, 0.0);

  // allocate aligned inds and clicks
  std::size_t total_size = num_meas * num_qubits * sizeof(int32_t);
  std::size_t total_size_padded = ((total_size + shadows::ALIGNMENT - 1) / shadows::ALIGNMENT) * shadows::ALIGNMENT;
  assert(total_size_padded % shadows::ALIGNMENT == 0);
  assert(total_size_padded >= total_size);
  int32_t* const inds = static_cast<int32_t*>(std::aligned_alloc(shadows::ALIGNMENT, total_size_padded));
  int32_t* const clicks = static_cast<int32_t*>(std::aligned_alloc(shadows::ALIGNMENT, total_size_padded));

#ifdef DEBUG
  std::cerr << "--------------- data from " << pars.in_file << " ---------------" << std::endl;
  data.print(2 * data.num_qubits, k_order, std::cerr);
#endif

  for (std::size_t run = 0; run < pars.num_runs; ++run) {

    std::memcpy(inds, data.inds + num_meas * num_qubits * run, total_size);
    std::memcpy(clicks, data.clicks + num_meas * num_qubits * run, total_size);

    if (running_ver) {
      double* const inter_Tvals_run = inter_Tvals.data() + pars.num_data_points * k_order * run;
      double* const kernel_times_run = kernel_times.data() + run;
      std::size_t* const inter_tidx_run = inter_tidx.data();

      stopwatch.start();
      Tvals_run =
          SQR::predict_Tvals(inds, clicks, num_qubits, num_meas, n_star, pars.var_offset, pars.num_data_points,
                             inter_Tvals_run, inter_tidx_run, kernel_times_run, pars.log2_num_threads, pars.use_simd);
      stopwatch.stop();
    }

    if (batched_ver) {
      double* const kernel_times_run = kernel_times.data() + run;

      stopwatch.start();
      Tvals_run = SQB::predict_Tvals(inds, clicks, num_qubits, num_meas, n_star, pars.var_offset, kernel_times_run,
                                     pars.log2_num_threads, pars.use_simd);
      stopwatch.stop();
    }

#ifdef DEBUG // print results
    Qutils::debug_print(num_qubits, num_meas, n_star, k_order, pars.num_data_points, Tvals_run, stopwatch[run],
                        data.QFI, inter_Tvals.data(), inter_tidx.data());
#endif

    for (std::size_t nidx = 1; nidx <= k_order; ++nidx) {
      std::size_t idx = k_order * run + (nidx - 1);
      Tvals[idx] = Tvals_run[nidx - 1];
    }
  }

  std::size_t num_threads;
  std::size_t vector_width;
  double rmMB;
  double rsptMB;
  double rfMF;
  double rmtMB;
  if (running_ver) {
    num_threads = SQR::get_num_threads(pars.log2_num_threads, std::size_t{1} << num_qubits);
    vector_width = SQR::get_vector_width(pars.use_simd, num_qubits);
    rmMB = SQR::rmMB(num_qubits, k_order, num_meas, num_threads);
    rsptMB = SQR::rsptMB(num_qubits, k_order);
    rfMF = SQR::rfMF(num_qubits, k_order, num_meas, num_threads, pars.num_data_points);
    rmtMB = SQR::rmtMB(num_qubits, k_order, num_meas, num_threads);
  }
  if (batched_ver) {
    num_threads = SQB::get_num_threads(pars.log2_num_threads, std::size_t{1} << num_qubits);
    vector_width = SQB::get_vector_width(pars.use_simd, num_qubits);
    rmMB = SQB::rmMB(num_qubits, k_order, num_meas, num_threads);
    rsptMB = SQB::rsptMB(num_qubits, k_order);
    rfMF = SQB::rfMF(num_qubits, k_order, num_meas, num_threads);
    rmtMB = SQB::rmtMB(num_qubits, k_order, num_meas, num_threads);
  }

  // write results
  data.write(pars.out_file, num_threads, vector_width, rmMB, rsptMB, rfMF, rmtMB, stopwatch.median(), Tvals.data(),
             inter_Tvals.data(), inter_tidx.data(), kernel_times.data());

  free(inds);
  free(clicks);

  return EXIT_SUCCESS;
}