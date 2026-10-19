#include "utils.hpp"

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <limits>

namespace utils {

namespace moments {

/* ----------------------- npz_parser class definition ----------------------- */
npz_parser::npz_parser() {}

npz_parser::npz_parser(std::string file_name) {
  npz = cnpy::npz_load(file_name);
  num_qubits = npz["num_qubits"].data<int32_t>()[0];
  num_meas = npz["num_meas"].data<int32_t>()[0]; // WARNING: this risks overflowing! (~2e9)
  max_order = npz["max_order"].data<int32_t>()[0];
  pt = npz["pt"].data<int32_t>()[0];
  werner = npz["werner"].data<int32_t>()[0];
  werner_par = npz["werner_par"].data<double>()[0];
  transposes = npz["transposes"].data<int32_t>();
  moments = npz["moments"].data<double>();
  moments_pt = npz["moments_pt"].data<double>();
  inds = npz["inds"].data<int32_t>();
  clicks = npz["clicks"].data<int32_t>();
}

void npz_parser::print(std::size_t max_num_meas, std::size_t order_, std::ostream& out) const {
  assert(max_num_meas <= num_meas);
  std::size_t idx;
  out << "num_qubits: "
      << " ";
  out << num_qubits << " " << std::endl;
  out << "num_meas: "
      << " ";
  out << num_meas << " " << std::endl;
  out << "max_order: "
      << " ";
  out << max_order << " " << std::endl;
  out << "pt: "
      << " ";
  out << pt << " " << std::endl;
  out << "werner: "
      << " ";
  out << werner << " " << std::endl;
  out << "werner_par: "
      << " ";
  out << werner_par << " " << std::endl;
  out << "transposes: "
      << " ";
  for (idx = 0; idx < num_qubits; idx += 1)
    out << transposes[idx] << " ";
  out << std::endl;
  out << "moments: "
      << " ";
  for (idx = 0; idx < order_; idx += 1)
    out << moments[idx] << " ";
  out << std::endl;
  out << "moments_pt: "
      << " ";
  for (idx = 0; idx < order_; idx += 1)
    out << moments_pt[idx] << " ";
  out << std::endl;
  out << "inds: "
      << " ";
  for (idx = 0; idx < max_num_meas; idx += 1)
    out << inds[idx] << " ";
  out << std::endl;
  out << "clicks: "
      << " ";
  for (idx = 0; idx < max_num_meas; idx += 1)
    out << clicks[idx] << " ";
  out << std::endl;
}

void npz_parser::add_used_pars(std::size_t cons_num_meas_, std::size_t order_, std::size_t num_runs_,
                               std::size_t num_data_points_) {
  assert(cons_num_meas_ <= num_meas);
  assert(order_ <= max_order);
  assert(num_runs_ > 0);
  cons_num_meas = cons_num_meas_;
  order = order_;
  num_runs = num_runs_;
  num_data_points = num_data_points_;
};

void npz_parser::write(std::string file, const bool PT, const std::size_t num_threads, const std::size_t num_ranks,
                       const std::size_t vector_width, const std::size_t var_offset, double rmMB, double rsptMB,
                       double rfMF, double rmtMB, double runtime, const double* const est_moments,
                       const double* const est_std_errs, const double* const est_inter_moments,
                       const double* const est_inter_std_errs, const std::size_t* const inter_tidx) const {

  // first must "w"rite, then we must "a"ppend
  cnpy::npz_save(file, "num_qubits", &num_qubits, {1}, "w");           // number of quibts
  cnpy::npz_save(file, "num_meas", &cons_num_meas, {1}, "a");          // used num_meas
  cnpy::npz_save(file, "order", &order, {1}, "a");                     // used max. order
  cnpy::npz_save(file, "num_threads", &num_threads, {1}, "a");         // used number of threads
  cnpy::npz_save(file, "num_ranks", &num_ranks, {1}, "a");             // used number of threads
  cnpy::npz_save(file, "vector_width", &vector_width, {1}, "a");       // used number of threads
  cnpy::npz_save(file, "PT", &PT, {1}, "a");                           // whether moments have been estimated
  cnpy::npz_save(file, "werner", &werner, {1}, "a");                   // whether used state is a werner state
  cnpy::npz_save(file, "werner_par", &werner_par, {1}, "a");           // used werner parameter
  cnpy::npz_save(file, "var_offset", &var_offset, {1}, "a");           // used var_offset, ratio for var. est.
  cnpy::npz_save(file, "rmMB", &rmMB, {1}, "a");                       // used var_offset, ratio for var. est.
  cnpy::npz_save(file, "rsptMB", &rsptMB, {1}, "a");                   // used var_offset, ratio for var. est.
  cnpy::npz_save(file, "rfMF", &rfMF, {1}, "a");                       // used var_offset, ratio for var. est.
  cnpy::npz_save(file, "rmtMB", &rmtMB, {1}, "a");                     // used var_offset, ratio for var. est.
  cnpy::npz_save(file, "num_runs", &num_runs, {1}, "a");               // number of runs that were averaged over
  cnpy::npz_save(file, "num_data_points", &num_data_points, {1}, "a"); // number of save interm. moments
  cnpy::npz_save(file, "runtime", &runtime, {1}, "a");                 // mean runtime (over num_runs runs) of dsc algo
  cnpy::npz_save(file, "est_moments", est_moments, {num_runs * order}, "a");   // estimated moments
  cnpy::npz_save(file, "est_std_errs", est_std_errs, {num_runs * order}, "a"); // standard error of moment estimation
  if (num_data_points > 0) {
    cnpy::npz_save(file, "est_inter_moments", est_inter_moments, {num_runs * num_data_points * order},
                   "a"); // estimated moments
    cnpy::npz_save(file, "est_inter_std_errs", est_inter_std_errs, {num_runs * num_data_points * order},
                   "a");                                                    // estimated moments
    cnpy::npz_save(file, "inter_tidx", inter_tidx, {num_data_points}, "a"); // estimated moments
  }
  if (!PT) {
    cnpy::npz_save(file, "moments", moments, {order}, "a"); // correct moments
  } else {
    cnpy::npz_save(file, "transposes", transposes, {num_qubits}, "a"); // what qubits were transposed (in pt case)
    cnpy::npz_save(file, "moments", moments_pt, {order}, "a");         // correct pt moments
  }
};
/* ----------------------- npz_parser class definition ----------------------- */

void debug_print(std::size_t num_qubits, std::size_t num_meas, std::size_t order, std::size_t var_offset,
                 std::size_t num_data_points, const std::vector<double>& moments,
                 const std::vector<double>& standard_errs, double time, double* corr_moments, double* corr_moments_pt,
                 const double* const inter_moments, const double* const inter_std_errs,
                 const std::size_t* const inter_tidx) {
  std::cerr << "------------------- results -------------------" << std::endl;
  std::cerr << "num_qubits: " << num_qubits << std::endl;
  std::cerr << "num_meas: " << num_meas << std::endl;
  std::cerr << "order: " << order << std::endl;
  std::cerr << "var_offset: " << var_offset << std::endl;
  std::cerr << "num_data_points: " << num_data_points << std::endl;
  std::cerr << "moments: ";
  for (std::size_t nidx = 1; nidx <= order; ++nidx)
    std::cerr << moments[nidx - 1] << " ";
  std::cerr << std::endl;
  std::cerr << "corr_moments: ";
  for (std::size_t nidx = 1; nidx <= order; ++nidx)
    std::cerr << corr_moments[nidx - 1] << " ";
  std::cerr << std::endl;
  std::cerr << "standard_errs: ";
  for (std::size_t nidx = 1; nidx <= order; ++nidx)
    std::cerr << standard_errs[nidx - 1] << " ";
  std::cerr << std::endl;
  std::cerr << "runtime: " << time << " s" << std::endl;
  std::cerr << std::endl;
  std::cerr << "corr_moments_pt: ";
  for (std::size_t nidx = 1; nidx <= order; ++nidx)
    std::cerr << corr_moments_pt[nidx - 1] << " ";
  std::cerr << std::endl;
  std::cerr << "inter_moments: ";
  for (std::size_t idx = 0; idx < num_data_points * order; ++idx)
    std::cerr << inter_moments[idx] << " ";
  std::cerr << std::endl;
  std::cerr << "inter_std_errs: ";
  for (std::size_t idx = 0; idx < num_data_points * order; ++idx)
    std::cerr << inter_std_errs[idx] << " ";
  std::cerr << std::endl;
  std::cerr << "inter_tidx: ";
  for (std::size_t idx = 0; idx < num_data_points * var_offset; ++idx)
    std::cerr << inter_tidx[idx] << " ";
  std::cerr << std::endl;
}

} // namespace moments

namespace QFI {

/* ----------------------- npz_parser class definition ----------------------- */
npz_parser::npz_parser() {}

npz_parser::npz_parser(std::string file_name) {
  npz = cnpy::npz_load(file_name);
  num_qubits = npz["num_qubits"].data<int32_t>()[0];
  num_meas = npz["num_meas"].data<int32_t>()[0]; // WARNING: this risks overflowing! (~2e9)
  n_star = npz["max_order"].data<int32_t>()[0];
  QFI = npz["QFI"].data<double>()[0];
  inds = npz["inds"].data<int32_t>();
  clicks = npz["clicks"].data<int32_t>();
}

void npz_parser::print(std::size_t max_num_meas, std::size_t order_, std::ostream& out) const {
  assert(max_num_meas <= num_meas);
  std::size_t idx;
  out << "num_qubits: "
      << " ";
  out << num_qubits << " " << std::endl;
  out << "num_meas: "
      << " ";
  out << num_meas << " " << std::endl;
  out << "n_star: " << n_star << std::endl;
  out << "order_: " << order_ << std::endl;
  out << "QFI: " << QFI << std::endl;
  out << std::endl;
  out << "inds: " << " ";
  for (idx = 0; idx < max_num_meas; idx += 1)
    out << inds[idx] << " ";
  out << std::endl;
  out << "clicks: " << " ";
  for (idx = 0; idx < max_num_meas; idx += 1)
    out << clicks[idx] << " ";
  out << std::endl;
}

void npz_parser::add_used_pars(std::size_t cons_num_meas_, std::size_t order_, std::size_t num_runs_,
                               std::size_t num_data_points_) {
  assert(cons_num_meas_ <= num_meas);
  assert(order_ > n_star);
  assert(num_runs_ > 0);
  assert(num_data_points_ > 0);
  order = order_;
  cons_num_meas = cons_num_meas_;
  num_runs = num_runs_;
  num_data_points = num_data_points_;
};

void npz_parser::write(std::string file, const std::size_t num_threads, const std::size_t vector_width, double rmMB,
                       double rsptMB, double rfMF, double rmtMB, double runtime, const double* const est_Tvals,
                       const double* const est_inter_Tvals, const std::size_t* const inter_tidx,
                       const double* const runtimesDSC) const {

  // first must "w"rite, then we must "a"ppend
  cnpy::npz_save(file, "num_qubits", &num_qubits, {1}, "w");           // number of quibts
  cnpy::npz_save(file, "num_meas", &cons_num_meas, {1}, "a");          // used num_meas
  cnpy::npz_save(file, "n_star", &n_star, {1}, "a");                   // used n_star
  cnpy::npz_save(file, "order", &order, {1}, "a");                     // used n_star
  cnpy::npz_save(file, "num_threads", &num_threads, {1}, "a");         // used number of threads
  cnpy::npz_save(file, "vector_width", &vector_width, {1}, "a");       // used number of threads
  cnpy::npz_save(file, "rmMB", &rmMB, {1}, "a");                       // used var_offset, ratio for var. est.
  cnpy::npz_save(file, "rsptMB", &rsptMB, {1}, "a");                   // used var_offset, ratio for var. est.
  cnpy::npz_save(file, "rfMF", &rfMF, {1}, "a");                       // used var_offset, ratio for var. est.
  cnpy::npz_save(file, "rmtMB", &rmtMB, {1}, "a");                     // used var_offset, ratio for var. est.
  cnpy::npz_save(file, "num_runs", &num_runs, {1}, "a");               // number of runs that were averaged over
  cnpy::npz_save(file, "num_data_points", &num_data_points, {1}, "a"); // number of save interm. moments
  cnpy::npz_save(file, "runtime", &runtime, {1}, "a");                 // mean runtime (over num_runs runs) of dsc algo
  cnpy::npz_save(file, "est_Tvals", est_Tvals, {num_runs * order}, "a"); // estimated moments
  cnpy::npz_save(file, "runtimesDSC", runtimesDSC, {num_runs}, "a");
  if (num_data_points > 0) {
    cnpy::npz_save(file, "est_inter_Tvals", est_inter_Tvals, {num_runs * num_data_points * order},
                   "a");                                                    // estimated moments
    cnpy::npz_save(file, "inter_tidx", inter_tidx, {num_data_points}, "a"); // estimated moments
  }

  cnpy::npz_save(file, "QFI", &QFI, {1}, "a");
};
/* ----------------------- npz_parser class definition ----------------------- */

void debug_print(std::size_t num_qubits, std::size_t num_meas, std::size_t n_star, std::size_t order,
                 std::size_t num_data_points, const std::vector<double>& Tvals, double time, double QFI,
                 const double* const inter_Tvals, const std::size_t* const inter_tidx) {
  std::cerr << "------------------- results -------------------" << std::endl;
  std::cerr << "num_qubits: " << num_qubits << std::endl;
  std::cerr << "num_meas: " << num_meas << std::endl;
  std::cerr << "n_star: " << n_star << std::endl;
  std::cerr << "order: " << order << std::endl;
  std::cerr << "QFI: " << QFI << std::endl;
  std::cerr << "num_data_points: " << num_data_points << std::endl;
  std::cerr << "Tvals: ";
  for (std::size_t nidx = 1; nidx <= order; ++nidx)
    std::cerr << Tvals[nidx - 1] << " ";
  std::cerr << std::endl;
  std::cerr << "runtime: " << time << " s" << std::endl;
  std::cerr << std::endl;
  std::cerr << "inter_moments: ";
  for (std::size_t idx = 0; idx < num_data_points * order; ++idx)
    std::cerr << inter_Tvals[idx] << " ";
  std::cerr << std::endl;
  std::cerr << "inter_tidx: ";
  for (std::size_t idx = 0; idx < num_data_points; ++idx)
    std::cerr << inter_tidx[idx] << " ";
  std::cerr << std::endl;
}

} // namespace QFI

} // namespace utils
