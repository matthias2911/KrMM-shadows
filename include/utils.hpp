#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#include "cnpy.h"
#pragma GCC diagnostic pop

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <vector>

namespace utils {

double f_rel_err(double ref, double approx);

std::vector<std::filesystem::path> parse_npz_files(std::string dir);

// low overhead stopwatch in seconds supporting start-stop and storing results
// IT IS YOUR RESPONSIBILITY TO NOT DO MORE MEASUREMENTS THAN max_meas
class Timer {
  using Clock = std::chrono::high_resolution_clock;
  using ChronoReturn_t = std::chrono::_V2::high_resolution_clock::time_point;
  using Duration = std::chrono::duration<double>;

  ChronoReturn_t start_time;
  ChronoReturn_t stop_time;
  std::vector<double> timings;
  const size_t max_meas;

public:
  size_t size = 0;

  Timer(size_t max_meas_) : max_meas(max_meas_) { timings = std::vector<double>(max_meas, 0); }

  void reset();                        // reset measurements
  void start();                        // start measurement
  void stop();                         // stop measurement
  double operator[](size_t idx) const; // random access
  double median() const;               // return median of measurements
  double mean() const;                 // return mean of measurements
};

class Config {
public:
  std::string in_file;
  std::string out_file;
  std::string version;

  std::size_t num_runs;
  std::size_t num_data_points;
  double sample_factor;
  bool PT;
  std::size_t order;
  int log2_num_threads;
  bool use_simd;
  std::size_t var_offset;

  Config(int argc, char** argv);
};

namespace moments {

class npz_parser {
public:
  cnpy::npz_t npz;
  std::size_t num_qubits;
  std::size_t num_meas;
  std::size_t max_order;
  int32_t pt;
  int32_t werner;
  double werner_par;
  int32_t* transposes;
  double* moments;
  double* moments_pt;
  int32_t* inds;
  int32_t* clicks;

  std::size_t cons_num_meas;
  std::size_t order;
  std::size_t num_runs;
  std::size_t num_data_points;

  npz_parser();

  npz_parser(std::string file_name);

  void print(std::size_t max_num_meas, std::size_t order, std::ostream& out = std::cout) const;

  void add_used_pars(std::size_t cons_num_meas_, std::size_t order_, std::size_t num_runs_,
                     std::size_t num_data_points_);

  void write(std::string file, const bool PT, const std::size_t num_threads, const std::size_t num_ranks,
             const std::size_t vector_width, const std::size_t var_offset, double rmMB, double rsptMB, double rfMF,
             double rmtMB, double runtime, const double* const est_moments, const double* const est_std_errs,
             const double* const est_inter_moments, const double* const est_inter_std_errs,
             const std::size_t* const inter_tidx) const;
};

void debug_print(std::size_t num_qubits, std::size_t num_meas, std::size_t order, std::size_t var_offset,
                 std::size_t num_data_points, const std::vector<double>& moments,
                 const std::vector<double>& standard_errs, double time, double* corr_moments, double* corr_moments_pt,
                 const double* const inter_moments, const double* const inter_std_errs,
                 const std::size_t* const inter_tidx);

} // namespace moments

namespace QFI {

class npz_parser {
public:
  cnpy::npz_t npz;
  std::size_t num_qubits;
  std::size_t num_meas;
  std::size_t n_star;
  double QFI;
  int32_t* inds;
  int32_t* clicks;

  std::size_t order;
  std::size_t cons_num_meas;
  std::size_t num_runs;
  std::size_t num_data_points;

  npz_parser();

  npz_parser(std::string file_name);

  void print(std::size_t max_num_meas, std::size_t order, std::ostream& out = std::cout) const;

  void add_used_pars(std::size_t cons_num_meas_, std::size_t order_, std::size_t num_runs_,
                     std::size_t num_data_points_);

  void write(std::string file, const std::size_t num_threads, const std::size_t vector_width, double rmMB,
             double rsptMB, double rfMF, double rmtMB, double runtime, const double* const est_Tvals,
             const double* const est_inter_Tvals, const std::size_t* const inter_tidx, const double* const runtimesDSC) const;
};

void debug_print(std::size_t num_qubits, std::size_t num_meas, std::size_t n_star, std::size_t order,
                 std::size_t num_data_points, const std::vector<double>& Tvals, double time, double QFI,
                 const double* const inter_Tvals, const std::size_t* const inter_tidx);

} // namespace QFI

} // namespace utils
