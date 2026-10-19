#include "utils.hpp"

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <limits>

namespace utils {

double f_rel_err(double ref, double approx) {
  double diff = std::fabs(approx - ref);
  double denom = std::fabs(ref);

  // Handle the case when ref == 0
  if (denom < std::numeric_limits<double>::epsilon()) {
    // If both are zero
    if (std::fabs(approx) < std::numeric_limits<double>::epsilon())
      return 0.0;
    else
      return std::numeric_limits<double>::infinity();
  }

  return diff / denom;
}

std::vector<std::filesystem::path> parse_npz_files(std::string dir) {
  std::vector<std::filesystem::path> npz_files;
  for (const auto& entry : std::filesystem::directory_iterator(dir))
    if (entry.path().extension() == ".npz")
      npz_files.push_back(entry.path());
  std::sort(npz_files.begin(), npz_files.end());
  return npz_files;
}

/* ----------------------- Timer class definition ----------------------- */
void Timer::reset() {
  std::fill(timings.begin(), timings.end(), 0.0);
  size = 0;
}

void Timer::start() { start_time = Clock::now(); }

void Timer::stop() {
  stop_time = Clock::now();
  timings[size++] = Duration(stop_time - start_time).count();
}

double Timer::operator[](size_t idx) const { return timings[idx]; }

double Timer::median() const {
  assert(size > 0);
  std::vector<double> tmp = timings;
  size_t mid = size / 2;
  std::nth_element(tmp.begin(), tmp.begin() + mid, tmp.end());
  if (size % 2 == 1) {
    return tmp[mid];
  } else {
    double upper = tmp[mid];
    double lower = *std::max_element(tmp.begin(), tmp.begin() + mid);
    return (lower + upper) / 2.0;
  }
}

double Timer::mean() const {
  assert(size > 0);
  double out = 0;
  for (std::size_t i = 0; i < size; ++i)
    out += timings[i];
  return out / static_cast<double>(size);
}
/* ----------------------- Timer class definition ----------------------- */

/* ----------------------- Config class definition ----------------------- */
Config::Config(int argc, char** argv) {
  assert(argc == 12);
  int i = 1;

  in_file = argv[i++];
  out_file = argv[i++];
  version = argv[i++];

  num_runs = static_cast<std::size_t>(std::stoull(argv[i++]));
  num_data_points = static_cast<std::size_t>(std::stoull(argv[i++]));
  sample_factor = std::stod(argv[i++]);
  PT = std::stoi(argv[i++]);
  order = static_cast<std::size_t>(std::stoull(argv[i++]));
  log2_num_threads = std::stoi(argv[i++]);
  use_simd = std::stoi(argv[i++]);
  var_offset = static_cast<std::size_t>(std::stoull(argv[i++]));
};
/* ----------------------- Config class definition ----------------------- */

} // namespace utils
