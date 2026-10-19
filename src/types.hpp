#pragma once

#include <array>
#include <complex>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <vector>

namespace shadows {

using z_t = std::complex<double>;

using tuple_dvec_t = std::tuple<std::vector<double>, std::vector<double>>;

template <bool _Cond, typename _Iftrue, typename _Iffalse> using tern_t = std::conditional_t<_Cond, _Iftrue, _Iffalse>;

} // namespace shadows
