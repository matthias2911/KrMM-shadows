#pragma once

#include "constants.hpp"

#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wconversion"
#define VCL_NAMESPACE vcl
#include "vcl/vectorclass.h"
#pragma GCC diagnostic pop

namespace shadows {

// template function for blend (shuffle) operation
// picking all even entries for variable vector-register size
template <typename VecXd> VecXd blend_even(VecXd a, VecXd b) {
  if constexpr (std::is_same_v<VecXd, vcl::Vec8d>) {
    return vcl::blend8<0, 2, 4, 6, 8, 10, 12, 14>(a, b);
  } else if constexpr (std::is_same_v<VecXd, vcl::Vec4d>) {
    return vcl::blend4<0, 2, 4, 6>(a, b);
  } else if constexpr (std::is_same_v<VecXd, vcl::Vec2d>) {
    return vcl::blend2<0, 2>(a, b);
  } else {
    static_assert(sizeof(VecXd) == 0, "blend_even: unsupported type");
  }
}

// template function for blend (shuffle) operation
// picking all odd entries for variable vector-register size
template <typename VecXd> VecXd blend_odd(VecXd a, VecXd b) {
  if constexpr (std::is_same_v<VecXd, vcl::Vec8d>) {
    return vcl::blend8<1, 3, 5, 7, 9, 11, 13, 15>(a, b);
  } else if constexpr (std::is_same_v<VecXd, vcl::Vec4d>) {
    return vcl::blend4<1, 3, 5, 7>(a, b);
  } else if constexpr (std::is_same_v<VecXd, vcl::Vec2d>) {
    return vcl::blend2<1, 3>(a, b);
  } else {
    static_assert(sizeof(VecXd) == 0, "blend_odd: unsupported type");
  }
}

} // namespace shadows
