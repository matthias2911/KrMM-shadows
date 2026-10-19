#pragma once

#include "blends.hpp"

namespace shadows {

// scalar
// kernel computing a single sliced row mult and addition
template <std::size_t num_qubits>
inline void SRM_ADD(z_t* __restrict__ const in_data, z_t* __restrict__ const in_dest,
                    const z_t* __restrict__ const in_src, const z_t* __restrict__ const kron_fac, const double alpha,
                    const double beta) {

  constexpr std::size_t ss_dim = std::size_t{1} << num_qubits;            // 2**num_qubits, state-space dimension
  constexpr std::size_t half_ss_dim = std::size_t{1} << (num_qubits - 1); // 2**num_qubits, state-space dimension

  const z_t f00 = kron_fac[0];
  const z_t f10 = kron_fac[1];
  const z_t f01 = kron_fac[2];
  const z_t f11 = kron_fac[3];

  std::size_t idx0 = 0;
  std::size_t idx1 = half_ss_dim;
  for (std::size_t col = 0; col < ss_dim; col += 2) {

    const z_t x0 = alpha * in_src[col];
    const z_t x1 = alpha * in_src[col + 1];

    const z_t y0 = beta * in_data[idx0] + x0 * f00 + x1 * f10;
    const z_t y1 = beta * in_data[idx1] + x0 * f01 + x1 * f11;

    in_dest[idx0] = y0;
    in_dest[idx1] = y1;

    in_data[idx0] = y0;
    in_data[idx1] = y1;

    ++idx0;
    ++idx1;
  }
}

// vectorized
// kernel computing a single sliced row mult of the last qubit
template <typename VecXd, std::size_t num_qubits>
inline void SRM_ADD(double* __restrict__ const in_data, double* __restrict__ const in_dest,
                    const double* __restrict__ const in_src, const double* __restrict__ const kron_fac,
                    const double alpha, const double beta) {

  constexpr std::size_t Vsize = VecXd::size();
  constexpr std::size_t half_ss_dim = std::size_t{1} << (num_qubits - 1); // 2**num_qubits, state-space dimension

  VecXd Y0r, Y0i, Y1r, Y1i;

  const double* src = in_src;
  double* dest = in_dest;
  double* data = in_data;

  for (std::size_t col_batch = 0; col_batch < half_ss_dim / Vsize; ++col_batch) {

    const double* const src0r = src;
    const double* const src0i = src + Vsize;
    const double* const src1r = src + Vsize * 2;
    const double* const src1i = src + Vsize * 3;

    double* const dest0r = dest;
    double* const dest0i = dest + Vsize;
    double* const dest1r = dest + 2 * half_ss_dim;
    double* const dest1i = dest + 2 * half_ss_dim + Vsize;

    double* const data0r = data;
    double* const data0i = data + Vsize;
    double* const data1r = data + 2 * half_ss_dim;
    double* const data1i = data + 2 * half_ss_dim + Vsize;

    // interleaved to (potentially) help pipelining
    Y0r.load_a(src0r);
    Y0i.load_a(src0i);
    Y1r.load_a(src1r);
    Y1i.load_a(src1i);
    const VecXd X0r = VecXd(alpha) * blend_even(Y0r, Y1r);
    const VecXd X1r = VecXd(alpha) * blend_odd(Y0r, Y1r);
    VecXd F0r(kron_fac[0]);
    VecXd F0i(kron_fac[1]);
    Y0r.load_a(data0r);
    Y0r *= VecXd(beta);
    Y0r = vcl::mul_add(X0r, F0r, Y0r);
    const VecXd X0i = VecXd(alpha) * blend_even(Y0i, Y1i);
    const VecXd X1i = VecXd(alpha) * blend_odd(Y0i, Y1i);
    VecXd F1r(kron_fac[2]);
    Y0i.load_a(data0i);
    Y0i *= VecXd(beta);
    Y0i = vcl::mul_add(X0r, F0i, Y0i);
    Y0r = vcl::nmul_add(X0i, F0i, Y0r);
    VecXd F1i(kron_fac[3]);
    Y0i = vcl::mul_add(X0i, F0r, Y0i);
    Y0r = vcl::mul_add(X1r, F1r, Y0r);
    F0r = VecXd(kron_fac[4]);
    Y1r.load_a(data1r);
    Y1r *= VecXd(beta);
    Y0i = vcl::mul_add(X1r, F1i, Y0i);
    Y0r = vcl::nmul_add(X1i, F1i, Y0r);
    F0i = VecXd(kron_fac[5]);
    Y1i.load_a(data1i);
    Y1i *= VecXd(beta);
    Y0i = vcl::mul_add(X1i, F1r, Y0i);
    Y1r = vcl::mul_add(X0r, F0r, Y1r);
    F1r = VecXd(kron_fac[6]);
    Y0r.store_a(data0r);
    Y0r.store_a(dest0r);
    Y1i = vcl::mul_add(X0r, F0i, Y1i);
    Y1r = vcl::nmul_add(X0i, F0i, Y1r);
    F1i = VecXd(kron_fac[7]);
    Y0i.store_a(data0i);
    Y0i.store_a(dest0i);
    Y1i = vcl::mul_add(X0i, F0r, Y1i);
    Y1r = vcl::mul_add(X1r, F1r, Y1r);
    Y1i = vcl::mul_add(X1r, F1i, Y1i);
    Y1r = vcl::nmul_add(X1i, F1i, Y1r);
    Y1r.store_a(data1r);
    Y1r.store_a(dest1r);
    Y1i = vcl::mul_add(X1i, F1r, Y1i);
    Y1i.store_a(data1i);
    Y1i.store_a(dest1i);

    src += 4 * Vsize;
    dest += 2 * Vsize;
    data += 2 * Vsize;
  }
}

// scalar
// kernel computing a single sliced row mult
template <std::size_t num_qubits>
inline void SRM(z_t* __restrict__ const dest, const z_t* __restrict__ const src,
                const z_t* __restrict__ const kron_fac) {

  constexpr std::size_t ss_dim = std::size_t{1} << num_qubits; // 2**num_qubits, state-space dimension
  constexpr std::size_t half_ss_dim = std::size_t{1}
                                      << (num_qubits - 1); // 2**(num_qubits - 1), half state-space dimension

  const z_t f00 = kron_fac[0];
  const z_t f10 = kron_fac[1];
  const z_t f01 = kron_fac[2];
  const z_t f11 = kron_fac[3];

  std::size_t idx0 = 0;
  std::size_t idx1 = half_ss_dim;
  for (std::size_t col = 0; col < ss_dim; col += 2) {

    const z_t x0 = src[col];
    const z_t x1 = src[col + 1];

    dest[idx0] = x0 * f00 + x1 * f10;
    dest[idx1] = x0 * f01 + x1 * f11;

    ++idx0;
    ++idx1;
  }
}

// vectorized
// kernel computing a single sliced row mult
template <typename VecXd, std::size_t num_qubits>
inline void SRM(double* __restrict__ const in_dest, const double* __restrict__ const in_src,
                const double* __restrict__ const kron_fac) {

  constexpr std::size_t Vsize = VecXd::size();
  constexpr std::size_t half_ss_dim = std::size_t{1} << (num_qubits - 1); // 2**num_qubits, state-space dimension

  VecXd Y0r, Y0i, Y1r, Y1i;

  const double* src = in_src;
  double* dest = in_dest;

  for (std::size_t col_batch = 0; col_batch < half_ss_dim / Vsize; ++col_batch) {

    const double* const src0r = src;
    const double* const src0i = src + Vsize;
    const double* const src1r = src + Vsize * 2;
    const double* const src1i = src + Vsize * 3;

    double* const dest0r = dest;
    double* const dest0i = dest + Vsize;
    double* const dest1r = dest + 2 * half_ss_dim;
    double* const dest1i = dest + 2 * half_ss_dim + Vsize;

    // interleaved to (potentially) help pipelining
    Y0r.load_a(src0r);
    Y0i.load_a(src0i);
    Y1r.load_a(src1r);
    Y1i.load_a(src1i);
    const VecXd X0r = blend_even(Y0r, Y1r);
    const VecXd X1r = blend_odd(Y0r, Y1r);
    VecXd F0r(kron_fac[0]);
    VecXd F0i(kron_fac[1]);
    Y0r = X0r * F0r;
    const VecXd X0i = blend_even(Y0i, Y1i);
    const VecXd X1i = blend_odd(Y0i, Y1i);
    VecXd F1r(kron_fac[2]);
    Y0i = X0r * F0i;
    Y0r = vcl::nmul_add(X0i, F0i, Y0r);
    VecXd F1i(kron_fac[3]);
    Y0i = vcl::mul_add(X0i, F0r, Y0i);
    Y0r = vcl::mul_add(X1r, F1r, Y0r);
    F0r = VecXd(kron_fac[4]);
    Y0i = vcl::mul_add(X1r, F1i, Y0i);
    Y0r = vcl::nmul_add(X1i, F1i, Y0r);
    Y0r.store_a(dest0r);
    F0i = VecXd(kron_fac[5]);
    Y0i = vcl::mul_add(X1i, F1r, Y0i);
    Y0i.store_a(dest0i);
    Y1r = X0r * F0r;
    F1r = VecXd(kron_fac[6]);
    Y1i = X0r * F0i;
    Y1r = vcl::nmul_add(X0i, F0i, Y1r);
    F1i = VecXd(kron_fac[7]);
    Y1i = vcl::mul_add(X0i, F0r, Y1i);
    Y1r = vcl::mul_add(X1r, F1r, Y1r);
    Y1i = vcl::mul_add(X1r, F1i, Y1i);
    Y1r = vcl::nmul_add(X1i, F1i, Y1r);
    Y1i = vcl::mul_add(X1i, F1r, Y1i);
    Y1r.store_a(dest1r);
    Y1i.store_a(dest1i);

    src += 4 * Vsize;
    dest += 2 * Vsize;
  }
}

} // namespace shadows
