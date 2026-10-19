#pragma once

#include "SRM.hpp"
#include "constants.hpp"

namespace shadows {

// scalar
// Kronecker Matrix-Vector, computes c = alpha * a @ F + beta * c
// specialized Kronecker factor access using transposes, inds, and clicks arrays
// afterwards: a,c,buff point to distinct, non-overlapping locations
//             c points to its original location
//             for even num_qubits, a and buff point to their original locations
//             for odd num_qubits, a and buff are swapped
//             a and c have the same content, buff contains garbage
// these properties can (and should) be used for fusing successive calls of KrMV
template <std::size_t num_qubits>
void KrMV(z_t* __restrict__ const c, z_t*& __restrict__ a, const double alpha, const double beta,
          const int32_t* __restrict__ const transposes, const int32_t* __restrict__ const inds,
          const int32_t* __restrict__ const clicks, z_t*& __restrict__ buff) {

  const z_t* a_c = a;

  // for the first N - 1 qubits
  for (std::size_t qubit = 0; qubit < num_qubits - 1; ++qubit) {

    const std::size_t qb_idx = num_qubits - 1 - qubit; // qubits must be accessed in reverse order (I think)
    const std::size_t sh_idx = 24 * transposes[qb_idx] + 8 * inds[qb_idx] + 4 * clicks[qb_idx];
    const z_t* const kron_fac = SQ_shadows.data() + sh_idx;

    SRM<num_qubits>(buff, a_c, kron_fac);

    std::swap(a, buff);
    a_c = a;
  } // qubit
  { // do the last qubit

    const std::size_t sh_idx = 24 * transposes[0] + 8 * inds[0] + 4 * clicks[0];
    const z_t* const kron_fac = SQ_shadows.data() + sh_idx;
    SRM_ADD<num_qubits>(c, buff, a_c, kron_fac, alpha, beta);

    std::swap(a, buff);
    a_c = a;
  }
}

// vectorized
// Kronecker Matrix-Vector, computes c = alpha * a @ F + beta * c
// specialized Kronecker factor access using transposes, inds, and clicks arrays
// afterwards: a,c,buff point to distinct, non-overlapping locations
//             c points to its original location
//             for even num_qubits, a and buff point to their original locations
//             for odd num_qubits, a and buff are swapped
//             a and c have the same content, buff contains garbage
// these properties can (and should) be used for fusing successive calls of KrMV
template <typename VecXd, std::size_t num_qubits>
void KrMV(double* __restrict__ const c, double*& __restrict__ a, const double alpha, const double beta,
          const int32_t* __restrict__ const transposes, const int32_t* __restrict__ const inds,
          const int32_t* __restrict__ const clicks, double*& __restrict__ buff) {

  const double* a_c = a;

  // for the first N - 1 qubits
  for (std::size_t qubit = 0; qubit < num_qubits - 1; ++qubit) {

    const std::size_t qb_idx = num_qubits - 1 - qubit; // qubits must be accessed in reverse order (I think)
    const std::size_t sh_idx = 48 * transposes[qb_idx] + 16 * inds[qb_idx] + 8 * clicks[qb_idx];
    const double* const kron_fac = SQ_shadows_d.data() + sh_idx;

    SRM<VecXd, num_qubits>(buff, a_c, kron_fac);

    std::swap(a, buff);
    a_c = a;
  } // qubit
  { // do the last qubit

    const std::size_t sh_idx = 48 * transposes[0] + 16 * inds[0] + 8 * clicks[0];
    const double* const kron_fac = SQ_shadows_d.data() + sh_idx;
    SRM_ADD<VecXd, num_qubits>(c, buff, a_c, kron_fac, alpha, beta);

    std::swap(a, buff);
    a_c = a;
  }
}

// scalar
// does not modify a, needs two temp buffers buff0 and buff1
template <std::size_t num_qubits>
void KrMV(z_t* __restrict__ const c, const z_t* __restrict__ const a, const double alpha, const double beta,
          const int32_t* __restrict__ const transposes, const int32_t* __restrict__ const inds,
          const int32_t* __restrict__ const clicks, z_t*& __restrict__ buff0, z_t*& __restrict__ buff1) {

  const z_t* cbuff = buff1;
  { // do the first qubit

    const std::size_t sh_idx = 24 * transposes[num_qubits - 1] + 8 * inds[num_qubits - 1] + 4 * clicks[num_qubits - 1];
    const z_t* const kron_fac = SQ_shadows.data() + sh_idx;

    SRM<num_qubits>(buff0, a, kron_fac);

    std::swap(buff0, buff1);
    cbuff = buff1;
  }
  // for the first N - 1 qubits
  for (std::size_t qubit = 1; qubit < num_qubits - 1; ++qubit) {

    const std::size_t qb_idx = num_qubits - 1 - qubit; // qubits must be accessed in reverse order (I think)
    const std::size_t sh_idx = 24 * transposes[qb_idx] + 8 * inds[qb_idx] + 4 * clicks[qb_idx];
    const z_t* const kron_fac = SQ_shadows.data() + sh_idx;

    SRM<num_qubits>(buff0, cbuff, kron_fac);

    std::swap(buff0, buff1);
    cbuff = buff1;
  } // qubit
  { // do the last qubit

    const std::size_t sh_idx = 24 * transposes[0] + 8 * inds[0] + 4 * clicks[0];
    const z_t* const kron_fac = SQ_shadows.data() + sh_idx;

    SRM_ADD<num_qubits>(c, buff0, cbuff, kron_fac, alpha, beta);

    std::swap(buff0, buff1);
    cbuff = buff1;
  }
}

// vectorized
// does not modify a, needs two temp buffers buff0 and buff1
template <typename VecXd, std::size_t num_qubits>
void KrMV(double* __restrict__ const c, const double* __restrict__ const a, const double alpha, const double beta,
          const int32_t* __restrict__ const transposes, const int32_t* __restrict__ const inds,
          const int32_t* __restrict__ const clicks, double*& __restrict__ buff0, double*& __restrict__ buff1) {

  const double* cbuff = buff1;
  { // do the first qubit

    const std::size_t sh_idx = 48 * transposes[num_qubits - 1] + 16 * inds[num_qubits - 1] + 8 * clicks[num_qubits - 1];
    const double* const kron_fac = SQ_shadows_d.data() + sh_idx;
    SRM<VecXd, num_qubits>(buff0, a, kron_fac);

    std::swap(buff0, buff1);
    cbuff = buff1;
  }
  // for the first N - 1 qubits
  for (std::size_t qubit = 1; qubit < num_qubits - 1; ++qubit) {

    const std::size_t qb_idx = num_qubits - 1 - qubit; // qubits must be accessed in reverse order (I think)
    const std::size_t sh_idx = 48 * transposes[qb_idx] + 16 * inds[qb_idx] + 8 * clicks[qb_idx];
    const double* const kron_fac = SQ_shadows_d.data() + sh_idx;

    SRM<VecXd, num_qubits>(buff0, cbuff, kron_fac);

    std::swap(buff0, buff1);
    cbuff = buff1;
  } // qubit
  { // do the last qubit

    const std::size_t sh_idx = 48 * transposes[0] + 16 * inds[0] + 8 * clicks[0];
    const double* const kron_fac = SQ_shadows_d.data() + sh_idx;
    SRM_ADD<VecXd, num_qubits>(c, buff0, cbuff, kron_fac, alpha, beta);

    std::swap(buff0, buff1);
    cbuff = buff1;
  }
}

// scalar
// Kronecker Matrix-Vector with Z Hamiltonian $H = 0.5 \sum_{i=1}^n Z_i$
// computes c = alpha * a @ H @ F + beta * c
// a and c have same content afterwards, needs two temp buffers buff0 and buff1
// if identity_ is true, then F = identity is used no matter what inds, clicks is
template <std::size_t num_qubits, bool identity_>
void KrMV_Z(z_t* __restrict__ const c, z_t*& __restrict__ a, const double alpha, const double beta,
            const int32_t* __restrict__ const inds, const int32_t* __restrict__ const clicks, z_t*& __restrict__ buff0,
            z_t*& __restrict__ buff1) {

  const z_t* kron_fac = nullptr;
  const z_t* a_c = a;

  const z_t* cbuff = buff1;

  for (std::size_t zidx = 0; zidx < num_qubits; ++zidx) {

    const double beta_ = (zidx == 0) ? beta : 1.0;

    { // do the first qubit

      if constexpr (identity_) {
        const std::size_t sh_idx = 4 * (num_qubits - 1 == zidx);
        kron_fac = QFI::id_facs.data() + sh_idx;
      } else {
        const std::size_t sh_idx =
            24 * (num_qubits - 1 == zidx) + 8 * inds[num_qubits - 1] + 4 * clicks[num_qubits - 1];
        kron_fac = QFI::SQ_shadows.data() + sh_idx;
      }

      SRM<num_qubits>(buff0, a_c, kron_fac);

      std::swap(buff0, buff1);
      cbuff = buff1;
    }
    // for the first N - 1 qubits
    for (std::size_t qubit = 1; qubit < num_qubits - 1; ++qubit) {

      const std::size_t qb_idx = num_qubits - 1 - qubit; // qubits must be accessed in reverse order (I think)

      if constexpr (identity_) {
        const std::size_t sh_idx = 4 * (qb_idx == zidx);
        kron_fac = QFI::id_facs.data() + sh_idx;
      } else {
        const std::size_t sh_idx = 24 * (qb_idx == zidx) + 8 * inds[qb_idx] + 4 * clicks[qb_idx];
        kron_fac = QFI::SQ_shadows.data() + sh_idx;
      }

      SRM<num_qubits>(buff0, cbuff, kron_fac);

      std::swap(buff0, buff1);
      cbuff = buff1;
    } // qubit
    { // do the last qubit

      if constexpr (identity_) {
        const std::size_t sh_idx = 4 * (0 == zidx);
        kron_fac = QFI::id_facs.data() + sh_idx;
      } else {
        const std::size_t sh_idx = 24 * (0 == zidx) + 8 * inds[0] + 4 * clicks[0];
        kron_fac = QFI::SQ_shadows.data() + sh_idx;
      }

      SRM_ADD<num_qubits>(c, buff0, cbuff, kron_fac, alpha/2.0, beta_); // 1/2 from Z Hamil

      std::swap(buff0, buff1);
      cbuff = buff1;
    }
  }

  std::swap(a, buff1);
}

// vectorized
// Kronecker Matrix-Vector with Z Hamiltonian $H = 0.5 \sum_{i=1}^n Z_i$
// computes c = alpha * a @ H @ F + beta * c
// a and c have same content afterwards, needs two temp buffers buff0 and buff1
// if identity_ is true, then F = identity is used no matter what inds, clicks is
template <typename VecXd, std::size_t num_qubits, bool identity_>
void KrMV_Z(double* __restrict__ const c, double*& __restrict__ a, const double alpha, const double beta,
            const int32_t* __restrict__ const inds, const int32_t* __restrict__ const clicks, double*& __restrict__ buff0,
            double*& __restrict__ buff1) {

  const double* kron_fac = nullptr;
  const double* a_c = a;

  const double* cbuff = buff1;

  for (std::size_t zidx = 0; zidx < num_qubits; ++zidx) {

    const double beta_ = (zidx == 0) ? beta : 1.0;

    { // do the first qubit

      if constexpr (identity_) {
        const std::size_t sh_idx = 8 * (num_qubits - 1 == zidx);
        kron_fac = QFI::id_facs_d.data() + sh_idx;
      } else {
        const std::size_t sh_idx =
            48 * (num_qubits - 1 == zidx) + 16 * inds[num_qubits - 1] + 8 * clicks[num_qubits - 1];
        kron_fac = QFI::SQ_shadows_d.data() + sh_idx;
      }

      SRM<VecXd, num_qubits>(buff0, a_c, kron_fac);

      std::swap(buff0, buff1);
      cbuff = buff1;
    }
    // for the first N - 1 qubits
    for (std::size_t qubit = 1; qubit < num_qubits - 1; ++qubit) {

      const std::size_t qb_idx = num_qubits - 1 - qubit; // qubits must be accessed in reverse order (I think)

      if constexpr (identity_) {
        const std::size_t sh_idx = 8 * (qb_idx == zidx);
        kron_fac = QFI::id_facs_d.data() + sh_idx;
      } else {
        const std::size_t sh_idx = 48 * (qb_idx == zidx) + 16 * inds[qb_idx] + 8 * clicks[qb_idx];
        kron_fac = QFI::SQ_shadows_d.data() + sh_idx;
      }

      SRM<VecXd, num_qubits>(buff0, cbuff, kron_fac);

      std::swap(buff0, buff1);
      cbuff = buff1;
    } // qubit
    { // do the last qubit

      if constexpr (identity_) {
        const std::size_t sh_idx = 8 * (0 == zidx);
        kron_fac = QFI::id_facs_d.data() + sh_idx;
      } else {
        const std::size_t sh_idx = 48 * (0 == zidx) + 16 * inds[0] + 8 * clicks[0];
        kron_fac = QFI::SQ_shadows_d.data() + sh_idx;
      }

      SRM_ADD<VecXd, num_qubits>(c, buff0, cbuff, kron_fac, alpha/2.0, beta_); // 1/2 from Z Hamil

      std::swap(buff0, buff1);
      cbuff = buff1;
    }
  }

  std::swap(a, buff1);
}

} // namespace shadows
