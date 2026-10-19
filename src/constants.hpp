#pragma once

#include "types.hpp"

namespace shadows {

// array alignment in bytes, for cache-line size and sse, avx, avx512
constexpr std::size_t ALIGNMENT = 64;

// lookup tables for Single-Qubit (PT) shadows using double precision complex numbers
alignas(ALIGNMENT) constexpr std::array<z_t, 48> SQ_shadows = { // access via: 24*pt + 8*ind + 4*click + 2*col + row
    // non-transposed
    z_t{ 2.0, 0.0}, z_t{ 0.0, 0.0}, z_t{ 0.0, 0.0}, z_t{-1.0, 0.0},
    z_t{-1.0, 0.0}, z_t{ 0.0, 0.0}, z_t{ 0.0, 0.0}, z_t{ 2.0, 0.0}, 
    z_t{ 0.5, 0.0}, z_t{ 1.5, 0.0}, z_t{ 1.5, 0.0}, z_t{ 0.5, 0.0}, 
    z_t{ 0.5, 0.0}, z_t{-1.5, 0.0}, z_t{-1.5, 0.0}, z_t{ 0.5, 0.0}, 
    z_t{ 0.5, 0.0}, z_t{ 0.0, 1.5}, z_t{ 0.0,-1.5}, z_t{ 0.5, 0.0}, 
    z_t{ 0.5, 0.0}, z_t{ 0.0,-1.5}, z_t{ 0.0, 1.5}, z_t{ 0.5, 0.0},

    // transposed
    z_t{ 2.0, 0.0}, z_t{ 0.0, 0.0}, z_t{ 0.0, 0.0}, z_t{-1.0, 0.0}, 
    z_t{-1.0, 0.0}, z_t{ 0.0, 0.0}, z_t{ 0.0, 0.0}, z_t{ 2.0, 0.0}, 
    z_t{ 0.5, 0.0}, z_t{ 1.5, 0.0}, z_t{ 1.5, 0.0}, z_t{ 0.5, 0.0}, 
    z_t{ 0.5, 0.0}, z_t{-1.5, 0.0}, z_t{-1.5, 0.0}, z_t{ 0.5, 0.0}, 
    z_t{ 0.5, 0.0}, z_t{ 0.0,-1.5}, z_t{ 0.0, 1.5}, z_t{ 0.5, 0.0}, 
    z_t{ 0.5, 0.0}, z_t{ 0.0, 1.5}, z_t{ 0.0,-1.5}, z_t{ 0.5, 0.0}
};

// lookup tables for real part of Single-Qubit (PT) shadows using double precision
alignas(ALIGNMENT) constexpr std::array<double, 96> SQ_shadows_d = { // access via: 48*pt + 16*ind + 8*click + 4*col + 2*row + imag
    // non-transposed
    2.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0,  0.0, 
   -1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  2.0,  0.0, 
    0.5,  0.0,  1.5,  0.0,  1.5,  0.0,  0.5,  0.0, 
    0.5,  0.0, -1.5,  0.0, -1.5,  0.0,  0.5,  0.0, 
    0.5,  0.0,  0.0,  1.5,  0.0, -1.5,  0.5,  0.0, 
    0.5,  0.0,  0.0, -1.5,  0.0,  1.5,  0.5,  0.0,

    // transposed
    2.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0,  0.0, 
   -1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  2.0,  0.0, 
    0.5,  0.0,  1.5,  0.0,  1.5,  0.0,  0.5,  0.0, 
    0.5,  0.0, -1.5,  0.0, -1.5,  0.0,  0.5,  0.0, 
    0.5,  0.0,  0.0, -1.5,  0.0,  1.5,  0.5,  0.0, 
    0.5,  0.0,  0.0,  1.5,  0.0, -1.5,  0.5,  0.0
};

namespace QFI {

// lookup tables for Single-Qubit shadows using double precision complex numbers
alignas(ALIGNMENT) constexpr std::array<z_t, 48> SQ_shadows = { // access via: 24*zidx + 8*ind + 4*click + 2*col + row
    // normal
    z_t{ 2.0, 0.0}, z_t{ 0.0, 0.0}, z_t{ 0.0, 0.0}, z_t{-1.0, 0.0},
    z_t{-1.0, 0.0}, z_t{ 0.0, 0.0}, z_t{ 0.0, 0.0}, z_t{ 2.0, 0.0},
    z_t{ 0.5, 0.0}, z_t{ 1.5, 0.0}, z_t{ 1.5, 0.0}, z_t{ 0.5, 0.0},
    z_t{ 0.5, 0.0}, z_t{-1.5, 0.0}, z_t{-1.5, 0.0}, z_t{ 0.5, 0.0},
    z_t{ 0.5, 0.0}, z_t{ 0.0, 1.5}, z_t{ 0.0,-1.5}, z_t{ 0.5, 0.0},
    z_t{ 0.5, 0.0}, z_t{ 0.0,-1.5}, z_t{ 0.0, 1.5}, z_t{ 0.5, 0.0},

    // multiplied with Pauli-Z matrix
    z_t{ 2.0, 0.0}, z_t{ 0.0, 0.0}, z_t{ 0.0, 0.0}, z_t{ 1.0, 0.0},
    z_t{-1.0, 0.0}, z_t{ 0.0, 0.0}, z_t{ 0.0, 0.0}, z_t{-2.0, 0.0},
    z_t{ 0.5, 0.0}, z_t{-1.5, 0.0}, z_t{ 1.5, 0.0}, z_t{-0.5, 0.0},
    z_t{ 0.5, 0.0}, z_t{ 1.5, 0.0}, z_t{-1.5, 0.0}, z_t{-0.5, 0.0},
    z_t{ 0.5, 0.0}, z_t{ 0.0,-1.5}, z_t{ 0.0,-1.5}, z_t{-0.5, 0.0},
    z_t{ 0.5, 0.0}, z_t{ 0.0, 1.5}, z_t{ 0.0, 1.5}, z_t{-0.5, 0.0}
};

// lookup tables for real part of Single-Qubit (PT) shadows using double precision
alignas(ALIGNMENT) constexpr std::array<double, 96> SQ_shadows_d = { // access via: 48*zidx + 16*ind + 8*click + 4*col + 2*row + imag
    // normal
    2.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0,  0.0,
   -1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  2.0,  0.0,
    0.5,  0.0,  1.5,  0.0,  1.5,  0.0,  0.5,  0.0,
    0.5,  0.0, -1.5,  0.0, -1.5,  0.0,  0.5,  0.0,
    0.5,  0.0,  0.0,  1.5,  0.0, -1.5,  0.5,  0.0,
    0.5,  0.0,  0.0, -1.5,  0.0,  1.5,  0.5,  0.0,

    // multiplied with Pauli-Z matrix
    2.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,
   -1.0,  0.0,  0.0,  0.0,  0.0,  0.0, -2.0,  0.0,
    0.5,  0.0, -1.5,  0.0,  1.5,  0.0, -0.5,  0.0,
    0.5,  0.0,  1.5,  0.0, -1.5,  0.0, -0.5,  0.0,
    0.5,  0.0,  0.0, -1.5,  0.0, -1.5, -0.5,  0.0,
    0.5,  0.0,  0.0,  1.5,  0.0,  1.5, -0.5,  0.0
};

// lookup tables for identity factors using double precision complex numbers
alignas(ALIGNMENT) constexpr std::array<z_t, 8> id_facs = { // access via: 4*zidx + 2*col + row
    // normal
    z_t{1.0, 0.0}, z_t{0.0, 0.0}, z_t{0.0, 0.0}, z_t{1.0, 0.0},

    // multiplied with Pauli-Z matrix
    z_t{1.0, 0.0}, z_t{0.0, 0.0}, z_t{0.0, 0.0}, z_t{-1.0, 0.0}
};

// lookup tables for identity factors using double precision
alignas(ALIGNMENT) constexpr std::array<double, 16> id_facs_d = { // access via: 8*zidx + 4*col + 2*row + imag
    // normal
    1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,

    // multiplied with Pauli-Z matrix
    1.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0,  0.0,
};

}

} // namespace shadows
