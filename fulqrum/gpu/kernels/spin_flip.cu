/**
 * This code is a part of Fulqrum.
 *
 * (C) Copyright IBM 2026.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */
#include <cstdint>
#include <cuda_runtime.h>

#include "device/tile_index.cuh"
#include "formats/pair_index.hpp"
#include "launch.hpp"
#include "launch_grid.hpp"

namespace
{

__device__ __forceinline__ int rank_below(const std::uint64_t bits, const int bit)
{
    return __popcll(bits & ((1ull << bit) - 1ull));
}

/**
 * Adds the opposite-direction correction of one source tile to each output element (a, b): for every
 * orbital i in b but not a and j in a but not b, (o - s) of pair {i, j} times the source element at
 * (a - j + i, b - i + j), with the parity of the orbitals strictly between i and j in a and in b.
 */
__global__ void __launch_bounds__(fq::launch::kArmThreads, 4)
    spin_flip_kernel(const FqTileGeometry geometry,
                     const FqVectors vectors,
                     const FqSpinFlipTables tables)
{
    const int row = int(fq::device::block_row());
    const int b = geometry.b_start + row;
    const int column = int(fq::device::thread_column());
    const int a = geometry.a_start + column;
    if(b >= geometry.b_stop || a >= geometry.a_stop)
    {
        return;
    }
    const std::uint64_t alpha = __ldg(tables.alpha_strings + geometry.bra_a0 + a);
    const std::uint64_t beta = __ldg(tables.beta_strings + geometry.bra_b0 + b);
    std::uint64_t in_beta = beta & ~alpha;
    const std::uint64_t in_alpha = alpha & ~beta;
    if(!in_beta || !in_alpha)
    {
        return;
    }
    const int alpha_empty = tables.num_orbitals - tables.alpha_electrons;
    const int beta_empty = tables.num_orbitals - tables.beta_electrons;
    const std::int32_t* alpha_moves =
        tables.alpha_moves + std::int64_t(a) * tables.alpha_electrons * alpha_empty;
    const std::int32_t* beta_moves =
        tables.beta_moves + std::int64_t(b) * tables.beta_electrons * beta_empty;
    double sum = 0.0;
    while(in_beta)
    {
        const int i = __ffsll(static_cast<long long>(in_beta)) - 1;
        in_beta &= in_beta - 1;
        std::uint64_t rest = in_alpha;
        while(rest)
        {
            const int j = __ffsll(static_cast<long long>(rest)) - 1;
            rest &= rest - 1;
            const double correction = __ldg(tables.corrections +
                                            fq::pair_index::gpu_pair(i < j ? i : j,
                                                                     i < j ? j : i,
                                                                     tables.num_orbitals));
            if(correction == 0.0)
            {
                continue;
            }
            // alpha moves j -> i, beta moves i -> j
            const int to_alpha =
                __ldg(alpha_moves + rank_below(alpha, j) * alpha_empty + rank_below(~alpha, i));
            if(to_alpha == 0)
            {
                continue;
            }
            const int to_beta =
                __ldg(beta_moves + rank_below(beta, i) * beta_empty + rank_below(~beta, j));
            if(to_beta == 0)
            {
                continue;
            }
            const int source_a = (to_alpha > 0 ? to_alpha : -to_alpha) - 1 - geometry.src_a0;
            const int source_b = (to_beta > 0 ? to_beta : -to_beta) - 1 - geometry.src_b0;
            if(source_a < 0 || source_a >= geometry.n_a_src || source_b < 0 ||
               source_b >= geometry.n_b_src)
            {
                continue;
            }
            const int alpha_slot = __ldg(vectors.alpha_src_slot + source_a);
            const int beta_slot = __ldg(vectors.beta_src_slot + source_b);
            if(alpha_slot < 0 || beta_slot < 0)
            {
                continue;
            }
            const double sign = ((to_alpha > 0) == (to_beta > 0)) ? correction : -correction;
            sum += sign * __ldg(vectors.src_in + std::int64_t(beta_slot) * geometry.n_alpha_src_compact +
                                alpha_slot);
        }
    }
    if(sum != 0.0)
    {
        vectors.out_local[fq::device::tile_index(row, geometry.a_stop - geometry.a_start, column)] +=
            sum;
    }
}

} // namespace

extern "C"
{

    int fq_launch_spin_flip(const FqTileGeometry* geometry,
                            const FqVectors* vectors,
                            const FqSpinFlipTables* tables,
                            void* stream)
    {
        if(fq::launch::empty_tile(*geometry))
        {
            return int(cudaSuccess);
        }
        spin_flip_kernel<<<fq::launch::alpha_column_grid<1>(*geometry),
                           geometry->threads_per_block,
                           0,
                           static_cast<cudaStream_t>(stream)>>>(*geometry, *vectors, *tables);
        return int(cudaGetLastError());
    }
}
