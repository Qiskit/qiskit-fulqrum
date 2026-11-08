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
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#    define FQ_KEY_WORDS_HD __host__ __device__ __forceinline__
#else
#    define FQ_KEY_WORDS_HD inline
#endif

namespace fq
{
namespace key_words
{

constexpr int kWordBits = 64;
constexpr std::uint64_t kEmptyWord = ~std::uint64_t(0);
constexpr std::uint64_t kHashWordOffset = 0x9E3779B97F4A7C15ULL;

template <int W>
struct Key
{
    static_assert(W >= 1, "a key has at least one word");
    std::uint64_t word[W];
};

FQ_KEY_WORDS_HD int popcount64(const std::uint64_t x)
{
#if defined(__CUDA_ARCH__)
    return __popcll(x);
#else
    return __builtin_popcountll(x);
#endif
}

FQ_KEY_WORDS_HD std::uint64_t splitmix64(std::uint64_t k)
{
    k = (k ^ (k >> 30)) * 0xBF58476D1CE4E5B9ULL;
    k = (k ^ (k >> 27)) * 0x94D049BB133111EBULL;
    return k ^ (k >> 31);
}

FQ_KEY_WORDS_HD std::uint64_t below(const int count)
{
    if(count <= 0)
    {
        return 0;
    }
    if(count >= kWordBits)
    {
        return kEmptyWord;
    }
    return (std::uint64_t(1) << count) - 1;
}

FQ_KEY_WORDS_HD int clamp_bits(const int value)
{
    return value < 0 ? 0 : (value > kWordBits ? kWordBits : value);
}

template <int W>
FQ_KEY_WORDS_HD Key<W> empty_w()
{
    Key<W> key;
    for(int w = 0; w < W; ++w)
    {
        key.word[w] = kEmptyWord;
    }
    return key;
}

template <int W>
FQ_KEY_WORDS_HD int popcount_w(const Key<W>& key)
{
    int count = 0;
    for(int w = 0; w < W; ++w)
    {
        count += popcount64(key.word[w]);
    }
    return count;
}

template <int W>
FQ_KEY_WORDS_HD int bit_w(const Key<W>& key, const int orbital)
{
    return int((key.word[orbital / kWordBits] >> (orbital % kWordBits)) & 1u);
}

template <int W>
FQ_KEY_WORDS_HD Key<W> between_mask_w(int low, int high)
{
    if(low > high)
    {
        const int t = low;
        low = high;
        high = t;
    }
    Key<W> mask;
    for(int w = 0; w < W; ++w)
    {
        const int base = w * kWordBits;
        mask.word[w] = below(clamp_bits(high - base)) ^ below(clamp_bits(low + 1 - base));
    }
    return mask;
}

template <int W>
FQ_KEY_WORDS_HD int between_parity_w(const Key<W>& key, const int low, const int high)
{
    const Key<W> mask = between_mask_w<W>(low, high);
    int count = 0;
    for(int w = 0; w < W; ++w)
    {
        count += popcount64(key.word[w] & mask.word[w]);
    }
    return (count & 1) ? -1 : 1;
}

template <int W>
FQ_KEY_WORDS_HD Key<W> xor_w(const Key<W>& a, const Key<W>& b)
{
    Key<W> out;
    for(int w = 0; w < W; ++w)
    {
        out.word[w] = a.word[w] ^ b.word[w];
    }
    return out;
}

template <int W>
FQ_KEY_WORDS_HD bool eq_w(const Key<W>& a, const Key<W>& b)
{
    bool same = true;
    for(int w = 0; w < W; ++w)
    {
        same = same && a.word[w] == b.word[w];
    }
    return same;
}

template <int W>
FQ_KEY_WORDS_HD bool is_empty_w(const Key<W>& key)
{
    return eq_w<W>(key, empty_w<W>());
}

template <int W>
FQ_KEY_WORDS_HD std::uint64_t hash_w(const Key<W>& key)
{
    std::uint64_t h = key.word[W - 1];
    for(int w = W - 2; w >= 0; --w)
    {
        h = key.word[w] ^ splitmix64(h + kHashWordOffset);
    }
    return splitmix64(h);
}

template <int W>
FQ_KEY_WORDS_HD Key<W> load_w(const std::uint64_t* words, const std::int64_t row)
{
    Key<W> key;
    for(int w = 0; w < W; ++w)
    {
        key.word[w] = words[row * W + w];
    }
    return key;
}

}
}

#undef FQ_KEY_WORDS_HD
