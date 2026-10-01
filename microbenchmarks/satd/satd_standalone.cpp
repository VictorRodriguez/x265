// satd_standalone.cpp
//
// Standalone SATD-style microbenchmark.
// No dependency on x265.
//
// Build:
//   g++ -O3 -g -fno-omit-frame-pointer \
//       satd_standalone.cpp -o satd_standalone
//
// Run:
//   ./satd_standalone avx2
//   ./satd_standalone avx512
//
// Profile:
//   taskset -c 0 perf stat -r 10 \
//     -e cycles,ref-cycles,instructions,task-clock \
//     ./satd_standalone avx2
//
//   taskset -c 0 perf stat -r 10 \
//     -e cycles,ref-cycles,instructions,task-clock \
//     ./satd_standalone avx512

#include <immintrin.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static constexpr uint64_t ITERATIONS = 100000000ULL;

// ------------------------------------------------------------
// Scalar 8-point Hadamard transform
// ------------------------------------------------------------

static inline void hadamard8(const int16_t* in, int32_t* out)
{
    const int32_t a0 = in[0] + in[4];
    const int32_t a1 = in[1] + in[5];
    const int32_t a2 = in[2] + in[6];
    const int32_t a3 = in[3] + in[7];

    const int32_t a4 = in[0] - in[4];
    const int32_t a5 = in[1] - in[5];
    const int32_t a6 = in[2] - in[6];
    const int32_t a7 = in[3] - in[7];

    const int32_t b0 = a0 + a2;
    const int32_t b1 = a1 + a3;
    const int32_t b2 = a0 - a2;
    const int32_t b3 = a1 - a3;

    const int32_t b4 = a4 + a6;
    const int32_t b5 = a5 + a7;
    const int32_t b6 = a4 - a6;
    const int32_t b7 = a5 - a7;

    out[0] = b0 + b1;
    out[1] = b0 - b1;
    out[2] = b2 + b3;
    out[3] = b2 - b3;

    out[4] = b4 + b5;
    out[5] = b4 - b5;
    out[6] = b6 + b7;
    out[7] = b6 - b7;
}

// ------------------------------------------------------------
// Common SATD transform
//
// SIMD functions below generate the 64 signed differences.
// The same integer Hadamard transform is then applied so that
// AVX2 and AVX-512 have identical mathematical semantics.
// ------------------------------------------------------------

static inline uint32_t hadamard_satd(const int16_t diff[64])
{
    alignas(64) int32_t rows[64];
    alignas(64) int16_t column[8];
    alignas(64) int32_t transformed[8];

    // Horizontal transform
    for (int y = 0; y < 8; ++y)
    {
        int32_t tmp[8];
        hadamard8(&diff[y * 8], tmp);

        for (int x = 0; x < 8; ++x)
            rows[y * 8 + x] = tmp[x];
    }

    uint64_t sum = 0;

    // Vertical transform
    for (int x = 0; x < 8; ++x)
    {
        for (int y = 0; y < 8; ++y)
            column[y] = static_cast<int16_t>(rows[y * 8 + x]);

        hadamard8(column, transformed);

        for (int y = 0; y < 8; ++y)
        {
            int32_t v = transformed[y];
            sum += static_cast<uint32_t>(v < 0 ? -v : v);
        }
    }

    // SATD normalization.
    return static_cast<uint32_t>((sum + 2) >> 2);
}

// ------------------------------------------------------------
// AVX2
//
// 32 source pixels are widened to 16-bit values per iteration.
// Two passes cover the complete 8x8 block.
// ------------------------------------------------------------

__attribute__((target("avx2"), noinline))
uint32_t satd8x8_avx2(
    const uint8_t* src,
    const uint8_t* ref)
{
    alignas(64) int16_t diff[64];

    for (int i = 0; i < 64; i += 32)
    {
        __m128i s0 =
            _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(src + i));

        __m128i s1 =
            _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(src + i + 16));

        __m128i r0 =
            _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(ref + i));

        __m128i r1 =
            _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(ref + i + 16));

        __m256i s0w = _mm256_cvtepu8_epi16(s0);
        __m256i s1w = _mm256_cvtepu8_epi16(s1);

        __m256i r0w = _mm256_cvtepu8_epi16(r0);
        __m256i r1w = _mm256_cvtepu8_epi16(r1);

        __m256i d0 = _mm256_sub_epi16(s0w, r0w);
        __m256i d1 = _mm256_sub_epi16(s1w, r1w);

        _mm256_store_si256(
            reinterpret_cast<__m256i*>(diff + i),
            d0);

        _mm256_store_si256(
            reinterpret_cast<__m256i*>(diff + i + 16),
            d1);
    }

    return hadamard_satd(diff);
}

// ------------------------------------------------------------
// AVX-512
//
// Each iteration converts 32 uint8 pixels into 32 int16 lanes.
// Two vectors cover the entire 8x8 block.
// ------------------------------------------------------------

__attribute__((target("avx512f,avx512bw"), noinline))
uint32_t satd8x8_avx512(
    const uint8_t* src,
    const uint8_t* ref)
{
    alignas(64) int16_t diff[64];

    __m256i s0 =
        _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(src));

    __m256i s1 =
        _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(src + 32));

    __m256i r0 =
        _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(ref));

    __m256i r1 =
        _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(ref + 32));

    __m512i s0w = _mm512_cvtepu8_epi16(s0);
    __m512i s1w = _mm512_cvtepu8_epi16(s1);

    __m512i r0w = _mm512_cvtepu8_epi16(r0);
    __m512i r1w = _mm512_cvtepu8_epi16(r1);

    __m512i d0 = _mm512_sub_epi16(s0w, r0w);
    __m512i d1 = _mm512_sub_epi16(s1w, r1w);

    _mm512_store_si512(
        reinterpret_cast<__m512i*>(diff),
        d0);

    _mm512_store_si512(
        reinterpret_cast<__m512i*>(diff + 32),
        d1);

    return hadamard_satd(diff);
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::fprintf(
            stderr,
            "Usage: %s avx2|avx512\n",
            argv[0]);

        return 1;
    }

    alignas(64) uint8_t src[64];
    alignas(64) uint8_t ref[64];

    // Same deterministic input pattern used in the
    // x265-linked experiment.
    for (int i = 0; i < 64; ++i)
    {
        src[i] =
            static_cast<uint8_t>(
                (i * 13 + 17) & 255);

        ref[i] =
            static_cast<uint8_t>(
                (i * 7 + 29) & 255);
    }

    using satd_fn =
        uint32_t (*)(const uint8_t*,
                     const uint8_t*);

    satd_fn fn = nullptr;

    if (std::strcmp(argv[1], "avx2") == 0)
    {
        fn = satd8x8_avx2;
        std::printf("ISA        : AVX2\n");
    }
    else if (std::strcmp(argv[1], "avx512") == 0)
    {
        fn = satd8x8_avx512;
        std::printf("ISA        : AVX-512\n");
    }
    else
    {
        std::fprintf(
            stderr,
            "Unknown ISA: %s\n",
            argv[1]);

        return 1;
    }

    // --------------------------------------------------------
    // Functional-equivalence check
    // --------------------------------------------------------

    const uint32_t avx2_result =
        satd8x8_avx2(src, ref);

    const uint32_t avx512_result =
        satd8x8_avx512(src, ref);

    std::printf(
        "SATD AVX2   : %u\n",
        avx2_result);

    std::printf(
        "SATD AVX512 : %u\n",
        avx512_result);

    if (avx2_result != avx512_result)
    {
        std::fprintf(
            stderr,
            "ERROR: results do not match\n");

        return 2;
    }

    volatile uint64_t sink = 0;

    // Warm-up
    for (uint64_t i = 0; i < 1000000; ++i)
        sink += fn(src, ref);

    const auto start =
        std::chrono::steady_clock::now();

    for (uint64_t i = 0; i < ITERATIONS; ++i)
        sink += fn(src, ref);

    const auto end =
        std::chrono::steady_clock::now();

    const double seconds =
        std::chrono::duration<double>(
            end - start).count();

    const double calls_per_second =
        static_cast<double>(
            ITERATIONS) / seconds;

    const double pixels_per_second =
        calls_per_second * 64.0;

    std::printf("\n");
    std::printf(
        "Iterations  : %lu\n",
        static_cast<unsigned long>(
            ITERATIONS));

    std::printf(
        "Time        : %.6f s\n",
        seconds);

    std::printf(
        "Calls/sec   : %.3f M\n",
        calls_per_second / 1e6);

    std::printf(
        "Pixels/sec  : %.3f G\n",
        pixels_per_second / 1e9);

    std::printf(
        "Sink        : %lu\n",
        static_cast<unsigned long>(
            sink));

    return 0;
}
