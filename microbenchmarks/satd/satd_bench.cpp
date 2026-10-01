#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

extern "C" {

int x265_pixel_satd_8x8_avx2(
    const uint8_t* pix1,
    intptr_t stride1,
    const uint8_t* pix2,
    intptr_t stride2);

int x265_pixel_satd_8x8_avx512(
    const uint8_t* pix1,
    intptr_t stride1,
    const uint8_t* pix2,
    intptr_t stride2);

}

static constexpr uint64_t ITERATIONS = 100000000ULL;

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::fprintf(stderr,
                     "Usage: %s avx2|avx512\n",
                     argv[0]);
        return 1;
    }

    alignas(64) uint8_t src[64];
    alignas(64) uint8_t ref[64];

    /*
     * Deterministic but non-trivial pixel data.
     * Avoid all-zero / identical blocks.
     */
    for (int i = 0; i < 64; ++i)
    {
        src[i] = static_cast<uint8_t>((i * 13 + 17) & 0xff);
        ref[i] = static_cast<uint8_t>((i * 7  + 29) & 0xff);
    }

    using satd_fn =
        int (*)(const uint8_t*, intptr_t,
                const uint8_t*, intptr_t);

    satd_fn fn = nullptr;

    if (std::strcmp(argv[1], "avx2") == 0)
    {
        fn = x265_pixel_satd_8x8_avx2;
        std::printf("ISA       : AVX2\n");
    }
    else if (std::strcmp(argv[1], "avx512") == 0)
    {
        fn = x265_pixel_satd_8x8_avx512;
        std::printf("ISA       : AVX-512\n");
    }
    else
    {
        std::fprintf(stderr,
                     "Unknown ISA: %s\n",
                     argv[1]);
        return 1;
    }

    /*
     * First verify that both x265 implementations
     * produce the same SATD result.
     */
    int result_avx2 =
        x265_pixel_satd_8x8_avx2(src, 8, ref, 8);

    int result_avx512 =
        x265_pixel_satd_8x8_avx512(src, 8, ref, 8);

    std::printf("SATD AVX2  : %d\n", result_avx2);
    std::printf("SATD AVX512: %d\n", result_avx512);

    if (result_avx2 != result_avx512)
    {
        std::fprintf(stderr,
                     "ERROR: AVX2 and AVX512 results differ!\n");
        return 2;
    }

    /*
     * Warm-up.
     *
     * Important for this experiment because we do not
     * want startup effects to dominate perf counters.
     */
    volatile uint64_t sink = 0;

    for (uint64_t i = 0; i < 1000000; ++i)
        sink += fn(src, 8, ref, 8);

    /*
     * Measured workload.
     */
    auto start = std::chrono::steady_clock::now();

    for (uint64_t i = 0; i < ITERATIONS; ++i)
        sink += fn(src, 8, ref, 8);

    auto end = std::chrono::steady_clock::now();

    double seconds =
        std::chrono::duration<double>(end - start).count();

    double calls_per_second =
        static_cast<double>(ITERATIONS) / seconds;

    double pixels_per_second =
        calls_per_second * 64.0;

    std::printf("\n");
    std::printf("Iterations : %lu\n",
                (unsigned long)ITERATIONS);

    std::printf("Time       : %.6f s\n", seconds);

    std::printf("Calls/sec  : %.3f M\n",
                calls_per_second / 1e6);

    std::printf("Pixels/sec : %.3f G\n",
                pixels_per_second / 1e9);

    std::printf("Sink       : %lu\n",
                (unsigned long)sink);

    return 0;
}
