# x265 SATD AVX2 vs AVX-512 Microbenchmark

This microbenchmark isolates the x265 8x8 SATD primitive and compares the performance of the native AVX2 and AVX-512 implementations used by x265.

The benchmark directly invokes the following x265 assembly routines:

```text
x265_pixel_satd_8x8_avx2
x265_pixel_satd_8x8_avx512
```

These are the same implementations present in the x265 encoder binary and observed during application-level profiling.

## Objective

The purpose of this benchmark is to quantify the performance difference between the AVX2 and AVX-512 implementations of the x265 8x8 SATD primitive while removing unrelated encoder effects such as:

- thread-pool scheduling
- WPP
- NUMA effects
- rate control
- motion-search orchestration
- frame-level parallelism
- other x265 primitives

The benchmark is intended to answer:

> How much performance benefit does the x265 AVX-512 SATD implementation provide relative to the corresponding AVX2 implementation when the primitive is measured in isolation?

## Relationship to x265

Application-level profiling of an 8-bit x265 encode identified AVX-512 SATD routines among the active AVX-512 hotspots.

The microbenchmark does not reimplement SATD using compiler intrinsics.

Instead, it links against the x265 static library and directly executes the existing x265 assembly implementations.

The benchmark therefore preserves the implementation being evaluated while simplifying the surrounding workload.

## Behavioral Equivalence

The benchmark is considered behaviorally representative of the selected x265 SATD hotspot for the following reasons:

1. It invokes the same x265 AVX2 and AVX-512 SATD symbols used by the encoder.
2. Both functions come directly from the x265 assembly implementation.
3. Disassembly confirms that the AVX2 implementation uses YMM/VEX instructions and the AVX-512 implementation uses ZMM/EVEX instructions.
4. Both implementations operate on the same 8x8 pixel input.
5. Both implementations produce the same SATD result.
6. The selected SATD primitive was observed as an actual hotspot during full x265 profiling.

This benchmark is not intended to reproduce the complete behavior or performance of the x265 encoder. It isolates one real x265 computational kernel.

## Directory Layout

Recommended location inside the x265 fork:

```text
x265/
├── source/
├── build/
└── microbenchmarks/
    └── satd/
        ├── satd_bench.cpp
        └── README.md
```

## Build x265

Create an 8-bit static x265 build with debug symbols and frame pointers:

```bash
cd x265

mkdir -p build/linux/8bit
cd build/linux/8bit

cmake ../../../source \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_FLAGS_RELWITHDEBINFO="-O3 -g -fno-omit-frame-pointer" \
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O3 -g -fno-omit-frame-pointer" \
  -DENABLE_SHARED=OFF \
  -DENABLE_CLI=ON

make -j"$(nproc)"
```

Confirm that the static library exists:

```bash
ls -lh libx265.a
```

## Build the Microbenchmark

From:

```text
x265/build/linux/8bit
```

run:

```bash
g++ -O3 -g \
  ../../../microbenchmarks/satd/satd_bench.cpp \
  ./libx265.a \
  -lpthread -ldl -lrt -lm -lnuma \
  -o satd_bench
```

## Verify the x265 Symbols

Confirm that the benchmark contains the expected implementations:

```bash
nm -C ./satd_bench | \
grep 'x265_pixel_satd_8x8_.*avx'
```

Expected symbols include:

```text
x265_pixel_satd_8x8_avx2
x265_pixel_satd_8x8_avx512
```

## Verify the Generated Assembly

AVX2:

```bash
objdump -d -Mintel ./satd_bench | \
grep -A40 '<x265_pixel_satd_8x8_avx2>'
```

The implementation should contain YMM/VEX instructions.

AVX-512:

```bash
objdump -d -Mintel ./satd_bench | \
grep -A40 '<x265_pixel_satd_8x8_avx512>'
```

The implementation should contain ZMM/EVEX instructions.

## Run the Benchmark

AVX2:

```bash
./satd_bench avx2
```

AVX-512:

```bash
./satd_bench avx512
```

Both implementations should produce the same SATD value.

Example:

```text
SATD AVX2  : 9504
SATD AVX512: 9504
```

## Performance Measurement

For reproducibility, pin the benchmark to one logical CPU.

### AVX2

```bash
taskset -c 0 \
perf stat -r 10 \
  -o satd-avx2-perf.txt \
  -e cycles,ref-cycles,instructions,task-clock \
  ./satd_bench avx2 \
  2>&1 | tee satd-avx2.log
```

### AVX-512

```bash
taskset -c 0 \
perf stat -r 10 \
  -o satd-avx512-perf.txt \
  -e cycles,ref-cycles,instructions,task-clock \
  ./satd_bench avx512 \
  2>&1 | tee satd-avx512.log
```

## Metrics

The primary metrics are:

```text
instructions
cycles
IPC
effective frequency
execution time
calls/second
pixels/second
```

Because each SATD call processes an 8x8 block:

```text
pixels per call = 64
```

Cycles per SATD call:

```text
cycles/call = total cycles / number of SATD calls
```

Cycles per pixel:

```text
cycles/pixel =
    total cycles /
    (number of SATD calls * 64)
```

Speedup:

```text
AVX-512 speedup =
    AVX2 execution time /
    AVX-512 execution time
```

## Initial 8-bit Result

Using 100 million SATD calls on one pinned CPU, the initial measurements were approximately:

| Metric | AVX2 | AVX-512 | Difference |
|---|---:|---:|---:|
| Instructions | 9.00 B | 6.78 B | -24.7% |
| Cycles | 3.37 B | 2.83 B | -16.2% |
| IPC | 2.67 | 2.40 | -10.1% |
| Effective frequency | 3.776 GHz | 3.570 GHz | -5.5% |
| Execution time | 0.893 s | 0.792 s | -11.3% |
| Throughput | ~113 M calls/s | ~128 M calls/s | ~+13% |

The AVX-512 implementation therefore executes substantially fewer instructions and cycles.

Although AVX-512 reduces the effective operating frequency, the reduction in computational work is large enough for the isolated SATD primitive to complete approximately 13% faster.

## Comparison With Full x265

The isolated SATD result should not be interpreted as the expected full-encoder speedup.

In the complete 8-bit x265 workload, AVX-512-tagged functions represented only a fraction of total sampled execution. Large portions of the workload remained in AVX2, AVX, SSE, and higher-level C++ code.

Therefore:

```text
strong local kernel improvement
            +
limited fraction of total execution
            +
AVX-512 frequency effect
            =
small or negligible application-level gain
```

This distinction between kernel-level acceleration and application-level acceleration is one of the primary motivations for this microbenchmark.

## Scope

This benchmark currently evaluates:

```text
8-bit x265
8x8 SATD
AVX2 vs AVX-512
single-thread execution
```

It does not attempt to model the complete x265 encoder.

Future experiments may extend the same methodology to other primitives or the high-bit-depth path.

## License

This benchmark is intended to be used within a fork of the upstream x265 project.

The x265 source and assembly implementations remain subject to the licensing terms of the upstream x265 project.
