# Bitwise Numerics

Comprehensive cross-language catalog of numeric formats from low-precision quantized integers to arbitrary-precision symbolic mathematics.

## Overview

This repository provides an exhaustive reference for numeric representations across programming languages, hardware platforms, and application domains. It documents how numbers are encoded at the bit level, covering everything from 1-bit binary quantization to arbitrary-precision symbolic computation.

The catalog emerged from analysis of AI/ML numeric requirements, particularly examining how format availability impacts system design. The DeepSeek R2 training experience highlighted this: NVIDIA's FP8 support enabled efficient training, while its absence on Huawei Ascend systems forced less efficient alternatives.

### Coverage

**Format Categories:**
- Integer formats: 1-bit to 128-bit, signed and unsigned, fixed and arbitrary precision
- Floating-point: FP4, FP6, FP8 variants, FP16, BF16, FP32, TF32, FP64, FP128, x87 extended
- Decimal: IEEE 754 decimal, .NET Decimal, BigDecimal, arbitrary precision
- Fixed-point: Q notation formats (Q7.8, Q15.16, Q0.31, etc.)
- Complex and quaternion types
- Rational numbers: exact fraction representation
- Symbolic: algebraic numbers, computer algebra systems
- Interval arithmetic: rigorous error bounds
- Vector/SIMD: packed formats for parallel processing
- Specialized: database types, graphics formats, time representations, legacy systems

**Languages:**
C, C++, C#, Java, Go, Rust, Python, Haskell, JavaScript, Excel Lambda, CUDA, Maple, Julia, Wolfram Language, R, Fortran, GLSL, HLSL, SQL

**Platforms:**
CPU (x86, x86-64, ARM), GPU (NVIDIA CUDA, AMD, Intel), NPU (Huawei Ascend), TPU (Google), database systems, graphics APIs (OpenGL, DirectX, Vulkan)

## Goals

1. Document comprehensive numeric type support across languages and platforms
2. Provide language-specific implementation details for each format
3. Analyze hardware support and performance characteristics
4. Establish format consistency and conversion requirements
5. Enable informed decisions about numeric precision in applications
6. Preserve knowledge of historical and legacy formats

## Structure

```
/docs/           # Format specifications organized by category
  int.json       # Integer formats
  float.json     # Floating-point formats
  decimal.json   # Decimal formats
  vector.json    # SIMD/vector formats
  arbitrary.json # Arbitrary precision formats
  specialized.json # Domain-specific formats
/src/            # Code examples for each language
/json/           # Universal JSON schema for format representation
/examples/       # Practical demonstrations and case studies
/viewer/         # Interactive HTML format explorer
```

## Format Categories

### Integer Formats

**Fixed Precision:**
- INT1 (binary): 2 values
- INT2: 4 values (-2 to +1 signed)
- INT4: 16 values (-8 to +7 signed)
- INT8/UINT8: 256 values
- INT16/UINT16: 65,536 values
- INT32/UINT32: ~4.3 billion values
- INT64/UINT64: ~18.4 quintillion values
- INT128/UINT128: 2^128 values (limited compiler support)

**Arbitrary Precision:**
- BigInteger (Java, C#, Python, Go, Rust, Haskell)
- Integer (Haskell, Wolfram, Maple)
- int (Python 3+)

**Platform-Dependent:**
- size_t, ptrdiff_t, intptr_t, uintptr_t
- Platform-specific int/uint (Go, Rust isize/usize)

### Floating-Point Formats

**Low Precision (ML/Graphics):**
- FP4 (E2M1, E1M2): 16 values, experimental
- FP6 (E3M2, E2M3): 64 values, experimental
- FP8 E4M3: ~240 values, NVIDIA ML training
- FP8 E5M2: ~192 values, NVIDIA ML training
- FP8 HiF8: Huawei proprietary (planned)

**Half Precision:**
- FP16 (IEEE 754): ~65,536 values, standard half precision
- BF16 (Brain Float): ~32,768 values, Google/Intel ML format

**Single Precision:**
- FP32 (IEEE 754): ~4.3 billion values, standard float
- TF32 (Tensor Float): ~524,288 values, NVIDIA tensor cores

**Double Precision:**
- FP64 (IEEE 754): ~18.4 quintillion values, standard double
- x87 Extended (80-bit): legacy x86 FPU format
- FP128 (Quadruple): IEEE 754 quadruple precision

**Arbitrary Precision:**
- MPFR: configurable precision floating-point
- BigFloat (Julia, Python via mpmath)
- Arbitrary precision (Wolfram, Maple)

### Decimal Formats

**Fixed Precision:**
- Decimal (.NET): 128-bit, 28-29 decimal digits
- Decimal32 (IEEE 754): ~10^7 values
- Decimal64 (IEEE 754): ~10^16 values
- Decimal128 (IEEE 754): ~10^34 values

**Arbitrary Precision:**
- BigDecimal (Java)
- Decimal (Python)
- Scientific (Haskell)

### Rational Numbers

- Ratio/Rational (Haskell)
- Fraction (Python)
- Rat (Go math/big)
- mpq_t (GMP)
- Native support in Wolfram, Maple, Julia

### Complex Numbers

- Complex&lt;float&gt; (32-bit components): C++, Go complex64
- Complex&lt;double&gt; (64-bit components): C++, C#, Python, Julia, Fortran, Go complex128, R
- Arbitrary precision complex (Julia, Wolfram, Maple)

### Vector/SIMD Formats

**CUDA:**
- half2, float4, double2, int4

**x86 SSE/AVX:**
- __m128, __m256, __m512 (4, 8, 16 floats)
- Integer variants for packed operations

**ARM NEON:**
- float32x4_t, int8x16_t, etc.

**Shader Languages:**
- GLSL: vec2/3/4, mat2/3/4, ivec, dvec
- HLSL: float2/3/4, half2/3/4, min16float

### Specialized Formats

**Time Representations:**
- Unix time_t (32-bit, 64-bit)
- Windows FILETIME (64-bit, 100ns ticks)
- .NET DateTime (62-bit ticks + 2-bit Kind)
- Java Instant (64-bit seconds + 32-bit nanos)
- JavaScript Date (FP64 milliseconds)
- Language-specific Duration types

**Database Types:**
- SQL: TINYINT, SMALLINT, INT, BIGINT, REAL, FLOAT, DOUBLE PRECISION, NUMERIC, DECIMAL, MONEY

**Graphics:**
- UNORM/SNORM (normalized integers)
- RGB10A2 (packed 10-bit color)
- R11G11B10F (packed HDR)
- sRGB (gamma-corrected)

**Historical/Legacy:**
- IBM Hexadecimal Float (base-16)
- DEC VAX F/D/G floating-point
- Cray-1 floating-point
- BCD (Binary Coded Decimal)

## Language Support Matrix

Each format entry includes comprehensive language support information:

- **Native support**: Built-in language types (e.g., float in C, Float64 in Julia)
- **Library support**: Available through standard or third-party libraries
- **Header/module requirements**: Include files or imports needed
- **Version requirements**: Minimum language version
- **Compiler requirements**: Specific compiler support (e.g., GCC extensions)
- **Platform restrictions**: Platform-specific availability
- **Notes**: Implementation details and caveats

## Universal JSON Schema

All formats are documented using a consistent JSON schema that captures:

```json
{
  "name": "Format name",
  "category": "integer|float|decimal|rational|symbolic|interval|other",
  "bits_total": "bit count or 'variable'",
  "signed": true|false|null,
  "range": "representable value range",
  "values": "count of distinct values",
  "structure": {
    "sign_bits": "count",
    "exponent_bits": "count or null",
    "mantissa_bits": "count or null", 
    "bias": "exponent bias or null",
    "specials": {
      "zeros": "representation",
      "infinities": true|false,
      "nans": true|false,
      "subnormals": true|false
    }
  },
  "bit_split": "human-readable layout",
  "governance": {
    "type": "IEEE|vendor|de-facto|language-standard|proprietary|proposed",
    "owner": "standards body or vendor",
    "status": "standardized|adopted|experimental|legacy",
    "notes": "additional context"
  },
  "language_support": [
    {
      "language": "language name",
      "native_support": "type name or none",
      "library_support": [
        {
          "library_name": "name",
          "data_type": "type",
          "operations": ["supported operations"],
          "version_required": "version"
        }
      ],
      "compiler_requirements": "requirements",
      "notes": "implementation details"
    }
  ],
  "arbitrary_precision_config": {
    "precision_type": "unlimited_integer|unlimited_rational|configurable_float|symbolic|interval",
    "representation": "internal representation",
    "memory_scaling": "complexity notation",
    "computational_complexity": {
      "addition": "O notation",
      "multiplication": "O notation",
      "division": "O notation"
    }
  }
}
```

## Key Technical Insights

### Precision vs Range Tradeoffs

Floating-point formats balance precision (mantissa bits) against dynamic range (exponent bits):

- **FP8 E4M3**: 4 exponent bits, 3 mantissa bits - prioritizes precision for gradients
- **FP8 E5M2**: 5 exponent bits, 2 mantissa bits - prioritizes range for activations
- **BF16**: 8 exponent bits, 7 mantissa bits - matches FP32 range, reduced precision
- **FP16**: 5 exponent bits, 10 mantissa bits - balanced for mixed-precision training

### Format Consistency

**Always Identical Across Platforms:**
- IEEE 754 formats: FP16, FP32, FP64, Decimal32/64/128
- Standard integers: INT8, INT16, INT32, INT64 (two's complement)
- Vendor formats with single specification: FP8 E4M3/E5M2 (NVIDIA), BF16, TF32

**Platform-Dependent:**
- Platform-sized integers: size_t, intptr_t, int/uint (language-dependent)
- Long double: may be 64, 80, or 128 bits
- wchar_t: 16 or 32 bits depending on platform

**Potentially Different:**
- Proprietary formats under development: HiF8 (Huawei)
- Experimental formats: Posit, Unum variants
- Historical formats: implementation-specific behaviors

### ML Training Format Requirements

Analysis of DeepSeek R2 training workflow:

1. **Forward Pass**: FP8 E4M3/E5M2 for matrix operations (memory bandwidth limited)
2. **Gradients**: FP8 E4M3 prioritized (needs higher precision)
3. **Weight Updates**: FP32 master weights (accumulation accuracy)
4. **Inference**: INT8 quantization (throughput optimized)

**NVIDIA H100 FP8 Advantage:**
- Tensor cores: 3,958 TFLOPS FP8 vs 989 TFLOPS FP16
- Memory: ~700GB model footprint with FP8
- Bandwidth: Higher effective bandwidth with 8-bit transfers

**Huawei Ascend Without FP8:**
- Forced FP16/BF16: ~1.4TB model footprint
- 2× memory bandwidth requirement
- Training instability reported
- Workarounds: gradient checkpointing, model parallelism inefficiencies

### Arithmetic Properties

**Integer Overflow Behavior:**
- Wrapping (default): C unsigned, Java, Go, JavaScript bitwise ops
- Checked (exception): C# checked blocks, Java Math.*Exact methods, Rust checked_*
- Saturating (clamp): Rust saturating_*, manual in other languages
- Undefined (error): C/C++ signed overflow is UB

**Floating-Point Special Values:**
- **NaN propagation**: Operations with NaN produce NaN (except some comparisons)
- **Infinity arithmetic**: ±∞ + ±∞ defined except ∞ - ∞ → NaN
- **Subnormal behavior**: Gradual underflow or flush-to-zero (configurable in some systems)

**Rounding Modes:**
- Round to nearest, ties to even (default)
- Round toward zero (truncation)
- Round toward +∞ (ceiling)
- Round toward -∞ (floor)
- Stochastic rounding (specialized ML hardware)

## Interactive Format Explorer

The repository includes an HTML-based interactive viewer (`/viewer/index.html`) that provides:

- Searchable catalog of all formats
- Side-by-side format comparison
- Language support filtering
- Bit layout visualization
- Links to specifications and documentation
- Platform compatibility matrix

## Case Studies

### DeepSeek R2 Training

NVIDIA H20 GPUs with FP8 support enabled successful training of the 671B parameter model:
- FP8 E4M3/E5M2 reduced memory footprint by ~50% compared to FP16
- Enabled fitting larger batch sizes within GPU memory
- Tensor core acceleration provided 4× throughput improvement
- Stable convergence with mixed precision training

Huawei Ascend systems without FP8 encountered challenges:
- FP16/BF16 fallback doubled memory requirements
- Memory bandwidth bottlenecks reduced effective throughput
- Interconnect stability issues under high memory pressure
- Software framework (CANN) maturity gaps

### Numeric Precision in Scientific Computing

Different domains have evolved different standard precisions:

- **Climate modeling**: FP64 for long-term accumulation accuracy
- **Graphics rendering**: FP16 sufficient for color values, FP32 for geometry
- **Financial calculations**: Decimal formats to avoid binary rounding errors
- **Cryptography**: Arbitrary precision integers for large number arithmetic
- **Signal processing**: Fixed-point Q formats on embedded systems
- **Machine learning inference**: INT8 quantization with per-layer scaling

## Getting Started

### Clone Repository

```bash
git clone https://github.com/Xll-ai/Bitwise-Numerics.git
cd Bitwise-Numerics
```

### Explore Format Catalog

```bash
# View all integer formats
cat docs/int.json

# View floating-point formats  
cat docs/float.json

# Open interactive viewer
open viewer/index.html
```

### Language-Specific Examples

Each `/src/<language>/` directory contains:
- Type declaration examples
- Encoding/decoding demonstrations
- Bit manipulation utilities
- Format conversion helpers
- Platform detection code

### Using the JSON Schema

The format definitions can be programmatically processed for:
- Automatic documentation generation
- Cross-language type mapping
- Numeric library development
- Format conversion utilities
- Educational visualization tools

## Contributing

Contributions are welcome for:

- Additional format documentation
- Language-specific implementation details
- Performance benchmarks
- Conversion utilities
- Bug fixes and corrections
- Platform-specific behavior notes

## References

### Standards Documents

- IEEE 754-2019: IEEE Standard for Floating-Point Arithmetic
- ISO/IEC 9899:2018: C Programming Language Standard  
- ISO/IEC 14882:2020: C++ Programming Language Standard
- ECMA-262: ECMAScript Language Specification

### Vendor Documentation

- NVIDIA CUDA Programming Guide
- Intel Intrinsics Guide
- ARM NEON Programmer's Guide
- AMD ROCm Documentation
- Huawei CANN Development Guide

### Research Papers

- Micikevicius et al., "Mixed Precision Training" (2018)
- Gustafson & Yonemoto, "Beating Floating Point at its Own Game: Posit Arithmetic" (2017)
- Muller et al., "Handbook of Floating-Point Arithmetic" (2018)

## License

This documentation is provided for educational and reference purposes. Individual code examples may have separate licenses as indicated.

## Acknowledgments

Format specifications compiled from official standards documents, vendor documentation, language references, and empirical testing across platforms. Special attention to ML numeric format requirements informed by analysis of production AI training systems.
