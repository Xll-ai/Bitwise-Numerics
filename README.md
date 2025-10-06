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

**Languages:** C, C++, C#, Java, Go, Rust, Python, Haskell, JavaScript, Excel Lambda, CUDA, Maple, Julia, Wolfram Language, R, Fortran, GLSL, HLSL, SQL

**Platforms:** CPU (x86, x86-64, ARM), GPU (NVIDIA CUDA, AMD, Intel), NPU (Huawei Ascend), TPU (Google), database systems, graphics APIs (OpenGL, DirectX, Vulkan)

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

## Numeric Formats Reference

### Integer Formats

| Format | Structure | Range | Values | IEEE 754 | Platform Support |
|--------|-----------|-------|--------|----------|------------------|
| **INT1** | 1-bit unsigned | 0 to 1 | 2 | No | ML hardware (binary quantization) |
| **INT2** | 2-bit signed | -2 to +1 | 4 | No | ML accelerators (experimental) |
| **INT4** | 4-bit signed | -8 to +7 | 16 | No | ML quantization, CUDA 12.0+ |
| **INT8** | 8-bit signed | -128 to +127 | 256 | No | Universal (Ascend inference) |
| **UINT8** | 8-bit unsigned | 0 to 255 | 256 | No | Universal |
| **INT16** | 16-bit signed | -32,768 to +32,767 | 65,536 | No | Universal |
| **INT32** | 32-bit signed | ±2.1B | ~4.3B | No | Universal (most common) |
| **INT64** | 64-bit signed | ±9.2×10¹⁸ | ~18.4Q | No | Universal |
| **INT128** | 128-bit signed | ±2¹²⁷ | 2¹²⁸ | No | GCC/Clang, Rust, Julia only |

### Floating-Point Formats

| Format | Structure | Range | Values | IEEE 754 | ML/AI Use |
|--------|-----------|-------|--------|----------|-----------|
| **FP4 E2M1** | 1s + 2e + 1m | ±6 | 16 | No | Experimental quantization |
| **FP6 E3M2** | 1s + 3e + 2m | ±28 | 64 | No | Research (ultra-low precision) |
| **FP8 E4M3** | 1s + 4e + 3m | ±448 | ~240 | No | NVIDIA training (H100) |
| **FP8 E5M2** | 1s + 5e + 2m | ±57,344 | ~192 | No | NVIDIA training (activations) |
| **HiF8** | 1s + 4e + 3m (est.) | ~±448 | ~240 | No | Huawei Ascend (planned) |
| **FP16** | 1s + 5e + 10m | ±65,504 | ~65K | Yes | Ascend fallback, mixed precision |
| **BF16** | 1s + 8e + 7m | ±3.4×10³⁸ | ~32K | No | Google/Intel ML, Ascend |
| **FP32** | 1s + 8e + 23m | ±3.4×10³⁸ | ~4.3B | Yes | Standard float, high precision |
| **TF32** | 1s + 8e + 10m | ±3.4×10³⁸ | ~524K | No | NVIDIA tensor cores (A100+) |
| **FP64** | 1s + 11e + 52m | ±1.8×10³⁰⁸ | ~18.4Q | Yes | Standard double precision |
| **x87 Extended** | 1s + 15e + 64m | ±1.2×10⁴⁹³² | ~2⁶⁴ | No | x86 FPU (legacy) |
| **FP128** | 1s + 15e + 112m | ±1.2×10⁴⁹³² | ~2¹¹³ | Yes | Quadruple precision (GCC) |

### Decimal Formats

| Format | Structure | Range | Precision | IEEE 754 | Use Case |
|--------|-----------|-------|-----------|----------|----------|
| **Decimal32** | 32-bit base-10 | ±9.99×10⁹⁶ | 7 digits | Yes | Compact decimal |
| **Decimal64** | 64-bit base-10 | ±9.99×10³⁸⁴ | 16 digits | Yes | Financial (IEEE) |
| **Decimal128** | 128-bit base-10 | ±9.99×10⁶¹⁴⁴ | 34 digits | Yes | High-precision financial |
| **.NET Decimal** | 128-bit base-10 | ±7.9×10²⁸ | 28-29 digits | No | C# financial calculations |

### Arbitrary Precision Formats

| Format | Type | Precision | Representation | Languages |
|--------|------|-----------|----------------|-----------|
| **BigInteger** | Integer | Unlimited | Array of limbs | Java, C#, Python, Go, Rust, Haskell, JS |
| **Rational** | Fraction | Unlimited | num/den pair | Haskell, Python, Julia, Go, Maple |
| **MPFR** | Float | Configurable | Arbitrary mantissa | C/C++ (lib), Julia (BigFloat), Python |
| **BigDecimal** | Decimal | Unlimited | Unscaled int + scale | Java, Python (Decimal) |
| **Symbolic** | Exact | Infinite | Expression tree | Wolfram, Maple, SymPy, Haskell |

### Vector/SIMD Formats

| Format | Elements | Element Type | Total Bits | Platform |
|--------|----------|--------------|------------|----------|
| **half2** | 2 | FP16 | 32 | CUDA (NVIDIA) |
| **float4** | 4 | FP32 | 128 | CUDA (NVIDIA) |
| **double2** | 2 | FP64 | 128 | CUDA (NVIDIA) |
| **__m128** | 4 | FP32 | 128 | x86 SSE |
| **__m256** | 8 | FP32 | 256 | x86 AVX |
| **__m512** | 16 | FP32 | 512 | x86 AVX-512 |
| **float32x4_t** | 4 | FP32 | 128 | ARM NEON |
| **vec4** | 4 | FP32 | 128 | GLSL/HLSL shaders |

### Specialized Formats

| Format | Type | Size | Use Case | Platform |
|--------|------|------|----------|----------|
| **size_t** | Unsigned int | 32/64-bit | Object sizes | C/C++, platform-dependent |
| **wchar_t** | Character | 16/32-bit | Wide characters | C/C++, platform-dependent |
| **time_t** | Integer | 32/64-bit | Unix timestamps | POSIX systems |
| **DateTime** | Integer | 64-bit | .NET timestamps | .NET (100ns ticks) |
| **UNORM/SNORM** | Normalized int | 8/16-bit | Graphics [0,1]/[-1,1] | DirectX, Vulkan |
| **RGB10A2** | Packed color | 32-bit | 10-bit RGB + 2-bit alpha | Graphics APIs |
| **SQL NUMERIC** | Decimal | Variable | Exact database values | SQL databases |

### Historical/Legacy Formats

| Format | Structure | Range | Platform | Status |
|--------|-----------|-------|----------|--------|
| **IBM Hex Float** | Base-16 exponent | ±7.2×10⁷⁵ | IBM mainframes | Legacy |
| **VAX F-float** | Non-IEEE 32-bit | ±1.7×10³⁸ | DEC VAX | Legacy |
| **VAX D-float** | Non-IEEE 64-bit | ±1.7×10³⁸ | DEC VAX | Legacy |
| **Cray Float** | 64-bit custom | ±10²⁴⁶⁶ | Cray supercomputers | Legacy |
| **BCD** | 4 bits per digit | 0-9 per nibble | Legacy systems | Legacy |

### Format Properties

**Two's Complement**: All signed integers use two's complement representation (e.g., INT8: -128 = 0x80 = 10000000₂), ensuring consistent behavior across platforms.

**Special Values**:
- **IEEE 754 formats** (FP16, FP32, FP64, Decimal32/64/128): Support ±0, ±Infinity, NaN, and subnormals per standard
- **Vendor formats** (FP8, BF16, TF32): Vendor-specific special value handling
- **HiF8**: Likely supports IEEE-style special values (to be confirmed)

**Precision vs Range Trade-offs**:
- **FP8 E4M3**: 4 exponent bits, 3 mantissa bits → high precision, limited range (±448)
- **FP8 E5M2**: 5 exponent bits, 2 mantissa bits → wide range (±57K), lower precision
- **BF16**: Matches FP32 range (8 exp bits) with reduced precision (7 mantissa bits)
- **FP16**: Balanced precision (10 mantissa bits) with limited range (5 exp bits)

**Consistency Guarantees**:
- **Always identical**: IEEE 754 formats (FP16, FP32, FP64, Decimal), standard integers (INT8-INT64)
- **Vendor-consistent**: FP8 E4M3/E5M2 (NVIDIA spec), BF16 (Google/Intel), TF32 (NVIDIA)
- **Platform-dependent**: size_t, intptr_t, long double, wchar_t
- **Potentially different**: HiF8 (Huawei, not yet released), Posit/Unum (experimental)

### DeepSeek R2 ML Training Context

| Format | Use Case | System | Performance Impact |
|--------|----------|--------|-------------------|
| **FP8 E4M3** | Forward pass gradients | NVIDIA H100 | 4× throughput vs FP16 |
| **FP8 E5M2** | Activations | NVIDIA H100 | ~700GB model size |
| **FP16** | Fallback training | Ascend 910B/C | ~1.4TB model size |
| **BF16** | Fallback training | Ascend 910B/C | 2× memory vs FP8 |
| **FP32** | Master weights | All platforms | Accumulation accuracy |
| **INT8** | Inference | Ascend | Throughput optimized |
| **TF32** | Tensor cores | NVIDIA A100+ | Automatic acceleration |

**Key Insights**:
- NVIDIA H100's FP8 support enabled successful DeepSeek R2 training with ~700GB memory footprint
- Huawei Ascend's lack of FP8 forced FP16/BF16 fallback, doubling memory to ~1.4TB
- Memory bandwidth bottlenecks and framework maturity (CANN) caused Ascend training failures
- INT8 quantization used for production inference across all platforms

### Format Selection Guidelines

**Machine Learning**:
- Training: FP8 E4M3/E5M2 (NVIDIA), FP16/BF16 (fallback), FP32 (master weights)
- Inference: INT8 quantization with per-layer scaling

**Scientific Computing**:
- General: FP64 for accumulation accuracy
- Climate/weather: FP64 for long-term stability
- Graphics: FP16 for colors, FP32 for geometry

**Financial**:
- Use decimal formats (Decimal64, .NET Decimal, BigDecimal) to avoid binary rounding errors
- Never use binary floating-point for currency calculations

**Embedded Systems**:
- Fixed-point formats (Q7.8, Q15.16, Q0.31) for deterministic performance
- Platform-native integer sizes for efficiency

**Cryptography**:
- Arbitrary-precision integers (BigInteger, GMP) for large number arithmetic
- Constant-time operations to prevent timing attacks

**Database**:
- SQL NUMERIC/DECIMAL for exact values
- FLOAT/DOUBLE for approximate scientific data
- INT64 for identifiers and counters

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
- Language support filtering
- Category-based navigation
- Detailed format specifications
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