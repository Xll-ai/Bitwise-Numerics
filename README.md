

# Bitwise-Numerics
**Cross-language exploration of bitwise numerics and floating-point formats — from Assembly to modern programming languages.**

---

## 📌 Overview
This repository dives into **low-level numeric representations**, exploring how numbers are encoded at the bitwise level across CPUs, GPUs, and programming languages. It examines their use in AI/ML workloads, inspired by DeepSeek’s R2 training, where NVIDIA’s FP8 enabled efficiency and Huawei Ascend’s lack of FP8 caused failures. Focus areas include:

- **Floating-Point Formats**: FP8 (E4M3/E5M2, HiF8), FP16, BF16, FP32, TF32, FP64, Decimal (IEEE, .NET).
- **Integer Encodings**: INT2, INT4, INT8 with two’s complement.
- **Special Values**: NaN, ±Infinity, subnormals.
- **Precision vs. Dynamic Range**: Trade-offs in AI and general computing.
- **Languages**: Assembly (x86/x86_64/ARM), C, C++, C#, Java, JavaScript, Python, Go, Rust.

---

## 🔍 Goals
- Document bitwise numeric handling across languages.
- Analyze CPU/GPU representations for AI/ML (e.g., DeepSeek R2).
- Compare assembler-level behavior with high-level abstractions.
- Provide reference code and a universal JSON adapter for number representation.

---

## 🛠 Languages & Environments
- **Assembly**: x86/x86_64/ARM instructions (e.g., F16C, SSE).
- **C/C++**: Native types, intrinsics for low-precision.
- **C#**: `System.Half`, `float`, `double`, `Decimal`, `sbyte`.
- **Java**: `HalfFloat`, `float`, `double`, `byte`, `BigDecimal`.
- **JavaScript**: `Float32Array`, `Int8Array`, library emulation.
- **Python**: `numpy.float16`, `float32`, `float64`, `int8`.
- **Go**: `float32`, `float64`, `int8`, `gonum`.
- **Rust**: `half::f16`, `bf16`, `f32`, `f64`, `i8`.

---

## 📂 Structure
- **/src**: Code snippets for encoding/decoding numbers in each language.
- **/docs**: Detailed analysis of formats, platforms, and AI use cases.
- **/json**: Universal JSON adapter for number representation.
- **/examples**: Practical demos, including DeepSeek R2 case study.

---

## 🔢 Numerical Formats
| Format       | Structure                     | Range                              | Values         | IEEE 754 | DeepSeek R2 Use |
|--------------|-------------------------------|------------------------------------|----------------|----------|-----------------|
| **INT2**     | 2-bit, signed/unsigned        | -2 to +1 / 0 to 3                 | 4              | No       | Not used        |
| **INT4**     | 4-bit, signed/unsigned        | -8 to +7 / 0 to 15                | 16             | No       | Not used        |
| **INT8**     | 8-bit, signed/unsigned        | -128 to +127 / 0 to 255           | 256            | No       | Inference (Ascend) |
| **FP8 E4M3** | 1 sign, 4 exp, 3 mantissa     | ±0.001953125 to ±448              | ~240           | No       | Training (NVIDIA) |
| **FP8 E5M2** | 1 sign, 5 exp, 2 mantissa     | ±0.000015259 to ±196,608          | ~128           | No       | Training (NVIDIA) |
| **HiF8**     | Speculated: variable exp/mant  | ~±0.001 to ±131,072               | ~128–256       | No       | Future Ascend   |
| **FP16**     | 1 sign, 5 exp, 10 mantissa    | ±0.000061035 to ±65,504           | ~65,536        | Yes      | Ascend fallback |
| **BF16**     | 1 sign, 8 exp, 7 mantissa     | ±1.18e-38 to ±3.4e38              | ~65,536        | No       | Ascend fallback |
| **FP32**     | 1 sign, 8 exp, 23 mantissa    | ±1.18e-38 to ±3.4e38              | ~4.3B          | Yes      | High-precision  |
| **TF32**     | 1 sign, 8 exp, 10 mantissa    | ±1.18e-38 to ±3.4e38              | ~524,288       | No       | NVIDIA training |
| **FP64**     | 1 sign, 11 exp, 52 mantissa   | ±2.23e-308 to ±1.8e308            | ~18.4 quintillion | Yes   | Not used        |
| **Decimal (.NET)** | 128-bit, base-10         | ±1.0e-28 to ±7.9e28               | ~10^28         | No       | Not used        |
| **Decimal32** | 32-bit, base-10             | ±1.0e-95 to ±9.9e96               | ~10^7          | Yes      | Not used        |
| **Decimal64** | 64-bit, base-10             | ±1.0e-383 to ±9.9e384             | ~10^16         | Yes      | Not used        |
| **Decimal128** | 128-bit, base-10           | ±1.0e-6143 to ±9.9e6144           | ~10^34         | Yes      | Not used        |

- **Two’s Complement**: Used for signed integers (e.g., INT8: -128 = 0x80, binary 10000000), consistent across platforms.
- **Special Values**: NaN, ±Infinity, subnormals in floating-point formats (IEEE 754 for FP16, FP32, FP64, Decimal32/64/128; vendor-specific for FP8, HiF8, BF16, TF32).
- **Precision vs. Range**: FP8 E4M3 (0.125 step near 1.0) prioritizes precision; E5M2/BF16 prioritize range (±196,608/±3.4e38). HiF8 may balance dynamically.

---

## 🌐 Universal JSON Adapter
A compact JSON structure to represent any number across formats, used in `/json/numerical_formats.json`:

```json
{
  "NumberRepresentation": {
    "Format": "{type}", // e.g., INT8, FP8_E4M3, FP64
    "Value": "{number}", // e.g., 1, 1.0
    "Hex": "{hex}", // e.g., 0x01 (INT8), 0x3C (FP8 E4M3)
    "Binary": "{binary}", // e.g., 00000001 (INT8)
    "Range": "{range}", // e.g., -128 to +127
    "Values": "{count}", // e.g., 256, ~240
    "IEEE": "{true/false}",
    "Platforms": {
      "Huawei_Ascend": "{support}",
      "NVIDIA_GPUs": "{support}",
      "AMD_GPUs": "{support}",
      "Intel_GPUs": "{support}",
      "CPUs": "{support}"
    },
    "Languages": {
      "Assembly": "{support}",
      "C": "{support}",
      "C++": "{support}",
      "C#": "{support}",
      "Python": "{support}",
      "Rust": "{support}",
      "Java": "{support}",
      "JavaScript": "{support}",
      "Go": "{support}"
    },
    "R2_Relevance": "{relevance}"
  }
}
```

**Example (INT8: 1)**:
```json
{
  "NumberRepresentation": {
    "Format": "INT8",
    "Value": "1",
    "Hex": "0x01",
    "Binary": "00000001",
    "Range": "-128 to +127",
    "Values": "256",
    "IEEE": false,
    "Platforms": {
      "Huawei_Ascend": "Native",
      "NVIDIA_GPUs": "Native",
      "AMD_GPUs": "Native",
      "Intel_GPUs": "Native",
      "CPUs": "Native"
    },
    "Languages": {
      "Assembly": "Native via CPU instructions",
      "C": "Native via int8_t",
      "C++": "Native via int8_t",
      "C#": "Native via sbyte",
      "Python": "Native via numpy.int8",
      "Rust": "Native via i8",
      "Java": "Native via byte",
      "JavaScript": "Native via Int8Array",
      "Go": "Native via int8"
    },
    "R2_Relevance": "Used for R2 inference on Ascend"
  }
}
```

---

## 🚀 Key Insights
- **DeepSeek R2 Case Study**: NVIDIA’s FP8 (E4M3/E5M2, ~240/~128 values) enabled efficient training (~700GB, 3,000 TFLOPS); Huawei Ascend’s lack of FP8 forced FP16/BF16 (1.4TB), causing failures due to memory inefficiency, unstable interconnects, and immature CANN framework. INT8 (256 values) used for inference.
- **China’s Pushback**: Rejects NVIDIA H20 GPUs due to security concerns (tracking/backdoors), promoting domestic chips like Ascend. Black market smuggling of FP8-capable GPUs (A100, H800) supports AI workloads.
- **Huawei’s HiF8**: Planned for Ascend 910D (Q4 2025), speculated ~128–256 values (~±0.001 to ±131,072), aims to match FP8’s efficiency but may differ in representation.
- **Consistency**: IEEE 754 formats (FP16, FP32, FP64, Decimal32/64/128) identical across platforms; INT8 standardized; FP8 E4M3/E5M2 consistent per NVIDIA spec; HiF8 likely unique.

---

## 📂 Repository Contents
- **/src**: Snippets for encoding/decoding numbers in Assembly, C, C++, C#, Java, JavaScript, Python, Go, Rust.
- **/docs**: Analysis of formats, platforms, and AI use cases (e.g., DeepSeek R2).
- **/json**: Universal JSON adapter (`numerical_formats.json`).
- **/examples**: Demos, including FP8 and INT8 in AI training/inference.

---

## 🛠 Getting Started
1. Clone the repo: `git clone https://github.com/your-username/Bitwise-Numerics.git`
2. Explore `/src` for language-specific code.
3. Use `/json/numerical_formats.json` to represent numbers.
4. Check `/docs` for detailed format specifications and AI applications.
