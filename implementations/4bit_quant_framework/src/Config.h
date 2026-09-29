#pragma once

// Disable all timers
// #define DISABLE_TIMING

namespace ML {
namespace Config {
constexpr bool ENABLE_SIMD = false;
constexpr bool FANCY_LOGGING = true;

// Floating Point Compare Epsilon
constexpr float EPSILON = 0.001;

// Quantization bit width
constexpr int QUANT_BITS = 4;
constexpr int QUANT_MAX = (1 << (QUANT_BITS - 1)) - 1;  // 7 for 4-bit
constexpr int QUANT_MIN = -(1 << (QUANT_BITS - 1));      // -8 for 4-bit
} // namespace Config
} // namespace ML::Config