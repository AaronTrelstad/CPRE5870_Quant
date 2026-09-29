#pragma once

namespace ML {
namespace Config {
constexpr bool ENABLE_SIMD = false;
constexpr bool FANCY_LOGGING = true;

constexpr float EPSILON = 0.001;

constexpr int QUANT_BITS = 8;
constexpr int QUANT_MAX = (1 << (QUANT_BITS - 1)) - 1;  
constexpr int QUANT_MIN = -(1 << (QUANT_BITS - 1));     
} 
} 
