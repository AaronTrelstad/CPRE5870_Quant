#pragma once

#include <cmath>
#include <vector>

#include "Config.h"
#include "Types.h"
#include "Utils.h"

namespace ML {

struct QuantParams {
    fp32 scale;
    i8 zero_point;
};

namespace Quantize {

    // Single-pass: compute avg and max|x - avg| simultaneously using online algorithm
    // (Welford-style: first pass gets avg, but we can't avoid 2 passes for max deviation
    //  without knowing avg first. However we CAN merge avg + quantize into 2 passes total
    //  instead of the current 3 passes: avg, max, quantize)
    inline QuantParams computeInputParams(const fp32* data, size count) {
        // Pass 1: compute average
        fp32 sum = 0.0f;
        for (size i = 0; i < count; i++) {
            sum += data[i];
        }
        fp32 avg = sum / (fp32)count;

        // Pass 2: max absolute deviation from average
        fp32 max_dev = 0.0f;
        for (size i = 0; i < count; i++) {
            fp32 d = fabsf(data[i] - avg);
            if (d > max_dev) max_dev = d;
        }

        fp32 scale = (max_dev > 0.0f)
            ? (fp32)Config::QUANT_MAX / max_dev
            : 1.0f;
        i8 zp = (i8)clamp(
            (int)(-roundf(avg * scale)),
            (int)Config::QUANT_MIN,
            (int)Config::QUANT_MAX
        );

        QuantParams p;
        p.scale = scale;
        p.zero_point = zp;
        return p;
    }

    // Compute input params AND quantize in 2 total passes instead of 3
    inline QuantParams computeAndQuantizeInput(
        const fp32* src, size count, std::vector<i8>& dst
    ) {
        // Pass 1: average
        fp32 sum = 0.0f;
        for (size i = 0; i < count; i++) {
            sum += src[i];
        }
        fp32 avg = sum / (fp32)count;

        // Pass 2: max deviation + quantize in one shot
        fp32 max_dev = 0.0f;
        for (size i = 0; i < count; i++) {
            fp32 d = fabsf(src[i] - avg);
            if (d > max_dev) max_dev = d;
        }

        fp32 scale = (max_dev > 0.0f)
            ? (fp32)Config::QUANT_MAX / max_dev : 1.0f;
        i8 zp = (i8)clamp(
            (int)(-roundf(avg * scale)),
            (int)Config::QUANT_MIN, (int)Config::QUANT_MAX
        );

        // Pass 3 (merged with nothing — but this is the minimum needed)
        dst.resize(count);
        for (size i = 0; i < count; i++) {
            int val = (int)roundf(scale * src[i]) + zp;
            dst[i] = (i8)clamp(val, (int)Config::QUANT_MIN, (int)Config::QUANT_MAX);
        }

        QuantParams p;
        p.scale = scale;
        p.zero_point = zp;
        return p;
    }

    inline QuantParams computeWeightParams(const fp32* data, size count) {
        fp32 max_abs = 0.0f;
        for (size i = 0; i < count; i++) {
            fp32 a = fabsf(data[i]);
            if (a > max_abs) max_abs = a;
        }

        fp32 scale = (max_abs > 0.0f)
            ? (fp32)Config::QUANT_MAX / max_abs : 1.0f;

        QuantParams p;
        p.scale = scale;
        p.zero_point = 0;
        return p;
    }

    inline void quantizeToInt8(
        const fp32* src, size count,
        const QuantParams& params,
        std::vector<i8>& dst
    ) {
        dst.resize(count);
        for (size i = 0; i < count; i++) {
            int val = (int)roundf(params.scale * src[i]) + params.zero_point;
            dst[i] = (i8)clamp(val, (int)Config::QUANT_MIN, (int)Config::QUANT_MAX);
        }
    }

    inline void quantizeBiasToInt32(
        const fp32* src, size count,
        fp32 Sb,
        std::vector<i32>& dst
    ) {
        dst.resize(count);
        for (size i = 0; i < count; i++) {
            dst[i] = (i32)roundf(Sb * src[i]);
        }
    }

    inline void computeWeightSums(
        const std::vector<i8>& qw,
        size filters_per_output,
        size num_outputs,
        size stride,
        std::vector<i32>& sums
    ) {
        sums.assign(num_outputs, 0);
        for (size o = 0; o < num_outputs; o++) {
            for (size i = 0; i < filters_per_output; i++) {
                sums[o] += (i32)qw[i * stride + o];
            }
        }
    }

    inline fp32 dequantize(i32 acc, i32 zi, i32 sum_qw_for_output, fp32 Si, fp32 Sw) {
        return (fp32)(acc - zi * sum_qw_for_output) / (Si * Sw);
    }

} // namespace Quantize
} // namespace ML
