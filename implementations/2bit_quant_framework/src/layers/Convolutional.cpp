#include "Convolutional.h"

#include <vector>

namespace ML {

    void ConvolutionalLayer::computeNaive(const LayerData& dataIn) const {
        const dimVec& in_dims = getInputParams().dims;
        const dimVec& out_dims = getOutputParams().dims;
        const dimVec& weight_dims = getWeightParams().dims;

        size H = in_dims[0], W = in_dims[1], C = in_dims[2];
        size P = out_dims[0], Q = out_dims[1], M = out_dims[2];
        size R = weight_dims[0], S = weight_dims[1];
        size U = (H - R) / (P - 1);
        size in_count = getInputParams().flat_count();

        // Quantize input
        std::vector<i8> qi;
        QuantParams iParams = Quantize::computeAndQuantizeInput(
            (const fp32*)dataIn.raw(), in_count, qi
        );

        // Bias quantization
        fp32 Sb = iParams.scale * getWeightScale();
        std::vector<i32> qb;
        Quantize::quantizeBiasToInt32(
            (const fp32*)getBiasData().raw(), M, Sb, qb
        );

        const i8* qw_ptr = getQuantizedWeights().data();
        const i8* qi_ptr = qi.data();
        const i32* qb_ptr = qb.data();
        const i32* sum_qw_ptr = getWeightSums().data();
        fp32 Sw = getWeightScale();
        fp32 Si = iParams.scale;
        i32 zi = (i32)iParams.zero_point;
        fp32 inv_scale = 1.0f / (Si * Sw);
        fp32* output = (fp32*)getOutputData().raw();

        // Accumulator buffer
        std::vector<i32> acc(P * Q * M);
        i32* acc_ptr = acc.data();

        // Initialize accumulators with quantized bias
        for (size p = 0; p < P; p++) {
            for (size q = 0; q < Q; q++) {
                i32* a = acc_ptr + p * Q * M + q * M;
                for (size m = 0; m < M; m++) {
                    a[m] = qb_ptr[m];
                }
            }
        }

        // Cache-friendly loop order: p→q→r→s→c→m
        for (size p = 0; p < P; p++) {
            for (size q = 0; q < Q; q++) {
                i32* a = acc_ptr + p * Q * M + q * M;
                for (size r = 0; r < R; r++) {
                    for (size s = 0; s < S; s++) {
                        const i8* in_row = qi_ptr + (p * U + r) * W * C + (q * U + s) * C;
                        const i8* w_base = qw_ptr + r * S * C * M + s * C * M;
                        for (size c = 0; c < C; c++) {
                            i32 in_val = (i32)in_row[c];
                            const i8* w_row = w_base + c * M;
                            for (size m = 0; m < M; m++) {
                                a[m] += in_val * (i32)w_row[m];
                            }
                        }
                    }
                }
            }
        }

        // Dequantize and apply activation
        ActivationType act = getActivationType();
        for (size pq = 0; pq < P * Q; pq++) {
            size idx = pq * M;
            for (size m = 0; m < M; m++) {
                fp32 result = (fp32)(acc_ptr[idx + m] - zi * sum_qw_ptr[m]) * inv_scale;
                output[idx + m] = Activation::apply(result, act);
            }
        }
    }

    void ConvolutionalLayer::computeThreaded(const LayerData& dataIn) const {}
    void ConvolutionalLayer::computeTiled(const LayerData& dataIn) const {}
    void ConvolutionalLayer::computeSIMD(const LayerData& dataIn) const {}

    size ConvolutionalLayer::getNumMACs() const {
        const dimVec& out_dims = getOutputParams().dims;
        const dimVec& weight_dims = getWeightParams().dims;

        const size P = out_dims[0], Q = out_dims[1];
        const size R = weight_dims[0], S = weight_dims[1];
        const size C = weight_dims[2], M = weight_dims[3];

        return P * Q * M * R * S * C;
    }
}  // namespace ML
