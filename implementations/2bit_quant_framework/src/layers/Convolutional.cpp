#include "Convolutional.h"

#include <cstring>
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
        size out_count = getOutputParams().flat_count();

        // Quantize input (merged: 2 passes instead of 3)
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

        const std::vector<i8>& qw = getQuantizedWeights();
        const std::vector<i32>& sum_qw = getWeightSums();
        fp32 Sw = getWeightScale();
        fp32 Si = iParams.scale;
        i32 zi = (i32)iParams.zero_point;
        fp32 inv_scale = 1.0f / (Si * Sw);

        // Allocate accumulator buffer for all output positions
        std::vector<i32> acc(out_count);

        // Initialize accumulators with bias
        for (size p = 0; p < P; p++) {
            for (size q = 0; q < Q; q++) {
                size base = p * Q * M + q * M;
                for (size m = 0; m < M; m++) {
                    acc[base + m] = qb[m];
                }
            }
        }

        // Cache-friendly loop order: p→q→r→s→c→m
        // Weight access qw[r*S*C*M + s*C*M + c*M + m] is now SEQUENTIAL
        // because c and m are the two innermost loops matching the memory layout
        for (size p = 0; p < P; p++) {
            for (size q = 0; q < Q; q++) {
                size out_base = p * Q * M + q * M;
                for (size r = 0; r < R; r++) {
                    for (size s = 0; s < S; s++) {
                        size in_base = (p * U + r) * W * C + (q * U + s) * C;
                        size w_base = r * S * C * M + s * C * M;
                        for (size c = 0; c < C; c++) {
                            i32 input_val = (i32)qi[in_base + c];
                            size w_off = w_base + c * M;
                            // m is innermost — sequential access to qw
                            for (size m = 0; m < M; m++) {
                                acc[out_base + m] += input_val * (i32)qw[w_off + m];
                            }
                        }
                    }
                }
            }
        }

        // Dequantize and apply activation
        for (size p = 0; p < P; p++) {
            for (size q = 0; q < Q; q++) {
                size base = p * Q * M + q * M;
                for (size m = 0; m < M; m++) {
                    fp32 result = (fp32)(acc[base + m] - zi * sum_qw[m]) * inv_scale;
                    getOutputData().get<fp32>(base + m) = Activation::apply(result, getActivationType());
                }
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
