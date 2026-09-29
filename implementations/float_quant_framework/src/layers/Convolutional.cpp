#include "Convolutional.h"

namespace ML {

    void ConvolutionalLayer::computeNaive(const LayerData& dataIn) const {
        const dimVec& in_dims = getInputParams().dims;
        const dimVec& out_dims = getOutputParams().dims;
        const dimVec& weight_dims = getWeightParams().dims;

        size H = in_dims[0], W = in_dims[1], C = in_dims[2];
        size P = out_dims[0], Q = out_dims[1], M = out_dims[2];
        size R = weight_dims[0], S = weight_dims[1];
        size U = (H - R) / (P - 1);

        const fp32* input = (const fp32*)dataIn.raw();
        const fp32* weights = (const fp32*)getWeightData().raw();
        const fp32* biases = (const fp32*)getBiasData().raw();
        fp32* output = (fp32*)getOutputData().raw();

        // Initialize output with bias
        for (size p = 0; p < P; p++) {
            for (size q = 0; q < Q; q++) {
                fp32* out_ptr = output + p * Q * M + q * M;
                for (size m = 0; m < M; m++) {
                    out_ptr[m] = biases[m];
                }
            }
        }

        // Cache-friendly loop order: p→q→r→s→c→m
        // m innermost = sequential weight access, enables auto-vectorization
        for (size p = 0; p < P; p++) {
            for (size q = 0; q < Q; q++) {
                fp32* out_ptr = output + p * Q * M + q * M;
                for (size r = 0; r < R; r++) {
                    for (size s = 0; s < S; s++) {
                        const fp32* in_ptr = input + (p * U + r) * W * C + (q * U + s) * C;
                        const fp32* w_ptr = weights + r * S * C * M + s * C * M;
                        for (size c = 0; c < C; c++) {
                            fp32 in_val = in_ptr[c];
                            const fp32* w_row = w_ptr + c * M;
                            for (size m = 0; m < M; m++) {
                                out_ptr[m] += in_val * w_row[m];
                            }
                        }
                    }
                }
            }
        }

        // Apply activation
        ActivationType act = getActivationType();
        if (act != ActivationType::NONE) {
            size out_count = P * Q * M;
            for (size i = 0; i < out_count; i++) {
                output[i] = Activation::apply(output[i], act);
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
