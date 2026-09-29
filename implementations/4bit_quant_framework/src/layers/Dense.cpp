#include "Dense.h"

#include <vector>

namespace ML {
    void DenseLayer::computeNaive(const LayerData& dataIn) const {
        size in_size = getInputParams().flat_count();
        size out_size = getOutputParams().flat_count();

        // Quantize input (only per-inference cost)
        QuantParams iParams = Quantize::computeInputParams(
            (const fp32*)dataIn.raw(), in_size
        );
        std::vector<i8> qi;
        Quantize::quantizeToInt8((const fp32*)dataIn.raw(), in_size, iParams, qi);

        // Compute biases with full scale: Sb = Si * Sw
        fp32 Sb = iParams.scale * getWeightScale();
        std::vector<i32> qb;
        Quantize::quantizeBiasToInt32(
            (const fp32*)getBiasData().raw(), out_size, Sb, qb
        );

        // Use pre-quantized weights from allocLayer()
        const std::vector<i8>& qw = getQuantizedWeights();
        const std::vector<i32>& sum_qw = getWeightSums();
        fp32 Sw = getWeightScale();
        fp32 Si = iParams.scale;
        i32 zi = (i32)iParams.zero_point;

        // Quantized dense with int32 accumulation
        for (size o = 0; o < out_size; o++) {
            i32 acc = qb[o];

            for (size i = 0; i < in_size; i++) {
                acc += (i32)qi[i] * (i32)qw[i * out_size + o];
            }

            fp32 result = Quantize::dequantize(acc, zi, sum_qw[o], Si, Sw);
            getOutputData().get<fp32>(o) = Activation::apply(result, getActivationType());
        }
    }

    void DenseLayer::computeThreaded(const LayerData& dataIn) const {};
    void DenseLayer::computeTiled(const LayerData& dataIn) const {};
    void DenseLayer::computeSIMD(const LayerData& dataIn) const {};

    size DenseLayer::getNumMACs() const {
        const dimVec& in_dims = getInputParams().dims;
        const dimVec& out_dims = getOutputParams().dims;

        const size N = in_dims[0];
        const size M = out_dims[0];

        return N * M;
    }
}
