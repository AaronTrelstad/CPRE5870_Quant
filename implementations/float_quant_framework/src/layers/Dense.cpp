#include "Dense.h"

namespace ML {
    void DenseLayer::computeNaive(const LayerData& dataIn) const {
        size in_size = getInputParams().flat_count();
        size out_size = getOutputParams().flat_count();

        const fp32* input = (const fp32*)dataIn.raw();
        const fp32* weights = (const fp32*)weightData.raw();
        const fp32* biases = (const fp32*)biasData.raw();
        fp32* output = (fp32*)getOutputData().raw();
        ActivationType act = getActivationType();

        // Local accumulator, loop order i→o for sequential weight access
        fp32 acc[256];
        for (size o = 0; o < out_size; o++) acc[o] = biases[o];

        for (size i = 0; i < in_size; i++) {
            fp32 in_val = input[i];
            const fp32* w_row = weights + i * out_size;
            for (size o = 0; o < out_size; o++) {
                acc[o] += in_val * w_row[o];
            }
        }

        for (size o = 0; o < out_size; o++) {
            output[o] = Activation::apply(acc[o], act);
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
