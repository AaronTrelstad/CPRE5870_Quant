#include "Dense.h"

namespace ML {
    void DenseLayer::computeNaive(const LayerData& dataIn) const {
        size in_size = getInputParams().flat_count();
        size out_size = getOutputParams().flat_count();

        for (size o = 0; o < out_size; o++) {
            fp32 sum = biasData.get<fp32>(o);

            for (size i = 0; i < in_size; i++) {
                sum += dataIn.get<fp32>(i) * weightData.get<fp32>(i * out_size + o);
            }

            getOutputData().get<fp32>(o) = Activation::apply(sum, getActivationType());
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
