#include "Flatten.h"

namespace ML {
    void FlattenLayer::computeNaive(const LayerData& dataIn) const {
        std::memcpy(getOutputData().raw(), dataIn.raw(), getOutputParams().byte_size());
    }

    void FlattenLayer::computeThreaded(const LayerData& dataIn) const {};
    void FlattenLayer::computeTiled(const LayerData& dataIn) const {};
    void FlattenLayer::computeSIMD(const LayerData& dataIn) const {};

    size FlattenLayer::getNumMACs() const {
        return 0;
    }
}
