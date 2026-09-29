#pragma once

#include <vector>

#include "../Types.h"
#include "../Utils.h"
#include "../Quantize.h"
#include "Activation.h"
#include "Layer.h"

namespace ML {
class DenseLayer : public Layer {
    public:
    DenseLayer(const LayerParams inParams, const LayerParams outParams, const LayerParams weightParams, const LayerParams biasParams, ActivationType activation = ActivationType::RELU)
        : Layer(inParams, outParams, LayerType::DENSE),
          weightParam(weightParams),
          weightData(weightParams),
          biasParam(biasParams),
          biasData(biasParams),
          activation(activation) {}

    // Getters
    const LayerParams& getWeightParams() const { return weightParam; }
    const LayerParams& getBiasParams() const { return biasParam; }
    const LayerData& getWeightData() const { return weightData; }
    const LayerData& getBiasData() const { return biasData; }
    const ActivationType getActivationType() const { return activation; }

    // Quantization accessors (used by all compute variants)
    const std::vector<i8>& getQuantizedWeights() const { return qWeights; }
    const std::vector<i32>& getWeightSums() const { return wSums; }
    fp32 getWeightScale() const { return wParams.scale; }

    // Allocate all resources needed for the layer & Load all of the required data for the layer
    virtual void allocLayer() override {
        Layer::allocLayer();
        weightData.loadData();
        biasData.loadData();
        preQuantizeWeights();
    }

    // Free all resources allocated for the layer
    virtual void freeLayer() override {
        Layer::freeLayer();
        weightData.freeData();
        biasData.freeData();
    }

    // Virtual functions
    virtual void computeNaive(const LayerData& dataIn) const override;
    virtual void computeThreaded(const LayerData& dataIn) const override;
    virtual void computeTiled(const LayerData& dataIn) const override;
    virtual void computeSIMD(const LayerData& dataIn) const override;
    virtual size getNumMACs() const override;

    private:
    void preQuantizeWeights() {
        size in_size = getInputParams().flat_count();
        size out_size = getOutputParams().flat_count();
        size weight_count = in_size * out_size;

        wParams = Quantize::computeWeightParams(
            (const fp32*)weightData.raw(), weight_count
        );
        Quantize::quantizeToInt8(
            (const fp32*)weightData.raw(), weight_count, wParams, qWeights
        );
        // weight layout {in, out}: each output o has in_size weights, stride out_size
        Quantize::computeWeightSums(qWeights, in_size, out_size, out_size, wSums);
    }

    LayerParams weightParam;
    LayerData weightData;
    LayerParams biasParam;
    LayerData biasData;
    ActivationType activation;

    // Pre-quantized at allocLayer()
    QuantParams wParams;
    std::vector<i8> qWeights;
    std::vector<i32> wSums;
};
}
