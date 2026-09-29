#pragma once

#include <vector>

#include "../Types.h"
#include "../Utils.h"
#include "../Quantize.h"
#include "Activation.h"
#include "Layer.h"

namespace ML {
class ConvolutionalLayer : public Layer {
   public:
    ConvolutionalLayer(const LayerParams inParams, const LayerParams outParams, const LayerParams weightParams, const LayerParams biasParams, ActivationType activation = ActivationType::RELU)
        : Layer(inParams, outParams, LayerType::CONVOLUTIONAL),
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

    // Quantization accessors (used by all compute variants: naive, tiled, threaded, SIMD)
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
    // Pre-quantize weights at load time (once, not per-inference)
    void preQuantizeWeights() {
        size weight_count = weightParam.flat_count();
        size R = weightParam.dims[0], S = weightParam.dims[1];
        size C = weightParam.dims[2], M = weightParam.dims[3];

        wParams = Quantize::computeWeightParams(
            (const fp32*)weightData.raw(), weight_count
        );
        Quantize::quantizeToInt8(
            (const fp32*)weightData.raw(), weight_count, wParams, qWeights
        );
        // weight layout {R,S,C,M}: each output m has R*S*C weights, stride M
        Quantize::computeWeightSums(qWeights, R * S * C, M, M, wSums);
    }

    LayerParams weightParam;
    LayerData weightData;

    LayerParams biasParam;
    LayerData biasData;

    ActivationType activation;

    // Pre-quantized at allocLayer() — shared by all compute variants
    QuantParams wParams;
    std::vector<i8> qWeights;
    std::vector<i32> wSums;
};

}  // namespace ML
