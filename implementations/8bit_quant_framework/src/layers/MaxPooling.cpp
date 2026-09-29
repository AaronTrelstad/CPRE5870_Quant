#include "MaxPooling.h"

namespace ML {
    void MaxPoolingLayer::computeNaive(const LayerData& dataIn) const {
        const dimVec& in_dims = getInputParams().dims;
        const dimVec& out_dims = getOutputParams().dims;

        size w_in = in_dims[1];

        size h_out = out_dims[0];
        size w_out = out_dims[1];
        size c_out = out_dims[2];

        for (size c = 0; c < c_out; c++) {
            for (size h = 0; h < h_out; h++) {
                for (size w = 0; w < w_out; w++) {
                    fp32 v00 = dataIn.get<fp32>((2 * h) * w_in * c_out + (2 * w) * c_out + c);
                    fp32 v01 = dataIn.get<fp32>((2 * h) * w_in * c_out + (2 * w + 1) * c_out + c);
                    fp32 v10 = dataIn.get<fp32>((2 * h + 1) * w_in * c_out + (2 * w) * c_out + c);
                    fp32 v11 = dataIn.get<fp32>((2 * h + 1) * w_in * c_out + (2 * w + 1) * c_out + c);

                    fp32 maxVal = v00;
                    if (v01 > maxVal) maxVal = v01;
                    if (v10 > maxVal) maxVal = v10;
                    if (v11 > maxVal) maxVal = v11;

                    getOutputData().get<fp32>(h * w_out * c_out + w * c_out + c) = maxVal;
                }
            }
        }
    }

    void MaxPoolingLayer::computeThreaded(const LayerData& dataIn) const {};
    void MaxPoolingLayer::computeTiled(const LayerData& dataIn) const {};
    void MaxPoolingLayer::computeSIMD(const LayerData& dataIn) const {};

    size MaxPoolingLayer::getNumMACs() const {
        return 0;
    }
}