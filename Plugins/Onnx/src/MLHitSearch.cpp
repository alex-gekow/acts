// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Plugins/Onnx/MLHitSearch.hpp"
#include <cassert>
#include <stdexcept>
#include <array>

// prediction function
std::vector<std::vector<float>> Acts::MLDetectorClassifier::predictVolumeAndLayer(Acts::NetworkBatchInput& inputTensorValues) {

  std::vector<std::vector<float>> outputs(inputTensorValues.rows(), std::vector<float>(45,0));
  // run the model over the input
  std::map<int, std::vector<std::vector<float>>> outputTensorValuesMap = runONNXInferenceMultilayerOutput(inputTensorValues);
  // The first layer should be (batch,15) volume OHE
  // The second layer should be (batch, 30) layer OHE
  int batchSize = sizeof(outputTensorValuesMap[0]) / sizeof(float);
  outputs.reserve(batchSize);
  for (int i=0; i<batchSize; i++){
    int predVolume = static_cast<int>(arg_max(outputTensorValuesMap[0][i]));
    int predLayer  = static_cast<int>(arg_max(outputTensorValuesMap[1][i]));

    std::vector<float> onehotencoding(45,0); 
    onehotencoding[predVolume] = 1;
    onehotencoding[predLayer] = 1;
    outputs.push_back(onehotencoding);
  }
  
  return outputs;
}

// prediction function
std::vector<float> Acts::MLHitPredictor::PredictHitCoordinate(std::vector<float>& inputFeatures) const {
  // run the model over the input
  std::vector<float> outputTensor = runONNXInference(inputFeatures);
  float output_x = outputTensor[0];
  float output_y = outputTensor[1];
  float output_z = outputTensor[2];
  std::vector<float> output = {output_x, output_y, output_z};
  return output;
}
