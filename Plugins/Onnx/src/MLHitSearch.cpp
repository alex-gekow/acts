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
std::vector<std::vector<float>> Acts::MLDetectorClassifier::PredictVolumeAndLayer(Acts::NetworkBatchInput& inputTensorValues) const {

  std::vector<std::vector<float>> outputs;
  // run the model over the input
  std::map<int, std::vector<std::vector<float>>> outputTensorValuesMap = runONNXInferenceMultilayerOutput(inputTensorValues);
  // The first layer should be (batch,15) volume OHE
  // The second layer should be (batch, 30) layer OHE
  int batchSize = outputTensorValuesMap[0].size();
  for (int i=0; i<batchSize; i++){
    int predVolume = static_cast<int>(arg_max(outputTensorValuesMap[1][i]));
    int predLayer  = static_cast<int>(arg_max(outputTensorValuesMap[0][i]));

    std::vector<float> onehotencoding(45,0); 
    onehotencoding[predVolume] = 1;
    onehotencoding[15+predLayer] = 1;
   
    outputs.push_back(onehotencoding);
  }
  
  return outputs;
}

// prediction function
std::vector<std::vector<float>> Acts::MLHitPredictor::PredictHitCoordinate(Acts::NetworkBatchInput& inputTensorValues) const {
  // run the model over the input
  // std::vector<float> outputTensor = runONNXInference(inputFeatures);
  std::vector<std::vector<float>> outputs;
  std::map<int, std::vector<std::vector<float>>> outputTensorValuesMap = runONNXInferenceMultilayerOutput(inputTensorValues);

  int batchSize = outputTensorValuesMap[0].size();
  for (int i=0; i<batchSize; i++){
    std::cout<<std::endl;
    outputs.push_back(outputTensorValuesMap[0][i]); // Only 1 layer output

  }

  // float output_x = outputTensor[0];
  // float output_y = outputTensor[1];
  // float output_z = outputTensor[2];
  // std::vector<float> output = {output_x, output_y, output_z};

  return outputs;
}
