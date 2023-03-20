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

// prediction function
std::vector<float> Acts::MLDetectorClassifier::predictVolumeAndLayer(std::vector<float>& inputFeatures) const {

  // run the model over the input
  std::vector<float> outputTensor = runONNXInference(inputFeatures);
  // this is binary classification, so only need first value
  auto volumeVec = outputTensor.front().GetTensorMutableData<float>;
  auto layerVec  = outputTensor.back().GetTensorMutableData<float>;
  auto predVolume = Argmax(volumeVec);
  auto predLayer  = Argmax(layerVec);
  
  // Output as floats to ensure the type is correct for ML hit predictor
  std::vector<float> outputs(45,0);
  outputs[predVolume] = 1;
  outputs[predLayer] = 1;
  
  return outputs;
}

// prediction function
std::vector<float> Acts::MLHitPredictor::PredictHitCoordinate(std::vector<float>& inputFeatures) const {
  // run the model over the input
  std::vector<float> outputTensor = runONNXInference(inputFeatures);
  output_x = outputTensor[0];
  output_y = outputTensor[1];
  output_z = outputTensor[2];
  std::vector<float> output = {output_x, output_y, output_z};
  return output;
}
