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
Eigen::MatrixXf Acts::MLDetectorClassifier::PredictVolumeAndLayer(Acts::NetworkBatchInput& inputTensorValues) const {

  const int batchSize = inputTensorValues.rows();
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> outputs(batchSize, 45);
  // run the model over the input
  std::map<int, Eigen::MatrixXf> outputTensorValuesMap = runONNXInferenceMultilayerOutput(inputTensorValues);
  // The first layer should be (batch,15) volume OHE
  // The second layer should be (batch, 30) layer OHE
  for (int i=0; i<batchSize; i++){

    // cast eigen vectors to std::vectors to use argmax
    // more efficient to write an argmax function to operate on Eigen::Vectors?
    auto volEigenVec = outputTensorValuesMap[1].row(i);
    auto layEigenVec = outputTensorValuesMap[0].row(i);

    Eigen::MatrixXf::Index predVolume;
    Eigen::MatrixXf::Index predLayer;
    volEigenVec.maxCoeff(&predVolume);
    layEigenVec.maxCoeff(&predLayer);

    Eigen::VectorXf onehotencoding = Eigen::VectorXf::Zero(45); 
    onehotencoding[ static_cast<int>(predVolume)] = 1;
    onehotencoding[ static_cast<int>(15+predLayer)] = 1;
    outputs.row(i) = onehotencoding;
  }
  return outputs;
}

// prediction function
// covnert to eigen functions
Eigen::MatrixXf Acts::MLHitPredictor::PredictHitCoordinate(Acts::NetworkBatchInput& inputTensorValues) const {
  // run the model over the input
  std::map<int, Eigen::MatrixXf> outputTensorValuesMap = runONNXInferenceMultilayerOutput(inputTensorValues);


  // float output_x = outputTensor[0];
  // float output_y = outputTensor[1];
  // float output_z = outputTensor[2];
  // std::vector<float> output = {output_x, output_y, output_z};

  auto output = outputTensorValuesMap[0];//Outputs for this model should have only one output layer
  output.col(0) *= getXScale();
  output.col(1) *= getYScale();
  output.col(2) *= getZScale();
  
  return output;
}
