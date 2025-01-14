// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once
#include <vector>
#include <map>
#include <Eigen/Dense>
// By default <core/session/onnxruntime_cxx_api.h> points to the wrong path
#include "/cvmfs/sft.cern.ch/lcg/views/LCG_103/x86_64-centos7-gcc11-opt/include/core/session/onnxruntime_cxx_api.h"

namespace Acts {

using NetworkBatchInput =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// General class that sets up the ONNX runtime framework for loading a ML model
// and using it for inference.
class OnnxRuntimeBase {
 public:
  /// @brief Default constructor
  OnnxRuntimeBase() = default;

  /// @brief Parametrized constructor
  ///
  /// @param env the ONNX runtime environment
  /// @param modelPath the path to the ML model in *.onnx format
  OnnxRuntimeBase(Ort::Env& env, const char* modelPath);

  /// @brief Default destructor
  ~OnnxRuntimeBase() = default;

  /// @brief Run the ONNX inference function
  ///
  /// @param inputTensorValues The input feature values used for prediction
  ///
  /// @return The output (predicted) values
  std::vector<float> runONNXInference(
      std::vector<float>& inputTensorValues) const;

  /// @brief Run the ONNX inference function for a batch of input
  ///
  /// @param inputTensorValues Vector of the input feature values of all the inputs used for prediction
  ///
  /// @return The vector of output (predicted) values
  std::vector<std::vector<float>> runONNXInference(
      NetworkBatchInput& inputTensorValues) const;

  std::map<int, Eigen::MatrixXf> runONNXInferenceMultilayerOutput(
    NetworkBatchInput& inputTensorValues) const;


 private:
  /// ONNX runtime session / model properties
  std::unique_ptr<Ort::Session> m_session;
  std::vector<Ort::AllocatedStringPtr> m_inputNodeNamesAllocated;
  
  std::vector<const char*> m_inputNodeNames;
  std::vector<int64_t> m_inputNodeDims;
  std::vector<Ort::AllocatedStringPtr> m_outputNodeNamesAllocated;
  std::vector<const char*> m_outputNodeNames;
  std::vector<int64_t> m_outputNodeDims;
};

}  // namespace Acts
