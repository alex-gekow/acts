// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Plugins/Onnx/OnnxRuntimeBase.hpp"

#include <vector>
#include <map>
#include <iterator>
#include <iostream>

namespace Acts {

// Specialized class that labels tracks as good/duplicate/fake using a
// deep neural network.
class MLDetectorClassifier : public OnnxRuntimeBase {
  using OnnxRuntimeBase::OnnxRuntimeBase;

 public:
  /// @brief Predict the next-hit volume and layer id
  ///
  /// @param inputFeatures The vector of input features comprising of n hits x,y,z coordinates 
  /// normalized such that x,y,z <=|1|
  /// @return One-hot-encoded volume and layer ID 
  Eigen::MatrixXf PredictVolumeAndLayer(Acts::NetworkBatchInput&  inputFeatures) const;

  // Argmax function to convert predictions to one-hot-encoding
  template <typename T, typename A>
  float arg_max(std::vector<T,A> vec) const {
    return static_cast<float>(std::distance(vec.begin(), max_element(vec.begin(), vec.end())));
  }

 private:
  float xScale = 1005;
  float yScale = 1005;
  float zScale = 3500;

 public:
  constexpr float getXScale() const {return xScale;};
  constexpr float getYScale() const {return yScale;};
  constexpr float getZScale() const {return zScale;};
};

class MLHitPredictor : public OnnxRuntimeBase {
  using OnnxRuntimeBase::OnnxRuntimeBase;

 public:
  /// @brief Predict the next-hit volume and layer id
  ///
  /// @param inputFeatures The vector of input features comprising of n hits x,y,z coordinates 
  /// normalized such that x,y,z <=|1| as well as the one-hot-encoded volume and layer id prediction
  /// @return The predicted next hit coordinate x,y,z <=|1|
  Eigen::MatrixXf PredictHitCoordinate(Acts::NetworkBatchInput& inputTensorValues) const;

 private:
  float xScale = 1005;
  float yScale = 1005;
  float zScale = 3500;

 public:
  constexpr float getXScale() const {return xScale;};
  constexpr float getYScale() const {return yScale;};
  constexpr float getZScale() const {return zScale;};

};

}  // namespace Acts