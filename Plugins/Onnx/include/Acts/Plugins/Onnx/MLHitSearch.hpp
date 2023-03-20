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
  std::vector<float> predictVolumeAndLayer(std::vector<float>& inputFeatures) const;

  // Argmax function to convert predictions to one-hot-encoding
  template <typename T, typename A>
  float arg_max(std::vector<T, A> const& vec) {
    return static_cast<float>(std::distance(vec.begin(), max_element(vec.begin(), vec.end())));
  }
};

class MLHitPredictor : public OnnxRuntimeBase {
  using OnnxRuntimeBase::OnnxRuntimeBase;

 public:
  /// @brief Predict the next-hit volume and layer id
  ///
  /// @param inputFeatures The vector of input features comprising of n hits x,y,z coordinates 
  /// normalized such that x,y,z <=|1| as well as the one-hot-encoded volume and layer id prediction
  /// @return The predicted next hit coordinate x,y,z <=|1|
  std::vector<float> PredictHitCoordinate(std::vector<float>& inputFeatures) const;
};

}  // namespace Acts