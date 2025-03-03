// This file is part of the Acts project.
//
// Copyright (C) 2022 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "ActsExamples/EventData/Trajectories.hpp"
#include "ActsExamples/Framework/WriterT.hpp"

#include <string>

#include "edm4hep/TrackCollection.h"
#include "podio/EventStore.h"
#include "podio/ROOTWriter.h"

namespace ActsExamples {

/// Write out the tracks reconstructed using Combinatorial Kalman Filter to
/// EDM4hep.
///
/// Inpersistent information:
/// - trajectory state incomplete
/// - relation to the particles
///
/// Known issues:
/// - curvature parameter
/// - track state local coordinates are written to (D0,Z0)
/// - covariance incorrect
class EDM4hepMultiTrajectoryWriter : public WriterT<TrajectoriesContainer> {
 public:
  struct Config {
    /// Input trajectory collection
    std::string inputTrajectories;
    /// Input hit-particles map collection
    std::string inputMeasurementParticlesMap;
    /// Where to place output file
    std::string outputPath;
  };

  /// constructor
  /// @param config is the configuration object
  /// @param level is the output logging level
  EDM4hepMultiTrajectoryWriter(
      const Config& config, Acts::Logging::Level level = Acts::Logging::INFO);

  ProcessCode finalize() final;

  /// Readonly access to the config
  const Config& config() const { return m_cfg; }

 protected:
  /// @brief Write method called by the base class
  /// @param [in] context is the algorithm context for consistency
  /// @param [in] tracks is the track collection
  ProcessCode writeT(const AlgorithmContext& context,
                     const TrajectoriesContainer& trajectories) final;

 private:
  Config m_cfg;

  podio::ROOTWriter m_writer;
  podio::EventStore m_store;

  edm4hep::TrackCollection* m_trackCollection;
};

}  // namespace ActsExamples
