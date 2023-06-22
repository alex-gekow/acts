// This file is part of the Acts project.
//
// Copyright (C) 2022 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Plugins/Onnx/OnnxRuntimeBase.hpp"
#include "Acts/Plugins/Onnx/MLHitSearch.hpp"
#include "ActsExamples/Framework/IAlgorithm.hpp"
#include "Acts/TrackFinding/CombinatorialKalmanFilter.hpp"
#include "ActsExamples/EventData/SimSpacePoint.hpp"
#include "Acts/EventData/TrackStatePropMask.hpp"



#include <string>
#include <vector>

namespace ActsExamples {

// Employs ML models to predict the next-hit-coordinate along a track
//
// Implemtation works as follows:
//  0) Provide track seeds
//  1) Loop over track seeds
//    a) Predict the next hit coordinate along the track
//    b) search for hits near the prediction within a user-defined uncertainty window
//    c) create a track for each found hit
//  2) repeat for each track until
//    a) no nearby hits are found
//    b) the track reaches the end of the detector

class HitSearchMLAlgorithm final: public IAlgorithm {
    public:
     struct Config  {
        // Input seed collection
        std::string inputSeeds;
        // Path to ONNX Files
        std::string NNDetectorClassifier, NNHitPredictor;
        // Output track collection
        std::string outputTracks;
        // Input source links
        std::string inputSourceLinks;
        // Input spacepoints
        std::string inputSpacePoints;
        // Minimum number of hits needed to consider a track
        int nHitsMin=6;
        // Uncertainty window to search for hits (in mm)
        float uncertainty = 10;
     };

    /// Construct the ambiguity resolution algorithm.
    ///
    /// @param cfg is the algorithm configuration
    /// @param lvl is the logging level
    HitSearchMLAlgorithm(Config cfg, Acts::Logging::Level lvl);
    /// Run the ambiguity resolution algorithm.
    ///
    /// @param cxt is the algorithm context with event information
    /// @return a process code indication success or failure
    ProcessCode execute(const AlgorithmContext& ctx) const final;

    /// Const access to the config
    const Config& config() const { return m_cfg; }

    Acts::NetworkBatchInput BatchTracksForGeoPrediction(std::vector<SimSpacePointContainer> hitTracks);
    Acts::NetworkBatchInput BatchTracksForGeoPrediction(Acts::CombinatorialKalmanFilterResult<Acts::VectorMultiTrajectory> tracks,
        std::vector<Acts::SourceLink> sourceLinks);


    private:
    Config m_cfg;
    // ONNX environement
    Ort::Env m_env; // do we need one env per model?
    // ONNX models for the classifier and hit coordinate predictor
    Acts::MLDetectorClassifier m_NNDetectorClassifier;
    Acts::MLHitPredictor m_NNHitPredictor;
};

struct MLPathFinderTipState {
    Acts::SourceLink& prevTip;
};

template <typename source_link_iterator_t, typename traj_t>
void createSourceLinkTrackStatesML(
                                     Acts::CombinatorialKalmanFilterResult<traj_t>& result,
                                     size_t prevTip,
                                     source_link_iterator_t slBegin,
                                     source_link_iterator_t slEnd) {
    result.trackStateCandidates.clear();
    result.trackStateCandidates.reserve(std::distance(slBegin, slEnd));
    result.stateBuffer->clear();
    for (auto it = slBegin; it != slEnd; ++it) {
        // get the source link
        const auto sourceLink = *it;
        Acts::TrackStatePropMask mask = Acts::TrackStatePropMask::Predicted;
        size_t tsi = result.stateBuffer->addTrackState(mask, prevTip);
        // CAREFUL! This trackstate has a previous index that is not in this
        // MultiTrajectory Visiting brackwards from this track state will
        // fail!
        auto ts = result.stateBuffer->getTrackState(tsi);
        ts.setUncalibratedSourceLink(sourceLink);
        result.trackStateCandidates.push_back(ts);
    }              
}

class SpacepointSourceLink {
 public:
  /// Construct from geometry identifier and spacepoint.
  SpacepointSourceLink(Acts::GeometryIdentifier gid, SimSpacePoint sp)
      : m_geometryId(gid), m_spacepoint(sp) {}

  // Construct an invalid source link. Must be default constructible to
  /// satisfy SourceLinkConcept.
  SpacepointSourceLink() = default;
  SpacepointSourceLink(const SpacepointSourceLink&) = default;
  SpacepointSourceLink(SpacepointSourceLink&&) = default;
  SpacepointSourceLink& operator=(const SpacepointSourceLink&) = default;
  SpacepointSourceLink& operator=(SpacepointSourceLink&&) = default;

  /// Access the index.
  const SimSpacePoint sp() const { return m_spacepoint; }

  Acts::GeometryIdentifier geometryId() const { return m_geometryId; }

 private:
  Acts::GeometryIdentifier m_geometryId;
  SimSpacePoint m_spacepoint;

  friend bool operator==(const SpacepointSourceLink& lhs,
                         const SpacepointSourceLink& rhs) {
    return (lhs.geometryId() == rhs.geometryId()) and
           (lhs.m_spacepoint == rhs.m_spacepoint);
  }
  friend bool operator!=(const SpacepointSourceLink& lhs,
                         const SpacepointSourceLink& rhs) {
    return not(lhs == rhs);
  }
};

using SpacepointSourceLinkContainer = std::vector<SpacepointSourceLink>;

} // namespace ActsExamples
