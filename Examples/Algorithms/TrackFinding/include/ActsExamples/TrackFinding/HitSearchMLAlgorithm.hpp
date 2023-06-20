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
    Acts::NetworkBatchInput BatchTracksForGeoPrediction(Acts::VectorMultiTrajectory tracks,
        std::map <const Acts::SourceLink*, const ActsExamples::SimSpacePoint*>& sourceLinkSPMap,
        IndexSourceLinkContainer sourceLinks, SimSpacePointContainer& spacePoints);


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

} // namespace ActsExamples