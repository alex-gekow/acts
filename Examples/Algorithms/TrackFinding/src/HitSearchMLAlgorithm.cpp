

#include "ActsExamples/TrackFinding/HitSearchMLAlgorithm.hpp"
#include "ActsExamples/EventData/SimSeed.hpp"
#include "ActsExamples/EventData/SimSpacePoint.hpp"
#include "ActsExamples/Framework/WhiteBoard.hpp"

ActsExamples::HitSearchMLAlgorithm::HitSearchMLAlgorithm(
    ActsExamples::HitSearchMLAlgorithm::Config cfg,
    Acts::Logging::Level lvl)
    : ActsExamples::IAlgorithm("HitSearchMLAlgorithm", lvl),
      m_cfg(std::move(cfg)),
      m_env(ORT_LOGGING_LEVEL_WARNING, "MLDetectorClassifier"),
      m_NNDetectorClassifier(m_env, m_cfg.NNDetectorClassifier.c_str()),
      m_NNHitPredictor(m_env, m_cfg.NNHitPredictor.c_str()) {
  if (m_cfg.inputSeeds.empty()) {
    throw std::invalid_argument("Missing seeds input collection");
  }
  if (m_cfg.outputTracks.empty()) {
    throw std::invalid_argument("Missing tracks output collection");
  }
  if (m_cfg.inputSourceLinks.empty()) {
    throw std::invalid_argument("Missing input source links collection");
  }
  if (m_cfg.inputSpacePoints.empty()) {
    throw std::invalid_argument("Missing input spacepoints collection");
  }

}

ActsExamples::ProcessCode ActsExamples::HitSearchMLAlgorithm::execute(const AlgorithmContext& ctx) const {

  /*
  1. Read in inputSeeds, sourceLinks, and measurements
  2. create multitrajectories out of the input objects to build upon
  3. Traverse the multitrajectories to get input to the NN
  4. Predict the next hit coordinate for each tip
  5. Hit Search for each tip. (Grow the multitrajectories)
  6. Repeat from step (3)
  */
  
  // Why not make sourceLinks and spacepoints shared_ptr??
  const auto& sourceLinks = ctx.eventStore.get<IndexSourceLinkContainer>(m_cfg.inputSourceLinks);
  const auto& seeds       = ctx.eventStore.get<SimSeedContainer>(m_cfg.inputSeeds);
  const auto& spacepoints = ctx.eventStore.get<SimSpacePointContainer>(m_cfg.inputSpacePoints);

  // Create mapping of sourcelinks to spacepoints. Spacepoints contain reference to sourcelinks but not the other way around
  // Multitrajectory is created from sourcelinks, not spacepoints
  // Need to use pointers as raw objects cannot be used as a map key
  std::map <const Acts::SourceLink*, const ActsExamples::SimSpacePoint*> slsp;
  for (const auto& sp:spacepoints){
    const Acts::SourceLink* sl_pointer;
    auto sl = sp.sourceLinks()[0];
    sl_pointer = &sl;
    slsp[sl_pointer] = &sp;
  }

  // Create multitrajectories out of the initial seeds. May need to use vectorMultiTrajectory insteaed of CKF Results
  // These should contain active tips that we can use to select hits for prediction
  std::vector<Acts::CombinatorialKalmanFilterResult<Acts::VectorMultiTrajectory>> seedTraj;
  for(int seedIdx=0; seedIdx<seeds.size(); seedIdx++){
    Acts::CombinatorialKalmanFilterResult<Acts::VectorMultiTrajectory> result;
    auto hitsList = seeds.at(seedIdx).sp();
    for(auto it = hitsList.begin(); it != hitsList.end(); it++){
      result.trackStateCandidates.clear();
      result.stateBuffer->clear();

      Acts::TrackStatePropMask mask = Acts::TrackStatePropMask::Predicted;
      size_t trackTip = (*it)->sourceLinks()[0].template get<IndexSourceLink>().index();
      Acts::CombinatorialKalmanFilterTipState tipState; // Not needed for NN yet. Leave it empty. nMeasurements or nHoles might be useful
      size_t tsi = result.stateBuffer->addTrackState(mask, trackTip);
      auto ts = result.stateBuffer->getTrackState(tsi);
      ts.setUncalibratedSourceLink((*it)->sourceLinks()[0]);
      result.trackStateCandidates.push_back(ts);
      result.lastTrackIndices.emplace_back(trackTip);
      result.activeTips.emplace_back(trackTip, tipState); 
    }
    seedTraj.emplace_back(result);
  }



  // size_t prevTip = SIZE_MAX;
  // TipState prevTipState;
  // if (not result.activeTips.empty()) {
  //   prevTip = result.activeTips.back().first;
  //   prevTipState = result.activeTips.back().second;
  //   // New state is to be added. Remove the last tip from active tips
  //   result.activeTips.erase(result.activeTips.end() - 1);
  // }


  // Create input for the classifier
  // Acts::NetworkBatchInput networkInput(1, 9); // 9 input features. 3 hits * 3 features each
  // for(int seedIdx=0; seedIdx<seeds.size(); seedIdx++){
  //   auto hitsList = seeds.at(seedIdx).sp();
  //   SimSpacePointContainer extrapolatedTrack;
  //   for(int i=0; i<hitsList.size(); i++){
  //     // networkInput(seedIdx, i*3)     = hitsList.at(i)->x() / m_NNDetectorClassifier.getXScale();
  //     // networkInput(seedIdx, i*3 + 1) = hitsList.at(i)->y() / m_NNDetectorClassifier.getYScale();
  //     // networkInput(seedIdx, i*3 + 2) = hitsList.at(i)->z() / m_NNDetectorClassifier.getZScale();

  //     extrapolatedTrack.emplace_back(hitsList.at(i));
  //   }
  //   extrapolatedTracks.emplace_back(extrapolatedTrack);
  // }



  return ActsExamples::ProcessCode::SUCCESS;
}

Acts::NetworkBatchInput ActsExamples::HitSearchMLAlgorithm::BatchTracksForGeoPrediction(std::vector<SimSpacePointContainer> hitTracks){
  
  Acts::NetworkBatchInput networkInput(hitTracks.size(), 9);
  // retrieve the last three hits from each track
  int trkIdx = 0;
  for (const auto& trk: hitTracks){
    for (int i=trk.size() - 3; i<trk.size(); i++){
      networkInput(trkIdx, i*3)     = trk.at(i).x() / m_NNDetectorClassifier.getXScale();
      networkInput(trkIdx, i*3 + 1) = trk.at(i).y() / m_NNDetectorClassifier.getYScale();
      networkInput(trkIdx, i*3 + 2) = trk.at(i).z() / m_NNDetectorClassifier.getZScale();
    }
    trkIdx++;
  }

  return networkInput;
}


/*
// Use active tips from a CKF Result as input. How to interface with MultiTrajectory to use visitBackwards?
Acts::NetworkBatchInput ActsExamples::HitSearchMLAlgorithm::BatchTracksForGeoPrediction(Acts::VectorMultiTrajectory tracks,
std::map <const Acts::SourceLink*, const ActsExamples::SimSpacePoint*>& sourceLinkSPMap,
IndexSourceLinkContainer sourceLinks, SimSpacePointContainer& spacePoints){

  Acts::NetworkBatchInput networkInput(tracks.activeTips.size(), 9);
  // retrieve the last three hits from each track
  int trkIndex = 0;
  for (const auto& tip: tracks.activeTips){

    tracks.visitBackwards(tip, [&](const auto& state){
      if (not state.typeFlags().test(Acts::TrackStateFlag::MeasurementFlag)) {
          return true;
        }

        auto source_link = state.getUncalibratedSourceLink().template get<IndexSourceLink>();
        // networkInput().push_back(source_link.index());

        return true;
    });
    trkIdx++;
  }

  return networkInput;
}
*/