

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
  1. Read in inputSeeds, sourceLinks, and spacepoints
  2. create multitrajectories out of the input objects to build upon
  3. Traverse the multitrajectories to get input to the NN
  4. Predict the next hit coordinate for each tip
  5. Hit Search for each tip. (Grow the multitrajectories)
  6. Repeat from step (3)
  */
  
  // Why not make sourceLinks and spacepoints shared_ptr??
  // const auto& sourceLinks = ctx.eventStore.get<IndexSourceLinkContainer>(m_cfg.inputSourceLinks);
  const auto& seeds       = ctx.eventStore.get<SimSeedContainer>(m_cfg.inputSeeds);
  const auto& spacepoints = ctx.eventStore.get<SimSpacePointContainer>(m_cfg.inputSpacePoints);

  // Create collection of sourceLinks built from spacepoints rather than measurement indices
  // Possible to use SpacepointSourceLinks instead of Acts::SourceLink?
  std::vector<Acts::SourceLink> sourceLinks;
  for (const auto& sp:spacepoints){
    // ActsExamples::SpacepointSourceLink sl(sp.sourceLinks()[0].geometryId(), sp);
    Acts::SourceLink sl(sp.sourceLinks()[0].geometryId(), sp);
    sourceLinks.push_back(sl);
  } 

  // Create multitrajectories out of the initial seeds. May need to use vectorMultiTrajectory insteaed of CKF Results
  // These should contain active tips that we can use to select hits for prediction
  std::vector<Acts::CombinatorialKalmanFilterResult<Acts::VectorMultiTrajectory>> seedTraj;
  for(int seedIdx=0; seedIdx<seeds.size(); seedIdx++){
    Acts::CombinatorialKalmanFilterResult<Acts::VectorMultiTrajectory> result;
    //auto hitsList = seeds.at(seedIdx).sp();
    
    for(auto it = sourceLinks.begin(); it != sourceLinks.end(); it++){
      result.trackStateCandidates.clear();
      // result.stateBuffer->clear();
      result.activeTips.clear();
      Acts::TrackStatePropMask mask = Acts::TrackStatePropMask::Predicted;
      // size_t trackTip = (*it)->sourceLinks()[0].template get<ActsExamples::IndexSourceLink>().index();
      Acts::CombinatorialKalmanFilterTipState tipState; // Not needed for NN yet. Leave it empty. nMeasurements or nHoles might be useful
      auto stateBuffer = std::make_shared<Acts::VectorMultiTrajectory>();
      result.stateBuffer = stateBuffer;
      size_t tsi = result.stateBuffer->addTrackState();
      auto ts = result.stateBuffer->getTrackState(tsi);
      ts.setUncalibratedSourceLink((*it));
      result.trackStateCandidates.push_back(ts);
      //result.lastTrackIndices.emplace_back(trackTip);
      result.activeTips.emplace_back(tsi, tipState); 
    }
    seedTraj.emplace_back(result);
  }

  auto tip  = seedTraj[0].activeTips[0];
  auto tmp = seedTraj[0].stateBuffer->getTrackState(tip.first).getUncalibratedSourceLink();
  auto sp = tmp.template get<ActsExamples::SimSpacePoint>();


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

// Use active tips from a CKF Result as input. How to interface with MultiTrajectory to use visitBackwards?
Acts::NetworkBatchInput ActsExamples::HitSearchMLAlgorithm::BatchTracksForGeoPrediction(
  Acts::CombinatorialKalmanFilterResult<Acts::VectorMultiTrajectory> tracks,
  std::vector<Acts::SourceLink> sourceLinks){

  // tracks.trackStateCandidattes returns the multitrajectory?
  Acts::NetworkBatchInput networkInput(tracks.activeTips.size(), 9);
  // retrieve the last three hits from each track
  int trkIndex = 0;
  for (const auto& tip: tracks.activeTips){
    int hitCounter = 0;
    auto tipIdx = tip.first;
    for(int hitCounter=2; hitCounter > -1; --hitCounter){
      auto ts  = tracks.stateBuffer->getTrackState(tipIdx);
      auto sourcelink = ts.getUncalibratedSourceLink(); 
      networkInput(trkIndex, hitCounter*3    ) =  sourcelink.template get<ActsExamples::SimSpacePoint>().x() / m_NNDetectorClassifier.getXScale();
      networkInput(trkIndex, hitCounter*3 + 1) =  sourcelink.template get<ActsExamples::SimSpacePoint>().y() / m_NNDetectorClassifier.getYScale();
      networkInput(trkIndex, hitCounter*3 + 2) =  sourcelink.template get<ActsExamples::SimSpacePoint>().z() / m_NNDetectorClassifier.getZScale();
      tipIdx = ts.previous();
    }
    trkIndex++;
  }

  return networkInput;
}

