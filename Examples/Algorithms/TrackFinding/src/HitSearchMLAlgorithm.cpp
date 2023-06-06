

#include "ActsExamples/TrackFinding/HitSearchMLAlgorithm.hpp"
#include "ActsExamples/EventData/SimSeed.hpp"
#include "ActsExamples/EventData/SimSpacepoint.hpp"
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
  
  const auto& sourceLinks = ctx.eventStore.get<IndexSourceLinkContainer>(m_cfg.inputSourceLinks);
  const auto& seeds       = ctx.eventStore.get<SimSeedContainer>(m_cfg.inputSeeds);
  const auto& spacepoints = ctx.eventStore.get<SimSpacePointContainer>(m_cfg.inputSpacePoints);
  // Store seeds in tracks to be extrapolated
  std::vector<SimSpacePointContainer> extrapolatedTracks;
  // Create input for the classifier
  // Acts::NetworkBatchInput networkInput(1, 9); // 9 input features. 3 hits * 3 features each
  for(int seedIdx=0; seedIdx<seeds.size(); seedIdx++){
    auto hitsList = seeds.at(seedIdx).sp();
    SimSpacePointContainer extrapolatedTrack
    for(int i=0; i<hitsList.size(); i++){
      // networkInput(seedIdx, i*3)     = hitsList.at(i)->x() / m_NNDetectorClassifier.getXScale();
      // networkInput(seedIdx, i*3 + 1) = hitsList.at(i)->y() / m_NNDetectorClassifier.getYScale();
      // networkInput(seedIdx, i*3 + 2) = hitsList.at(i)->z() / m_NNDetectorClassifier.getZScale();

      extrapolatedTrack.emplace_back(hitsList.at(i));
    }
    extrapolatedTracks.emplace_back(extrapolatedTrack);
  }

  std::vector<SimSpacePointContainer> tmp_extrapolatedTracks;
  std::vector<SimSpacePointContainer> completeTracks;

  while(extrapolatedTracks.size() > 0){

  
    auto networkInput = ActsExamples::HitSearchMLAlgorithm::BatchTracksForGeoPrediction(extrapolatedTracks);
    auto encDetectorID = m_NNDetectorClassifier.PredictVolumeAndLayer(networkInput);
    // Create input to the hit predictor
    Acts::NetworkBatchInput networkAppendedInput(networkInput.rows(), networkInput.cols()+encDetectorID.cols());
    networkAppendedInput << networkInput, encDetectorID;
    auto predCoor = m_NNHitPredictor.PredictHitCoordinate(networkAppendedInput); //matrix of predicted hit coordinates x,y,z


    // Perform the hit search. This step requires further optimization
    float uncertainty = 15; // Uncertainty window to search for hits
    for (int r=0; r<predCoor.rows();r++){ //This should be the same length as the number of input tracks
      for (auto&: sp:spacepoints){
        float distance  = std::hypot((predCoor[r][0]*m_NNDetectorClassifier.getXScale() - sp.x()) + 
                                    ( predCoor[r][1]*m_NNDetectorClassifier.getYScale() - sp.y()) +
                                    ( predCoor[r][2]*m_NNDetectorClassifier.getZScale() - sp.z())
                                    );
        std::cout<<distance<<std::endl;
        if (distance < uncertainty){
          SimSpacePointContainer newTrack = track;
          newTrack.emplace_back(sp);
          if(sp.z() > 3000 || sp.rho() > 1050 || newTrack.size() >= 18){
            completeTracks.emplace_back(newTrack);
          }
          else{
            tmp_extrapolatedTracks.emplace_back(newTrack);
          }
        } // Matched Hit loop
      } // Hit search loop
    } // Input tracks loop
  extrapolatedTracks = tmp_extrapolatedTracks;
  tmp_extrapolatedTracks.clear();
  }

  return ActsExamples::ProcessCode::SUCCESS;
}

Acts::NetworkBatchInput ActsExamples::HitSearchMLAlgorithm::BatchTracksForGeoPrediction(std::vector<SimSpacePointContainer> hitTracks){
  
  Acts::NetworkBatchInput networkInput(hitTracks.size(), 9);
  // retrieve the last three hits from each track
  trkIdx = 0;
  for (const auto& trk: hitTracks){
    for (int i=trk.size() - 3; i<trk.size(); i++){
      networkInput(trkIdx, i*3)     = trk.at(i)->x() / m_NNDetectorClassifier.getXScale();
      networkInput(trkIdx, i*3 + 1) = trk.at(i)->y() / m_NNDetectorClassifier.getYScale();
      networkInput(trkIdx, i*3 + 2) = trk.at(i)->z() / m_NNDetectorClassifier.getZScale();
    }
    trkIdx++;
  }

  return networkInput;
}

