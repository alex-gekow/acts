

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

  float uncertainty = 15; // in mm
  
  // Why not make sourceLinks and spacepoints shared_ptr??
  // const auto& sourceLinks = ctx.eventStore.get<IndexSourceLinkContainer>(m_cfg.inputSourceLinks);
  const auto& seeds       = ctx.eventStore.get<SimSeedContainer>(m_cfg.inputSeeds);
  const auto& spacepoints = ctx.eventStore.get<SimSpacePointContainer>(m_cfg.inputSpacePoints);

  // std::cout<<"--- spacepoints in event ---"<<std::endl;
  // for(const auto& sp:spacepoints){
  //   std::cout<<sp.x()<<" "<<sp.y()<<" "<<sp.z()<<std::endl;
  // }

  // Create collection of sourceLinks built from spacepoints rather than measurement indices
  // Possible to use SpacepointSourceLinks instead of Acts::SourceLink?
  std::vector<Acts::SourceLink> spacepointSourceLinks;

  // Create new seeds out of the sourceLinks which contain the spacepoints
  std::vector<std::vector<Acts::SourceLink>> sourceLinkSeeds;
  sourceLinkSeeds.reserve(seeds.size());
  int seedIdx = seeds.size();
  for (int i=0; i<seedIdx; i++){
    auto hitsList = seeds.at(i).sp();
    std::vector<Acts::SourceLink> newSeed;
    newSeed.reserve(3);
    int hitIdx = 0;
    // std::cout<<"--- seeds ---"<<std::endl;
    for (const ActsExamples::SimSpacePoint* sp:hitsList){
      Acts::SourceLink sl(sp->sourceLinks()[0].geometryId(), *sp);
      // std::cout<<sp->x()<<" "<<sp->y()<<" "<<sp->z();
      newSeed.push_back(sl);
      spacepointSourceLinks.push_back(sl);
    }
    // std::cout<<std::endl;
    sourceLinkSeeds.push_back(newSeed);
  } 

  // Create multitrajectories out of the initial seeds. May need to use vectorMultiTrajectory insteaed of CKF Results
  // These should contain active tips that we can use to select hits for prediction
  std::vector<Acts::CombinatorialKalmanFilterResult<Acts::VectorMultiTrajectory>> seedTraj;
  for(auto& seed:sourceLinkSeeds){
    Acts::CombinatorialKalmanFilterResult<Acts::VectorMultiTrajectory> result;
    auto stateBuffer = std::make_shared<Acts::VectorMultiTrajectory>();
    result.stateBuffer = stateBuffer;
    size_t tsi = std::numeric_limits<std::uint32_t>::max(); //kInvalid original type
    for(auto it = seed.begin(); it != seed.end(); it++){
      result.trackStateCandidates.clear();
      // result.stateBuffer->clear();
      result.activeTips.clear();
      Acts::TrackStatePropMask mask = Acts::TrackStatePropMask::Predicted;
      Acts::CombinatorialKalmanFilterTipState tipState; // Not needed for NN yet. Leave it empty. nMeasurements or nHoles might be useful
      tsi = result.stateBuffer->addTrackState(mask, tsi);
      auto ts = result.stateBuffer->getTrackState(tsi);
      ts.setUncalibratedSourceLink((*it));
      result.trackStateCandidates.push_back(ts);
      //result.lastTrackIndices.emplace_back(trackTip);
      result.activeTips.emplace_back(tsi, tipState); 
    }
    seedTraj.emplace_back(result);
  }

  ActsExamples::HitSearchMLAlgorithm::BatchedHitSearch(seedTraj, spacepointSourceLinks,20);

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

// Use active tips from a CKF Result as input
Acts::NetworkBatchInput ActsExamples::HitSearchMLAlgorithm::BatchTracksForGeoPrediction(
  Acts::CombinatorialKalmanFilterResult<Acts::VectorMultiTrajectory> tracks) const {

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

void ActsExamples::HitSearchMLAlgorithm::BatchedHitSearch(std::vector<Acts::CombinatorialKalmanFilterResult<Acts::VectorMultiTrajectory>>& seedTrajectories,
  const std::vector<Acts::SourceLink> spacepointSourceLinks, float uncertainty) const{
  
  int totalActiveTips = 999;
  while (totalActiveTips > 0){
    int tmpActiveTipCount = 0;
      
    // Batch all active tips for volume/layer classification
    // Get the first batch
    // Acts::NetworkBatchInput batchedInput = ActsExamples::HitSearchMLAlgorithm::BatchTracksForGeoPrediction(seedTrajectories[0]);
    Acts::NetworkBatchInput batchedInput(0,0);
    // std::vector<Acts::CombinatorialKalmanFilterResult<Acts::VectorMultiTrajectory>>::const_iterator traj;
    // Concatenate the other batches
    for (auto& traj: seedTrajectories){
      if (traj.activeTips.size() == 0) continue;
      tmpActiveTipCount = tmpActiveTipCount + traj.activeTips.size();
      if(batchedInput.rows() == 0){
        Acts::NetworkBatchInput networkInput = ActsExamples::HitSearchMLAlgorithm::BatchTracksForGeoPrediction(traj);
        batchedInput = networkInput;
      }
      else{
        Acts::NetworkBatchInput networkInput = ActsExamples::HitSearchMLAlgorithm::BatchTracksForGeoPrediction(traj);
        Acts::NetworkBatchInput tmpMatrix = batchedInput;
        batchedInput.resize(batchedInput.rows()+networkInput.rows(), batchedInput.cols() );
        batchedInput << tmpMatrix, networkInput;
      }
    }

    //predict the detectorIDs
    auto encDetectorID = m_NNDetectorClassifier.PredictVolumeAndLayer(batchedInput);
    //append predictions to the original input
    Acts::NetworkBatchInput batchedAppendedInput(batchedInput.rows(), batchedInput.cols()+encDetectorID.cols());
    batchedAppendedInput << batchedInput, encDetectorID;

    // Predict the hit coordinates
    auto predCoor = m_NNHitPredictor.PredictHitCoordinate(batchedAppendedInput);
    std::cout<<std::endl; 
    // std::cout<<predCoor<<std::endl;
    // Perform the hit search
    unsigned int globalRowIndex = 0; //Keep count of which row we are in the prediction matrix
    for (auto& traj:seedTrajectories){
      if(traj.activeTips.size()==0) continue;
      std::vector<Acts::MultiTrajectoryTraits::IndexType> tipBuffer; // Store activeTip state indices
      tipBuffer.reserve(traj.activeTips.size());
      for(const auto& tip:traj.activeTips){
        tipBuffer.push_back(tip.first);
      }
      traj.activeTips.clear();
      // Compute the distance between predicted hit coordinates and hits in the detector
      // TODO: Can be more heavily optimized by caching hits by volume and layer first
      // order of activeTips corresponds to row order in prediction matrix
      for(int idx=0; idx<tipBuffer.size(); idx++){
        auto row = predCoor.row( globalRowIndex + idx );
        for(const auto& sl:spacepointSourceLinks){
          auto sp = sl.template get<ActsExamples::SimSpacePoint>();
          float distance = std::hypot( row[0] - sp.x(), row[1]-sp.y(), row[2]-sp.z() );
          
          if (distance < uncertainty){
            std::cout<<"distance: "<<distance<<std::endl;
            Acts::TrackStatePropMask mask = Acts::TrackStatePropMask::Predicted;
            auto tsi = tipBuffer[idx];
            tsi = traj.stateBuffer->addTrackState(mask, tsi);
            auto ts = traj.stateBuffer->getTrackState(tsi);
            ts.setUncalibratedSourceLink(sl);
            // Do not create active tip if the last hit is at the edge of the detector
            // TODO: constrain by volume/layer rather than coordinate
            if(sp.r() < 980 && sp.z() < 2950){
              Acts::CombinatorialKalmanFilterTipState tipState;
              traj.activeTips.emplace_back(tsi, tipState);
            }
            else{
              traj.lastTrackIndices.push_back(tsi);
            }
          }
        }
      } // active tip loop
      globalRowIndex = globalRowIndex + tipBuffer.size();
    }
    totalActiveTips = tmpActiveTipCount;
  }
}

// void ActsExamples::HitSearchMLAlgorithm::BatchedHitSearch(std::vector<Acts::CombinatorialKalmanFilterResult<Acts::VectorMultiTrajectory>>& seedTrajectories,
//   const std::vector<Acts::SourceLink> spacepointSourceLinks, float uncertainty) const{
  // Predict and hit search
  // int totalActiveTips = 999;
  // while (totalActiveTips > 0){
  //   int tmpActiveTipCount = 0;
  //   for (auto& traj:seedTraj){
  //     if(traj.activeTips.size() == 0){
  //       continue;
  //     }
  //     tmpActiveTipCount = tmpActiveTipCount + traj.activeTips.size();
  //     Acts::NetworkBatchInput networkInput = ActsExamples::HitSearchMLAlgorithm::BatchTracksForGeoPrediction(traj);
  //     std::cout<<networkInput<<std::endl;
  //     std::cout<<"------pred detector id -------"<<std::endl;
  //     auto encDetectorID = m_NNDetectorClassifier.PredictVolumeAndLayer(networkInput);
  //     std::cout<<encDetectorID<<std::endl;
  //     Acts::NetworkBatchInput networkAppendedInput(networkInput.rows(), networkInput.cols()+encDetectorID.cols());
  //     networkAppendedInput << networkInput, encDetectorID;
  //     std::cout<<networkAppendedInput<<std::endl;
  //     std::cout<<"-------------pred--------------"<<std::endl;
  //     auto predCoor = m_NNHitPredictor.PredictHitCoordinate(networkAppendedInput); //matrix of predicted hit coordinates x,y,z
  //     std::cout<<predCoor<<std::endl;
  //     std::vector<Acts::MultiTrajectoryTraits::IndexType> tipBuffer; // Store activeTip state indices
  //     tipBuffer.reserve(traj.activeTips.size());
  //     for(const auto& tip:traj.activeTips){
  //       tipBuffer.push_back(tip.first);
  //     }
  //     traj.activeTips.clear();
  //     // Compute the distance between predicted hit coordinates and hits in the detector
  //     // TODO: Can be more heavily optimized by caching hits by volume and layer first
  //     // order of activeTips corresponds to row order in prediction matrix
  //     for(int idx=0; idx<tipBuffer.size(); idx++){
  //       auto row = predCoor.row(idx);
  //       for(const auto& sl:spacepointSourceLinks){
  //         auto sp = sl.template get<ActsExamples::SimSpacePoint>();
  //         float distance = std::hypot( row[0] - sp.x(), row[1]-sp.y(), row[2]-sp.z() );
  //         if (distance < uncertainty){
  //           Acts::TrackStatePropMask mask = Acts::TrackStatePropMask::Predicted;
  //           auto tsi = tipBuffer[idx];
  //           tsi = traj.stateBuffer->addTrackState(mask, tsi);
  //           auto ts = traj.stateBuffer->getTrackState(tsi);
  //           ts.setUncalibratedSourceLink(sl);
  //           // Do not create active tip if the last hit is at the edge of the detector
  //           // TODO: constrain by volume/layer rather than coordinate
  //           if(sp.r() < 1000 && sp.z() < 2950){
  //             Acts::CombinatorialKalmanFilterTipState tipState;
  //             traj.activeTips.emplace_back(tsi, tipState);
  //           }
  //         }
  //       }
  //     } // active tip loop
  //   } // SeedTraj loop
  //   totalActiveTips = tmpActiveTipCount;
  // } // While loop
  // }