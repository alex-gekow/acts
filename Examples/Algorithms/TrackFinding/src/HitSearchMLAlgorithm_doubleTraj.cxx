

#include "ActsExamples/TrackFinding/HitSearchMLAlgorithm.hpp"
#include "ActsExamples/EventData/SimSeed.hpp"
#include "ActsExamples/EventData/SimSpacePoint.hpp"
#include "ActsExamples/Framework/WhiteBoard.hpp"

using SourceLinkPtr = std::shared_ptr<Acts::SourceLink>;

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
/*Usees two trajectories. One to store the index source links. One to store the spacepoints*/
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
  const auto& sourceLinks = ctx.eventStore.get<IndexSourceLinkContainer>(m_cfg.inputSourceLinks);
  const auto& seeds       = ctx.eventStore.get<SimSeedContainer>(m_cfg.inputSeeds);
  const auto& spacepoints = ctx.eventStore.get<SimSpacePointContainer>(m_cfg.inputSpacePoints);

  // remove redundant spacepoints from event
  std::set<int> indexToRemove;
  SimSpacePointContainer thinned_spacepoints;
  for(int i=0; i<spacepoints.size(); i++){
    if(indexToRemove.find(i) != indexToRemove.end()) continue;
    for(int j=i+1; j<spacepoints.size(); j++){
      auto sp1 = spacepoints[i];
      auto sp2 = spacepoints[j];
      float dRRho = std::hypot(sp1.x()-sp2.x(), sp1.y()-sp2.y());
      if ( (dRRho < 1) && (std::abs(sp1.z()-sp2.z() < 6)) ){
        indexToRemove.insert(j);
      }
    }
  }
  for(int i = 0; i < spacepoints.size(); i++)
  {
      if(indexToRemove.find(i) != indexToRemove.end()) continue;
      thinned_spacepoints.push_back (spacepoints[i]);
  }

  // Remove redundant seeds from event. Ensure the seed uses only spacepoints from the thinned_spacepoints container
  indexToRemove.clear();
  SimSeedContainer thinned_seeds;
  for(int i=0; i<seeds.size(); i++){
    auto hits = seeds.at(i).sp();
    for(const auto& hit:hits){
      if(std::find(thinned_spacepoints.begin(), thinned_spacepoints.end(), *hit) == thinned_spacepoints.end()) indexToRemove.insert(i);
    }
    for(int j=i+1; j<seeds.size(); j++){
      float dR = 0;
      for(int k=0; k<3; k++){ // seed size is 3
        auto sp1 = seeds[i].sp().at(k);
        auto sp2 = seeds[j].sp().at(k);
        dR = dR + std::hypot(sp1->x()-sp2->x(), sp1->y()-sp2->y(),sp1->z()-sp2->z());
      }
      if(dR/seeds.size() < 1) {indexToRemove.insert(i);}
    }
  }

  for(int i = 0; i < seeds.size(); i++)
  {
      if(indexToRemove.find(i) != indexToRemove.end()) continue;
      thinned_seeds.push_back (seeds[i]);
  }
  

  std::cout<<"spacepoints size: "<<spacepoints.size()<<" thinned sp size: "<<thinned_spacepoints.size()<<" seed size: "<<seeds.size()<<" thinned seed size: "<<thinned_seeds.size()<<std::endl;
  
  // std::cout<<"--- spacepoints in event ---"<<std::endl;
  // for(const auto& sp:spacepoints){
  //   std::cout<<sp.x()<<" "<<sp.y()<<" "<<sp.z()<<std::endl;
  // }

  // std::cout<<"--- thinned spacepoints in event ---"<<std::endl;
  // for(const auto& sp:thinned_spacepoints){
  //   std::cout<<sp.x()<<" "<<sp.y()<<" "<<sp.z()<<std::endl;
  // }

  std::cout<<"--- thinned seeds in event ---"<<std::endl;
  for(const auto& seed:thinned_seeds){
    auto hitsList = seed.sp();
    for(const auto& sp:hitsList){
      std::cout<<sp->x()<<" "<<sp->y()<<" "<<sp->z()<<std::endl;
    }
    std::cout<<"\n";
  }

  // Create collection of sourceLinks built from spacepoints rather than measurement indices
  // Possible to use SpacepointSourceLinks instead of Acts::SourceLink?
  // Better to make shared_ptr<Acts::SourceLink> for more efficient storage in multiple containers such as cached hits

  std::vector<SourceLinkPtr> spacepointSourceLinks;
  std::map<std::pair<int,int>,std::vector<SourceLinkPtr>> cachedSpacePoints; // cache spacepoints by volume & layer

  // Create new seeds out of the sourceLinks which contain the spacepoints
  std::vector<std::vector<SourceLinkPtr>> sourceLinkSeeds;
  sourceLinkSeeds.reserve(thinned_seeds.size());
  int seedSize = thinned_seeds.size();
  for (int i=0; i<seedSize; i++){
    auto hitsList = thinned_seeds.at(i).sp();
    std::vector<SourceLinkPtr> newSeed;
    newSeed.reserve(3);
    int hitIdx = 0;
    // std::cout<<"--- seeds ---"<<std::endl;
    for (const ActsExamples::SimSpacePoint* sp:hitsList){
      SourceLinkPtr sl = std::make_shared<Acts::SourceLink>(sp->sourceLinks()[0].geometryId(), *sp);
      // std::cout<<sp->x()<<" "<<sp->y()<<" "<<sp->z();
      newSeed.push_back(sl);
      spacepointSourceLinks.push_back(sl);
      auto volLayerPair = std::make_pair(sp->sourceLinks()[0].geometryId().volume(), sp->sourceLinks()[0].geometryId().layer());
      cachedSpacePoints[volLayerPair].push_back(sl);
    }
    sourceLinkSeeds.push_back(newSeed);
  } 

  // Create multitrajectories out of the initial seeds
  // These should contain active tips that we can use to select hits for prediction
  std::vector<Acts::CombinatorialKalmanFilterResult<Acts::VectorMultiTrajectory>> seedTraj;

  // MTJ to store indexSourceLinks necessary for precision fit.
  // Wasetful duplication of VMTJ, but no better way to do this yet
  std::vector<Acts::CombinatorialKalmanFilterResult<Acts::VectorMultiTrajectory>> IndexTraj;

  for(auto& seed:sourceLinkSeeds){
    Acts::CombinatorialKalmanFilterResult<Acts::VectorMultiTrajectory> result;
    Acts::CombinatorialKalmanFilterResult<Acts::VectorMultiTrajectory> idx_result;
    auto stateBuffer = std::make_shared<Acts::VectorMultiTrajectory>();
    auto idx_stateBuffer = std::make_shared<Acts::VectorMultiTrajectory>();
    result.stateBuffer = stateBuffer;
    idx_result.stateBuffer = idx_stateBuffer;
    size_t tsi = std::numeric_limits<std::uint32_t>::max(); //kInvalid original type

    for(auto it = seed.begin(); it != seed.end(); it++){
      result.trackStateCandidates.clear();
      // result.stateBuffer->clear();
      result.activeTips.clear();
      Acts::TrackStatePropMask mask = Acts::TrackStatePropMask::Predicted;
      Acts::CombinatorialKalmanFilterTipState tipState; // Not needed for NN yet. Leave it empty. nMeasurements or nHoles attributes might be useful in the future
      
      auto idx_tsi = idx_result.stateBuffer->addTrackState(mask, tsi); // idx_tsi should be the same as tsi. No need to use both
      tsi = result.stateBuffer->addTrackState(mask, tsi);

      auto ts = result.stateBuffer->getTrackState(tsi);
      auto idx_ts = idx_result.stateBuffer->getTrackState(tsi);

      ts.setUncalibratedSourceLink((**it));
      idx_ts.setUncalibratedSourceLink((*it)->template get<ActsExamples::SimSpacePoint>().sourceLinks()[0]);

      result.trackStateCandidates.push_back(ts);
      result.activeTips.emplace_back(tsi, tipState); 
    }
    seedTraj.emplace_back(result);
    IndexTraj.emplace_back(idx_result);
  }

  // ActsExamples::HitSearchMLAlgorithm::BatchedHitSearch(seedTraj, spacepointSourceLinks,cachedSpacePoints, 20);
  ActsExamples::HitSearchMLAlgorithm::BatchedHitSearch(seedTraj, IndexTraj, spacepointSourceLinks,cachedSpacePoints, 20);

  // for(const auto& v:seedTraj){
  //   std::cout<<v.lastTrackIndices.size()<<std::endl;
  //   if(v.lastTrackIndices.size() > 1){
  //     std::cout<<"built track"<<std::endl;
  //     auto iendpoint = v.lastTrackIndices[0];
  //     while (true) {
  //       auto ts = v.stateBuffer->getTrackState(iendpoint);
  //       auto sourcelink = ts.getUncalibratedSourceLink(); 
  //       std::cout<<sourcelink.template get<ActsExamples::SimSpacePoint>().x()<<" "<<sourcelink.template get<ActsExamples::SimSpacePoint>().y()<<" "<<sourcelink.template get<ActsExamples::SimSpacePoint>().z()<<std::endl;
  //       if(!ts.hasPrevious()) break;
  //       iendpoint = ts.previous();
  //     }
  //     break;
  //   }
  // }

  std::cout<<"trajectory sizes"<<std::endl;
  for(const auto& v:seedTraj){
    std::cout<<v.lastTrackIndices.size()<<std::endl;
  }

  // ActsExamples::HitSearchMLAlgorithm::MultiTrajectorySpacepointToIndex(seedTraj);

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
  // retrieve the last three unique hits from each track
  int trkIndex = 0;
  for (const auto& tip: tracks.activeTips){
    int hitCounter = 2;
    auto tipIdx = tip.first;
    Acts::VectorMultiTrajectory::ConstTrackStateProxy* prev_ts = nullptr;
    while(hitCounter >= 0){
      auto ts  = tracks.stateBuffer->getTrackState(tipIdx);

      // If the previous spacepoint is the same, get the one before it
      if(prev_ts){
        if(ts.getUncalibratedSourceLink().template get<ActsExamples::SimSpacePoint>() == prev_ts->getUncalibratedSourceLink().template get<ActsExamples::SimSpacePoint>()){
          tipIdx = ts.previous();
          continue;
        }
      }
      auto sourcelink = ts.getUncalibratedSourceLink(); 
      networkInput(trkIndex, hitCounter*3    ) =  sourcelink.template get<ActsExamples::SimSpacePoint>().x() / m_NNDetectorClassifier.getXScale();
      networkInput(trkIndex, hitCounter*3 + 1) =  sourcelink.template get<ActsExamples::SimSpacePoint>().y() / m_NNDetectorClassifier.getYScale();
      networkInput(trkIndex, hitCounter*3 + 2) =  sourcelink.template get<ActsExamples::SimSpacePoint>().z() / m_NNDetectorClassifier.getZScale();
      // std::cout<<"trackIndex: "<<trkIndex<<" hitCounter: "<<hitCounter<<" "<<sourcelink.template get<ActsExamples::SimSpacePoint>().x()<<" "<<sourcelink.template get<ActsExamples::SimSpacePoint>().y()<<" "<<sourcelink.template get<ActsExamples::SimSpacePoint>().z()<<std::endl;
      
      tipIdx = ts.previous();
      hitCounter--;
      
    }
    // std::cout<<"\n"<<std::endl;
    trkIndex++;
  }

  return networkInput;
}

void ActsExamples::HitSearchMLAlgorithm::BatchedHitSearch(std::vector<Acts::CombinatorialKalmanFilterResult<Acts::VectorMultiTrajectory>>& seedTrajectories, std::vector<Acts::CombinatorialKalmanFilterResult<Acts::VectorMultiTrajectory>>& indexTrajectories,
  const std::vector<SourceLinkPtr> spacepointSourceLinks, const std::map<std::pair<int,int>,std::vector<SourceLinkPtr>>& cachedSpacePoints, float uncertainty) const{
  
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
    // std::cout<<"batched input"<<std::endl;
    // std::cout<<batchedAppendedInput<<std::endl;
    // Predict the hit coordinates
    auto predCoor = m_NNHitPredictor.PredictHitCoordinate(batchedAppendedInput);
    // std::cout<<"predicted"<<std::endl; 
    // std::cout<<predCoor<<std::endl;
    // Perform the hit search
    unsigned int globalRowIndex = 0; //Keep count of which row we are in the prediction matrix
    for(int trajIdx=0; trajIdx<seedTrajectories.size(); trajIdx++){
      auto traj = seedTrajectories.at(trajIdx);
      auto idxTraj = indexTrajectories.at(trajIdx);
    //for (auto& traj:seedTrajectories){
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
        bool matched = false;
        auto row = predCoor.row( globalRowIndex + idx );

        // Get the predicted volume and layer 
        // auto detectorRow = encDetectorID.row( globalRowIndex + idx);
        // Eigen::MatrixXf::Index predVolume, predLayer;
        // detectorRow.head<15>().maxCoeff(&predVolume);
        // detectorRow.tail<30>().maxCoeff(&predLayer);
        // auto volLayerPair = std::make_pair( static_cast<int>(predVolume), static_cast<int>(predLayer) );
        // auto spacePointsInDetector = cachedSpacePoints.at(volLayerPair);
        // for(auto& sl_ptr:spacePointsInDetector){

        for(const auto& sl_ptr:spacepointSourceLinks){
          auto sl = *sl_ptr;
          auto sp = sl.template get<ActsExamples::SimSpacePoint>();
          float distance = std::hypot( row[0] - sp.x(), row[1]-sp.y(), row[2]-sp.z() );
          
          // If the spacepoint is in the SCT we need to add the spacepoint twice to be able to access both measurements downstream
          // These will be replaced by IndexSourceLinks downstream
          // Prediction must account for repeated spacepoint track states when fetching input to NNs.
          
          if (distance < uncertainty){
            // std::cout<<"distance: "<<distance<<std::endl;
            matched = true;
            auto volumeID = sl.geometryId().volume();
            auto tsi = tipBuffer[idx];
            Acts::TrackStatePropMask mask = Acts::TrackStatePropMask::Predicted;
            tsi = traj.stateBuffer->addTrackState(mask, tsi);
            auto idx_tsi = idxTraj.stateBuffer->addTrackState(mask, tsi); 

            auto ts = traj.stateBuffer->getTrackState(tsi);
            auto idx_ts = idxTraj.stateBuffer->getTrackState(tsi);
            ts.setUncalibratedSourceLink(sl);
            idx_ts.setUncalibratedSourceLink(sl.template get<ActsExamples::SimSpacePoint>().sourceLinks()[0]);

            // Add SCT spacepoints twice
            if (volumeID == 22 || volumeID == 23 || volumeID == 24){
              tsi = traj.stateBuffer->addTrackState(mask, tsi);
              ts = traj.stateBuffer->getTrackState(tsi);
              ts.setUncalibratedSourceLink(sl);

              idx_tsi = idxTraj.stateBuffer->addTrackState(mask, tsi);
              idx_ts = idxTraj.stateBuffer->getTrackState(tsi);
              idx_ts.setUncalibratedSourceLink(sl.template get<ActsExamples::SimSpacePoint>().sourceLinks()[1]);

            }
            // Do not create active tip if the last hit is at the edge of the detector
            // TODO: constrain by volume/layer rather than coordinate
            if(sp.r() < 980 && sp.z() < 2950){
              Acts::CombinatorialKalmanFilterTipState tipState;
              traj.activeTips.emplace_back(tsi, tipState);
            }
            else{
              traj.lastTrackIndices.push_back(tsi);
              idxTraj.lastTrackIndices.push_back(tsi);
            }
          }
        }
        // If there was not found hit we end the track. Alternitvely we can create a spacepoint, add it to the track and continue searching
        if(!matched){
          traj.lastTrackIndices.push_back(tipBuffer[idx]);
        }
      } // active tip loop
      globalRowIndex = globalRowIndex + tipBuffer.size();
    }
    totalActiveTips = tmpActiveTipCount;
  }
}

void ActsExamples::HitSearchMLAlgorithm::MultiTrajectorySpacepointToIndex(std::vector<Acts::CombinatorialKalmanFilterResult<Acts::VectorMultiTrajectory>>& Trajectories) const {
  
  for(auto& v:Trajectories){
    std::cout<<"v"<<std::endl;
    for(auto iendpoint: v.lastTrackIndices){
      const Acts::SourceLink* prev_sp_sl = nullptr;
      const Acts::SourceLink* index_sourcelink = nullptr;
      std::cout<<"endpoint: "<<iendpoint<<std::endl;
      while (true) {
        auto ts = v.stateBuffer->getTrackState(iendpoint);
        const auto sp_sourcelink = ts.getUncalibratedSourceLink();
        if (!prev_sp_sl){
          std::cout<<"0"<<std::endl;
          index_sourcelink = &(sp_sourcelink.template get<ActsExamples::SimSpacePoint>().sourceLinks()[0]);
          prev_sp_sl = &sp_sourcelink;
          std::cout<<"b"<<std::endl;
          ts.setUncalibratedSourceLink(*index_sourcelink);
          std::cout<<"a"<<std::endl;
          if(!ts.hasPrevious()) break;
          iendpoint = ts.previous();
          std::cout<<"c"<<std::endl;
          continue;
          std::cout<<"-0"<<std::endl;
        }

        if (sp_sourcelink.template get<ActsExamples::SimSpacePoint>() == (*prev_sp_sl).template get<ActsExamples::SimSpacePoint>()){ 
          std::cout<<"1"<<std::endl;
          index_sourcelink = &(sp_sourcelink.template get<ActsExamples::SimSpacePoint>().sourceLinks()[1]);
          std::cout<<"-1"<<std::endl;
        }
        else{
          std::cout<<"2"<<std::endl;
          index_sourcelink = &(sp_sourcelink.template get<ActsExamples::SimSpacePoint>().sourceLinks()[0]);
          std::cout<<"-2"<<std::endl;
        }
        std::cout<<"3"<<std::endl;
        prev_sp_sl = &sp_sourcelink;
        ts.setUncalibratedSourceLink(*index_sourcelink);
        std::cout<<"4"<<std::endl;
        if(!ts.hasPrevious()) break;
        iendpoint = ts.previous();
      }
      break;
    }
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