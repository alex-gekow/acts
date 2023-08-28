

#include "ActsExamples/TrackFinding/HitSearchMLAlgorithm.hpp"
#include "ActsExamples/EventData/SimSeed.hpp"
#include "ActsExamples/EventData/SimSpacePoint.hpp"
#include "ActsExamples/Framework/WhiteBoard.hpp"

using SourceLinkPtr = std::shared_ptr<Acts::SourceLink>;
using SpacePointPtr = std::shared_ptr<ActsExamples::SimSpacePoint>;

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
  // Get infor from the outside world
  // Why not make sourceLinks and spacepoints shared_ptr??
  const auto& indexSourceLinks = ctx.eventStore.get<IndexSourceLinkContainer>(m_cfg.inputSourceLinks);
  const auto& seeds       = ctx.eventStore.get<SimSeedContainer>(m_cfg.inputSeeds);
  const auto& spacepoints = ctx.eventStore.get<SimSpacePointContainer>(m_cfg.inputSpacePoints);


  ////////////////////////////
  // Space point cleaning
  ////////////////////////////
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

  ////////////////////////////
  // Seed point cleaning
  ////////////////////////////
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
  // for(const auto& sp:spacepoints)
  // {
  //   auto sl = sp.sourceLinks()[0].template get<ActsExamples::IndexSourceLink>();
  //   auto volumeID = sl.geometryId().volume();
  //   auto layerID = sl.geometryId().layer();

  //   std::cout<<sp.x()<<" "<<sp.y()<<" "<<sp.z()<<" "<<volumeID<<" "<<layerID<<std::endl;
  // }

  // std::cout<<"--- thinned spacepoints in event ---"<<std::endl;
  // for(const auto& sp:thinned_spacepoints){
  //   std::cout<<sp.x()<<" "<<sp.y()<<" "<<sp.z()<<std::endl;
  // }

  // std::cout<<"--- thinned seeds in event ---"<<std::endl;
  // for(const auto& seed:thinned_seeds){
  //   auto hitsList = seed.sp();
  //   for(const auto& sp:hitsList){
  //     std::cout<<sp->x()<<" "<<sp->y()<<" "<<sp->z()<<std::endl;
  //   }
  //   std::cout<<"\n";
  // }

  // Create collection of sourceLinks built from spacepoints rather than measurement indices
  // Possible to use SpacepointSourceLinks instead of Acts::SourceLink?
  // Better to make shared_ptr<Acts::SourceLink> for more efficient storage in multiple containers such as cached hits


  ////////////////////////////
  // Caching for easier lookup
  ////////////////////////////
  // Create map from measurement index to spacepoint
  std::map<std::pair<int,int>,std::vector<SpacePointPtr>> cachedSpacePoints; // cache spacepoints by volume & layer
  std::map<Index, std::shared_ptr<ActsExamples::SimSpacePoint>> IndexSpacePointMap;
  for(const auto& sp:spacepoints)
  {
    auto sl = sp.sourceLinks()[0].template get<ActsExamples::IndexSourceLink>();
    auto sp_ptr = std::make_shared<ActsExamples::SimSpacePoint>(sp);
    
    auto idx = sl.index();
    IndexSpacePointMap[idx] = sp_ptr;
    
    auto volumeID = sl.geometryId().volume();
    auto layerID = sl.geometryId().layer();
    std::pair<int, int> geo_id = std::make_pair(static_cast<int>(volumeID), static_cast<int>(layerID));
    cachedSpacePoints[geo_id].push_back(sp_ptr);

    // TODO:: From H for A This is not working...
    if (inStripLayer(volumeID))
    {
      idx = sp.sourceLinks()[1].template get<ActsExamples::IndexSourceLink>().index();
      IndexSpacePointMap[idx] = sp_ptr;
    }
  }

  ////////////////////////////
  // Multi trajectory from initial seends
  ////////////////////////////
  // Create multitrajectories out of the initial seeds
  // These should contain active tips that we can use to select hits for prediction
  std::vector<Acts::CombinatorialKalmanFilterResult<Acts::VectorMultiTrajectory>> seedTraj;
  std::vector<Acts::CombinatorialKalmanFilterResult<Acts::VectorMultiTrajectory>> seedTraj2;
  for(auto& seed:seeds)
  {
    // Object to hold the muli trajectory
    Acts::CombinatorialKalmanFilterResult<Acts::VectorMultiTrajectory> result;

    // Actually adding the multitrajectory to the track
    auto stateBuffer = std::make_shared<Acts::VectorMultiTrajectory>();
    result.stateBuffer = stateBuffer;

    // Current track state index
    size_t tsi = std::numeric_limits<std::uint32_t>::max(); //kInvalid original type

    // For each hit add to the track state with the link to the previous tsi
    for(const auto& hit: seed.sp())
    {
      // Clear the active states to make space for the next hit
      result.trackStateCandidates.clear();
      // result.stateBuffer->clear();
      result.activeTips.clear();

      // Create inforation about the new hit
      Acts::TrackStatePropMask mask = Acts::TrackStatePropMask::Predicted;
      Acts::CombinatorialKalmanFilterTipState tipState; // Not needed for NN yet. Leave it empty. nMeasurements or nHoles attributes might be useful in the future

      // Create an empty track state, connected to the previous one
      tsi = result.stateBuffer->addTrackState(mask, tsi);

      // Get the track state
      auto ts = result.stateBuffer->getTrackState(tsi);

      // Add hit to the track state
      auto sl = hit->sourceLinks()[0]; // ok to use first sourcelink as seed is in pixel
      ts.setUncalibratedSourceLink(sl);

      // Push information into the track states
      result.trackStateCandidates.push_back(ts);
      // result.lastTrackIndices.emplace_back(trackTip);
      result.activeTips.emplace_back(tsi, tipState); 
    }
    seedTraj.emplace_back(result);
    seedTraj2.emplace_back(result);
  }

  ////////////////////////////
  //Bactched hit search
  ////////////////////////////
  // ActsExamples::HitSearchMLAlgorithm::BatchedHitSearch(seedTraj, thinned_spacepoints, IndexSpacePointMap, cachedSpacePoints, 20, 128);

  ActsExamples::HitSearchMLAlgorithm::BatchedHitSearch_recommendedBatchSize(seedTraj, spacepoints, IndexSpacePointMap, cachedSpacePoints, 20, 128);

  // ActsExamples::HitSearchMLAlgorithm::BatchedHitSearch(seedTraj2, spacepoints, IndexSpacePointMap, cachedSpacePoints, 20, 128);

  // // For debug printing
  // for(const auto& v:seedTraj)
  // {
  //   std::cout<<"trajectory sizes"<<std::endl;
  //   std::cout<<v.lastTrackIndices.size()<<std::endl;
  //   std::cout<<"built track"<<std::endl;
  //   for(auto iendpoint:v.lastTrackIndices)
  //   {
  //     std::cout<<"-----------------"<<std::endl;
  //     while (true) {
  //       auto ts = v.stateBuffer->getTrackState(iendpoint);
  //       auto sourcelink = ts.getUncalibratedSourceLink();
  //       auto idx = sourcelink.template get<IndexSourceLink>().index();
  //       auto sp = IndexSpacePointMap[idx];
  //       std::cout<<sp->x()<<" "<<sp->y()<<" "<<sp->z()<<std::endl;
  //       if(!ts.hasPrevious()) break;
  //       iendpoint = ts.previous();
  //     }
  //     std::cout<<std::endl;
  //     // break;
  //   }
  // }

  return ActsExamples::ProcessCode::SUCCESS;
}




void ActsExamples::HitSearchMLAlgorithm::BatchTracksForGeoPrediction_recommendedBatchSize(
  std::vector<Acts::CombinatorialKalmanFilterResult<Acts::VectorMultiTrajectory>>& tracks, 
  std::vector<int> indexToProcess,
  Acts::NetworkBatchInput& networkInput, 
  std::map<Index, std::shared_ptr<ActsExamples::SimSpacePoint>>& IndexSpacePointMap) const {
  
  int rowIndex = 0;
  // Loop over all the tracks
  for (const auto& traj_idx: indexToProcess) 
  {
    // Get the current trajector
    auto traj = tracks[traj_idx];

    if (traj.activeTips.size() == 0)
    {
      std::cout<<"ERROR: Empty trajectory given to the create input for"<<std::endl;
      exit(1);
    }

    // Loop over all trajectories
    for (int i = 0; i < traj.activeTips.size(); i++) 
    {
      auto tip = traj.activeTips.at(i);

      int hitCounter = 2;
      auto tipIdx = tip.first;

      decltype(tipIdx) prev_tipIndex = 0;
      bool hasPrev_tipIndex = false;
      
      while (hitCounter >= 0) 
      {
        auto ts = traj.stateBuffer->getTrackState(tipIdx);
        auto index_sl = ts.getUncalibratedSourceLink().template get < ActsExamples::IndexSourceLink > ();
        auto sp = IndexSpacePointMap[index_sl.index()];

        if (hasPrev_tipIndex)
        {
          auto prev_ts = traj.stateBuffer->getTrackState(prev_tipIndex);
          auto prev_index_sl = prev_ts.getUncalibratedSourceLink().template get < ActsExamples::IndexSourceLink > ();
          auto prev_sp = IndexSpacePointMap[prev_index_sl.index()];
          if (sp == prev_sp) 
          {
            std::cout<<"Found matching SP. It shouldn't be finding one right now, till we fix volume id"<<std::endl;
            prev_tipIndex = tipIdx;
            tipIdx = ts.previous();
            hasPrev_tipIndex = true;
            continue;
          }
        }
        networkInput(rowIndex, hitCounter * 3 + 0) = sp -> x() / m_NNDetectorClassifier.getXScale();
        networkInput(rowIndex, hitCounter * 3 + 1) = sp -> y() / m_NNDetectorClassifier.getYScale();
        networkInput(rowIndex, hitCounter * 3 + 2) = sp -> z() / m_NNDetectorClassifier.getZScale();

        hasPrev_tipIndex = true;
        prev_tipIndex = tipIdx;
        tipIdx = ts.previous();

        hitCounter--;
      }
      rowIndex++;
     
    }
  }
}

void ActsExamples::HitSearchMLAlgorithm::BatchedHitSearch_recommendedBatchSize(
  std::vector<Acts::CombinatorialKalmanFilterResult<Acts::VectorMultiTrajectory>>& seedTrajectories,
  const SimSpacePointContainer& spacepoints, 
  std::map<Index, std::shared_ptr<ActsExamples::SimSpacePoint>>& IndexSpacePointMap, 
  const std::map<std::pair<int,int>,std::vector<SpacePointPtr>>& cachedSpacePoints, 
  float uncertainty, 
  int batch_size) const
  {
  
    // For easier counter
    int iteration = 0;
    do 
    {
      iteration++;

      ////////////////////////////////////
      // Extract trajectories to process
      ////////////////////////////////////
      std::vector<int> trajIndexToProcess;
      int currentIterTipSize = 0;
      for (int trajIndex = 0; trajIndex < seedTrajectories.size(); trajIndex++)
      {
        // If the current trajectory is done, continue;
        if (seedTrajectories.at(trajIndex).activeTips.size() == 0) continue;

        // Process the current trajectories, ~ the recommended batch size
        trajIndexToProcess.push_back(trajIndex);
        currentIterTipSize += seedTrajectories.at(trajIndex).activeTips.size();
        if(currentIterTipSize > batch_size) break;
      }

      if(trajIndexToProcess.size() == 0) break;

      // for(auto& index: trajIndexToProcess)
      // {
      //   std::cout<<"Iteration: "<<iteration<<" processing index: "<<index<<std::endl;
      // }

      //////////////////////////////////
      // Prepare and get the prediction
      //////////////////////////////////
      Acts::NetworkBatchInput batchedInput(currentIterTipSize, 9);
      ActsExamples::HitSearchMLAlgorithm::BatchTracksForGeoPrediction_recommendedBatchSize(seedTrajectories, trajIndexToProcess, batchedInput, IndexSpacePointMap);
      std::cout<<"batchedInput: size: "<<currentIterTipSize<<std::endl;;
      // std::cout<<batchedInput<<std::endl;

      //predict the detectorIDs
      auto encDetectorID = m_NNDetectorClassifier.PredictVolumeAndLayer(batchedInput);
      // std::cout<<"encDetectorID: \n";
      // std::cout<<encDetectorID<<std::endl;

      //append predictions to the original input
      Acts::NetworkBatchInput batchedAppendedInput(batchedInput.rows(), batchedInput.cols()+encDetectorID.cols());
      batchedAppendedInput << batchedInput, encDetectorID;

      // Predict the hit coordinates
      auto predCoor = m_NNHitPredictor.PredictHitCoordinate(batchedAppendedInput);
      // std::cout<<"predCoor: \n";
      // std::cout<<predCoor<<std::endl;

      //////////////////////////////////
      // Hit search
      //////////////////////////////////
      int currentRow = 0;
      for (const auto& i: trajIndexToProcess) 
      {
        auto& traj = seedTrajectories.at(i);

        if (traj.activeTips.size() == 0)
        {
          std::cout<<"ERROR: Empty trajectory given to do hit search for"<<std::endl;
          exit(1);
        }
        
        int activeTipSize = traj.activeTips.size();
        for(int tip_idx = 0; tip_idx < activeTipSize; tip_idx++)
        {

          // Get the spacepoint from this tip
          auto tsi = traj.activeTips.at(tip_idx).first;
          auto ts = traj.stateBuffer->getTrackState(tsi);
          auto sl = ts.getUncalibratedSourceLink();
          auto sp = IndexSpacePointMap[sl.template get<IndexSourceLink>().index()];

          // Get the prediction
          auto row = predCoor.row(currentRow );
          currentRow++;

          // Do the hit search
          bool matched = ActsExamples::HitSearchMLAlgorithm::FullHitSearch(traj, sp, spacepoints, row, tip_idx, cachedSpacePoints, uncertainty);
         
          // If there was not found hit we end the track. 
          // Alternitvely we can create a spacepoint, add it to the track and continue searching
          if(!matched)
          {
            traj.lastTrackIndices.push_back(traj.activeTips[tip_idx].first);
          }

          // if(matched) std::cout<<"Found a match for traj: "<<i<<std::endl;
          // else std::cout<<"Not Found a match for traj: "<<i<<std::endl;

        } // active tip loop
       
        traj.activeTips.clear();
        // Change the active tips to the ones in the buffer
        traj.activeTips.swap(traj.activeTipBuffer);
        traj.activeTipBuffer.clear();
      }
    }while(true);



    // finsih and return 

  }


// Hit searning
bool ActsExamples::HitSearchMLAlgorithm::FullHitSearch(Acts::CombinatorialKalmanFilterResult<Acts::VectorMultiTrajectory>& traj, 
  const std::shared_ptr<SimSpacePoint> spacepoint,
  const SimSpacePointContainer& spacepoints, 
  Eigen::VectorXf row, 
  Acts::MultiTrajectoryTraits::IndexType tipIndex, 
  const std::map<std::pair<int,int>, std::vector<SpacePointPtr>>& cachedSpacePoints, 
  float uncertainty) const {
  
  bool matched=false;

  Eigen::VectorXf detectorRow = row.tail<45>(); // Get the one hot encoded detector id
  Eigen::MatrixXf::Index predVolume;
  Eigen::MatrixXf::Index predLayer;
  detectorRow.head<15>().maxCoeff(&predVolume);
  detectorRow.tail<30>().maxCoeff(&predLayer);
  auto volLayerPair = std::make_pair( static_cast<int>(predVolume), static_cast<int>(predLayer) );

  // If there are no hits in the predicted vol/layer it will not exist in the cache
  if (cachedSpacePoints.find(volLayerPair) == cachedSpacePoints.end())
  {
    // std::cout<<"ERROR: no cached sp found for the predVolume: "<<predVolume<<" predLayer: "<<predLayer<<std::endl;
    // return false;
  }
  // Get the cached space point
  // auto spacePointsInDetector = cachedSpacePoints.at(volLayerPair);
  for(const auto& spList: cachedSpacePoints)
  {
    auto spacePointsInDetector = spList.second;
    // Find the closest one
    for(auto& sp_ptr:spacePointsInDetector)
    {
      auto sp = *sp_ptr;
      float distance = std::hypot( row[0] - sp.x(), row[1]-sp.y(), row[2]-sp.z() );

      // std::cout<<"Distance: "<<distance << " " <<row[0] << " " << sp.x()  << " " << row[1] << " " << sp.y() << " " <<  row[2] << " " << sp.z()<<std::endl;
    
      // If the spacepoint is in the SCT we need to add the spacepoint twice to be able to access both measurements downstream
      // These will be replaced by IndexSourceLinks downstream
      // Prediction must account for repeated spacepoint track states when fetching input to NNs.  
      if (distance < uncertainty)
      {
        if( (*spacepoint) == sp) continue;
        matched = true;

        // Info for the new track state
        Acts::TrackStatePropMask mask = Acts::TrackStatePropMask::Predicted;

        // Create and retrieve the track state index, connected to the old index
        auto tsi = traj.activeTips[tipIndex].first;      // Get the track state index of the active tip
        tsi = traj.stateBuffer->addTrackState(mask, tsi);

        // Get the track state
        auto ts = traj.stateBuffer->getTrackState(tsi);

        // Add hit to the track state
        auto sl = sp.sourceLinks()[0];
        ts.setUncalibratedSourceLink(sl);

        // Add SCT spacepoints twice
        auto volumeID = sp.sourceLinks()[0].geometryId().volume();
        if (inStripLayer(volumeID))
        {
          tsi = traj.stateBuffer->addTrackState(mask, tsi);
          ts = traj.stateBuffer->getTrackState(tsi);
          auto sl1 = sp.sourceLinks()[1]; // Note different source link from above
          ts.setUncalibratedSourceLink(sl1); 
        }

        // Do not create active tip if the last hit is at the edge of the detector
        // TODO: constrain by volume/layer rather than coordinate
        if(sp.r() > 1000 || std::abs(sp.z()) > 3450)
        {
          traj.lastTrackIndices.push_back(tsi);
        }
        else
        {
          Acts::CombinatorialKalmanFilterTipState tipState;
          traj.activeTipBuffer.emplace_back(tsi, tipState);
        }
      }
    }
    // TODO:: if there is a match in this layer, we don't expect a match in any other, so break
    if(matched) break;
  }

  return matched;
}