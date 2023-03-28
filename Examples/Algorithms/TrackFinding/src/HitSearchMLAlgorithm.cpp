

#include "ActsExamples/TrackFinding/HitSearchMLAlgorithm.hpp"
#include "ActsExamples/EventData/SimSeed.hpp"
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
}

ActsExamples::ProcessCode ActsExamples::HitSearchMLAlgorithm::execute(const AlgorithmContext& ctx) const {

  // Read in container of seeds
  const auto& seeds = ctx.eventStore.get<SimSeedContainer>(m_cfg.inputSeeds);
  // Create input for the classifier
  Acts::NetworkBatchInput networkInput(seeds.size(), 9); // 9 input features. 3 hits * 3 features each
  for(int seedIdx=0; seedIdx<seeds.size(); seedIdx++){
    auto hitsList = seeds.at(seedIdx).sp();
    for(int i=0; i<hitsList.size(); i++){
      networkInput(seedIdx, i*3)     = hitsList.at(i)->x() / m_NNDetectorClassifier.getXScale();
      networkInput(seedIdx, i*3 + 1) = hitsList.at(i)->y() / m_NNDetectorClassifier.getYScale();
      networkInput(seedIdx, i*3 + 2) = hitsList.at(i)->z() / m_NNDetectorClassifier.getZScale();
    }
  }
  
  auto encDetectorID = m_NNDetectorClassifier.PredictVolumeAndLayer(networkInput);
  std::cout<<"encDetectorID size "<<encDetectorID.size()<<std::endl;
  for (auto& v:encDetectorID[0]) std::cout<<v<<", ";
  std::cout<<std::endl;

  // Create input to the hit predictor
  // More efficient to output eigen matrix from network prediction rather than vectors?

  networkInput.conservativeResize(noChange_t, 54);
  for(int i=9; i<54;i++){
    mat.col(i) = encDetectorID.at(i);
  }
  // Acts::NetworkBatchInput networkInput(seeds.size(), 54); // 54 input features. 3 hits * 3 features each + ohe vector
  // for(int seedIdx=0; seedIdx<seeds.size(); seedIdx++){
  //   auto hitsList = seeds.at(seedIdx).sp();
    
    
    
  //   for(int i=0; i<hitsList.size(); i++){
  //     networkInput(seedIdx, i*3)     = hitsList.at(i)->x() / m_NNDetectorClassifier.getXScale();
  //     networkInput(seedIdx, i*3 + 1) = hitsList.at(i)->y() / m_NNDetectorClassifier.getYScale();
  //     networkInput(seedIdx, i*3 + 2) = hitsList.at(i)->z() / m_NNDetectorClassifier.getZScale();
  //   }
  //   std::vector<float> 
  //   networkInput(seedIdx).insert(networkInput(seedIdx))
  // }
  auto testOHE = m_NNDetectorClassifier.predictVolumeAndLayer(networkInput);
  auto predcoor = m_NNHitPredictor.PredictHitCoordinate(predNetworkInput);


  return ActsExamples::ProcessCode::SUCCESS;
}
    

