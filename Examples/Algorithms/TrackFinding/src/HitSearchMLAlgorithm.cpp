

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
  Acts::NetworkBatchInput networkInput(1, 9); // 9 input features. 3 hits * 3 features each
  for(int seedIdx=0; seedIdx<1; seedIdx++){
    auto hitsList = seeds.at(seedIdx).sp();
    for(int i=0; i<hitsList.size(); i++){
      networkInput(seedIdx, i*3)     = hitsList.at(i)->x() / m_NNDetectorClassifier.getXScale();
      networkInput(seedIdx, i*3 + 1) = hitsList.at(i)->y() / m_NNDetectorClassifier.getYScale();
      networkInput(seedIdx, i*3 + 2) = hitsList.at(i)->z() / m_NNDetectorClassifier.getZScale();
    }
  }

  std::cout<<"org inp \n";
  std::cout<<networkInput;
  auto encDetectorID = m_NNDetectorClassifier.PredictVolumeAndLayer(networkInput);
  // Create input to the hit predictor
  // More efficient to output eigen matrix from network prediction rather than vectors?
  std::cout<<"first output \n";
  std::cout<<encDetectorID<<std::endl;
  // networkInput.conservativeResize(Eigen::NoChange, 54);
  // for(int i=9; i<54;i++){
  //   networkInput.col(i) = encDetectorID.col(i);
  // }
  Acts::NetworkBatchInput networkAppendedInput(networkInput.rows(), networkInput.cols()+encDetectorID.cols());
  networkAppendedInput << networkInput, encDetectorID;
  std::cout<<"new inp \n";
  std::cout<<networkAppendedInput<<std::endl;
  auto predCoor = m_NNHitPredictor.PredictHitCoordinate(networkAppendedInput);
  std::cout<<"pred hit \n";
  std::cout<<predCoor<<std::endl;

  return ActsExamples::ProcessCode::SUCCESS;
}