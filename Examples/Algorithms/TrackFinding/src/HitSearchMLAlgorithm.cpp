

#include "ActsExamples/TrackFinding/HitSearchMLAlgorithm.hpp"
#include "ActsExamples/EventData/SimSeed.hpp"


ActsExamples::HitSearchMLAlgorithm::HitSearchMLAlgorithm(
    ActsExamples::HitSearchMLAlgorithm::Config cfg,
    Acts::Logging::Level lvl)
    : ActsExamples::IAlgorithm("HitSearchMLAlgorithm", lvl),
      m_cfg(std::move(cfg)),
      m_env(ORT_LOGGING_LEVEL_WARNING, "MLDetectorClassifier"),
      m_NNDetectorClassifier(m_env, m_cfg.NNDetectorClassifier.c_str()),
      m_NNHitPredictor(m_env, m_cfg.NNDetectorClassifier.c_str()) {
  if (m_cfg.inputSeeds.empty()) {
    throw std::invalid_argument("Missing seeds input collection");
  }
  if (m_cfg.outputTracks.empty()) {
    throw std::invalid_argument("Missing tracks output collection");
  }
}

ActsExamples::ProcessCode ActsExamples::HitSearchMLAlgorithm::execute(const AlgorithmContext& ctx){

   // Read in container of seeds
   // const auto& seeds = ctx.eventStore.get<SimSeedContainer>(m_cfg.inputSeeds);
  // test input
  Acts::NetworkBatchInput networkInput(1, 9);
  std::vector<float> testSeed = {0.1,0.2,0.3,0.1,0.2,0.3,0.1,0.2,0.3};
  for (int v=0; v<testSeed.size(); v++){  
    networkInput(0, v) = testSeed[v];
  }
  
  auto testOHE = m_NNDetectorClassifier.predictVolumeAndLayer(networkInput);
  // testSeed.insert(testSeed.begin(), testOHE.begin(), testOHE.end());
  // auto predHitCoordinate = m_NNHitPredictor.PredictHitCoordinate(testSeed);

  return ActsExamples::ProcessCode::SUCCESS;
}
    

