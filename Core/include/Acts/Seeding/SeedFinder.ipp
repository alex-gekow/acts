// This file is part of the Acts project.
//
// Copyright (C) 2023 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cmath>
#include <numeric>
#include <type_traits>

namespace Acts {

template <typename external_spacepoint_t, typename platform_t>
SeedFinder<external_spacepoint_t, platform_t>::SeedFinder(
    const Acts::SeedFinderConfig<external_spacepoint_t>& config)
    : m_config(config) {
  if (not config.isInInternalUnits) {
    throw std::runtime_error(
        "SeedFinderConfig not in ACTS internal units in SeedFinder");
  }
  if (std::isnan(config.deltaRMaxTopSP)) {
    throw std::runtime_error("Value of deltaRMaxTopSP was not initialised");
  }
  if (std::isnan(config.deltaRMinTopSP)) {
    throw std::runtime_error("Value of deltaRMinTopSP was not initialised");
  }
  if (std::isnan(config.deltaRMaxBottomSP)) {
    throw std::runtime_error("Value of deltaRMaxBottomSP was not initialised");
  }
  if (std::isnan(config.deltaRMinBottomSP)) {
    throw std::runtime_error("Value of deltaRMinBottomSP was not initialised");
  }
}

template <typename external_spacepoint_t, typename platform_t>
template <template <typename...> typename container_t, typename sp_range_t>
void SeedFinder<external_spacepoint_t, platform_t>::createSeedsForGroup(
    const Acts::SeedFinderOptions& options, SeedingState& state,
    Acts::SpacePointGrid<external_spacepoint_t>& grid,
    std::back_insert_iterator<container_t<Seed<external_spacepoint_t>>> outIt,
    const sp_range_t& bottomSPsIdx, const std::size_t middleSPsIdx,
    const sp_range_t& topSPsIdx,
    const Acts::Range1D<float>& rMiddleSPRange) const {
  if (not options.isInInternalUnits) {
    throw std::runtime_error(
        "SeedFinderOptions not in ACTS internal units in SeedFinder");
  }

  // This is used for seed filtering later
  const std::size_t max_num_seeds_per_spm =
      m_config.seedFilter->getSeedFilterConfig().maxSeedsPerSpMConf;
  const std::size_t max_num_quality_seeds_per_spm =
      m_config.seedFilter->getSeedFilterConfig().maxQualitySeedsPerSpMConf;

  state.candidates_collector.setMaxElements(max_num_seeds_per_spm,
                                            max_num_quality_seeds_per_spm);

  // If there are no bottom or top bins, just return and waste no time
  if (bottomSPsIdx.size() == 0 or topSPsIdx.size() == 0) {
    return;
  }

  // Get the middle space point candidates
  auto& middleSPs = grid.at(middleSPsIdx);

  // neighbours
  // clear previous results
  state.bottomNeighbours.clear();
  state.topNeighbours.clear();

  // Fill
  // bottoms
  for (const std::size_t idx : bottomSPsIdx) {
    state.bottomNeighbours.emplace_back(
        grid, idx, middleSPs.front()->radius() - m_config.deltaRMaxBottomSP);
  }
  // tops
  for (const std::size_t idx : topSPsIdx) {
    state.topNeighbours.emplace_back(
        grid, idx, middleSPs.front()->radius() + m_config.deltaRMinTopSP);
  }

  for (auto& spM : middleSPs) {
    float rM = spM->radius();
    float zM = spM->z();

    // check if spM is outside our radial region of interest
    if (m_config.useVariableMiddleSPRange) {
      if (rM < rMiddleSPRange.min()) {
        continue;
      }
      if (rM > rMiddleSPRange.max()) {
        // break because SPs are sorted in r
        break;
      }
    } else if (not m_config.rRangeMiddleSP.empty()) {
      /// get zBin position of the middle SP
      auto pVal = std::lower_bound(m_config.zBinEdges.begin(),
                                   m_config.zBinEdges.end(), zM);
      int zBin = std::distance(m_config.zBinEdges.begin(), pVal);
      /// protects against zM at the limit of zBinEdges
      zBin == 0 ? zBin : --zBin;
      if (rM < m_config.rRangeMiddleSP[zBin][0]) {
        continue;
      }
      if (rM > m_config.rRangeMiddleSP[zBin][1]) {
        // break because SPs are sorted in r
        break;
      }
    } else {
      if (rM < m_config.rMinMiddle) {
        continue;
      }
      if (rM > m_config.rMaxMiddle) {
        // break because SPs are sorted in r
        break;
      }
    }

    getCompatibleDoublets(options, grid, state.topNeighbours, *spM.get(),
                          state.compatTopSP, m_config.deltaRMinTopSP,
                          m_config.deltaRMaxTopSP, false);

    // no top SP found -> try next spM
    if (state.compatTopSP.empty()) {
      continue;
    }

    // apply cut on the number of top SP if seedConfirmation is true
    SeedFilterState seedFilterState;
    if (m_config.seedConfirmation) {
      // check if middle SP is in the central or forward region
      SeedConfirmationRangeConfig seedConfRange =
          (zM > m_config.centralSeedConfirmationRange.zMaxSeedConf ||
           zM < m_config.centralSeedConfirmationRange.zMinSeedConf)
              ? m_config.forwardSeedConfirmationRange
              : m_config.centralSeedConfirmationRange;
      // set the minimum number of top SP depending on whether the middle SP is
      // in the central or forward region
      seedFilterState.nTopSeedConf = rM > seedConfRange.rMaxSeedConf
                                         ? seedConfRange.nTopForLargeR
                                         : seedConfRange.nTopForSmallR;
      // set max bottom radius for seed confirmation
      seedFilterState.rMaxSeedConf = seedConfRange.rMaxSeedConf;
      // continue if number of top SPs is smaller than minimum
      if (state.compatTopSP.size() < seedFilterState.nTopSeedConf) {
        continue;
      }
    }

    getCompatibleDoublets(options, grid, state.bottomNeighbours, *spM.get(),
                          state.compatBottomSP, m_config.deltaRMinBottomSP,
                          m_config.deltaRMaxBottomSP, true);

    // no bottom SP found -> try next spM
    if (state.compatBottomSP.empty()) {
      continue;
    }

    // filter candidates
    filterCandidates(*spM.get(), options, seedFilterState, state);

    m_config.seedFilter->filterSeeds_1SpFixed(
        state.candidates_collector, seedFilterState.numQualitySeeds, outIt);

  }  // loop on mediums
}

template <typename external_spacepoint_t, typename platform_t>
template <typename out_range_t>
inline void
SeedFinder<external_spacepoint_t, platform_t>::getCompatibleDoublets(
    const Acts::SeedFinderOptions& options,
    Acts::SpacePointGrid<external_spacepoint_t>& grid,
    boost::container::small_vector<Neighbour<external_spacepoint_t>, 9>&
        otherSPsNeighbours,
    const InternalSpacePoint<external_spacepoint_t>& mediumSP,
    out_range_t& outVec, const float& deltaRMinSP, const float& deltaRMaxSP,
    bool isBottom) const {
  const int sign = isBottom ? -1 : 1;

  outVec.clear();

  const float& rM = mediumSP.radius();
  const float& xM = mediumSP.x();
  const float& yM = mediumSP.y();
  const float& zM = mediumSP.z();
  const float ratio_xM_rM = xM / rM;
  const float ratio_yM_rM = yM / rM;

  for (auto& otherSPCol : otherSPsNeighbours) {
    auto& otherSPs = grid.at(otherSPCol.index);
    if (otherSPs.size() == 0) {
      continue;
    }

    /// we make a copy of the iterator here since we need it to remain
    /// the same in the Neighbour object
    auto min_itr = otherSPCol.itr;
    bool found = false;

    for (; min_itr != otherSPs.end(); ++min_itr) {
      auto& otherSP = *min_itr;
      const float rO = otherSP->radius();
      float deltaR = sign * (rO - rM);

      if (deltaR < deltaRMinSP) {
        if (isBottom) {
          break;
        }
        continue;
      }

      // if r-distance is too big, try next SP in bin
      if (deltaR > deltaRMaxSP) {
        if (not isBottom) {
          break;
        }
        continue;
      }

      /// We update the iterator in the Neighbout object
      /// that mean that we have changed the middle space point
      /// and the lower bound has moved accordingly
      if (not found) {
        found = true;
        otherSPCol.itr = min_itr;
      }

      const float zO = otherSP->z();
      float deltaZ = sign * (zO - zM);
      if (deltaZ > m_config.deltaZMax or deltaZ < -m_config.deltaZMax) {
        continue;
      }

      // ratio Z/R (forward angle) of space point duplet
      float cotTheta = deltaZ / deltaR;
      if (cotTheta > m_config.cotThetaMax or cotTheta < -m_config.cotThetaMax) {
        continue;
      }

      // check if duplet origin on z axis within collision region
      float zOrigin = zM - rM * cotTheta;
      if (zOrigin < m_config.collisionRegionMin ||
          zOrigin > m_config.collisionRegionMax) {
        continue;
      }

      if (not m_config.interactionPointCut) {
        outVec.push_back(otherSP.get());
        continue;
      }

      const float xVal =
          (otherSP->x() - xM) * ratio_xM_rM + (otherSP->y() - yM) * ratio_yM_rM;
      const float yVal =
          (otherSP->y() - yM) * ratio_xM_rM - (otherSP->x() - xM) * ratio_yM_rM;

      if (std::abs(rM * yVal) <= sign * m_config.impactMax * xVal) {
        outVec.push_back(otherSP.get());
        continue;
      }

      // conformal transformation u=x/(x²+y²) v=y/(x²+y²) transform the
      // circle into straight lines in the u/v plane the line equation can
      // be described in terms of aCoef and bCoef, where v = aCoef * u +
      // bCoef
      const float uT = xVal / (xVal * xVal + yVal * yVal);
      const float vT = yVal / (xVal * xVal + yVal * yVal);
      // in the rotated frame the interaction point is positioned at x = -rM
      // and y ~= impactParam
      const float uIP = -1. / rM;
      float vIP = m_config.impactMax / (rM * rM);
      if (sign * yVal > 0.) {
        vIP = -vIP;
      }
      // we can obtain aCoef as the slope dv/du of the linear function,
      // estimated using du and dv between the two SP bCoef is obtained by
      // inserting aCoef into the linear equation
      const float aCoef = (vT - vIP) / (uT - uIP);
      const float bCoef = vIP - aCoef * uIP;
      // the distance of the straight line from the origin (radius of the
      // circle) is related to aCoef and bCoef by d^2 = bCoef^2 / (1 +
      // aCoef^2) = 1 / (radius^2) and we can apply the cut on the curvature
      if ((bCoef * bCoef) * options.minHelixDiameter2 > (1 + aCoef * aCoef)) {
        continue;
      }
      outVec.push_back(otherSP.get());
    }
  }
}

template <typename external_spacepoint_t, typename platform_t>
inline void SeedFinder<external_spacepoint_t, platform_t>::filterCandidates(
    InternalSpacePoint<external_spacepoint_t>& spM,
    const Acts::SeedFinderOptions& options, SeedFilterState& seedFilterState,
    SeedingState& state) const {
  float rM = spM.radius();
  float varianceRM = spM.varianceR();
  float varianceZM = spM.varianceZ();

  state.linCircleBottom.clear();
  state.linCircleTop.clear();

  std::size_t numTopSP = state.compatTopSP.size();

  transformCoordinates(state.compatBottomSP, spM, true, state.linCircleBottom);
  transformCoordinates(state.compatTopSP, spM, false, state.linCircleTop);

  // sort: make index vector
  std::vector<std::size_t> sorted_bottoms(state.linCircleBottom.size());
  for (std::size_t i(0); i < sorted_bottoms.size(); ++i) {
    sorted_bottoms[i] = i;
  }

  std::vector<std::size_t> sorted_tops(state.linCircleTop.size());
  for (std::size_t i(0); i < sorted_tops.size(); ++i) {
    sorted_tops[i] = i;
  }

  std::sort(sorted_bottoms.begin(), sorted_bottoms.end(),
            [&state](const std::size_t& a, const std::size_t& b) -> bool {
              return state.linCircleBottom[a].cotTheta <
                     state.linCircleBottom[b].cotTheta;
            });

  std::sort(sorted_tops.begin(), sorted_tops.end(),
            [&state](const std::size_t& a, const std::size_t& b) -> bool {
              return state.linCircleTop[a].cotTheta <
                     state.linCircleTop[b].cotTheta;
            });

  // Reserve enough space, in case current capacity is too little
  state.topSpVec.reserve(numTopSP);
  state.curvatures.reserve(numTopSP);
  state.impactParameters.reserve(numTopSP);

  size_t t0 = 0;

  // clear previous results and then loop on bottoms and tops
  state.candidates_collector.clear();

  for (const std::size_t& b : sorted_bottoms) {
    // break if we reached the last top SP
    if (t0 == numTopSP) {
      break;
    }

    auto lb = state.linCircleBottom[b];
    seedFilterState.zOrigin = lb.Zo;
    float cotThetaB = lb.cotTheta;
    float Vb = lb.V;
    float Ub = lb.U;
    float ErB = lb.Er;
    float iDeltaRB = lb.iDeltaR;

    // 1+(cot^2(theta)) = 1/sin^2(theta)
    float iSinTheta2 = (1. + cotThetaB * cotThetaB);
    float sigmaSquaredSPtDependent = iSinTheta2 * options.sigmapT2perRadius;
    // calculate max scattering for min momentum at the seed's theta angle
    // scaling scatteringAngle^2 by sin^2(theta) to convert pT^2 to p^2
    // accurate would be taking 1/atan(thetaBottom)-1/atan(thetaTop) <
    // scattering
    // but to avoid trig functions we approximate cot by scaling by
    // 1/sin^4(theta)
    // resolving with pT to p scaling --> only divide by sin^2(theta)
    // max approximation error for allowed scattering angles of 0.04 rad at
    // eta=infinity: ~8.5%
    float scatteringInRegion2 = m_config.maxScatteringAngle2 * iSinTheta2;
    // multiply the squared sigma onto the squared scattering
    scatteringInRegion2 *= m_config.sigmaScattering * m_config.sigmaScattering;

    float sinTheta = 1 / std::sqrt(iSinTheta2);
    float cosTheta = cotThetaB * sinTheta;

    // clear all vectors used in each inner for loop
    state.topSpVec.clear();
    state.curvatures.clear();
    state.impactParameters.clear();

    // coordinate transformation and checks for middle spacepoint
    // x and y terms for the rotation from UV to XY plane
    float rotationTermsUVtoXY[2] = {0, 0};
    if (m_config.useDetailedDoubleMeasurementInfo) {
      rotationTermsUVtoXY[0] = spM.x() * sinTheta / spM.radius();
      rotationTermsUVtoXY[1] = spM.y() * sinTheta / spM.radius();
    }

    // minimum number of compatible top SPs to trigger the filter for a certain
    // middle bottom pair if seedConfirmation is false we always ask for at
    // least one compatible top to trigger the filter
    size_t minCompatibleTopSPs = 2;
    if (!m_config.seedConfirmation or
        state.compatBottomSP[b]->radius() > seedFilterState.rMaxSeedConf) {
      minCompatibleTopSPs = 1;
    }
    if (m_config.seedConfirmation and seedFilterState.numQualitySeeds) {
      minCompatibleTopSPs++;
    }

    for (size_t index_t = t0; index_t < numTopSP; index_t++) {
      const std::size_t& t = sorted_tops[index_t];

      auto lt = state.linCircleTop[t];

      float cotThetaT = lt.cotTheta;
      float rMxy = 0.;
      float ub = 0.;
      float vb = 0.;
      float ut = 0.;
      float vt = 0.;

      if (m_config.useDetailedDoubleMeasurementInfo) {
        // protects against division by 0
        float dU = lt.U - Ub;
        if (dU == 0.) {
          continue;
        }
        // A and B are evaluated as a function of the circumference parameters
        // x_0 and y_0
        float A0 = (lt.V - Vb) / dU;

        // position of Middle SP converted from UV to XY assuming cotTheta
        // evaluated from the Bottom and Middle SPs double
        double positionMiddle[3] = {
            rotationTermsUVtoXY[0] - rotationTermsUVtoXY[1] * A0,
            rotationTermsUVtoXY[0] * A0 + rotationTermsUVtoXY[1],
            cosTheta * std::sqrt(1 + A0 * A0)};

        double rMTransf[3];
        if (!xyzCoordinateCheck(m_config, spM, positionMiddle, rMTransf)) {
          continue;
        }

        // coordinate transformation and checks for bottom spacepoint
        float B0 = 2. * (Vb - A0 * Ub);
        float Cb = 1. - B0 * lb.y;
        float Sb = A0 + B0 * lb.x;
        double positionBottom[3] = {
            rotationTermsUVtoXY[0] * Cb - rotationTermsUVtoXY[1] * Sb,
            rotationTermsUVtoXY[0] * Sb + rotationTermsUVtoXY[1] * Cb,
            cosTheta * std::sqrt(1 + A0 * A0)};

        auto spB = state.compatBottomSP[b];
        double rBTransf[3];
        if (!xyzCoordinateCheck(m_config, *spB, positionBottom, rBTransf)) {
          continue;
        }

        // coordinate transformation and checks for top spacepoint
        float Ct = 1. - B0 * lt.y;
        float St = A0 + B0 * lt.x;
        double positionTop[3] = {
            rotationTermsUVtoXY[0] * Ct - rotationTermsUVtoXY[1] * St,
            rotationTermsUVtoXY[0] * St + rotationTermsUVtoXY[1] * Ct,
            cosTheta * std::sqrt(1 + A0 * A0)};

        auto spT = state.compatTopSP[t];
        double rTTransf[3];
        if (!xyzCoordinateCheck(m_config, *spT, positionTop, rTTransf)) {
          continue;
        }

        // bottom and top coordinates in the spM reference frame
        float xB = rBTransf[0] - rMTransf[0];
        float yB = rBTransf[1] - rMTransf[1];
        float zB = rBTransf[2] - rMTransf[2];
        float xT = rTTransf[0] - rMTransf[0];
        float yT = rTTransf[1] - rMTransf[1];
        float zT = rTTransf[2] - rMTransf[2];

        float iDeltaRB2 = 1. / (xB * xB + yB * yB);
        float iDeltaRT2 = 1. / (xT * xT + yT * yT);

        cotThetaB = -zB * std::sqrt(iDeltaRB2);
        cotThetaT = zT * std::sqrt(iDeltaRT2);

        rMxy = std::sqrt(rMTransf[0] * rMTransf[0] + rMTransf[1] * rMTransf[1]);
        float Ax = rMTransf[0] / rMxy;
        float Ay = rMTransf[1] / rMxy;

        ub = (xB * Ax + yB * Ay) * iDeltaRB2;
        vb = (yB * Ax - xB * Ay) * iDeltaRB2;
        ut = (xT * Ax + yT * Ay) * iDeltaRT2;
        vt = (yT * Ax - xT * Ay) * iDeltaRT2;
      }

      // use geometric average
      float cotThetaAvg2 = cotThetaB * cotThetaT;
      if (m_config.arithmeticAverageCotTheta) {
        // use arithmetic average
        float averageCotTheta = 0.5 * (cotThetaB + cotThetaT);
        cotThetaAvg2 = averageCotTheta * averageCotTheta;
      } else if (cotThetaAvg2 <= 0) {
        continue;
      }

      // add errors of spB-spM and spM-spT pairs and add the correlation term
      // for errors on spM
      float error2 =
          lt.Er + ErB +
          2 * (cotThetaAvg2 * varianceRM + varianceZM) * iDeltaRB * lt.iDeltaR;

      float deltaCotTheta = cotThetaB - cotThetaT;
      float deltaCotTheta2 = deltaCotTheta * deltaCotTheta;
      // Apply a cut on the compatibility between the r-z slope of the two
      // seed segments. This is done by comparing the squared difference
      // between slopes, and comparing to the squared uncertainty in this
      // difference - we keep a seed if the difference is compatible within
      // the assumed uncertainties. The uncertainties get contribution from
      // the  space-point-related squared error (error2) and a scattering term
      // calculated assuming the minimum pt we expect to reconstruct
      // (scatteringInRegion2). This assumes gaussian error propagation which
      // allows just adding the two errors if they are uncorrelated (which is
      // fair for scattering and measurement uncertainties)
      if (deltaCotTheta2 > (error2 + scatteringInRegion2)) {
        // skip top SPs based on cotTheta sorting when producing triplets
        if (not m_config.skipPreviousTopSP) {
          continue;
        }
        // break if cotTheta from bottom SP < cotTheta from top SP because
        // the SP are sorted by cotTheta
        if (cotThetaB - cotThetaT < 0) {
          break;
        }
        t0 = index_t + 1;
        continue;
      }

      float dU = 0;
      float A = 0;
      float S2 = 0;
      float B = 0;
      float B2 = 0;

      if (m_config.useDetailedDoubleMeasurementInfo) {
        dU = ut - ub;
        // protects against division by 0
        if (dU == 0.) {
          continue;
        }
        A = (vt - vb) / dU;
        S2 = 1. + A * A;
        B = vb - A * ub;
        B2 = B * B;
      } else {
        dU = lt.U - Ub;
        // protects against division by 0
        if (dU == 0.) {
          continue;
        }
        // A and B are evaluated as a function of the circumference parameters
        // x_0 and y_0
        A = (lt.V - Vb) / dU;
        S2 = 1. + A * A;
        B = Vb - A * Ub;
        B2 = B * B;
      }

      // sqrt(S2)/B = 2 * helixradius
      // calculated radius must not be smaller than minimum radius

      if (S2 < B2 * options.minHelixDiameter2) {
        continue;
      }

      // refinement of the cut on the compatibility between the r-z slope of
      // the two seed segments using a scattering term scaled by the actual
      // measured pT (p2scatterSigma)
      float iHelixDiameter2 = B2 / S2;
      // convert p(T) to p scaling by sin^2(theta) AND scale by 1/sin^4(theta)
      // from rad to deltaCotTheta
      float p2scatterSigma = iHelixDiameter2 * sigmaSquaredSPtDependent;
      if (!std::isinf(m_config.maxPtScattering)) {
        // if pT > maxPtScattering, calculate allowed scattering angle using
        // maxPtScattering instead of pt.
        float pT = options.pTPerHelixRadius * std::sqrt(S2 / B2) / 2.;
        if (pT > m_config.maxPtScattering) {
          float pTscatterSigma =
              (m_config.highland / m_config.maxPtScattering) *
              m_config.sigmaScattering;
          p2scatterSigma = pTscatterSigma * pTscatterSigma * iSinTheta2;
        }
      }

      // if deltaTheta larger than allowed scattering for calculated pT, skip
      if (deltaCotTheta2 > (error2 + p2scatterSigma)) {
        if (not m_config.skipPreviousTopSP) {
          continue;
        }
        if (cotThetaB - cotThetaT < 0) {
          break;
        }
        t0 = index_t;
        continue;
      }
      // A and B allow calculation of impact params in U/V plane with linear
      // function
      // (in contrast to having to solve a quadratic function in x/y plane)
      float Im = m_config.useDetailedDoubleMeasurementInfo
                     ? std::abs((A - B * rMxy) * rMxy)
                     : std::abs((A - B * rM) * rM);

      if (Im > m_config.impactMax) {
        continue;
      }

      state.topSpVec.push_back(state.compatTopSP[t]);
      // inverse diameter is signed depending if the curvature is
      // positive/negative in phi
      state.curvatures.push_back(B / std::sqrt(S2));
      state.impactParameters.push_back(Im);
    }  // loop on tops

    // continue if number of top SPs is smaller than minimum required for filter
    if (state.topSpVec.size() < minCompatibleTopSPs) {
      continue;
    }

    m_config.seedFilter->filterSeeds_2SpFixed(
        *state.compatBottomSP[b], spM, state.topSpVec, state.curvatures,
        state.impactParameters, seedFilterState, state.candidates_collector);
  }  // loop on bottoms
}

template <typename external_spacepoint_t, typename platform_t>
template <typename sp_range_t>
std::vector<Seed<external_spacepoint_t>>
SeedFinder<external_spacepoint_t, platform_t>::createSeedsForGroup(
    const Acts::SeedFinderOptions& options,
    Acts::SpacePointGrid<external_spacepoint_t>& grid,
    const sp_range_t& bottomSPs, const std::size_t middleSPs,
    const sp_range_t& topSPs) const {
  SeedingState state;
  const Acts::Range1D<float> rMiddleSPRange;
  std::vector<Seed<external_spacepoint_t>> ret;

  createSeedsForGroup(options, state, grid, std::back_inserter(ret), bottomSPs,
                      middleSPs, topSPs, rMiddleSPRange);

  return ret;
}
}  // namespace Acts
