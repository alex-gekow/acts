// This file is part of the Acts project.
//
// Copyright (C) 2022 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

namespace Acts {

struct SpacePointBuilderOptions {
  std::pair<const std::pair<Vector3, Vector3>,
            const std::pair<Vector3, Vector3>>
      stripEndsPair;
};

}  // namespace Acts
