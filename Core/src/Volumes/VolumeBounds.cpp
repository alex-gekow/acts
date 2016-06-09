// This file is part of the ACTS project.
//
// Copyright (C) 2016 ACTS project team
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

///////////////////////////////////////////////////////////////////
// VolumeBounds.cpp, ACTS project
///////////////////////////////////////////////////////////////////

#include "ACTS/Volumes/VolumeBounds.hpp"

/**Overload of << operator for std::ostream for debug output*/
std::ostream&
Acts::operator<<(std::ostream& sl, const Acts::VolumeBounds& vb)
{
  return vb.dump(sl);
}