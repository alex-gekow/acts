// This file is part of the ACTS project.
//
// Copyright (C) 2016 ACTS project team
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

///////////////////////////////////////////////////////////////////
// AbstractVolume.cpp, ACTS project
///////////////////////////////////////////////////////////////////

// Geometry module
#include "ACTS/Volumes/AbstractVolume.hpp"
#include "ACTS/Surfaces/CylinderSurface.hpp"
#include "ACTS/Surfaces/DiscSurface.hpp"
#include "ACTS/Surfaces/PlaneSurface.hpp"
#include "ACTS/Surfaces/Surface.hpp"
#include "ACTS/Volumes/BoundaryCylinderSurface.hpp"
#include "ACTS/Volumes/BoundaryDiscSurface.hpp"
#include "ACTS/Volumes/BoundaryPlaneSurface.hpp"
#include "ACTS/Volumes/BoundarySurface.hpp"
#include "ACTS/Volumes/VolumeBounds.hpp"
// STD/STL
#include <iostream>

// Default constructor
Acts::AbstractVolume::AbstractVolume() : Volume(), m_boundarySurfaces(nullptr)
{
}

// constructor with Acts::Transform3D
Acts::AbstractVolume::AbstractVolume(Acts::Transform3D*        htrans,
                                     const Acts::VolumeBounds* volbounds)
  : Volume(htrans, volbounds), m_boundarySurfaces(nullptr)
{
  createBoundarySurfaces();
}

// constructor with Acts::Transform3D
Acts::AbstractVolume::AbstractVolume(
    std::shared_ptr<Acts::Transform3D>        htrans,
    std::shared_ptr<const Acts::VolumeBounds> volbounds)
  : Volume(htrans, volbounds), m_boundarySurfaces(nullptr)
{
  createBoundarySurfaces();
}

// destructor
Acts::AbstractVolume::~AbstractVolume()
{
  delete m_boundarySurfaces;
}

// assignment operator
Acts::AbstractVolume&
Acts::AbstractVolume::operator=(const Acts::AbstractVolume& vol)
{
  if (this != &vol) {
    Volume::operator=(vol);
    delete m_boundarySurfaces;
    m_boundarySurfaces = new std::
        vector<std::shared_ptr<const Acts::
                                   BoundarySurface<Acts::AbstractVolume>>>(
            *vol.m_boundarySurfaces);
  }
  return *this;
}

const std::
    vector<std::shared_ptr<const Acts::BoundarySurface<Acts::AbstractVolume>>>&
    Acts::AbstractVolume::boundarySurfaces() const
{
  return (*m_boundarySurfaces);
}

void
Acts::AbstractVolume::createBoundarySurfaces()
{
  // prepare the BoundarySurfaces
  m_boundarySurfaces = new std::
      vector<std::
                 shared_ptr<const Acts::BoundarySurface<Acts::AbstractVolume>>>;
  // transform Surfaces To BoundarySurfaces
  const std::vector<const Acts::Surface*>* surfaces
      = Acts::Volume::volumeBounds().decomposeToSurfaces(m_transform);
  std::vector<const Acts::Surface*>::const_iterator surfIter
      = surfaces->begin();

  // counter to flip the inner/outer position for Cylinders
  int sfCounter = 0;
  int sfNumber  = surfaces->size();

  for (; surfIter != surfaces->end(); ++surfIter) {
    sfCounter++;
    const Acts::PlaneSurface* psf
        = dynamic_cast<const Acts::PlaneSurface*>(*surfIter);
    if (psf) {
      m_boundarySurfaces->push_back(
          std::shared_ptr<const Acts::BoundarySurface<Acts::AbstractVolume>>(
              new Acts::BoundaryPlaneSurface<Acts::AbstractVolume>(
                  this, 0, *psf)));
      delete psf;
      continue;
    }
    const Acts::DiscSurface* dsf
        = dynamic_cast<const Acts::DiscSurface*>(*surfIter);
    if (dsf) {
      m_boundarySurfaces->push_back(
          std::shared_ptr<const Acts::BoundarySurface<Acts::AbstractVolume>>(
              new Acts::BoundaryDiscSurface<Acts::AbstractVolume>(
                  this, 0, *dsf)));
      delete dsf;
      continue;
    }
    const Acts::CylinderSurface* csf
        = dynamic_cast<const Acts::CylinderSurface*>(*surfIter);
    if (csf) {
      Acts::AbstractVolume* inner = (sfCounter == 3 && sfNumber > 3) ? 0 : this;
      Acts::AbstractVolume* outer = (inner) ? 0 : this;
      m_boundarySurfaces->push_back(
          std::shared_ptr<const Acts::BoundarySurface<Acts::AbstractVolume>>(
              new Acts::BoundaryCylinderSurface<Acts::AbstractVolume>(
                  inner, outer, *csf)));
      delete csf;
      continue;
    }
  }

  delete surfaces;
}