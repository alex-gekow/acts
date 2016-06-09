// This file is part of the ACTS project.
//
// Copyright (C) 2016 ACTS project team
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

///////////////////////////////////////////////////////////////////
// IDetExtension.h, ACTS project, DD4hepDetector plugin
///////////////////////////////////////////////////////////////////

#ifndef ACTS_DD4HEPDETECTORELEMENT_IDETEXTENSION_H
#define ACTS_DD4HEPDETECTORELEMENT_IDETEXTENSION_H 1

// Algebra
#include "ACTS/Utilities/Definitions.hpp"

namespace DD4hep {
namespace Geometry {
  class DetElement;
  class Segmentation;
}
}

namespace Acts {

/** @class IDetExtension

 Interface class for making extensions to the DD4hep::DetElement class, needed
 for the translation from the DD4hep geometry into the tracking geometry of the
 ATS package.
 In this way, the segmentation of the sensitive detector elements can be
 directly accessed from DD4hep to ensure consistency between the full and the
 tracking geometry.
 Since in DD4hep volumes used as a cylinder (detector layers are binned in r and
 z, e.g. central barrel volume) and discs (detector layers are binned in r and
 phi, e.g. end caps) are both described as a ROOT TGeoConeSeg one needs to
 distinguish between these volume types by setting the shape.
 @TODO find replacement for Gaudi exeption and message stream

 */

class Module;

enum ShapeType {

  None     = 0,
  Cylinder = 1,
  Disc     = 2
};

class IDetExtension
{
public:
  /* virtual destructor **/
  virtual ~IDetExtension() {}
  /* hand over shape **/
  virtual void
  setShape(ShapeType shape)
      = 0;
  /* possibility to hand over shape of a volume **/
  virtual ShapeType
  shape() const = 0;
  /* method to hand over the DD4hep segmentation **/
  virtual void
  setSegmentation(const DD4hep::Geometry::Segmentation segmentation)
      = 0;
  /* access segmentation **/
  virtual const DD4hep::Geometry::Segmentation
  segmentation() const = 0;
  /* possibility to hand over supporting structure of a layer*/
  virtual void
  setSupportStructure(const DD4hep::Geometry::DetElement support)
      = 0;
  /* access supporting structure */
  virtual const DD4hep::Geometry::DetElement&
  supportStructure() const = 0;
  /* possibility to set contained sensitive DetectorModules by a layer*/
  virtual void
  setModules(std::vector<DD4hep::Geometry::DetElement> mod)
      = 0;
  /* access modules */
  virtual std::vector<DD4hep::Geometry::DetElement>
  modules() const = 0;

protected:
  /* protected constructor **/
  IDetExtension() {}
};
}

#endif  // ACTS_DD4HEPDETECTORELEMENT_DET_IDETEXTENSION_H