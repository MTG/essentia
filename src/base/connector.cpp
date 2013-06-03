/*
 * Copyright (C) 2006-2010 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "connector.h"
#include "streamingalgorithm.h"
using namespace std;

namespace essentia {
namespace streaming {


string Connector::parentName() const {
  if (_parent) return _parent->name();
  return "<NoParent>";
}

string Connector::fullName() const {
  ostringstream fullname;
  fullname << parentName() << "::" << name();

  return fullname.str();
}



} // namespace streaming
} // namespace essentia

