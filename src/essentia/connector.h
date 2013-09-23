/*
 * Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
 *
 * This file is part of Essentia
 *
 * Essentia is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License as published by the Free
 * Software Foundation (FSF), either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the Affero GNU General Public License
 * version 3 along with this program.  If not, see http://www.gnu.org/licenses/
 */

#ifndef ESSENTIA_CONNECTOR_H
#define ESSENTIA_CONNECTOR_H

#include "types.h"
#include "streamconnector.h"

namespace essentia {
namespace streaming {

class SinkBase;
class Algorithm;


/**
 * This is the base class for connectors in Essentia.
 * It is the highest-level class which is shared both by Sources and Sinks,
 * and is a slightly better StreamConnector which has a parent Algorithm
 * and is aware (through TypeProxy) of which type of data is supposed to flow
 * through it.
 */
class Connector : public TypeProxy, public StreamConnector {
 protected:
  Algorithm* _parent;

 public:
  Connector(Algorithm* parent = 0, const std::string& name = "Unnamed") :
    TypeProxy(name),
    _parent(parent) {}

  Connector(const std::string& name) :
    TypeProxy(name),
    _parent(0) {}

  const Algorithm* parent() const { return _parent; }
  Algorithm* parent() { return _parent; }
  void setParent(Algorithm* parent) { _parent = parent; }

  /**
   * Return parent's name if parent is set, "<NoParent>" otherwise.
   */
  std::string parentName() const;

  /**
   * Return a fully qualified name consisting of:
   * "<Parent name>::<Connector name>".
   */
  std::string fullName() const;

};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_CONNECTOR_H
