/*
 * Copyright (C) 2006-2010 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
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
