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

#ifndef ESSENTIA_CONFIGURABLE_H
#define ESSENTIA_CONFIGURABLE_H

#include "parameter.h"

namespace essentia {

/**
 * A Configurable instance is an object that has a given name, and can be
 * configured using a certain number of @c Parameters. These parameters have
 * to be declared beforehand using the @c declareParameters method.
 *
 * Whenever a Configurable instance gets reconfigured with new parameters,
 * it will first save them internally and then call the @c configure() method.
 * You should reimplement this method and do anything necessary for your object
 * to be up-to-date and synchronized with the new parameters. These are
 * accessible using the @c parameter() method.
 */
class ESSENTIA_API Configurable {

 public:

  // NB: virtual destructor needed because of virtual methods.
  virtual ~Configurable() {}

  /**
   * Return the name of this Configurable.
   */
  const std::string& name() const { return _name; }

  /**
   * Set the name for this Configurable.
   */
  void setName(const std::string& name) { _name = name; }


  /**
   * Declare the parameters that this @c Configurable can accept.
   * You have to implement this method in derived classes, even though you don't
   * need any parameters. In that case, just define it as empty.
   *
   * In this method you should only be calling the @c declareParameter method,
   * once for each parameter, with optional default values.
   */
  virtual void declareParameters() = 0;

  /**
   * Set the given parameters as the current ones. Parameters which are not
   * redefined will keep their old values, while this method will throw an
   * @c EssentiaException if passing it an unknown parameter (i.e.: not declared
   * using @c declareParameters() ).
   * As a general rule, it is better to use the configure(const ParameterMap&)
   * method, but in certain cases you may want to set parameters _without_
   * reconfiguring the object.
   */
  virtual void setParameters(const ParameterMap& params);

  /**
   * Set the given parameters as the current ones and reconfigure the object.
   * @see setParameters
   */
  virtual void configure(const ParameterMap& params);

  /**
   * This function will be automatically called after some parameters have been
   * set. This is the place where you should write your specific code which
   * needs to be called when configuring this object.
   *
   * You can access the newly set parameters using the @c parameter() method.
   */
  virtual void configure() {}


  /**
   * Return a map filled with the parameters that have been declared, along with
   * their default value if defined.
   */
  const ParameterMap& defaultParameters() const {
    return _defaultParams;
  }


  /**
   * Returns the parameter corresponding to the given name.
   */
  const Parameter& parameter(const std::string& key) const { return _params[key]; }

 protected:

  /**
   * Use this method to declare the list of parameters that this algorithm in
   * the @c declareParameters() method.
   * If a Parameter doesn't have a default value, use its type instead, e.g.:
   * Parameter::STRING, or Parameter::VECTOR_REAL, etc...
   */
  void declareParameter(const std::string& name, const std::string& desc,
                        const std::string& range,
                        const Parameter& defaultValue);


 public:

  // make doxygen skip all the following macros...
  /// @cond

  // These are conveniency functions, for configuring an algo with fewer
  // characters per line
  // WARNING: when declaring the function, you HAVE to omit the P(1) (because it
  //          is already included in the CONFIGURE macro (for comma ',' reasons)
#define CONFIGURE void configure(const std::string& name1, const Parameter& value1
#define P(n) , const std::string& name##n, const Parameter& value##n
#define AP(n) params.add(name##n, value##n);
#define PBEG ) { ParameterMap params;
#define PEND configure(params); }

  CONFIGURE PBEG AP(1) PEND
  CONFIGURE P(2) PBEG AP(1) AP(2) PEND
  CONFIGURE P(2) P(3) PBEG AP(1) AP(2) AP(3) PEND
  CONFIGURE P(2) P(3) P(4) PBEG AP(1) AP(2) AP(3) AP(4) PEND
  CONFIGURE P(2) P(3) P(4) P(5) PBEG AP(1) AP(2) AP(3) AP(4) AP(5) PEND
  CONFIGURE P(2) P(3) P(4) P(5) P(6) PBEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) PEND
  CONFIGURE P(2) P(3) P(4) P(5) P(6) P(7) PBEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) AP(7) PEND
  CONFIGURE P(2) P(3) P(4) P(5) P(6) P(7) P(8)
    PBEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) AP(7) AP(8) PEND
  CONFIGURE P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9)
    PBEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) AP(7) AP(8) AP(9) PEND
  CONFIGURE P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9) P(10)
    PBEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) AP(7) AP(8) AP(9) AP(10) PEND
  CONFIGURE P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9) P(10) P(11)
    PBEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) AP(7) AP(8) AP(9) AP(10) AP(11) PEND
  CONFIGURE P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9) P(10) P(11) P(12)
    PBEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) AP(7) AP(8) AP(9) AP(10) AP(11) AP(12) PEND
  CONFIGURE P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9) P(10) P(11) P(12) P(13)
    PBEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) AP(7) AP(8) AP(9) AP(10) AP(11) AP(12) AP(13) PEND
  CONFIGURE P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9) P(10) P(11) P(12) P(13) P(14)
    PBEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) AP(7) AP(8) AP(9) AP(10) AP(11) AP(12) AP(13) AP(14) PEND
  CONFIGURE P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9) P(10) P(11) P(12) P(13) P(14) P(15)
    PBEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) AP(7) AP(8) AP(9) AP(10) AP(11) AP(12) AP(13) AP(14) AP(15) PEND
  CONFIGURE P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9) P(10) P(11) P(12) P(13) P(14) P(15) P(16)
    PBEG AP(1) AP(2) AP(3) AP(4) AP(5) AP(6) AP(7) AP(8) AP(9) AP(10) AP(11) AP(12) AP(13) AP(14) AP(15) AP(16) PEND


#undef PEND
#undef PBEG
#undef AP
#undef P
#undef CONFIGURE

  /// @endcond

 protected:
  std::string _name;
  ParameterMap _params;
  ParameterMap _defaultParams;

 public:
  DescriptionMap parameterDescription;
  DescriptionMap parameterRange;

};

// macro which is little useful, but allows to write configure() in a very clean
// and understandable way for StreamingAlgorithmComposite
#define INHERIT(x) x, parameter(x)

template <typename T> bool compareByName(const T* a, const T* b) {
  return a->name() < b->name();
}

} // namespace essentia

#endif // ESSENTIA_CONFIGURABLE_H
