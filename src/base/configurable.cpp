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

#include <memory>
#include "configurable.h"
#include "range.h"
#include "essentiautil.h"
using namespace std;

namespace essentia {

void Configurable::configure(const ParameterMap& params) {
  setParameters(params);
  configure();
}


void Configurable::declareParameter(const string& name, const string& desc,
                                    const string& range,
                                    const Parameter& defaultValue) {
  // do not use addParam here so that in case it already had a value, it
  // doesn't overwrite it
  _params.insert(name, defaultValue);
  _defaultParams.insert(name, defaultValue);
  parameterDescription.insert(name, desc);
  parameterRange.insert(name, range);
}

void Configurable::setParameters(const ParameterMap& params) {

#if !ALLOW_DEFAULT_PARAMETERS

  vector<string> allParams = _defaultParams.keys();
  vector<string> givenParams = params.keys();
  if (givenParams != allParams) {
    ostringstream msg;
    msg << "Trying to configure algorithm '" << _name
        << "' with parameters: " << givenParams
        << "\nbut you need to define the following ones: " << allParams;
    throw EssentiaException(msg);
  }

#endif // !ALLOW_DEFAULT_PARAMETERS


  // merge the parameters from the new map into the existing one, so that we
  // can only specify those parameters that change without overwriting the
  // ones that are already in there and are not in the new map
  for (ParameterMap::const_iterator it = params.begin(); it != params.end(); ++it) {
    const string& name = it->first;
    Parameter value = it->second;

    // throw an exception if a parameter is not recognized by the algorithm
    if (!contains(_params, name)) {
      ostringstream msg;
      msg << "Trying to configure algorithm '" << _name
          << "' with parameter '" << name
          << "' but it only accepts the following ones: " << _params.keys();
      throw EssentiaException(msg);
    }

    Parameter::ParamType valueType = value.type();
    Parameter::ParamType definedType = _params.find(name)->second.type();

    // throw an exception if the parameter exists, but is of the wrong type
    // before throwing, try to see if there is no possible coercion into a type
    // of greater precision (ie: int -> float, ...)
    if (valueType != definedType) {

      // make special case that ints can be configured to Reals
      if (definedType == Parameter::REAL &&
          valueType == Parameter::INT) {
        value = Parameter(value.toReal());
      }
      // make special case that reals can be configured to ints
      else if (definedType == Parameter::INT &&
               valueType == Parameter::REAL) {
        ostringstream msg;
        msg << "Warning: Trying to configure algorithm '" << _name
            << "' 's parameter '" << name << "'"
            << " with a parameter of type '" << valueType << "'"
            << " but the required parameter type is '" << definedType
            << ". Losing resolution while truncating to integer.";
        E_WARNING(msg.str());
        value = Parameter(value.toInt());
      }
      else {
        ostringstream msg;
        msg << "Trying to configure algorithm '" << _name
            << "' 's parameter '" << name << "'"
            << " with a parameter of type '" << valueType << "'"
            << " but the required parameter type is '" << definedType << "'";
        throw EssentiaException(msg);
      }
    }

    // check that the parameter fits in its valid range, if specified
    const string& srange = parameterRange[name];
    auto_ptr<Range> r(Range::create(srange));

    if (!r->contains(value)) {
      ostringstream msg;
      msg << "Parameter " << name << " = " << value << " is not within specified range: " << srange;
      throw EssentiaException(msg);
    }

    // otherwise, just set the new value
    _params.add(name, value);
  }
}

} // namespace essentia
