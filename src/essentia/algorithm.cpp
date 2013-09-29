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

#include "algorithm.h"
#include "algorithmfactory.h"
using namespace std;

namespace essentia {
namespace standard {

const string Algorithm::processingMode = "Standard ";

InputBase& Algorithm::input(const string& name) {
  try {
    return *_inputs[name];
  }
  catch (EssentiaException&) {
    ostringstream msg;
    msg << "Couldn't find '" << name << "' in " << this->name() << "::inputs.";
    msg << " Available input names are: " << _inputs.keys();
    throw EssentiaException(msg);
  }
}

OutputBase& Algorithm::output(const string& name) {
  try {
    return *_outputs[name];
  }
  catch (EssentiaException&) {
    ostringstream msg;
    msg << "Couldn't find '" << name << "' in " << this->name() << "::outputs.";
    msg << " Available output names are: " << _outputs.keys();
    throw EssentiaException(msg);
  }
}


vector<const type_info*> Algorithm::inputTypes() const {
  vector<const type_info*> types;
  types.reserve(_inputs.size());
  for (InputMap::const_iterator it = _inputs.begin(); it != _inputs.end(); ++it) {
    types.push_back(&it->second->typeInfo());
  }
  return types;
}

vector<const type_info*> Algorithm::outputTypes() const {
  vector<const type_info*> types;
  types.reserve(_outputs.size());
  for (OutputMap::const_iterator it = _outputs.begin(); it != _outputs.end(); ++it) {
    types.push_back(&it->second->typeInfo());
  }
  return types;
}


void Algorithm::declareInput(InputBase& input, const string& name,
                             const string& desc) {
  input._parent = this;
  input._name = name;
  _inputs.insert(name, &input);
  inputDescription.insert(name, desc);
}

void Algorithm::declareOutput(OutputBase& output, const string& name,
                              const string& desc) {
  output._parent = this;
  output._name = name;
  _outputs.insert(name, &output);
  outputDescription.insert(name, desc);
}

} // namespace standard
} // namespace essentia
