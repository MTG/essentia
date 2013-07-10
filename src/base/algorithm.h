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

#ifndef ESSENTIA_ALGORITHM_H
#define ESSENTIA_ALGORITHM_H

#include "types.h"
#include "configurable.h"
#include "iotypewrappers.h"


namespace essentia {
namespace standard {

class ESSENTIA_API Algorithm : public Configurable {

 public:
  static const std::string processingMode;

  typedef OrderedMap<InputBase> InputMap;
  typedef OrderedMap<OutputBase> OutputMap;

  DescriptionMap inputDescription;
  DescriptionMap outputDescription;


 public:
  /**
   * Empty virtual destructor, needed because we have some virtual functions.
   */
  virtual ~Algorithm() {}

  const InputMap& inputs() const { return _inputs; }
  const OutputMap& outputs() const { return _outputs; }

  /**
   * Return the input wrapper associated with the given name.
   */
  InputBase& input(const std::string& name);

  /**
   * Return the output wrapper associated with the given name.
   */
  OutputBase& output(const std::string& name);

  /**
   * Return the names of all the inputs that have been defined for this object.
   */
  std::vector<std::string> inputNames() const { return _inputs.keys(); }

  /**
   * Return the names of all the outputs that have been defined for this object.
   */
  std::vector<std::string> outputNames() const { return _outputs.keys(); }

  /**
   * Do the actual computation once that everything is set and configured.
   * The recommended use for this function is to first get the inputs and
   * outputs into local ref variables (const for the inputs) and then do the
   * processing.
   * This allow you also to write a "classic" function call with parameters
   * which you would just wrap with the parameterless function.
   */
  virtual void compute() = 0;

  /**
   * This function will be called when doing batch computations between each
   * file that is processed. That is, if your algorithm is some sort of state
   * machine, it allows you to reset it to its original state to process
   * another file without having to delete and reinstantiate it.
   */
  virtual void reset() {}


  // methods for having access to the types of the inputs/outputs
  std::vector<const std::type_info*> inputTypes() const;
  std::vector<const std::type_info*> outputTypes() const;


 protected:

  void declareInput(InputBase& input, const std::string& name, const std::string& desc);
  void declareOutput(OutputBase& output, const std::string& name, const std::string& desc);

  InputMap _inputs;
  OutputMap _outputs;

};

} // namespace standard
} // namespace essentia

#include "iotypewrappers_impl.h"

#endif // ESSENTIA_ALGORITHM_H
