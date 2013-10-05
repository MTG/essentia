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

#include "streamingalgorithmwrapper.h"
#include "algorithmfactory.h"


using namespace std;


namespace essentia {
namespace streaming {


StreamingAlgorithmWrapper::~StreamingAlgorithmWrapper() {
  delete _algorithm;
  _algorithm = 0;
}


void StreamingAlgorithmWrapper::synchronizeIO() {
  for (InputMap::const_iterator input = _inputs.begin(); input!=_inputs.end(); ++input) {
    synchronizeInput(input->first);
  }

  for (OutputMap::const_iterator output = _outputs.begin(); output!=_outputs.end(); ++output) {
    synchronizeOutput(output->first);
  }
}

void StreamingAlgorithmWrapper::declareAlgorithm(const std::string& name) {
  _algorithm = standard::AlgorithmFactory::create(name);
  _name = name;
}

void StreamingAlgorithmWrapper::declareInput(SinkBase& sink, NumeralType type, const std::string& name) {
  declareInput(sink, type, 1, name);
}

void StreamingAlgorithmWrapper::declareInput(SinkBase& sink, NumeralType type,
                                             int n, const std::string& name) {
  if (!_algorithm) {
    throw EssentiaException("When defining a StreamingAlgorithmWrapper, you need to call declareAlgorithm before any declareInput/Output function.");
  }

  if ( (!_inputType.empty() && _inputType.begin()->second != type) ||
       (!_outputType.empty() && _outputType.begin()->second != type) ) {
    throw EssentiaException("StreamingAlgorithmWrapper::declareInput: all inputs and outputs must have the same NumeralType (", _algorithm->name()+":"+name, ")");
  }

  if (type == TOKEN && n != 1) {
    throw EssentiaException("StreamingAlgorithmWrapper::declareInput: when using the TOKEN NumeralType, only a size of 1 can be declared as the number tokens for this input (", _algorithm->name()+":"+name, ")");
  }
  else if (type == STREAM &&
           ( (!inputs().empty() && inputs().begin()->second->acquireSize() != n) ||
             (!outputs().empty() && outputs().begin()->second->acquireSize() != n) ) ) {
    throw EssentiaException("StreamingAlgorithmWrapper::declareInput: all input and output STREAM sizes must be the same (", _algorithm->name()+":"+name, ")");
  }

  Algorithm::declareInput(sink, n, name, _algorithm->inputDescription[name]);
  _inputType.insert(name, type);
}

void StreamingAlgorithmWrapper::declareOutput(SourceBase& source, NumeralType type, const std::string& name) {
  declareOutput(source, type, 1, name);
}

void StreamingAlgorithmWrapper::declareOutput(SourceBase& source, NumeralType type,
                                              int n, const std::string& name) {
  if (!_algorithm) {
    throw EssentiaException("When defining a StreamingAlgorithmWrapper, you need to call declareAlgorithm before any declareInput/Output function.");
  }

  if ( (!_inputType.empty() && _inputType.begin()->second != type) ||
       (!_outputType.empty() && _outputType.begin()->second != type) ) {
    throw EssentiaException("StreamingAlgorithmWrapper::declareOutput: all inputs and outputs must have the same NumeralType (", _algorithm->name()+":"+name, ")");
  }

  if (type == TOKEN && n != 1) {
    throw EssentiaException("StreamingAlgorithmWrapper::declareOutput: when using the TOKEN NumeralType, only a size of 1 can be declared as the number tokens for this output (", _algorithm->name()+":"+name, ")");
  }
  else if (type == STREAM &&
           ( (!inputs().empty() && inputs().begin()->second->acquireSize() != n) ||
             (!outputs().empty() && outputs().begin()->second->acquireSize() != n) ) ) {
    throw EssentiaException("StreamingAlgorithmWrapper::declareOutput: all input and output STREAM sizes must be the same (", _algorithm->name()+":"+name, ")");
  }

  Algorithm::declareOutput(source, n, name, _algorithm->outputDescription[name]);
  _outputType.insert(name, type);
}


void StreamingAlgorithmWrapper::synchronizeInput(const std::string& name) {
  if (_inputType[name] == TOKEN) {
    _algorithm->input(name).setSinkFirstToken(*_inputs[name]);
  }
  else if (_inputType[name] == STREAM) {
    _algorithm->input(name).setSinkTokens(*_inputs[name]);
  }
}


void StreamingAlgorithmWrapper::synchronizeOutput(const std::string& name) {
  if (_outputType[name] == TOKEN) {
    _algorithm->output(name).setSourceFirstToken(*_outputs[name]);
  }
  else if (_outputType[name] == STREAM) {
    _algorithm->output(name).setSourceTokens(*_outputs[name]);
  }
}


/**
 * Look for implementation using a mutexlocker, instead of dealing by hand with
 * mutexes all over the place
 */
AlgorithmStatus StreamingAlgorithmWrapper::process() {

  EXEC_DEBUG("acquiring data");
  AlgorithmStatus status = acquireData();
  EXEC_DEBUG("done acquiring data locks");

  if (status != OK) {
    // if shouldStop is true, that means there is no more input to follow, so
    // we need to take what's left, instead of waiting for more data to come
    // in (which would have done by returning from this function)
    if (!shouldStop()) {
      EXEC_DEBUG("returning because no more input data available");
      return status;
    }

    E_DEBUG(EAlgorithm, name() << "::shouldStop(), " << inputs().begin()->second->available() << " tokens available on input, "
            << outputs().begin()->second->available() << " tokens available on output");

#if DEBUGGING_ENABLED
    for (OutputMap::const_iterator it = outputs().begin(); it != outputs().end(); ++it) {
        const std::string& name = it->first;
      SourceBase& source = *it->second;
      E_DEBUG(EAlgorithm, " + " << name << ": " << source.totalProduced() << " tokens produced");
    }
#endif

    // use the following heuristic (it will only work in a specific case)
    // - take as many tokens as are available on all the inputs we can
    //   find, and the same number on the first output (in case of stream)
    //   in case of Token that should be taken one by one, the fact that
    //   status != OK means there is nothing left at all, so we can
    //   forget about this case.
    EXEC_DEBUG("consuming all input data left as a stream");

    // if all inputs don't have the same number of tokens, it means one of them is late
    // and we should wait for it to receive the data. Just return and wait to be called
    // back again later
    int minAvailable = inputs().begin()->second->available(), maxAvailable = 0;
    for (InputMap::const_iterator it = inputs().begin(); it != inputs().end(); ++it) {
      SinkBase& sink = *it->second;
      minAvailable = min(minAvailable, sink.available());
      maxAvailable = max(maxAvailable, sink.available());
    }
    if (minAvailable != maxAvailable) {
      E_WARNING("something strange happened in " << name() << ":");
      E_WARNING("we are at the end of the stream, but there is a different number of tokens available on the inputs:");
      for (InputMap::const_iterator it = inputs().begin(); it != inputs().end(); ++it) {
        SinkBase& sink = *it->second;
        E_WARNING(" - " << sink.fullName() << ": " << sink.available());
      }
      return NO_INPUT;
    }

    // if nothing is left, return now saying so
    int available = minAvailable;
    if (available == 0) return NO_INPUT;

    // otherwise, just grab all of those on the inputs and go on
    for (InputMap::const_iterator it = inputs().begin(); it != inputs().end(); ++it) {
      SinkBase& sink = *it->second;
      sink.setAcquireSize(available);
      sink.setReleaseSize(available);
    }

    // make sure we also have an output to this algo (ie: it is not a file
    // writer or some other storage algo that doesn't have any output)
    for (OutputMap::const_iterator it = outputs().begin(); it != outputs().end(); ++it) {
      SourceBase& source = *it->second;
      source.setAcquireSize(available);
      source.setReleaseSize(available);
    }

    return process();
  }

  synchronizeIO();

  EXEC_DEBUG("computing");
  _algorithm->compute();
  EXEC_DEBUG("done computing, releasing data");

  releaseData();
  EXEC_DEBUG("data released");

#if DEBUGGING_ENABLED
  if (shouldStop()) {
    for (OutputMap::const_iterator it = outputs().begin(); it != outputs().end(); ++it) {
        const std::string& name = it->first;
      SourceBase& source = *it->second;
      E_DEBUG(EAlgorithm, " - " << name << ": " << source.totalProduced() << " tokens produced");
    }
  }
#endif

  return OK;
}


} // namespace streaming
} // namespace essentia
