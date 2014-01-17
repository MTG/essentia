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

#include "streamingalgorithm.h"
using namespace std;

namespace essentia {
namespace streaming {


const string Algorithm::processingMode = "Streaming";

// here we check the types as well
void connect(SourceBase& source, SinkBase& sink) {
  try {
    // NB: source needs to connect to sink first to have a ReaderID as soon as possible.
    //     this is a requirement for ProxyConnectors
    E_DEBUG(EConnectors, "Connecting " << source.fullName() << " to " << sink.fullName());
    source.connect(sink);
    sink.connect(source);
  }
  catch (EssentiaException& e) {
    std::ostringstream msg;
    msg << "While connecting " << source.fullName()
        << " to " << sink.fullName() << ":\n"
        << e.what();
    throw EssentiaException(msg);
  }
}

void connect(Algorithm* sourceAlgo, const std::string& sourcePort,
             Algorithm* sinkAlgo, const std::string& sinkPort) {
  try {
    SourceBase& source = sourceAlgo->output(sourcePort);
    SinkBase& sink = sinkAlgo->input(sinkPort);

    connect(source, sink);
  }
  catch (EssentiaException& e) {
    std::ostringstream msg;
    msg << "While connecting " << sourceAlgo->name() << "::" << sourcePort
        << " to " << sinkAlgo->name() << "::" << sinkPort << ":\n"
        << e.what();
    throw EssentiaException(msg);
  }
}


void disconnect(SourceBase& source, SinkBase& sink) {
  try {
    E_DEBUG(EConnectors, "Disconnecting " << source.fullName() << " from " << sink.fullName());

    source.disconnect(sink);
    sink.disconnect(source);
  }
  catch (const EssentiaException& e) {
    std::ostringstream msg;
    msg << "While disconnecting " << source.fullName()
        << " (output) from " << sink.fullName() << " (input):\n"
        << e.what();
    throw EssentiaException(msg);
  }
}


void Algorithm::declareInput(SinkBase& sink, const std::string& name,
                             const std::string& desc) {
  sink.setName(name);
  sink.setParent(this);

  _inputs.insert(name, &sink);
  inputDescription.insert(name, desc);
}

void Algorithm::declareInput(SinkBase& sink, int n,
                             const std::string& name, const std::string& desc) {
  declareInput(sink, n, n, name, desc);
}

void Algorithm::declareInput(SinkBase& sink,
                             int acquireSize,
                             int releaseSize,
                             const std::string& name,
                             const std::string& desc) {
  sink.setAcquireSize(acquireSize);
  sink.setReleaseSize(releaseSize);
  declareInput(sink, name, desc);
}

void Algorithm::declareOutput(SourceBase& source, const std::string& name, const std::string& desc) {
  source.setName(name);
  source.setParent(this);

  _outputs.insert(name, &source);
  outputDescription.insert(name, desc);
}

void Algorithm::declareOutput(SourceBase& source, int n, const std::string& name, const std::string& desc) {
  declareOutput(source, n, n, name, desc);
}

void Algorithm::declareOutput(SourceBase& source,
                              int acquireSize,
                              int releaseSize,
                              const std::string& name,
                              const std::string& desc) {
  source.setAcquireSize(acquireSize);
  source.setReleaseSize(releaseSize);
  declareOutput(source, name, desc);
}


SinkBase& Algorithm::input(const std::string& name) {
  try {
    return *_inputs[name];
  }
  catch (EssentiaException&) {
    std::ostringstream msg;
    msg << "Couldn't find '" << name << "' in " << this->name() << "::inputs.";
    msg << " Available input names are:";
    std::vector<std::string> availableInputs = _inputs.keys();
    for (uint i=0; i<availableInputs.size(); i++) {
      msg << ' ' << availableInputs[i];
    }
    throw EssentiaException(msg);
  }
}

SourceBase& Algorithm::output(const std::string& name) {
  try {
    return *_outputs[name];
  }
  catch (EssentiaException&) {
    std::ostringstream msg;
    msg << "Couldn't find '" << name << "' in " << this->name() << "::outputs.";
    msg << " Available output names are:";
    std::vector<std::string> availableOutputs = _outputs.keys();
    for (uint i=0; i<availableOutputs.size(); i++) {
      msg << ' ' << availableOutputs[i];
    }
    throw EssentiaException(msg);
  }
}

SinkBase& Algorithm::input(int idx) {
  if ((idx < 0) || (idx >= _inputs.size())) {
    ostringstream msg;
    msg << "Cannot access input number " << idx << " because " << this->name() << " only has " << _inputs.size() << " inputs.";
    throw EssentiaException(msg);
  }

  return *_inputs[idx].second;
}

SourceBase& Algorithm::output(int idx) {
  if ((idx < 0) || (idx >= _outputs.size())) {
    ostringstream msg;
    msg << "Cannot access output number " << idx << " because " << this->name() << " only has " << _outputs.size() << " outputs.";
    throw EssentiaException(msg);
  }

  return *_outputs[idx].second;
}


void Algorithm::disconnectAll() {
  // disconnect all outputs
  for (OutputMap::const_iterator output = _outputs.begin();
       output != _outputs.end();
       ++output) {
    vector<SinkBase*>& sinks = output->second->sinks();

    for (vector<SinkBase*>::iterator it = sinks.begin(); it != sinks.end(); ++it) {
      disconnect(*output->second, **it);
    }
  }

  // disconnect all inputs
  for (InputMap::const_iterator input = _inputs.begin();
       input != _inputs.end();
       ++input) {
    SourceBase* source = input->second->source();
    if (source) disconnect(*source, *input->second);
  }
}


// returns true if it succeeded, otherwise returns false (also frees the
// resources it may have managed to acquire.
AlgorithmStatus Algorithm::acquireData() {
  // Note: check for inputs first, so we're sure we're done and no need to reschedule
  // the algorithm because of no output
  for (InputMap::const_iterator input = _inputs.begin(); input!=_inputs.end(); ++input) {
    if (!input->second->acquire()) {
      return NO_INPUT;
    }
  }

  for (OutputMap::const_iterator output = _outputs.begin(); output!=_outputs.end(); ++output) {
    if (!output->second->acquire()) {
      return NO_OUTPUT;
    }
  }


  return OK;
}

void Algorithm::releaseData() {
  for (OutputMap::const_iterator output = _outputs.begin(); output!=_outputs.end(); ++output) {
    output->second->release();
  }

  for (InputMap::const_iterator input = _inputs.begin(); input!=_inputs.end(); ++input) {
    input->second->release();
  }
}


void Algorithm::reset() {
  E_DEBUG(EAlgorithm, "Streaming: " << name() << "::reset()");
  shouldStop(false);

  // reset the buffers of the sources of this algorithm
  for (OutputMap::iterator it = _outputs.begin();
       it != _outputs.end();
       ++it) {
    E_DEBUG(EAlgorithm, "resetting buffer for " << it->second->fullName());
    it->second->reset();
  }

  // Note: we don't need to reset the inputs because they share a Multi-rate
  // buffer with the outputs
  E_DEBUG(EAlgorithm, "Streaming: " << name() << "::reset() ok!");
}


void Algorithm::shouldStop(bool stop) {
#if DEBUGGING_ENABLED
  std::ostringstream msg;
  msg << "Streaming: " << name() << "::shouldStop[" << nProcess << "] = "
      << (stop ? "true" : "false");
  E_DEBUG(EAlgorithm, msg.str());
#else
  E_DEBUG(EAlgorithm, "Streaming: " << name() << "::shouldStop = " << (stop?"true":"false"));
#endif
  _shouldStop = stop;
}



} // namespace streaming
} // namespace essentia
