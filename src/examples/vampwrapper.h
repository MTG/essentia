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

#ifndef ESSENTIA_VAMPWRAPPER_H_
#define ESSENTIA_VAMPWRAPPER_H

#include "vamp-sdk/Plugin.h"
#include <essentia/types.h>
#include <essentia/algorithm.h>
#include <essentia/algorithmfactory.h>
#include <essentia/pool.h>

#include <iostream>

class VampWrapper : public Vamp::Plugin {

protected:
  essentia::standard::Algorithm* _algo;
  float _sampleRate;
  int _stepSize;
  int _blockSize;
  int _pluginId;
  std::string _name, _description;
  std::vector<float> _spectrum;
  std::vector<float> _phase;
  essentia::standard::Algorithm* _peaks;
  std::vector<float> _peakmags;
  std::vector<float> _peakfreqs;
  essentia::standard::Algorithm* _bbands;
  std::vector<float> _barkBands;
  essentia::standard::Algorithm* _mbands;
  std::vector<float> _melBands;
  essentia::Pool _pool;

  static int essentiaVampPluginId;

public:
  VampWrapper(essentia::standard::Algorithm* algo, float inputSampleRate);
  ~VampWrapper();

  bool initialise(size_t channels, size_t stepSize, size_t blockSize);

  void reset();

  // by default, freq domain, but should be overriden for temporal domain
  InputDomain getInputDomain() const { return FrequencyDomain; }

  std::string getMaker() const { return "Music Technology Group"; }
  std::string getCopyright() const { return "(C) 2012 MTG, Universitat Pompeu Fabra"; }
  int getPluginVersion() const { return 2; }

  essentia::AlgorithmInfo<essentia::standard::Algorithm> info() const {
    return essentia::standard::AlgorithmFactory::getInfo(_algo->name());
  }

  // we cannot always use the name (i.e. info().name) of the _algo as a unique
  // identifier for the vamp plugin. That is, because several algorithms (
  // (i.e. onsets and onsetDetection) may use the same essentia algo to
  // initalize the plugin. The same applies for the name

  std::string getIdentifier() const  {
    std::stringstream s;
    s << info().name  << "_" << _pluginId;
    return s.str();
  }
  std::string getName() const        { return _name; };
  std::string getDescription() const { return _description; };

  void setName(const std::string& name) { _name = name; };
  void setDescription(const std::string& desc) {
    // get rid of the references if there are any
    std::string::size_type pos = desc.rfind("\nReferences:\n");
   _description = desc.substr(0, pos);
  };

  void computeSpectrum(const float *const *inputBuffers);
  void computePeaks(const float *const *inputBuffers);
  void computeBarkBands(const float *const *inputBuffers);
  void computeMelBands(const float *const *inputBuffers);

  OutputList genericDescriptor(const std::string& unit,
                               int ndim,
                               const std::string& prefix = "") const;

  Feature makeFeature(float f) const;
  Feature makeFeature(const std::vector<float>& f) const;

  template <typename T>
  FeatureSet returnFeature(T f) const {
    FeatureSet result;
    result[0].push_back(makeFeature(f));
    return result;
  }

  FeatureSet getRemainingFeatures() {
    return FeatureSet();
  }

};


#endif // ESSENTIA_VAMPWRAPPER_H
