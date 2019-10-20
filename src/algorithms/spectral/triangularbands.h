/*
 * Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_TRIANGULARBANDS_H
#define ESSENTIA_TRIANGULARBANDS_H

#include "algorithm.h"
#include "essentiautil.h"

using namespace std;

namespace essentia {
namespace standard {

class TriangularBands : public Algorithm {

 protected:
  Input<std::vector<Real> > _spectrumInput;
  Output<std::vector<Real> > _bandsOutput;
  
  std::vector<Real> _bandFrequencies;
  std::vector<Real> _freqBands;
  int _nBands;
  Real _sampleRate;
  bool _isLog;
  std::vector<std::vector<Real> > _filterCoefficients;
  Real _inputSize;
  std::string _normalize;
  std::string _type;
  void createFilters(int spectrumSize);
  void setWeightingFunctions(std::string weighting);

  typedef  Real (*funcPointer)(Real);
  funcPointer _weighter;

 public:
  TriangularBands() {
    declareInput(_spectrumInput, "spectrum", "the input spectrum (must be greater than size one)");
    declareOutput(_bandsOutput, "bands", "the energy in each band");
  }

  void declareParameters() {
    // FIXME These are the bands used for SuperFlux. It is better to update the default bands to something else.
    Real freqBands[] = {21.533203125, 43.06640625, 64.599609375, 86.1328125, 107.666015625, 129.19921875, 150.732421875, 172.265625, 193.798828125, 215.33203125, 236.865234375, 258.3984375, 279.931640625, 301.46484375, 322.998046875, 344.53125, 366.064453125, 387.59765625, 409.130859375, 430.6640625, 452.197265625, 473.73046875, 495.263671875, 516.796875, 538.330078125, 559.86328125, 581.396484375, 602.9296875, 624.462890625, 645.99609375, 667.529296875, 689.0625, 710.595703125, 732.12890625, 753.662109375, 775.1953125, 796.728515625, 839.794921875, 861.328125, 882.861328125, 904.39453125, 925.927734375, 968.994140625, 990.52734375, 1012.060546875, 1055.126953125, 1076.66015625, 1098.193359375, 1141.259765625, 1184.326171875, 1205.859375, 1248.92578125, 1270.458984375, 1313.525390625, 1356.591796875, 1399.658203125, 1442.724609375, 1485.791015625, 1528.857421875, 1571.923828125, 1614.990234375, 1658.056640625, 1701.123046875, 1765.72265625, 1808.7890625, 1873.388671875, 1916.455078125, 1981.0546875, 2024.12109375, 2088.720703125, 2153.3203125, 2217.919921875, 2282.51953125, 2347.119140625, 2411.71875, 2497.8515625, 2562.451171875, 2627.05078125, 2713.18359375, 2799.31640625, 2885.44921875, 2950.048828125, 3036.181640625, 3143.84765625, 3229.98046875, 3316.11328125, 3423.779296875, 3509.912109375, 3617.578125, 3725.244140625, 3832.91015625, 3940.576171875, 4069.775390625, 4177.44140625, 4306.640625, 4435.83984375, 4565.0390625, 4694.23828125, 4844.970703125, 4974.169921875, 5124.90234375, 5275.634765625, 5426.3671875, 5577.099609375, 5749.365234375, 5921.630859375, 6093.896484375, 6266.162109375, 6459.9609375, 6653.759765625, 6847.55859375, 7041.357421875, 7256.689453125, 7450.48828125, 7687.353515625, 7902.685546875, 8139.55078125, 8376.416015625, 8613.28125, 8871.6796875, 9130.078125, 9388.4765625, 9668.408203125, 9948.33984375, 10249.8046875, 10551.26953125, 10852.734375, 11175.732421875, 11498.73046875, 11843.26171875, 12187.79296875, 12553.857421875, 12919.921875, 13285.986328125, 13673.583984375, 14082.71484375, 14491.845703125, 14922.509765625, 15353.173828125, 15805.37109375, 16257.568359375};
    declareParameter("inputSize", "the size of the spectrum", "(1,inf)", 1025);
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("frequencyBands", "list of frequency ranges into which the spectrum is divided (these must be in ascending order and connot contain duplicates),"\
    					"each triangle is build as x(i-1)=0, x(i)=1, x(i+1)=0 over i, the resulting number of bands is size of input array - 2", "", arrayToVector<Real>(freqBands));
    declareParameter("log", "compute log-energies (log10 (1 + energy))","{true,false}", true);
    declareParameter("weighting", "type of weighting function for determining triangle area","{linear,slaneyMel,htkMel}","linear");
    declareParameter("normalize", "spectrum bin weights to use for each triangular band: 'unit_max' to make each triangle vertex equal to 1, 'unit_sum' to make each triangle area equal to 1 summing the actual weights of spectrum bins, 'unit_area' to make each triangle area equal to 1 normalizing the weights of each triangle by its bandwidth","{unit_sum,unit_tri,unit_max}", "unit_sum");
    declareParameter("type", "use magnitude or power spectrum","{magnitude,power}", "power");
  }

  void compute();
  void configure();


  static const char* name;
  static const char* category;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class TriangularBands : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrumInput;
  Source<std::vector<Real> > _bandsOutput;


 public:
  TriangularBands() {
    declareAlgorithm("TriangularBands");
    declareInput(_spectrumInput, TOKEN, "spectrum");
    declareOutput(_bandsOutput, TOKEN, "bands");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_TRIANGULARBANDS_H
