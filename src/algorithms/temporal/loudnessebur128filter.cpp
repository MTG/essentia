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

#include "loudnessebur128filter.h"
#include "essentiamath.h"
#include "algorithmfactory.h"

using namespace std;

namespace essentia {
namespace streaming {

const char* LoudnessEBUR128Filter::name = "LoudnessEBUR128Filter";
const char* LoudnessEBUR128Filter::category = "Loudness/dynamics";
const char* LoudnessEBUR128Filter::description = DOC("An auxilary signal preprocessing algorithm used within the LoudnessEBUR128 algorithm. It applies the pre-processing K-weighting filter and computes signal representation requiered by LoudnessEBUR128 in accordance with the EBU R128 recommendation.\n"
"\n"
"References:\n"
"  [2] ITU-R BS.1770-2. \"Algorithms to measure audio programme loudness and true-peak audio level\n\n"
);

LoudnessEBUR128Filter::LoudnessEBUR128Filter() : AlgorithmComposite() {

  AlgorithmFactory& factory = AlgorithmFactory::instance();
  _stereoDemuxer  = factory.create("StereoDemuxer");

  _filterLeft     = factory.create("IIR");
  _filterRight    = factory.create("IIR");
  _squareLeft     = factory.create("UnaryOperatorStream");
  _squareRight    = factory.create("UnaryOperatorStream");
  _sum            = factory.create("BinaryOperatorStream");

  declareInput(_signal, "signal", "the input stereo audio signal");
  declareOutput(_signalFiltered, "signal", "the filtered signal (the sum of squared amplitudes of both channels filtered by ITU-R BS.1770 algorithm");

  // Connect input proxy
  _signal >> _stereoDemuxer->input("audio");

  // Connect algos  
  _stereoDemuxer->output("left")    >> _filterLeft->input("signal");
  _stereoDemuxer->output("right")   >> _filterRight->input("signal");

  _filterLeft->output("signal")    >> _squareLeft->input("array");
  _filterRight->output("signal")   >> _squareRight->input("array");

  // NOTE: It is not recommended to use scheduler with diamond shape graphs 
  // according to documentation. However, this works in practise. The agorithm 
  // BinaryOperator sums both left and right values.
  _squareLeft->output("array")      >> _sum->input("array1");
  _squareRight->output("array")     >> _sum->input("array2");

  // Connect output proxy
  _sum->output("array")             >> _signalFiltered;

  _network = new scheduler::Network(_stereoDemuxer);
}

LoudnessEBUR128Filter::~LoudnessEBUR128Filter() {
  delete _network;
}

void LoudnessEBUR128Filter::configure() {

  Real sampleRate = parameter("sampleRate").toReal(); 

  vector<Real> filterB1(3, 0.), filterA1(3, 0.), 
               filterB2(3, 0.), filterA2(3, 0.);

  // NOTE: ITU-R BS.1770-2 provides precomputed values for filter coefficients.
  // However, our tests on reference files revealed incorrect integrated loudness 
  // when using these values. Therefore, instead of hardcoding the coeffcients, 
  // we use a formula to compute them for any sample rate taken from: 
  // https://github.com/jiixyj/libebur128/blob/v1.0.2/ebur128/ebur128.c#L82
  // The original code is released under MIT license: 
  //         https://github.com/jiixyj/libebur128/blob/v1.0.2/COPYING
  // This formula is generic for any sample rate, therefore we do not need to 
  // resample the signal
  
  double f0 = 1681.974450955533;
  double G  = 3.999843853973347;
  double Q  = 0.7071752369554196;

  double K  = tan(M_PI * f0 / (double) sampleRate);
  double Vh = pow(10.0, G / 20.0);
  double Vb = pow(Vh, 0.4996667741545416);
  double a0 = 1.0 + K / Q + K * K;
  
  filterB1[0] = (Vh + Vb * K / Q + K * K) / a0;
  filterB1[1] = 2.0 * (K * K -  Vh) / a0;
  filterB1[2] = (Vh - Vb * K / Q + K * K) / a0;

  filterA1[0] = 1.;
  filterA1[1] = 2.0 * (K * K - 1.0) / a0;
  filterA1[2] = (1.0 - K / Q + K * K) / a0;

  f0 = 38.13547087602444;
  Q  = 0.5003270373238773;
  K  = tan(M_PI * f0 / (double) sampleRate);

  filterB2[0] = 1.;
  filterB2[1] = -2.;
  filterB2[2] = 1.;

  filterA2[0] = 1.;
  filterA2[1] = 2.0 * (K * K - 1.0) / (1.0 + K / Q + K * K);
  filterA2[2] = (1.0 - K / Q + K * K) / (1.0 + K / Q + K * K);

  // combine two filters into one
  vector<Real> filterB(5, 0.), filterA(5, 0.);

  filterB[0] = filterB1[0] * filterB2[0];
  filterB[1] = filterB1[0] * filterB2[1] + filterB1[1] * filterB2[0];
  filterB[2] = filterB1[0] * filterB2[2] + filterB1[1] * filterB2[1] + filterB1[2] * filterB2[0];
  filterB[3] = filterB1[1] * filterB2[2] + filterB1[2] * filterB2[1];
  filterB[4] = filterB1[2] * filterB2[2];

  filterA[0] = filterA1[0] * filterA2[0];
  filterA[1] = filterA1[0] * filterA2[1] + filterA1[1] * filterA2[0];
  filterA[2] = filterA1[0] * filterA2[2] + filterA1[1] * filterA2[1] + filterA1[2] * filterA2[0];
  filterA[3] = filterA1[1] * filterA2[2] + filterA1[2] * filterA2[1];
  filterA[4] = filterA1[2] * filterA2[2];

  _filterLeft->configure("numerator", filterB, "denominator", filterA);
  _filterRight->configure("numerator", filterB, "denominator", filterA);

  _squareLeft->configure("type", "square");
  _squareRight->configure("type", "square");

  _sum->configure("type", "add");
}


void LoudnessEBUR128Filter::reset() {
  AlgorithmComposite::reset();
}

} // namespace streaming
} // namespace essentia
