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

#include "loudnessebur128filter.h"
#include "algorithmfactory.h"

using namespace std;

namespace essentia {
namespace streaming {

const char* LoudnessEBUR128Filter::name = "LoudnessEBUR128Filter";
const char* LoudnessEBUR128Filter::description = DOC("This is an auxilary signal preprocessing algorithm used within the LoudnessEBUR128 algorithm. It applies the TODO filter and computes signal representation requiered by LoudnessEBUR128 in accordance with the EBU R128 recommendation (TODO refs to both specifications).\n"
"\n"
"References:\n"
"  [1] TODO: J. Salamon and E. Gómez, \"Melody extraction from polyphonic music\n"
"  signals using pitch contour characteristics,\" IEEE Transactions on Audio,\n"
"  Speech, and Language Processing, vol. 20, no. 6, pp. 1759–1770, 2012.\n\n"
"  [2] TODO http://mtg.upf.edu/technologies/melodia\n"
);

LoudnessEBUR128Filter::LoudnessEBUR128Filter() : AlgorithmComposite() {

  AlgorithmFactory& factory = AlgorithmFactory::instance();
  _stereoDemuxer  = factory.create("StereoDemuxer");
  _resampleLeft   = factory.create("Resample");
  _resampleRight  = factory.create("Resample");
  _filterLeft1    = factory.create("IIR");
  _filterLeft2    = factory.create("IIR");
  _filterRight1   = factory.create("IIR");
  _filterRight2   = factory.create("IIR");
  _squareLeft     = factory.create("UnaryOperatorStream");
  _squareRight    = factory.create("UnaryOperatorStream");
  _sum            = factory.create("BinaryOperatorStream");

  declareInput(_signal, "signal", "the input stereo audio signal");
  declareOutput(_signalFiltered, "signal", "the filtered signal (the sum of squared amplitudes of both channels filtered by ITU-R BS.1770 algorithm");

  // Connect input proxy
  _signal >> _stereoDemuxer->input("audio");

  // Connect algos 
  _stereoDemuxer->output("left")    >> _resampleLeft->input("signal");
  _stereoDemuxer->output("right")   >> _resampleRight->input("signal");

  _resampleLeft->output("signal")   >> _filterLeft1->input("signal");
  _resampleRight->output("signal")  >> _filterRight1->input("signal");

  _filterLeft1->output("signal")    >> _filterLeft2->input("signal");
  _filterRight1->output("signal")   >> _filterRight2->input("signal");

  _filterLeft2->output("signal")    >> _squareLeft->input("array");
  _filterRight2->output("signal")   >> _squareRight->input("array");

  // TODO ideally, scheduler should work with diamond shape graphs, so that we
  // can use here an algorithm BinaryOperator to sum both left and right values.
  // Test if this works in practise 
  _squareLeft->output("array")      >> _sum->input("array1");
  _squareRight->output("array")     >> _sum->input("array2");

  // Connect output proxy
  _sum->output("array")             >> _signalFiltered;
}

LoudnessEBUR128Filter::~LoudnessEBUR128Filter() {
}

void LoudnessEBUR128Filter::configure() {

  vector<Real> filterB1, filterA1, filterB2, filterA2;

  Real inputSampleRate = parameter("sampleRate").toReal();                           
  Real outputSampleRate = (inputSampleRate == 44100.) ? 44100. : 48000.;

  if (inputSampleRate == 44100.) {                                                   
    // TODO test this C++11 feature works on mingw_port
    filterB1.push_back(1.535);
    filterB1.push_back(-2.633);
    filterB1.push_back(1.151);

    filterA1.push_back(1.);
    filterA1.push_back(-1.647);
    filterA1.push_back(0.701);
    
    filterB2.push_back(1.);
    filterB2.push_back(-2.);
    filterB2.push_back(1.);

    filterA2.push_back(1.);
    filterA2.push_back(-1.9891);
    filterA2.push_back(0.98913);
  }                  
  else { // values for 48000 Hz                                                 
    filterB1.push_back(1.53512485958697);
    filterB1.push_back(-2.69169618940638);
    filterB1.push_back(1.19839281085285);

    filterA1.push_back(1.0);
    filterA1.push_back(-1.69065929318241);
    filterA1.push_back(0.73248077421585);

    filterB2.push_back(1.0);
    filterB2.push_back(-2.0);
    filterB2.push_back(1.0);

    filterA2.push_back(1.0);
    filterA2.push_back(-1.99004745483398);
    filterA2.push_back(0.99007225036621);
  } 

  _resampleLeft->configure("inputSampleRate", inputSampleRate,
                           "outputSampleRate", outputSampleRate);
  _resampleRight->configure("inputSampleRate", inputSampleRate,
                            "outputSampleRate", outputSampleRate);

  _filterLeft1->configure("numerator", filterB1, "denominator", filterA1);
  _filterRight1->configure("numerator", filterB1, "denominator", filterA1);
  _filterLeft2->configure("numerator", filterB2, "denominator", filterA2);
  _filterRight2->configure("numerator", filterB2, "denominator", filterA2);
}


void LoudnessEBUR128Filter::reset() {
  AlgorithmComposite::reset();
  _network->reset();

}

} // namespace streaming
} // namespace essentia
