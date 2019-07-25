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

#include "mfcc.h"
#include "essentiamath.h" // lin2db

using namespace std;
using namespace essentia;
using namespace standard;

const char* MFCC::name = "MFCC";
const char* MFCC::category = "Spectral";
const char* MFCC::description = DOC("This algorithm computes the mel-frequency cepstrum coefficients of a spectrum. As there is no standard implementation, the MFCC-FB40 is used by default:\n"
"  - filterbank of 40 bands from 0 to 11000Hz\n"
"  - take the log value of the spectrum energy in each mel band. Bands energy values below silence threshold will be clipped to its value before computing log-energies\n"
"  - DCT of the 40 bands down to 13 mel coefficients\n"
"There is a paper describing various MFCC implementations [1].\n"
"\n"
"The parameters of this algorithm can be configured in order to behave like HTK [3] as follows:\n"
"  - type = 'magnitude'\n"
"  - warpingFormula = 'htkMel'\n"
"  - weighting = 'linear'\n"
"  - highFrequencyBound = 8000\n"
"  - numberBands = 26\n"
"  - numberCoefficients = 13\n"
"  - normalize = 'unit_max'\n"
"  - dctType = 3\n"
"  - logType = 'log'\n"
"  - liftering = 22\n"
"\n"
"In order to completely behave like HTK the audio signal has to be scaled by 2^15 before the processing and if the Windowing and FrameCutter algorithms are used they should also be configured as follows. \n"
"\n"
"FrameGenerator:\n"
"  - frameSize = 1102\n"
"  - hopSize = 441\n"
"  - startFromZero = True\n"
"  - validFrameThresholdRatio = 1\n"
"\n"
"Windowing:\n"
"  - type = 'hamming'\n"
"  - size = 1102\n"
"  - zeroPadding = 946\n"
"  - normalized = False\n"
"\n"
"This algorithm depends on the algorithms MelBands and DCT and therefore inherits their parameter restrictions. An exception is thrown if any of these restrictions are not met. The input \"spectrum\" is passed to the MelBands algorithm and thus imposes MelBands' input requirements. Exceptions are inherited by MelBands as well as by DCT.\n"
"\n"
"IDCT can be used to compute smoothed Mel Bands. In order to do this:\n"
"  - compute MFCC\n"
"  - smoothedMelBands = 10^(IDCT(MFCC)/20)\n"
"\n"                                
"Note: The second step assumes that 'logType' = 'dbamp' was used to compute MFCCs, otherwise that formula should be changed in order to be consistent.\n"
"\n"
"References:\n"
"  [1] T. Ganchev, N. Fakotakis, and G. Kokkinakis, \"Comparative evaluation\n"
"  of various MFCC implementations on the speaker verification task,\" in\n"
"  International Conference on Speach and Computer (SPECOM’05), 2005,\n"
"  vol. 1, pp. 191–194.\n\n"
"  [2] Mel-frequency cepstrum - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Mel_frequency_cepstral_coefficient\n\n"
"  [3] Young, S. J., Evermann, G., Gales, M. J. F., Hain, T., Kershaw, D.,\n"
"  Liu, X., … Woodland, P. C. (2009). The HTK Book (for HTK Version 3.4).\n"
"  Construction, (July 2000), 384, https://doi.org/http://htk.eng.cam.ac.uk\n\n"
"  [4] Slaney, M. Auditory Toolbox: A MATLAB Toolbox for Auditory Modeling Work.\n"
"  Technical Report, version 2, Interval Research Corporation, 1998.");



void MFCC::configure() {
  _melFilter->configure(INHERIT("inputSize"),
                        INHERIT("sampleRate"),
                        INHERIT("numberBands"),
                        INHERIT("lowFrequencyBound"),
                        INHERIT("highFrequencyBound"),
                        INHERIT("warpingFormula"),
                        INHERIT("weighting"),
                        INHERIT("normalize"),
                        INHERIT("type"));

  _dct->configure("inputSize", parameter("numberBands"),
                  "outputSize", parameter("numberCoefficients"),
                  INHERIT("dctType"),
                  INHERIT("liftering"));
  _logbands.resize(parameter("numberBands").toInt());

  _logType = parameter("logType").toLower();
  _silenceThreshold = parameter("silenceThreshold").toReal();
  _dbSilenceThreshold = 10 * log10(_silenceThreshold);
  _logSilenceThreshold = log(_silenceThreshold);
}


void MFCC::compute() {

  // get the inputs and outputs
  const vector<Real>& spectrum = _spectrum.get();
  vector<Real>& mfcc = _mfcc.get();
  vector<Real>& bands = _bands.get();

  // filter the spectrum using a mel-scaled filterbank
  _melFilter->input("spectrum").set(spectrum);
  _melFilter->output("bands").set(bands);
  _melFilter->compute();

  // take the dB amplitude of the spectrum
  for (int i=0; i<int(bands.size()); ++i) {
    if (_logType == "dbpow") {
      _logbands[i] = pow2db(bands[i], _silenceThreshold, _dbSilenceThreshold);
    }
    else if (_logType == "dbamp") {
      _logbands[i] = amp2db(bands[i], _silenceThreshold, _dbSilenceThreshold);
    }
    else if (_logType == "log") {
      _logbands[i] = lin2log(bands[i], _silenceThreshold, _logSilenceThreshold);
    }
    else if (_logType == "natural") {
      _logbands[i] = bands[i]; 
    }
    else {
      throw EssentiaException("MFCC: Bad 'logType' parameter");
    }
  }

  // compute the DCT of these bands
  _dct->input("array").set(_logbands);
  _dct->output("dct").set(mfcc);
  _dct->compute();
}
