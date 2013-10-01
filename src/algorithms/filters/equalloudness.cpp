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

#include "equalloudness.h"

using namespace essentia;
using namespace standard;
using namespace std;

const char* EqualLoudness::name = "EqualLoudness";
const char* EqualLoudness::description = DOC("This algorithm implements an equal-loudness filter. The human ear does not perceive sounds of all frequencies as having equal loudness, and to account for this, the signal is filtered by an inverted approximation of the equal-loudness curves. Technically, the filter is a cascade of a 10th order Yulewalk filter with a 2nd order Butterworth high pass filter.\n"
"\n"
"This algorithm depends on the IIR algorithm. Any requirements of the IIR algorithm are imposed for this algorithm. This algorithm is only defined for the sampling rates specified in parameters. It will throw an exception if attempting to configure with any other sampling rate.\n"
"\n"
"References:\n"
"  [1] Replay Gain - Equal Loudness Filter,\n"
"  http://replaygain.hydrogenaudio.org/proposal/equal_loudness.html");


void EqualLoudness::reset() {
  _yulewalkFilter->reset();
  _butterworthFilter->reset();
}

void EqualLoudness::configure() {
  Real fs = parameter("sampleRate").toReal();

  if ((fs != 44100.0) && (fs != 48000.0) && (fs != 32000.0)) {
    throw EssentiaException("EqualLoudness: the sample rate is neither 44100, 48000 nor 32000 Hz, it must be one of these values");
  }

  vector<Real> By(11, 0.0);
  vector<Real> Ay(11, 0.0);
  vector<Real> Bb(3, 0.0);
  vector<Real> Ab(3, 0.0);

  if (fs == 44100.0) {

    // Yulewalk filter
    By[0] =   0.05418656406430;
    By[1] =  -0.02911007808948;
    By[2] =  -0.00848709379851;
    By[3] =  -0.00851165645469;
    By[4] =  -0.00834990904936;
    By[5] =   0.02245293253339;
    By[6] =  -0.02596338512915;
    By[7] =   0.01624864962975;
    By[8] =  -0.00240879051584;
    By[9] =   0.00674613682247;
    By[10] = -0.00187763777362;

    Ay[0] =   1.00000000000000;
    Ay[1] =  -3.47845948550071;
    Ay[2] =   6.36317777566148;
    Ay[3] =  -8.54751527471874;
    Ay[4] =   9.47693607801280;
    Ay[5] =  -8.81498681370155;
    Ay[6] =   6.85401540936998;
    Ay[7] =  -4.39470996079559;
    Ay[8] =   2.19611684890774;
    Ay[9] =  -0.75104302451432;
    Ay[10] =  0.13149317958808;

    // Butterworth filter
    Bb[0] =  0.98500175787242;
    Bb[1] = -1.97000351574484;
    Bb[2] =  0.98500175787242;

    Ab[0] =  1.00000000000000;
    Ab[1] = -1.96977855582618;
    Ab[2] =  0.97022847566350;

  }
  else if (fs == 48000.0) {

    // Yulewalk filtering
    By[0] =  0.03857599435200;
    By[1] = -0.02160367184185;
    By[2] = -0.00123395316851;
    By[3] = -0.00009291677959;
    By[4] = -0.01655260341619;
    By[5] =  0.02161526843274;
    By[6] = -0.02074045215285;
    By[7] =  0.00594298065125;
    By[8] =  0.00306428023191;
    By[9] =  0.00012025322027;
    By[10] = 0.00288463683916;

    Ay[0] =  1.00000000000000;
    Ay[1] = -3.84664617118067;
    Ay[2] =  7.81501653005538;
    Ay[3] = -11.34170355132042;
    Ay[4] =  13.05504219327545;
    Ay[5] = -12.28759895145294;
    Ay[6] =  9.48293806319790;
    Ay[7] = -5.87257861775999;
    Ay[8] =  2.75465861874613;
    Ay[9] = -0.86984376593551;
    Ay[10] = 0.13919314567432;

    // Butterworth filtering
    Bb[0] =  0.98621192462708;
    Bb[1] = -1.97242384925416;
    Bb[2] =  0.98621192462708;

    Ab[0] =  1.00000000000000;
    Ab[1] = -1.97223372919527;
    Ab[2] =  0.97261396931306;

  }
  else if (fs == 32000.0) {

    // Yulewalk filtering
    By[0] =   0.15457299681924;
    By[1] =  -0.09331049056315;
    By[2] =  -0.06247880153653;
    By[3] =   0.02163541888798;
    By[4] =  -0.05588393329856;
    By[5] =   0.04781476674921;
    By[6] =   0.00222312597743;
    By[7] =   0.03174092540049;
    By[8] =  -0.01390589421898;
    By[9] =   0.00651420667831;
    By[10] = -0.00881362733839;

    Ay[0] =   1.00000000000000;
    Ay[1] =  -2.37898834973084;
    Ay[2] =   2.84868151156327;
    Ay[3] =  -2.64577170229825;
    Ay[4] =   2.23697657451713;
    Ay[5] =  -1.67148153367602;
    Ay[6] =   1.00595954808547;
    Ay[7] =  -0.45953458054983;
    Ay[8] =   0.16378164858596;
    Ay[9] =  -0.05032077717131;
    Ay[10] =  0.02347897407020;

    // Butterworth filtering
    Bb[0] =  0.97938932735214;
    Bb[1] =  -1.95877865470428;
    Bb[2] =  0.97938932735214;

    Ab[0] =  1.00000000000000;
    Ab[1] = -1.95835380975398;
    Ab[2] =  0.95920349965459;

  }

  // configure both filters and set them ready to go
  _yulewalkFilter->configure("numerator", By, "denominator", Ay);

  _butterworthFilter->configure("numerator", Bb, "denominator", Ab);

  _yulewalkFilter->output("signal").set(_z);
  _butterworthFilter->input("signal").set(_z);

}

void EqualLoudness::compute() {
  _yulewalkFilter->input("signal").set(_x.get());
  _butterworthFilter->output("signal").set(_y.get());

  _yulewalkFilter->compute();
  _butterworthFilter->compute();
}
