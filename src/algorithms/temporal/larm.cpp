/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "larm.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* Larm::name = "Larm";
const char* Larm::description = DOC("This algorithm estimates the long-term loudness of an audio signal. The LARM model is based on the asymmetrical low-pass filtering of the Peak Program Meter (PPM), combined with Revised Low-frequency B-weighting (RLB) and power mean calculations. LARM has shown to be a reliable and objective loudness estimate of music and speech.\n"
"\n"
"It accepts a power parameter to define the exponential for computing the power mean. Note that if the parameter's value is 2, this algorithm would be equivalent to RMS and if 1, this algorithm would be the mean of the absolute value.\n"
"\n"
"References:\n"
"  [1] Skovenborg, E. and Nielsen, S., Evaluation of Different Loudness \n"
"      Models with Music and Speech Material, Proceedings of the AES 117th \n"
"      Convention, San Francisco, CA, USA, 2004");


void Larm::configure() {

  _envelope->configure("sampleRate", parameter("sampleRate").toInt(),
                       "attackTime", parameter("attackTime").toReal(),
                       "releaseTime", parameter("releaseTime").toReal());

  _powerMean->configure("power", parameter("power"));
}

void Larm::compute() {

  const vector<Real>& signal = _signal.get();
  Real& larm = _larm.get();

  vector<Real> envelope;
  Real powerMean;

  _envelope->input("signal").set(signal);
  _envelope->output("signal").set(envelope);
  _envelope->compute();

  _powerMean->input("array").set(envelope);
  _powerMean->output("powerMean").set(powerMean);
  _powerMean->compute();

  if (powerMean < 1e-5) {
    larm = -100.0;
  }
  else {
    larm = 20.0*log10(powerMean);
  }
}
