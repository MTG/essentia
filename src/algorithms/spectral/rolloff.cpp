/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "rolloff.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* RollOff::name = "RollOff";
const char* RollOff::description = DOC("This algorithm computes the roll-off frequency of a spectrum. The roll-off frequency is defined as the frequency under which some percentage (cutoff) of the total energy of the spectrum is contained. The roll-off frequency can be used to distinguish between harmonic (below roll-off) and noisy sounds (above roll-off).\n"
"\n"
"An exception is thrown if the input audio spectrum is smaller than 2.\n"
"References:\n"
"  [1] G. Peeters, A large set of audio features for sound description (similarity and classification) in the CUIDADO project,"
"      CUIDADO I.S.T. Project Report, 2004");


void RollOff::compute() {

  const std::vector<Real>& spectrum = _spectrum.get();
  Real& rolloff = _rolloff.get();
  rolloff = 0.0;

  if (spectrum.size() < 2) {
    throw EssentiaException("RollOff: input audio spectrum is smaller than 2");
  }

  Real cumEnergy = 0.0; // cumulative energy
  Real cutoff = parameter("cutoff").toReal() * energy(spectrum);

  // sum the energy until cutoff reached
  for (int i=0; i<int(spectrum.size()); ++i) {
    cumEnergy += spectrum[i]*spectrum[i];
    if (cumEnergy >= cutoff) {
      rolloff = Real(i);
      break;
    }
  }

  // normalize the rolloff to the desired frequency range
  rolloff *= (parameter("sampleRate").toReal()/2.0) / (spectrum.size()-1);
}
