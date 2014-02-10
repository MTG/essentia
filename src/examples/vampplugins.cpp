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

#include "vamp/vamp.h"
#include "vamp-sdk/PluginAdapter.h"
#include "vampeasywrapper.h"

#include "vamppluginsextra.cpp"
using namespace std;
using namespace essentia;


WRAP_ALGO(Centroid, "Hz", 1, float);

class SpectralCentroid : public Centroid {
public:
  SpectralCentroid(float sr) : Centroid(sr) {
    _algo->configure("range", _sampleRate/2);
  }

};

static Vamp::PluginAdapter<SpectralCentroid> aSpectralCentroid;



#define WRAP_PLUGIN(algoname, unit, ndim, dtype)  \
WRAP_ALGO(algoname, unit, ndim, dtype);           \
static Vamp::PluginAdapter<algoname> a##algoname;

#define WRAP_PEAKS_PLUGIN(algoname, unit, ndim, dtype) \
WRAP_PEAKS_ALGO(algoname, unit, ndim, dtype);          \
static Vamp::PluginAdapter<algoname> p##algoname;

#define WRAP_TEMPORAL_PLUGIN(algoname, unit, ndim, dtype) \
WRAP_TEMPORAL_ALGO(algoname, unit, ndim, dtype);          \
static Vamp::PluginAdapter<algoname> t##algoname;

#define WRAP_BARK_PLUGIN(algoname, unit, ndim, dtype) \
WRAP_BARK_ALGO(algoname, unit, ndim, dtype);          \
static Vamp::PluginAdapter<B##algoname> b##algoname;

#define WRAP_MEL_PLUGIN(algoname, unit, ndim, dtype) \
WRAP_MEL_ALGO(algoname, unit, ndim, dtype);          \
static Vamp::PluginAdapter<M##algoname> m##algoname;

// Spectral
WRAP_PLUGIN(BarkBands, "", _algo->parameter("numberBands").toInt(), vector<float>);
WRAP_PLUGIN(Flux, "", 1, float);
WRAP_BARK_PLUGIN(Flux, "", 1, float);
//WRAP_PLUGIN(HFC, "", 1, float);
WRAP_PLUGIN(MaxMagFreq, "Hz", 1, float);
WRAP_PLUGIN(MelBands, "", _algo->parameter("numberBands").toInt(), vector<float>);
//WRAP_PLUGIN(MFCC, "", _algo->parameter("numberCoefficients").toInt(), vector<float>);
WRAP_PLUGIN(ERBBands, "", _algo->parameter("numberBands").toInt(), vector<float>);
WRAP_PLUGIN(RollOff, "Hz", 1, float);
WRAP_PLUGIN(SpectralComplexity, "", 1, float);
WRAP_PLUGIN(StrongPeak, "", 1, float);

static Vamp::PluginAdapter<SpectralContrast> aSpectralContrast;
static Vamp::PluginAdapter<MFCC> aMFCC;
static Vamp::PluginAdapter<GFCC> aGFCC;
static Vamp::PluginAdapter<HFC> aHFC;
static Vamp::PluginAdapter<OnsetDetection> aOnsetDetection;
static Vamp::PluginAdapter<Onsets> aOnsets;
static Vamp::PluginAdapter<RhythmTransform> mRhythmTransform;
static Vamp::PluginAdapter<SBic> mSbic;
static Vamp::PluginAdapter<NoveltyCurve> aNoveltyCurve;
static Vamp::PluginAdapter<BpmHistogram> aBpmHistogram;

// SFX
WRAP_PLUGIN(PitchSalience, "", 1, float);
WRAP_PEAKS_PLUGIN(Dissonance, "", 1, float);
WRAP_PEAKS_PLUGIN(Inharmonicity, "", 1, float);
WRAP_PEAKS_PLUGIN(Tristimulus, "", 3, vector<float>);
WRAP_PEAKS_PLUGIN(OddToEvenHarmonicEnergyRatio, "", 1, float);

static Vamp::PluginAdapter<Friction> pFriction;
static Vamp::PluginAdapter<Crunchiness> pCrunchiness;
static Vamp::PluginAdapter<Perculleys> aPerculleys;
static Vamp::PluginAdapter<HPCP> pHpcp;
static Vamp::PluginAdapter<SpectralWhitening> pSpectralWhithening;
static Vamp::PluginAdapter<SpectralPeaks> pSpectralPeaks;

// Stats
/*
WRAP_PLUGIN(Spread, "", 1, float);
WRAP_PLUGIN(Skewness, "", 1, float);
WRAP_PLUGIN(Kurtosis, "", 1, float);
WRAP_BARK_PLUGIN(Spread, "", 1, float);
WRAP_BARK_PLUGIN(Skewness, "", 1, float);
WRAP_BARK_PLUGIN(Kurtosis, "", 1, float);
*/
WRAP_BARK_PLUGIN(Flatness, "", 1, float);
WRAP_PLUGIN(Crest, "", 1, float);

// Temporal
static Vamp::PluginAdapter<LPC> aLPC;
WRAP_TEMPORAL_PLUGIN(RMS, "", 1, float);
WRAP_TEMPORAL_PLUGIN(StrongDecay, "", 1, float);
WRAP_TEMPORAL_PLUGIN(Larm, "", 1, float);
WRAP_TEMPORAL_PLUGIN(LoudnessVickers, "", 1, float);
WRAP_TEMPORAL_PLUGIN(ZeroCrossingRate, "", 1, float);

static Vamp::PluginAdapter<Pitch> aPitch;
static Vamp::PluginAdapter<DistributionShape> aDistributionShape;
static Vamp::PluginAdapter<BarkShape> aBarkShape;

ESSENTIA_API
const VampPluginDescriptor *vampGetPluginDescriptor(unsigned int version,
                                                    unsigned int index) {
  if (version < 1) return 0;

  switch (index) {
  case 0: return aSpectralCentroid.getDescriptor();
  case 1: return aBarkBands.getDescriptor();
  case 2: return aFlux.getDescriptor();
  case 3: return aHFC.getDescriptor();
  case 4: return aMaxMagFreq.getDescriptor();
  case 5: return aMelBands.getDescriptor();
  case 6: return aMFCC.getDescriptor();
  case 7: return aRollOff.getDescriptor();
  case 8: return aSpectralComplexity.getDescriptor();
  case 9: return aStrongPeak.getDescriptor();

  case 10: return aPitchSalience.getDescriptor();
  case 11: return aPitch.getDescriptor();
  case 12: return pDissonance.getDescriptor();
  case 13: return pInharmonicity.getDescriptor();
  case 14: return pTristimulus.getDescriptor();

  case 15: return aDistributionShape.getDescriptor();
  case 16: return aBarkShape.getDescriptor();

  case 17: return aSpectralContrast.getDescriptor();
  case 18: return bFlatness.getDescriptor();

  case 19: return aLPC.getDescriptor();

  case 20: return tRMS.getDescriptor();
  case 21: return tStrongDecay.getDescriptor();
  case 22: return tLarm.getDescriptor();

  case 23: return bFlux.getDescriptor();

  case 24: return pFriction.getDescriptor();
  case 25: return pCrunchiness.getDescriptor();

  case 26: return aCrest.getDescriptor();
  case 27: return aPerculleys.getDescriptor();
  case 28: return tLoudnessVickers.getDescriptor();
  case 29: return pHpcp.getDescriptor();
  case 30: return pSpectralWhithening.getDescriptor();
  case 31: return pSpectralPeaks.getDescriptor();
  case 32: return aOnsetDetection.getDescriptor();
  case 33: return aOnsets.getDescriptor();
  case 34: return mRhythmTransform.getDescriptor();
  case 35: return mSbic.getDescriptor();
  case 36: return aNoveltyCurve.getDescriptor();
  case 37: return pOddToEvenHarmonicEnergyRatio.getDescriptor();
  case 38: return aBpmHistogram.getDescriptor();

  case 39: return aERBBands.getDescriptor();
  case 40: return aGFCC.getDescriptor();
  case 41: return tZeroCrossingRate.getDescriptor();

  default:
    return 0;
  }
}


class Gloub {
public:
  Gloub() {
    essentia::init();
  }
};

static Gloub goulou;

//int main() { essentia::init(); return 0; }
