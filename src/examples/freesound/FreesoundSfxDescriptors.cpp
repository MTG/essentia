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


#include "FreesoundSfxDescriptors.h"
using namespace std;
using namespace essentia;
using namespace essentia::streaming;


using namespace std;

const string FreesoundSfxDescriptors::nameSpace="sfx.";  

void  FreesoundSfxDescriptors::createNetwork(SourceBase& source, Pool& pool){
  
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  // Envelope
  Algorithm* envelope = factory.create("Envelope");
  connect(source, envelope->input("signal"));

  // Temporal Statistics
  Algorithm* decrease = factory.create("Decrease");
  Algorithm* accu = factory.create("RealAccumulator");
  connect(envelope->output("signal"), accu->input("data"));
  connect(accu->output("array"), decrease->input("array"));
  connect(decrease->output("decrease"), pool, nameSpace + "temporal_decrease");

  // Audio Central Moments Statistics
  Algorithm* cm = factory.create("CentralMoments");
  Algorithm* ds = factory.create("DistributionShape");
  connect(accu->output("array"), cm->input("array"));
  connect(cm->output("centralMoments"), ds->input("centralMoments"));
  connect(ds->output("kurtosis"), pool, nameSpace + "temporal_kurtosis");
  connect(ds->output("spread"), pool, nameSpace + "temporal_spread");
  connect(ds->output("skewness"), pool, nameSpace + "temporal_skewness");

  // Spectral Centroid
  Algorithm* centroid = factory.create("Centroid");
  connect(accu->output("array"), centroid->input("array"));
  connect(centroid->output("centroid"), pool, nameSpace + "temporal_centroid");

  // Effective Duration
  Algorithm* duration = factory.create("EffectiveDuration");
  connect(accu->output("array"), duration->input("signal"));
  connect(duration->output("effectiveDuration"), pool, nameSpace + "effective_duration");

  // Log Attack Time
  Algorithm* log = factory.create("LogAttackTime");
  connect(accu->output("array"), log->input("signal"));
  connect(log->output("logAttackTime"), pool, nameSpace + "logattacktime");

  // Strong Decay
  Algorithm* decay = factory.create("StrongDecay");
  connect(envelope->output("signal"), decay->input("signal"));
  connect(decay->output("strongDecay"), pool, nameSpace + "strongdecay");

  // Flatness
  Algorithm* flatness = factory.create("FlatnessSFX");
  connect(accu->output("array"), flatness->input("envelope"));
  connect(flatness->output("flatness"), pool, nameSpace + "flatness");

  // Morphological Descriptors
  Algorithm* max1 = factory.create("MaxToTotal");
  connect(envelope->output("signal"), max1->input("envelope"));
  connect(max1->output("maxToTotal"), pool, nameSpace + "max_to_total");

  Algorithm* tc = factory.create("TCToTotal");
  connect(envelope->output("signal"), tc->input("envelope"));
  connect(tc->output("TCToTotal"), pool, nameSpace + "tc_to_total");

  Algorithm* der = factory.create("DerivativeSFX");
  connect(accu->output("array"), der->input("envelope"));
  connect(der->output("derAvAfterMax"), pool, nameSpace + "der_av_after_max");
  connect(der->output("maxDerBeforeMax"), pool, nameSpace + "max_der_before_max");
}


void  FreesoundSfxDescriptors::createPitchNetwork(SourceBase& source, Pool& pool){

  vector<Real> pitch = pool.value<vector<Real> >("lowlevel.pitch");

  standard::Algorithm* maxtt = standard::AlgorithmFactory::create("MaxToTotal");
  Real maxToTotal;
  maxtt->input("envelope").set(pitch);
  maxtt->output("maxToTotal").set(maxToTotal);
  maxtt->compute();
  pool.set(nameSpace + "pitch_max_to_total", maxToTotal);

  standard::Algorithm* mintt = standard::AlgorithmFactory::create("MinToTotal");
  Real minToTotal;
  mintt->input("envelope").set(pitch);
  mintt->output("minToTotal").set(minToTotal);
  mintt->compute();
  pool.set(nameSpace + "pitch_min_to_total", minToTotal);

  standard::Algorithm* pc = standard::AlgorithmFactory::create("Centroid");
  Real centroid;
  pc->configure("range", (uint)pitch.size() - 1);
  pc->input("array").set(pitch);
  pc->output("centroid").set(centroid);
  pc->compute();
  pool.set(nameSpace + "pitch_centroid", centroid);

  standard::Algorithm* amt = standard::AlgorithmFactory::create("AfterMaxToBeforeMaxEnergyRatio");
  Real ratio;
  amt->input("pitch").set(pitch);
  amt->output("afterMaxToBeforeMaxEnergyRatio").set(ratio);
  amt->compute();
  pool.set(nameSpace + "pitch_after_max_to_before_max_energy_ratio", ratio);
}


