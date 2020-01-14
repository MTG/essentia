/*
 * Copyright (C) 2006-2020  Music Technology Group - Universitat Pompeu Fabra
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
 * You should ha ve received a copy of the Affero GNU General Public License
 * version 3 along with this program.  If not, see http://www.gnu.org/licenses/
 */

#include "harmonicmodelanal.h"
#include "essentiamath.h"
#include <essentia/utils/synth_utils.h>

using namespace essentia;
using namespace standard;

const char* HarmonicModelAnal::name = "HarmonicModelAnal";
const char* HarmonicModelAnal::category = "Synthesis";
const char* HarmonicModelAnal::description = DOC("This algorithm computes the harmonic model analysis.\n"
"\n"
"This algorithm uses SineModelAnal and keeps only the harmonic partials. It receives an external pitch value as input. You can use PitchYinFft algorithm to compute the pitch per frame.\n"
"\n"
"References:\n"
"  https://github.com/MTG/sms-tools\n"
"  http://mtg.upf.edu/technologies/sms\n"
);



void HarmonicModelAnal::configure() {                      
  _sineModelAnal->configure( "sampleRate", parameter("sampleRate").toReal(),
                            // sinusoidal tracking
                              "maxnSines", parameter("maxnSines").toInt() ,
                              "maxPeaks", parameter("maxPeaks").toInt() ,                           
                              "freqDevOffset", parameter("freqDevOffset").toReal(),
                              "freqDevSlope",  parameter("freqDevSlope").toReal(),
                              "magnitudeThreshold", parameter("magnitudeThreshold").toReal(),
                              "orderBy", parameter("orderBy")
                              );


  // get parameters
  _sampleRate =  parameter("sampleRate").toReal();
  _nH = parameter("nHarmonics").toInt() ; // number of harmonics
 _harmDevSlope =  parameter("harmDevSlope").toReal();
 
 _lasthfreq.clear();
}


void HarmonicModelAnal::compute() {

  // inputs and outputs
  const std::vector<std::complex<Real> >& fftin = _fft.get();
  const Real& pitch = _pitch.get();

  std::vector<Real>& hpeakMagnitude = _magnitudes.get();
  std::vector<Real>& hpeakFrequency = _frequencies.get();
  std::vector<Real>& hpeakPhase = _phases.get();
 
  std::vector<Real> fftmag;
  std::vector<Real> fftphase;

  std::vector<Real> peakMagnitude ;
  std::vector<Real> peakFrequency;
  std::vector<Real> peakPhase;
 

  _sineModelAnal->input("fft").set(fftin);
  _sineModelAnal->output("magnitudes").set(peakMagnitude);
  _sineModelAnal->output("frequencies").set(peakFrequency);
  _sineModelAnal->output("phases").set(peakPhase);
  _sineModelAnal->compute();


  harmonicDetection(peakFrequency, peakMagnitude, peakPhase, pitch, _nH, _lasthfreq,  _sampleRate,  _harmDevSlope,  hpeakFrequency,  hpeakMagnitude,  hpeakPhase);

	_lasthfreq = hpeakFrequency;  // copy last harmonic frequencies for tracking


}


// ---------------------------
// additional methods


void HarmonicModelAnal::harmonicDetection(const std::vector<Real> pfreq, const std::vector<Real> pmag, const std::vector<Real> pphase, const Real f0, const int nH,  std::vector<Real> hfreqp, Real fs, Real harmDevSlope/*=0.01*/,  std::vector<Real> &hfreq,  std::vector<Real> &hmag,  std::vector<Real> &hphase)
{
/*
	Detection of the harmonics of a frame from a set of spectral peaks using f0
	to the ideal harmonic series built on top of a fundamental frequency
	pfreq, pmag, pphase: peak frequencies, magnitudes and phases
	f0: fundamental frequency, nH: number of harmonics,
	hfreqp: harmonic frequencies of previous frame,
	fs: sampling rate; harmDevSlope: slope of change of the deviation allowed to perfect harmonic
	returns hfreq, hmag, hphase: harmonic frequencies, magnitudes, phases

*/


	// init vectors  
	hfreq.resize(_nH); // initialize harmonic frequencies
	std::fill(hfreq.begin(), hfreq.end(), 0.);
	hmag.resize(nH); // initialize harmonic magnitudes
	std::fill(hmag.begin(), hmag.end(), -100.);
	hphase.resize(nH); // initialize harmonic phases
	std::fill(hphase.begin(), hphase.end(), 0.);


	if (f0 > 0)
		{
			std::vector<Real> hf(nH);
			for (int i=1;i<=nH;i++)
			{
				hf [i-1]= f0*i;       
			}		
      
		int hi = 0 ;                                                          // initialize harmonic index
		if (hfreqp.size() == 0)                                   // if no incomming harmonic tracks initialize to harmonic series
			{
				hfreqp = hf;			
			}
			
			std::vector<Real> difftmp =  pfreq;
      
      
		while ((f0>0) && (hi<nH) && (hf[hi]<fs/2.))          // find harmonic peaks
		{	
			
			for (int j=0;j<(int) difftmp.size(); j++)
			{
				difftmp[j] = abs(pfreq[j] - hf[hi]);      
			}
      
      
			 // get closest peak
			int pei = std::min_element(difftmp.begin(), difftmp.end()) - difftmp.begin();
			      
			Real dev1 = abs(pfreq[pei] - hf[hi])  ;                // deviation from perfect harmonic			      
			 Real dev2 =  fs; 														// deviation from previous frame
			 if (hfreqp[hi]>0 )
			 {
				 dev2 = abs(pfreq[pei] - hfreqp[hi]);
			 }
			
			Real threshold = f0/3. + harmDevSlope * pfreq[pei];
        
			if ((dev1<threshold) || (dev2<threshold))         //accept peak if deviation is small
			{
				hfreq[hi] = pfreq[pei]  ;                        // harmonic frequencies
				hmag[hi] = pmag[pei] ;                         // harmonic magnitudes
				hphase[hi] = pphase[pei] ;                  // harmonic phases                     
			}
			hi += 1 ;                                   				// increase harmonic index
		}
	}	
  
	return ;

}
