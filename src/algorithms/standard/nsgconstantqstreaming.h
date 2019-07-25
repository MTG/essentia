#ifndef ESSENTIA_NSGCONSTANTQSTREAMING_H
#define ESSENTIA_NSGCONSTANTQSTREAMING_H

#include "streamingalgorithmcomposite.h"
#include "algorithmfactory.h"
#include "network.h"


namespace essentia {
namespace streaming {

class NSGConstantQStreaming : public Algorithm{
 protected:
  SinkProxy<std::vector<Real> >_frame;

  Sink<std::vector<std::vector<std::complex<Real> > > >_constantQinner;
  Sink<std::vector<std::complex<Real> > > _constantQDCinner;
  Sink<std::vector<std::complex<Real> > > _constantQNFinner;

  Source<std::vector<std::complex<Real> > > _constantQ;
  Source<std::vector<std::complex<Real> > > _constantQDC;
  Source<std::vector<std::complex<Real> > > _constantQNF;

  Source<int> _frameStamps;

  int _frameStampsCount;

  Algorithm* _wrapper;

 public:
  NSGConstantQStreaming();

  ~NSGConstantQStreaming() {};

  AlgorithmStatus process();
 
  void declareParameters() {
    declareParameter("inputSize", "the size of the input", "(0,inf)", 4096);
    declareParameter("minFrequency", "the minimum frequency", "(0,inf)", 27.5);
    declareParameter("maxFrequency", "the maximum frequency", "(0,inf)", 7040.);
    declareParameter("binsPerOctave", "the number of bins per octave", "[1,inf)", 48);
    declareParameter("sampleRate", "the desired sampling rate [Hz]", "[0,inf)", 44100.);
    declareParameter("rasterize", "hop sizes for each frequency channel. With 'none' each frequency channel is distinct. 'full' sets the hop sizes of all the channels to the smallest. 'piecewise' rounds down the hop size to a power of two", "{none,full,piecewise}", "full");
    declareParameter("phaseMode", "'local' to use zero-centered filters. 'global' to use a phase mapping function as described in [1]", "{local,global}", "global");
    declareParameter("gamma", "The bandwidth of each filter is given by Bk = 1/Q * fk + gamma", "[0,inf)", 0);
    declareParameter("normalize", "coefficient normalization", "{sine,impulse,none}", "none");
    declareParameter("window","the type of window for the frequency filters. It is not recommended to change the default window.","{hamming,hann,hannnsgcq,triangular,square,blackmanharris62,blackmanharris70,blackmanharris74,blackmanharris92}","hannnsgcq");
    declareParameter("minimumWindow", "minimum size allowed for the windows", "[2,inf)", 4);
    declareParameter("windowSizeFactor", "window sizes are rounded to multiples of this", "[1,inf)", 1);
    }

  void configure();

  static const char* name;
  static const char* category;
  static const char* description;

};

}
}

#endif // ESSENTIA_NSGCONSTANTQSTREAMING_H
