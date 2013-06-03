#include "essentia.h"
#include "algorithmfactory.h"
#include "fileoutputproxy.h"
//#include "testsignal.h"
#include "inputcopy.h"
#include "audioengine.h"
#include "devnull.h"
#include <cstdio>

int main() {
  using essentia::streaming::NOWHERE;
  using essentia::streaming::runGenerator;
  using essentia::streaming::runTask;
  using essentia::streaming::checkConnections;
  using essentia::streaming::checkBufferSizes;
  using essentia::streaming::AlgorithmFactory;
  using essentia::StreamingAlgorithm;
  using essentia::streaming::FileOutputProxy;
  using essentia::streaming::InputCopy;
     
  essentia::init();
  
  essentia::streaming::AlgorithmFactory::Registrar<essentia::streaming::InputCopy> regInputCopy;

  //StreamingAlgorithm* generator = AlgorithmFactory::create("MonoLoader","filename","test.wav");
  StreamingAlgorithm* generator =  AlgorithmFactory::create("InputCopy");
  StreamingAlgorithm* frameCutter = AlgorithmFactory::create("FrameCutter");
  StreamingAlgorithm* windowing = AlgorithmFactory::create("Windowing");
  StreamingAlgorithm* spectrum = AlgorithmFactory::create("Spectrum");
  StreamingAlgorithm* spectralWhitening = AlgorithmFactory::create("SpectralWhitening");
  StreamingAlgorithm* spectralPeaks = AlgorithmFactory::create("SpectralPeaks");
  StreamingAlgorithm* hpcp = AlgorithmFactory::create("HPCP","size",36);
  StreamingAlgorithm* valueFile = AlgorithmFactory::create("FileOutput");

  connect(generator,"signal",frameCutter,"signal");
  //connect(generator,"audio",frameCutter,"signal");
  connect(frameCutter,"frame",windowing,"frame");
  connect(windowing,"frame",spectrum,"frame");
  connect(spectrum,"spectrum",spectralPeaks,"spectrum");
  connect(spectrum,"spectrum",spectralWhitening,"spectrum");
  connect(spectralPeaks,"magnitudes",spectralWhitening,"magnitudes");
  connect(spectralPeaks,"frequencies",spectralWhitening,"frequencies");
  connect(spectralPeaks,"frequencies",hpcp,"frequencies");
  connect(spectralWhitening,"magnitudes",hpcp,"magnitudes");
  //connect(hpcp->output("hpcp"),*dynamic_cast<FileOutputProxy*>(valueFile));
  connect(hpcp->output("hpcp"),NOWHERE); 


  checkConnections(generator);
  checkBufferSizes(generator);
  
  InputCopy* inputCopy = dynamic_cast<InputCopy*>(generator);

  AudioEngine engine(inputCopy);

  engine.run();
 
  getchar();

  //float tmp[1024];
  //while (!generator->shouldStop())
  //{
  //  inputCopy->setInput(tmp,1024);
  //  runTask(generator);
  //}
}

