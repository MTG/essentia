#import "avfoundationmonoloader.hpp"
#import "AVFoundationLoader.hpp"
#import "algorithmfactory.h"

using namespace std;

namespace essentia {
namespace streaming {

const char* AVMonoLoader::name = "AVMonoLoader";
const char* AVMonoLoader::category = "Input/output";
const char* AVMonoLoader::description = DOC("A MonoLoader using AVFoundation.");


AVMonoLoader::AVMonoLoader() : AlgorithmComposite(),
               _audioLoader(0), _mixer(0), _resample(0), _configured(false) {

  declareOutput(_audio, "audio", "the mono audio signal");

  AlgorithmFactory& factory = AlgorithmFactory::instance();

  _audioLoader = factory.create("AVAudioLoader");
  _mixer       = factory.create("MonoMixer");
  _resample    = factory.create("Resample");

  _audioLoader->output("audio")           >>  _mixer->input("audio");
  _audioLoader->output("numberChannels")  >>  _mixer->input("numberChannels");
  _mixer->output("audio")                 >>  _resample->input("signal");

  _audioLoader->output("md5")        >> NOWHERE;
  _audioLoader->output("bit_rate")   >> NOWHERE;
  _audioLoader->output("codec")      >> NOWHERE;
  _audioLoader->output("sampleRate") >> NOWHERE;

  attach(_resample->output("signal"), _audio);
}

void AVMonoLoader::configure() {
  Parameter filename = parameter("filename");
  // if no file has been specified, do not do anything
  if (!filename.isConfigured()) return;

  _audioLoader->configure("filename", filename,
              "computeMD5", false,
              INHERIT("audioStream"));

  int inputSampleRate = (int)lastTokenProduced<Real>(_audioLoader->output("sampleRate"));

  // TODO: this should probably be turned into a source as well, same as what's done above for audioLoader->sampleRate
  // also keep it as a parameter (ugly), but act as an optional source (no need
  // to connect, etc...)
  _params.add("originalSampleRate", inputSampleRate);

  _resample->configure("inputSampleRate", inputSampleRate,
             "outputSampleRate", parameter("sampleRate"));

  _mixer->configure("type", parameter("downmix"));

}

} // namespace streaming
} // namespace essentia
