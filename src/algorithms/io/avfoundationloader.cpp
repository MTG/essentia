#import "avfoundationloader.hpp"
#import "algorithmfactory.h"
#import <iomanip>  //  setw()

using namespace std;

namespace essentia {
namespace streaming {

const char* AVAudioLoader::name = "AVAudioLoader";
const char* AVAudioLoader::category = "Input/output";
const char* AVAudioLoader::description = DOC("An AudioLoader using AVFoundation.");


AVAudioLoader::~AVAudioLoader() {
	closeAudioFile();
}

void AVAudioLoader::configure() {
	_computeMD5 = parameter("computeMD5").toBool();
	_selectedStream = parameter("audioStream").toInt();
	reset();
}


void AVAudioLoader::openAudioFile(const string& filename) {
	E_DEBUG(EAlgorithm, "AVAudioLoader: opening file: " << filename);
	
	_file = new AVFoundationAudioFile(filename, BUFFER_SIZE);
}


void AVAudioLoader::closeAudioFile() {
	if (_file) {
		delete _file;
		_file = NULL;
	}
}


void AVAudioLoader::pushChannelsSampleRateInfo(int nChannels, Real sampleRate) {
	if (nChannels > 2) {
		throw EssentiaException("AVAudioLoader: could not load audio. Audio file has more than 2 channels.");
	}
	if (sampleRate <= 0) {
		throw EssentiaException("AVAudioLoader: could not load audio. Audio sampling rate must be greater than 0.");
	}

	_nChannels = nChannels;

	_channels.push(nChannels);
	_sampleRate.push(sampleRate);
}


void AVAudioLoader::pushCodecInfo(std::string codec, int bit_rate) {
	_codec.push(codec);
	_bit_rate.push(bit_rate);
}


AlgorithmStatus AVAudioLoader::process() {
	if (!parameter("filename").isConfigured()) {
		throw EssentiaException("AVAudioLoader: Trying to call process() on an AVAudioLoader algo which hasn't been correctly configured.");
	}
	if (_computeMD5) {
		throw EssentiaException("AVAudioLoader: computeMD5 is not implemented.");
	}

	int framesRead = _file->readNext();
	if (!framesRead) {
		shouldStop(true);
//		copyOutput();
		closeAudioFile();
		
		string md5 = "";
		_md5.push(md5);
		
		return FINISHED;
	}
	
	copyOutput();
	
	return OK;
}

void AVAudioLoader::copyOutput() {
	int nsamples = _file->frameLength;
	int stride = _file->stride;

	bool ok = _audio.acquire(nsamples);
	if (!ok) {
		throw EssentiaException("AudioLoader: could not acquire output for audio");
	}

	vector<StereoSample>& audio = *((vector<StereoSample>*)_audio.getTokens());

	if (_nChannels == 1) {
		float *buffer = _file->buffers[0];
		
		for (int i=0; i<nsamples; i++) {
			audio[i].left() = *buffer;
			buffer += stride;
		}
	}
	else { // _nChannels == 2
		float *left = _file->buffers[0];
		float *right = _file->buffers[1];

		for (int i=0, loc=0; i<nsamples; i++) {
			audio[i].left() = *left;
			audio[i].right() = *right;
			left += stride;
			right += stride;
		}
	}

	_audio.release(nsamples);
}

void AVAudioLoader::reset() {
	Algorithm::reset();

	if (!parameter("filename").isConfigured()) return;

	string filename = parameter("filename").toString();

	closeAudioFile();
	openAudioFile(filename);

	pushChannelsSampleRateInfo(_file->channels, _file->sample_rate);
	pushCodecInfo(_file->codec, _file->bit_rate);
}

} // namespace streaming
} // namespace essentia
