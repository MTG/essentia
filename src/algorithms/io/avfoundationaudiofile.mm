#import "avfoundationaudiofile.hpp"

#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>

AVFoundationAudioFile::~AVFoundationAudioFile() {
	if (_file) {
		CFBridgingRelease(_file);
		_file = NULL;
	}
	if (_buffer) {
		CFBridgingRelease(_buffer);
		_buffer = NULL;
	}
}

AVFoundationAudioFile::AVFoundationAudioFile(const std::string& filename, const int bufferLength) {
	// Read file
	
	NSError *error = nil;
	NSURL *url = [NSURL fileURLWithPath: [NSString stringWithUTF8String: filename.c_str()]];
	AVAudioFile *file = [[AVAudioFile alloc] initForReading: url error: &error];

	if (error) {
		throw std::runtime_error([[error localizedDescription] UTF8String]);
	}
	
	_file = (__bridge_retained void*) file;
	
	// Read metadata for convenient access
	
	length = [file length];
	
	AVAudioFormat *format = [file processingFormat];
	
	channels = [format channelCount];
	sample_rate = [format sampleRate];
	codec = [[format className] UTF8String];  // TODO
	int bitDepth = [[[format settings] objectForKey: AVLinearPCMBitDepthKey] intValue];
	bit_rate = (int) (channels * sample_rate * bitDepth);
	is_interleaved = [format isInterleaved];
	
	// Create buffer
	
	AVAudioPCMBuffer *pcmBuffer = [[AVAudioPCMBuffer alloc] initWithPCMFormat:format frameCapacity: bufferLength / bitDepth];
	_buffer = (__bridge_retained void*) pcmBuffer;
	buffers = (float **)[pcmBuffer floatChannelData];
	AVFoundationAudioFile::frameLength = pcmBuffer.frameLength = 0;
	stride = [pcmBuffer stride];
}

int AVFoundationAudioFile::readNext() {
	AVAudioFile *file = (__bridge AVAudioFile *) _file;
	AVAudioPCMBuffer *pcmBuffer = (__bridge AVAudioPCMBuffer *) _buffer;
	NSError *error = nil;
	
	[file readIntoBuffer:pcmBuffer error: &error];
	
	if (error) {
		throw std::runtime_error([[error localizedDescription] UTF8String]);
	}
	
	frameLength = pcmBuffer.frameLength;
	return frameLength;
}

void AVFoundationAudioFile::setFramePosition(const uint64_t framePosition) {
	AVAudioFile *file = (__bridge AVAudioFile *) _file;
	file.framePosition = framePosition;
}

const uint64_t AVFoundationAudioFile::getFramePosition() {
	AVAudioFile *file = (__bridge AVAudioFile *) _file;
	return file.framePosition;
}
