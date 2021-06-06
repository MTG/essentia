#ifndef AVFoundationCPPGlue_hpp
#define AVFoundationCPPGlue_hpp

#include <stdio.h>
#include <string>
#include <stdexcept>

class AVFoundationAudioFile {
private:
	void *_file;
	void *_buffer;

public:
	// Note: Unfortunately, Apple's API demands reading buffers into AVAudioPCMBuffer.
	// Reading into custom memory is not allowed. So this class maintains a buffer - if
	// reading into outside buffers is desired, read using this class, and then memcpy.

	AVFoundationAudioFile(const std::string& filename, const int bufferLength);
	~AVFoundationAudioFile();
	
	// Returns new frameLength
	int readNext();
	
	const uint64_t getFramePosition();
	void setFramePosition(const uint64_t framePosition);

	float **buffers;
	// Current number of valid frames in the buffer
	int frameLength;
	
	uint64_t length;

	int channels;
	float sample_rate;
	std::string codec;
	int bit_rate;
	bool is_interleaved;
	int stride;
};

#endif /* AVFoundationCPPGlue_hpp */
