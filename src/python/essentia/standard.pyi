from . import _essentia
import numpy as np
from numpy.typing import NDArray
from essentia import Pool
from typing import Any


class AfterMaxToBeforeMaxEnergyRatio(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Computes the ratio between the pitch energy after the pitch maximum and the pitch energy before the pitch maximum.
		
		Sounds having an monotonically ascending pitch or one unique pitch will show a value of (0,1], while sounds having a monotonically descending pitch will show a value of [1,inf). In case there is no energy before the max pitch, the algorithm will return the energy after the maximum pitch.
		
		The algorithm throws exception when input is either empty or contains only zeros.		""" 
		... 
	def __call__(self, pitch:NDArray[np.float32]) -> float:
		"""compute
		Args:
			pitch (NDArray[np.float32]): the array of pitch values [Hz]. Defaults to None. 
		Returns:
			afterMaxToBeforeMaxEnergyRatio (float): the ratio between the pitch energy after the pitch maximum to the pitch energy before the pitch maximum
		""" 
		... 


class AllPass(_essentia.Algorithm): 
	def __init__(self, bandwidth:float=500.0, cutoffFrequency:float=1500.0, order:int=1, sampleRate:float=44100.0) -> None:
		"""Implements a IIR all-pass filter of order 1 or 2.
		
		Because of its dependence on IIR, IIR's requirements are inherited.
		
		References:
		  [1] U. Zölzer, DAFX - Digital Audio Effects, p. 43,
		  John Wiley & Sons, 2002

		Args:
			bandwidth (float): the bandwidth of the filter [Hz] (used only for 2nd-order filters). Defaults to 500.0. Range (0,inf)
			cutoffFrequency (float): the cutoff frequency for the filter [Hz]. Defaults to 1500.0. Range (0,inf)
			order (int): the order of the filter. Defaults to 1. Range {1,2}
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input signal. Defaults to None. 
		Returns:
			signal (NDArray[np.float32]): the filtered signal
		""" 
		... 


class Audio2Midi(_essentia.Algorithm): 
	def __init__(self, applyTimeCompensation:bool=True, hopSize:int=32, loudnessThreshold:float=-51.0, maxFrequency:float=2300.0, midiBufferDuration:float=0.05, minFrequency:float=60.0, minNoteChangePeriod:float=0.03, minOccurrenceRate:float=0.5, minOffsetCheckPeriod:float=0.2, minOnsetCheckPeriod:float=0.075, pitchConfidenceThreshold:float=0.25, sampleRate:int=44100, transpositionAmount:int=0, tuningFrequency:int=440) -> None:
		"""Audio2Pitch and Pitch2Midi for real time application.
		
		This algorithm has a state that is used to estimate note on/off events based on consequent compute() calls.

		Args:
			applyTimeCompensation (bool): whether to apply time compensation correction to MIDI note detection. Defaults to True. Range {true,false}
			hopSize (int): equivalent to I/O buffer size. Defaults to 32. Range [1,inf)
			loudnessThreshold (float): loudness level above/below which note ON/OFF start to be considered, in decibels. Defaults to -51.0. Range [-inf,0]
			maxFrequency (float): maximum frequency to detect in Hz. Defaults to 2300.0. Range [10,20000]
			midiBufferDuration (float): duration in seconds of buffer used for voting in MidiPool algorithm. Defaults to 0.05. Range [0.005,0.5]
			minFrequency (float): minimum frequency to detect in Hz. Defaults to 60.0. Range [10,20000]
			minNoteChangePeriod (float): minimum time to wait until a note change is detected (testing only). Defaults to 0.03. Range (0,1]
			minOccurrenceRate (float): rate of predominant pitch occurrence in MidiPool buffer to consider note ON event. Defaults to 0.5. Range [0,1]
			minOffsetCheckPeriod (float): minimum time to wait until an offset is detected (testing only). Defaults to 0.2. Range (0,1]
			minOnsetCheckPeriod (float): minimum time to wait until an onset is detected (testing only). Defaults to 0.075. Range (0,1]
			pitchConfidenceThreshold (float): level of pitch confidence above which note ON/OFF start to be considered. Defaults to 0.25. Range [0,1]
			sampleRate (int): sample rate of incoming audio frames. Defaults to 44100. Range [8000,inf)
			transpositionAmount (int): Apply transposition (in semitones) to the detected MIDI notes.. Defaults to 0. Range (-69,50)
			tuningFrequency (int): tuning frequency for semitone index calculation, corresponding to A3 [Hz]. Defaults to 440. Range {432,440} 
		""" 
		... 
	def __call__(self, frame:NDArray[np.float32]) -> tuple[float, float, list[str], NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			frame (NDArray[np.float32]): the input frame to analyse. Defaults to None. 
		Returns:
			pitch (float): pitch given in Hz
			loudness (float): detected loudness in decibels
			messageType (list[str]): the output of MIDI message type, as string, {noteoff, noteon, noteoff-noteon}
			midiNoteNumber (NDArray[np.float32]): the output of detected MIDI note number, as integer, in range [0,127]
			timeCompensation (NDArray[np.float32]): time to be compensated in the messages
		""" 
		... 


class Audio2Pitch(_essentia.Algorithm): 
	def __init__(self, frameSize:int=1024, loudnessThreshold:float=-51.0, maxFrequency:float=2300.0, minFrequency:float=60.0, pitchAlgorithm:str='pitchyinfft', pitchConfidenceThreshold:float=0.25, sampleRate:int=44100, tolerance:float=1.0, weighting:str='custom') -> None:
		"""Computes pitch with various pitch algorithms, specifically targeted for real-time pitch detection on audio signals.
		
		The algorithm internally uses pitch estimation with PitchYin (pitchyin) and PitchYinFFT (pitchyinfft).

		Args:
			frameSize (int): size of input frame in samples. Defaults to 1024. Range [1,inf)
			loudnessThreshold (float): loudness level above/below which note ON/OFF start to be considered, in decibels. Defaults to -51.0. Range [-inf,0]
			maxFrequency (float): maximum frequency to detect in Hz. Defaults to 2300.0. Range [10,20000]
			minFrequency (float): minimum frequency to detect in Hz. Defaults to 60.0. Range [10,20000]
			pitchAlgorithm (str): pitch algorithm to use. Defaults to 'pitchyinfft'. Range {pitchyin,pitchyinfft}
			pitchConfidenceThreshold (float): level of pitch confidence above/below which note ON/OFF start to be considered. Defaults to 0.25. Range [0,1]
			sampleRate (int): sample rate of incoming audio frames. Defaults to 44100. Range [8000,inf)
			tolerance (float): sets tolerance for peak detection on pitch algorithm. Defaults to 1.0. Range [0,1]
			weighting (str): string to assign a weighting function. Defaults to 'custom'. Range {custom,A,B,C,D,Z} 
		""" 
		... 
	def __call__(self, frame:NDArray[np.float32]) -> tuple[float, float, float, int]:
		"""compute
		Args:
			frame (NDArray[np.float32]): the input frame to analyse. Defaults to None. 
		Returns:
			pitch (float): detected pitch in Hz
			pitchConfidence (float): confidence of detected pitch (from 0.0 to 1.0)
			loudness (float): detected loudness in decibels
			voiced (int): voiced frame categorization, 1 for voiced and 0 for unvoiced frame
		""" 
		... 


class AudioLoader(_essentia.Algorithm): 
	def __init__(self, filename:str, audioStream:int=0, computeMD5:bool=False) -> None:
		"""Loads the single audio stream contained in a given audio or video file.
		
		Supported formats are all those supported by the FFmpeg library including wav, aiff, flac, ogg and mp3.
		
		This algorithm will throw an exception if it was not properly configured which is normally due to not specifying a valid filename. Invalid names comprise those with extensions different than the supported  formats and non existent files. If using this algorithm on Windows, you must ensure that the filename is encoded as UTF-8
		
		Note: ogg files are decoded in reverse phase, due to be using ffmpeg library.
		
		References:
		  [1] WAV - Wikipedia, the free encyclopedia,
		      http://en.wikipedia.org/wiki/Wav
		  [2] Audio Interchange File Format - Wikipedia, the free encyclopedia,
		      http://en.wikipedia.org/wiki/Aiff
		  [3] Free Lossless Audio Codec - Wikipedia, the free encyclopedia,
		      http://en.wikipedia.org/wiki/Flac
		  [4] Vorbis - Wikipedia, the free encyclopedia,
		      http://en.wikipedia.org/wiki/Vorbis
		  [5] MP3 - Wikipedia, the free encyclopedia,
		      http://en.wikipedia.org/wiki/Mp3

		Args:
			audioStream (int): audio stream index to be loaded. Other streams are no taken into account (e.g. if stream 0 is video and 1 is audio use index 0 to access it.). Defaults to 0. Range [0,inf)
			computeMD5 (bool): compute the MD5 checksum. Defaults to False. Range {true,false}
			filename (str): the name of the file from which to read. Defaults to None. Range None 
		""" 
		... 
	def __call__(self, ) -> tuple[NDArray[np.float32], float, int, str, int, str]:
		"""compute
		Returns:
			audio (NDArray[np.float32]): the input audio signal
			sampleRate (float): the sampling rate of the audio signal [Hz]
			numberChannels (int): the number of channels
			md5 (str): the MD5 checksum of raw undecoded audio payload
			bit_rate (int): the bit rate of the input audio, as reported by the decoder codec
			codec (str): the codec that is used to decode the input audio
		""" 
		... 


class AudioOnsetsMarker(_essentia.Algorithm): 
	def __init__(self, onsets:NDArray[np.float32]=np.array([]), sampleRate:float=44100.0, type:str='beep') -> None:
		"""Creates a wave file in which a given audio signal is mixed with a series of time onsets.
		
		The sonification of the onsets can be heard as beeps, or as short white noise pulses if configured to do so.
		
		This algorithm will throw an exception if parameter "filename" is not supplied

		Args:
			onsets (NDArray[np.float32]): the list of onset locations [s]. Defaults to np.array([]). Range None
			sampleRate (float): the sampling rate of the output signal [Hz]. Defaults to 44100.0. Range (0,inf)
			type (str): the type of sound to be added on the event. Defaults to 'beep'. Range {beep,noise} 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input signal. Defaults to None. 
		Returns:
			signal (NDArray[np.float32]): the input signal mixed with bursts at onset locations
		""" 
		... 


class AudioWriter(_essentia.Algorithm): 
	def __init__(self, filename:str, bitrate:int=192, format:str='wav', sampleRate:float=44100.0) -> None:
		"""Encodes an input stereo signal into a stereo audio file.
		
		The algorithm uses the FFmpeg library. Supported formats are wav, aiff, mp3, flac and ogg. The default FFmpeg encoders are used for each format.
		
		An exception is thrown when other extensions are given. Note that to encode in mp3 format it is mandatory that FFmpeg was configured with mp3 enabled.

		Args:
			bitrate (int): the audio bit rate for compressed formats [kbps]. Defaults to 192. Range {32,40,48,56,64,80,96,112,128,144,160,192,224,256,320}
			filename (str): the name of the encoded file. Defaults to None. Range None
			format (str): the audio output format. Defaults to 'wav'. Range {wav,aiff,mp3,ogg,flac}
			sampleRate (float): the audio sampling rate [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, audio:NDArray[np.float32]) -> None:
		"""compute
		Args:
			audio (NDArray[np.float32]): the audio signal. Defaults to None. 
		""" 
		... 


class AutoCorrelation(_essentia.Algorithm): 
	def __init__(self, frequencyDomainCompression:float=0.5, generalized:bool=False, normalization:str='standard') -> None:
		"""Computes the autocorrelation vector of a signal.
		
		It uses the version most commonly used in signal processing, which doesn't remove the mean from the observations.
		Using the 'generalized' option this algorithm computes autocorrelation as described in [3].
		
		References:
		  [1] Autocorrelation -- from Wolfram MathWorld,
		  http://mathworld.wolfram.com/Autocorrelation.html
		
		  [2] Autocorrelation - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Autocorrelation
		
		  [3] Tolonen T., and Karjalainen, M. (2000). A computationally efficient multipitch analysis model.
		  IEEE Transactions on Audio, Speech, and Language Processing, 8(6), 708-716.
		
		

		Args:
			frequencyDomainCompression (float): factor at which FFT magnitude is compressed (only used if 'generalized' is set to true, see [3]). Defaults to 0.5. Range (0,inf)
			generalized (bool): bool value to indicate whether to compute the 'generalized' autocorrelation as described in [3]. Defaults to False. Range {true,false}
			normalization (str): type of normalization to compute: either 'standard' (default) or 'unbiased'. Defaults to 'standard'. Range {standard,unbiased} 
		""" 
		... 
	def __call__(self, array:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			array (NDArray[np.float32]): the array to be analyzed. Defaults to None. 
		Returns:
			autoCorrelation (NDArray[np.float32]): the autocorrelation vector
		""" 
		... 


class BFCC(_essentia.Algorithm): 
	def __init__(self, dctType:int=2, highFrequencyBound:float=11000.0, inputSize:int=1025, liftering:int=0, logType:str='dbamp', lowFrequencyBound:float=0.0, normalize:str='unit_sum', numberBands:int=40, numberCoefficients:int=13, sampleRate:float=44100.0, type:str='power', weighting:str='warping') -> None:
		"""Computes the bark-frequency cepstrum coefficients of a spectrum.
		
		Bark bands and their subsequent usage in cepstral analysis have shown to be useful in percussive content [1, 2]
		This algorithm is implemented using the Bark scaling approach in the Rastamat version of the MFCC algorithm and in a similar manner to the MFCC-FB40 default specs:
		
		http://www.ee.columbia.edu/ln/rosa/matlab/rastamat/
		  - filterbank of 40 bands from 0 to 11000Hz
		  - take the log value of the spectrum energy in each bark band
		  - DCT of the 40 bands down to 13 mel coefficients
		
		The parameters of this algorithm can be configured in order to behave like Rastamat [3] as follows:
		  - type = 'power' 
		  - weighting = 'linear'
		  - lowFrequencyBound = 0
		  - highFrequencyBound = 8000
		  - numberBands = 26
		  - numberCoefficients = 13
		  - normalize = 'unit_max'
		  - dctType = 3
		  - logType = 'log'
		  - liftering = 22
		
		In order to completely behave like Rastamat the audio signal has to be scaled by 2^15 before the processing and if the Windowing and FrameCutter algorithms are used they should also be configured as follows. 
		
		FrameGenerator:
		  - frameSize = 1102 
		  - hopSize = 441 
		  - startFromZero = True 
		  - validFrameThresholdRatio = 1 
		
		Windowing:
		  - type = 'hann' 
		  - size = 1102 
		  - zeroPadding = 946 
		  - normalized = False 
		
		This algorithm depends on the algorithms TriangularBarkBands (not the regular BarkBands algo as it is non-configurable) and DCT and therefore inherits their parameter restrictions. An exception is thrown if any of these restrictions are not met. The input "spectrum" is passed to the TriangularBarkBands algorithm and thus imposes TriangularBarkBands' input requirements. Exceptions are inherited by TriangualrBarkBands as well as by DCT.
		
		References:
		  [1] P. Herrera, A. Dehamel, and F. Gouyon, "Automatic labeling of unpitched percussion sounds in
		  Audio Engineering Society 114th Convention, 2003,
		  [2] W. Brent, "Cepstral Analysis Tools for Percussive Timbre Identification in
		  Proceedings of the 3rd International Pure Data Convention, Sao Paulo, Brazil, 2009,
		

		Args:
			dctType (int): the DCT type. Defaults to 2. Range [2,3]
			highFrequencyBound (float): the upper bound of the frequency range [Hz]. Defaults to 11000.0. Range (0,inf)
			inputSize (int): the size of input spectrum. Defaults to 1025. Range (1,inf)
			liftering (int): the liftering coefficient. Use '0' to bypass it. Defaults to 0. Range [0,inf)
			logType (str): logarithmic compression type. Use 'dbpow' if working with power and 'dbamp' if working with magnitudes. Defaults to 'dbamp'. Range {natural,dbpow,dbamp,log}
			lowFrequencyBound (float): the lower bound of the frequency range [Hz]. Defaults to 0.0. Range [0,inf)
			normalize (str): 'unit_max' makes the vertex of all the triangles equal to 1, 'unit_sum' makes the area of all the triangles equal to 1. Defaults to 'unit_sum'. Range {unit_sum,unit_max}
			numberBands (int): the number of bark bands in the filter. Defaults to 40. Range [1,inf)
			numberCoefficients (int): the number of output cepstrum coefficients. Defaults to 13. Range [1,inf)
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			type (str): use magnitude or power spectrum. Defaults to 'power'. Range {magnitude,power}
			weighting (str): type of weighting function for determining triangle area. Defaults to 'warping'. Range {warping,linear} 
		""" 
		... 
	def __call__(self, spectrum:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			spectrum (NDArray[np.float32]): the audio spectrum. Defaults to None. 
		Returns:
			bands (NDArray[np.float32]): the energies in bark bands
			bfcc (NDArray[np.float32]): the bark frequency cepstrum coefficients
		""" 
		... 


class BPF(_essentia.Algorithm): 
	def __init__(self, xPoints:NDArray[np.float32]=np.array([0, 1]), yPoints:NDArray[np.float32]=np.array([0, 1])) -> None:
		"""Implements a break point function which linearly interpolates between discrete xy-coordinates to construct a continuous function.
		
		Exceptions are thrown when the size the vectors specified in parameters is not equal and at least they contain two elements. Also if the parameter vector for x-coordinates is not sorted ascendantly. A break point function cannot interpolate outside the range specified in parameter "xPoints". In that case an exception is thrown.
		 
		References:
		  [1] Linear interpolation - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Linear_interpolation

		Args:
			xPoints (NDArray[np.float32]): the x-coordinates of the points forming the break-point function (the points must be arranged in ascending order and cannot contain duplicates). Defaults to np.array([0, 1]). Range None
			yPoints (NDArray[np.float32]): the y-coordinates of the points forming the break-point function. Defaults to np.array([0, 1]). Range None 
		""" 
		... 
	def __call__(self, x:float) -> float:
		"""compute
		Args:
			x (float): the input coordinate (x-axis). Defaults to None. 
		Returns:
			y (float): the output coordinate (y-axis)
		""" 
		... 


class BandPass(_essentia.Algorithm): 
	def __init__(self, bandwidth:float=500.0, cutoffFrequency:float=1500.0, sampleRate:float=44100.0) -> None:
		"""Implements a 2nd order IIR band-pass filter.
		
		Because of its dependence on IIR, IIR's requirements are inherited.
		
		References:
		  [1] U. Zölzer, DAFX - Digital Audio Effects, 2nd edition, p. 55,
		  John Wiley & Sons, 2011

		Args:
			bandwidth (float): the bandwidth of the filter [Hz]. Defaults to 500.0. Range (0,inf)
			cutoffFrequency (float): the cutoff frequency for the filter [Hz]. Defaults to 1500.0. Range (0,inf)
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input audio signal. Defaults to None. 
		Returns:
			signal (NDArray[np.float32]): the filtered signal
		""" 
		... 


class BandReject(_essentia.Algorithm): 
	def __init__(self, bandwidth:float=500.0, cutoffFrequency:float=1500.0, sampleRate:float=44100.0) -> None:
		"""Implements a 2nd order IIR band-reject filter.
		
		Because of its dependence on IIR, IIR's requirements are inherited.
		
		References:
		  [1] U. Zölzer, DAFX - Digital Audio Effects, 2nd edition, p. 55,
		  John Wiley & Sons, 2011

		Args:
			bandwidth (float): the bandwidth of the filter [Hz]. Defaults to 500.0. Range (0,inf)
			cutoffFrequency (float): the cutoff frequency for the filter [Hz]. Defaults to 1500.0. Range (0,inf)
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input signal. Defaults to None. 
		Returns:
			signal (NDArray[np.float32]): the filtered signal
		""" 
		... 


class BarkBands(_essentia.Algorithm): 
	def __init__(self, numberBands:int=27, sampleRate:float=44100.0) -> None:
		"""Computes energy in Bark bands of a spectrum.
		
		The band frequencies are: [0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0, 920.0, 1080.0, 1270.0, 1480.0, 1720.0, 2000.0, 2320.0, 2700.0, 3150.0, 3700.0, 4400.0, 5300.0, 6400.0, 7700.0, 9500.0, 12000.0, 15500.0, 20500.0, 27000.0]. The first two Bark bands [0,100] and [100,200] have been split in half for better resolution (because of an observed better performance in beat detection). For each bark band the power-spectrum (mag-squared) is summed.
		
		This algorithm uses FrequencyBands and thus inherits its input requirements and exceptions.
		
		References:
		  [1] The Bark Frequency Scale,
		  http://ccrma.stanford.edu/~jos/bbt/Bark_Frequency_Scale.html

		Args:
			numberBands (int): the number of desired barkbands. Defaults to 27. Range [1,28]
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range [0,inf) 
		""" 
		... 
	def __call__(self, spectrum:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			spectrum (NDArray[np.float32]): the input spectrum. Defaults to None. 
		Returns:
			bands (NDArray[np.float32]): the energy of the bark bands
		""" 
		... 


class BeatTrackerDegara(_essentia.Algorithm): 
	def __init__(self, maxTempo:int=208, minTempo:int=40) -> None:
		"""Estimates the beat positions given an input signal.
		
		It computes 'complex spectral difference' onset detection function and utilizes the beat tracking algorithm (TempoTapDegara) to extract beats [1]. The algorithm works with the optimized settings of 2048/1024 frame/hop size for the computation of the detection function, with its posterior x2 resampling.) While it has a lower accuracy than BeatTrackerMultifeature (see the evaluation results in [2]), its computational speed is significantly higher, which makes reasonable to apply this algorithm for batch processings of large amounts of audio signals.
		
		Note that the algorithm requires the audio input with the 44100 Hz sampling rate in order to function correctly.
		
		References:
		  [1] N. Degara, E. A. Rua, A. Pena, S. Torres-Guijarro, M. E. Davies, and
		  M. D. Plumbley, "Reliability-informed beat tracking of musical signals,"
		  IEEE Transactions on Audio, Speech, and Language Processing, vol. 20,
		  no. 1, pp. 290–301, 2012.
		
		  [2] J.R. Zapata, M.E.P. Davies and E. Gómez, "Multi-feature beat tracking,"
		  IEEE Transactions on Audio, Speech, and Language Processing, vol. 22,
		  no. 4, pp. 816-825, 2014.

		Args:
			maxTempo (int): the fastest tempo to detect [bpm]. Defaults to 208. Range [60,250]
			minTempo (int): the slowest tempo to detect [bpm]. Defaults to 40. Range [40,180] 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the audio input signal. Defaults to None. 
		Returns:
			ticks (NDArray[np.float32]):  the estimated tick locations [s]
		""" 
		... 


class BeatTrackerMultiFeature(_essentia.Algorithm): 
	def __init__(self, maxTempo:int=208, minTempo:int=40) -> None:
		"""Estimates the beat positions given an input signal.
		
		It computes a number of onset detection functions and estimates beat location candidates from them using TempoTapDegara algorithm. Thereafter the best candidates are selected using TempoTapMaxAgreement. The employed detection functions, and the optimal frame/hop sizes used for their computation are:
		  - complex spectral difference (see 'complex' method in OnsetDetection algorithm, 2048/1024 with posterior x2 upsample or the detection function)
		  - energy flux (see 'rms' method in OnsetDetection algorithm, the same settings)
		  - spectral flux in Mel-frequency bands (see 'melflux' method in OnsetDetection algorithm, the same settings)
		  - beat emphasis function (see 'beat_emphasis' method in OnsetDetectionGlobal algorithm, 2048/512)
		  - spectral flux between histogrammed spectrum frames, measured by the modified information gain (see 'infogain' method in OnsetDetectionGlobal algorithm, 2048/512)
		
		You can follow these guidelines [2] to assess the quality of beats estimation based on the computed confidence value:
		  - [0, 1)      very low confidence, the input signal is hard for the employed candidate beat trackers
		  - [1, 1.5]    low confidence
		  - (1.5, 3.5]  good confidence, accuracy around 80% in AMLt measure
		  - (3.5, 5.32] excellent confidence
		
		Note that the algorithm requires the audio input with the 44100 Hz sampling rate in order to function correctly.
		
		References:
		  [1] J. Zapata, M. Davies and E. Gómez, "Multi-feature beat tracker,"
		  IEEE/ACM Transactions on Audio, Speech and Language Processing. 22(4),
		  816-825, 2014
		
		  [2] J.R. Zapata, A. Holzapfel, M.E.P. Davies, J.L. Oliveira, F. Gouyon,
		  "Assigning a confidence threshold on automatic beat annotation in large
		  datasets", International Society for Music Information Retrieval Conference
		  (ISMIR'12), pp. 157-162, 2012
		

		Args:
			maxTempo (int): the fastest tempo to detect [bpm]. Defaults to 208. Range [60,250]
			minTempo (int): the slowest tempo to detect [bpm]. Defaults to 40. Range [40,180] 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> tuple[NDArray[np.float32], float]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the audio input signal. Defaults to None. 
		Returns:
			ticks (NDArray[np.float32]):  the estimated tick locations [s]
			confidence (float): confidence of the beat tracker [0, 5.32]
		""" 
		... 


class Beatogram(_essentia.Algorithm): 
	def __init__(self, size:int=16) -> None:
		"""Filters the loudness matrix given by BeatsLoudness algorithm in order to keep only the most salient beat band representation.
		
		This algorithm has been found to be useful for estimating time signatures.
		
		Quality: experimental (not evaluated, do not use)

		Args:
			size (int): number of beats for dynamic filtering. Defaults to 16. Range [1,inf) 
		""" 
		... 
	def __call__(self, loudness:NDArray[np.float32], loudnessBandRatio:np.ndarray) -> np.ndarray:
		"""compute
		Args:
			loudness (NDArray[np.float32]): the loudness at each beat. Defaults to None. 
			loudnessBandRatio (np.ndarray): matrix of loudness ratios at each band and beat. Defaults to None. 
		Returns:
			beatogram (np.ndarray): filtered matrix loudness
		""" 
		... 


class BeatsLoudness(_essentia.Algorithm): 
	def __init__(self, beatDuration:float=0.05, beatWindowDuration:float=0.1, beats:NDArray[np.float32]=np.array([]), frequencyBands:NDArray[np.float32]=np.array([20, 150, 400, 3200, 7000, 22000]), sampleRate:float=44100.0) -> None:
		"""Computes the spectrum energy of beats in an audio signal given their positions.
		
		The energy is computed both on the whole frequency range and for each of the specified frequency bands. See the SingleBeatLoudness algorithm for a more detailed explanation.
		
		Note that the algorithm will output empty results in the case if no beats are specified in the "beats" parameter.

		Args:
			beatDuration (float): the duration of the window in which the beat will be restricted [s]. Defaults to 0.05. Range (0,inf)
			beatWindowDuration (float): the duration of the window in which to look for the beginning of the beat (centered around the positions in 'beats') [s]. Defaults to 0.1. Range (0,inf)
			beats (NDArray[np.float32]): the list of beat positions (each position is in seconds). Defaults to np.array([]). Range None
			frequencyBands (NDArray[np.float32]): the list of bands to compute energy ratios [Hz. Defaults to np.array([20, 150, 400, 3200, 7000, 22000]). Range None
			sampleRate (float): the audio sampling rate [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> tuple[NDArray[np.float32], np.ndarray]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input audio signal. Defaults to None. 
		Returns:
			loudness (NDArray[np.float32]): the beat's energy in the whole spectrum
			loudnessBandRatio (np.ndarray): the ratio of the beat's energy on each frequency band
		""" 
		... 


class BinaryOperator(_essentia.Algorithm): 
	def __init__(self, type:str='add') -> None:
		"""Performs basic arithmetical operations element by element given two arrays.
		
		Note:
		  - using this algorithm in streaming mode can cause diamond shape graphs which have not been tested with the current scheduler. There is NO GUARANTEE of its correct work for diamond shape graphs.
		  - for y=0, x/y is invalid

		Args:
			type (str): the type of the binary operator to apply to the input arrays. Defaults to 'add'. Range {add,subtract,multiply,divide} 
		""" 
		... 
	def __call__(self, array1:NDArray[np.float32], array2:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			array1 (NDArray[np.float32]): the first operand input array. Defaults to None. 
			array2 (NDArray[np.float32]): the second operand input array. Defaults to None. 
		Returns:
			array (NDArray[np.float32]): the array containing the result of binary operation
		""" 
		... 


class BinaryOperatorStream(_essentia.Algorithm): 
	def __init__(self, type:str='add') -> None:
		"""Performs basic arithmetical operations element by element given two arrays.
		
		Note:
		  - using this algorithm in streaming mode can cause diamond shape graphs which have not been tested with the current scheduler. There is NO GUARANTEE of its correct work for diamond shape graphs.
		  - for y=0, x/y is invalid

		Args:
			type (str): the type of the binary operator to apply to the input arrays. Defaults to 'add'. Range {add,subtract,multiply,divide} 
		""" 
		... 
	def __call__(self, array1:NDArray[np.float32], array2:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			array1 (NDArray[np.float32]): the first operand input array. Defaults to None. 
			array2 (NDArray[np.float32]): the second operand input array. Defaults to None. 
		Returns:
			array (NDArray[np.float32]): the array containing the result of binary operation
		""" 
		... 


class BpmHistogram(_essentia.Algorithm): 
	def __init__(self, bpm:float=0.0, constantTempo:bool=False, frameRate:float=86.1328, frameSize:float=4.0, maxBpm:float=560.0, maxPeaks:int=50, minBpm:float=30.0, overlap:int=16, tempoChange:float=5.0, weightByMagnitude:bool=True, windowType:str='hann', zeroPadding:int=0) -> None:
		"""Analyzes predominant periodicities in a signal given its novelty curve [1] (see NoveltyCurve algorithm) or another onset detection function (see OnsetDetection and OnsetDetectionGlobal).
		
		It estimates pulse BPM values and time positions together with a half-wave rectified sinusoid whose peaks represent the pulses present in the audio signal and their magnitudes. The analysis is based on the FFT of the input novelty curve from which salient periodicities are detected by thresholding. Temporal evolution of these periodicities is output in the "tempogram". Candidate BPMs are then detected based on a histogram of the observed periodicities weighted by their energy in the tempogram. The sinusoidal model is constructed based on the observed periodicities and their magnitudes with the estimated overall BPM as a reference.
		
		The algorithm outputs: 
		 - bpm: the mean of the most salient BPM values representing periodicities in the signal (the mean BPM).
		 - bpmCandidates and bpmMagnitudes: list of the most salient BPM values and their magnitudes (intensity). These two outputs can be helpful for taking an alternative decision on estimation of the overall BPM.
		 - tempogram: spectrogram-like representation of the estimated salient periodicities and their intensities over time (per-frame BPM magnitudes). It is useful for detecting tempo variations and visualization of tempo evolution.
		 - frameBpms: list of candidate BPM values at each frame. The candidate values are similar to the mean BPM. If no candidates are found to be similar, the mean value itself is used unless "tempoChange" seconds have triggered a variation in tempo.
		 - ticks: time positions of ticks in seconds.
		 - ticksMagnitude: magnitude of each tick. Higher values correspond to higher probability of correctly identified ticks.
		 - sinusoid: a sinusoidal model of the ticks' positions. The previous outputs are based on detecting peaks of this half-wave rectified sinusoid. This model can be used to obtain ticks using alternative peak detection algorithms if necessary. Beware that the last few ticks may exceed the length of the audio signal due to overlap factors. Therefore, this output should be always checked against the length of audio signal.
		
		Note:
		 - This algorithm is outdated. For beat tracking it is recommended to use RhythmExtractor2013 algorithm found to perform better than NoveltyCurve with BpmHistogram in evaluations.
		 - The "frameRate" parameter refers to the frame rate at which the novelty curve has been computed. It is equal to the audio sampling rate divided by the hop size at which the signal was processed.
		 - Although the algorithm tries to find beats that fit the mean BPM the best, the tempo is not assumed to be constant unless specified in the corresponding parameter. For this reason and if tempo differs too much from frame to frame, there may be phase discontinuities when constructing the sinusoid which can yield to too many ticks. One can recursively run this algorithm on the sinusoid output until the ticks stabilize. At this point it may be useful to infer a specific BPM and set the constant tempo parameter to true.
		 - Another useful trick is to run the algorithm one time to get an estimation of the mean BPM and re-run it again with a "frameSize" parameter set to a multiple of the mean BPM.
		
		Quality: outdated (use RhythmExtractor2013 instead, still this algorithm might be useful when working with other onset detection functions apart from NoveltyCurve)
		
		References:
		  [1] P. Grosche and M. Müller, "A mid-level representation for capturing
		  dominant tempo and pulse information in music recordings," in
		  International Society for Music Information Retrieval Conference
		  (ISMIR’09), 2009, pp. 189–194.

		Args:
			bpm (float): bpm to induce a certain tempo tracking. Zero if unknown. Defaults to 0.0. Range [0,inf)
			constantTempo (bool): whether to consider constant tempo. Set to true when inducina specific tempo. Defaults to False. Range {true,false}
			frameRate (float): the sampling rate of the novelty curve [frame/s]. Defaults to 86.1328. Range [1,inf)
			frameSize (float): the minimum length to compute the FFT [s]. Defaults to 4.0. Range [1,inf)
			maxBpm (float): the maximum bpm to consider. Defaults to 560.0. Range (0,inf)
			maxPeaks (int): the number of peaks to be considered at each spectrum. Defaults to 50. Range (0,inf]
			minBpm (float): the minimum bpm to consider. Defaults to 30.0. Range [0,inf)
			overlap (int): the overlap factor. Defaults to 16. Range (0,inf)
			tempoChange (float): the minimum length to consider a change in tempo as stable [s]. Defaults to 5.0. Range [0,inf)
			weightByMagnitude (bool): whether to consider peaks' magnitude when building the histogram. Defaults to True. Range {true,false}
			windowType (str): the window type to be used when computing the FFT. Defaults to 'hann'. Range None
			zeroPadding (int): zero padding factor to compute the FFT [s]. Defaults to 0. Range [0,inf) 
		""" 
		... 
	def __call__(self, novelty:NDArray[np.float32]) -> tuple[float, NDArray[np.float32], NDArray[np.float32], np.ndarray, NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			novelty (NDArray[np.float32]): the novelty curve. Defaults to None. 
		Returns:
			bpm (float): mean BPM of the most salient tempo
			bpmCandidates (NDArray[np.float32]): list of the most salient BPM values
			bpmMagnitudes (NDArray[np.float32]): magnitudes of the most salient BPM values
			tempogram (np.ndarray): spectrogram-like representation of tempo over time (frames of BPM magnitudes)
			frameBpms (NDArray[np.float32]): BPM values at each frame
			ticks (NDArray[np.float32]): time positions of ticks [s]
			ticksMagnitude (NDArray[np.float32]): ticks' strength (magnitude)
			sinusoid (NDArray[np.float32]): sinusoid whose peaks indicate tick positions
		""" 
		... 


class BpmHistogramDescriptors(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Computes beats per minute histogram and its statistics for the highest and second highest peak.
		
		Note: histogram vector contains occurance frequency for each bpm value, 0-th element corresponds to 0 bpm value.		""" 
		... 
	def __call__(self, bpmIntervals:NDArray[np.float32]) -> tuple[float, float, float, float, float, float, NDArray[np.float32]]:
		"""compute
		Args:
			bpmIntervals (NDArray[np.float32]): the list of bpm intervals [s]. Defaults to None. 
		Returns:
			firstPeakBPM (float): value for the highest peak [bpm]
			firstPeakWeight (float): weight of the highest peak
			firstPeakSpread (float): spread of the highest peak
			secondPeakBPM (float): value for the second highest peak [bpm]
			secondPeakWeight (float): weight of the second highest peak
			secondPeakSpread (float): spread of the second highest peak
			histogram (NDArray[np.float32]): bpm histogram [bpm]
		""" 
		... 


class BpmRubato(_essentia.Algorithm): 
	def __init__(self, longRegionsPruningTime:float=20.0, shortRegionsMergingTime:float=4.0, tolerance:float=0.08) -> None:
		"""Extracts the locations of large tempo changes from a list of beat ticks.
		
		An exception is thrown if the input beats are not in ascending order and/or if the input beats contain duplicate values.
		
		Quality: experimental (non-reliable, poor accuracy).
		
		References:
		  [1] Tempo Rubato - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Rubato

		Args:
			longRegionsPruningTime (float): time for the longest constant tempo region inside a rubato region [s]. Defaults to 20.0. Range [0,inf)
			shortRegionsMergingTime (float): time for the shortest constant tempo region from one tempo region to another [s]. Defaults to 4.0. Range [0,inf)
			tolerance (float): minimum tempo deviation to look for. Defaults to 0.08. Range [0,1] 
		""" 
		... 
	def __call__(self, beats:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32], int]:
		"""compute
		Args:
			beats (NDArray[np.float32]): list of detected beat ticks [s]. Defaults to None. 
		Returns:
			rubatoStart (NDArray[np.float32]): list of timestamps where the start of a rubato region was detected [s]
			rubatoStop (NDArray[np.float32]): list of timestamps where the end of a rubato region was detected [s]
			rubatoNumber (int): number of detected rubato regions
		""" 
		... 


class CartesianToPolar(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Converts an array of complex numbers from cartesian to polar form.
		
		It uses the Euler formula:
		  z = x + i*y = |z|(cos(α) + i sin(α))
		    where x = Real part, y = Imaginary part,
		    and |z| = modulus = magnitude, α = phase in (-pi,pi]
		
		It returns the magnitude and the phase as 2 separate vectors.
		
		References:
		  [1] Polar Coordinates -- from Wolfram MathWorld,
		  http://mathworld.wolfram.com/PolarCoordinates.html
		
		  [2] Polar coordinate system - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Polar_coordinates		""" 
		... 
	def __call__(self, complex:np.ndarray) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			complex (np.ndarray): the complex input vector. Defaults to None. 
		Returns:
			magnitude (NDArray[np.float32]): the magnitude vector
			phase (NDArray[np.float32]): the phase vector
		""" 
		... 


class CentralMoments(_essentia.Algorithm): 
	def __init__(self, mode:str='pdf', range:float=1.0) -> None:
		"""Extracts the 0th, 1st, 2nd, 3rd and 4th central moments of an array.
		
		It returns a 5-tuple in which the index corresponds to the order of the moment.
		
		Central moments cannot be computed on arrays which size is less than 2, in which case an exception is thrown.
		
		Note: the 'mode' parameter defines whether to treat array values as a probability distribution function (pdf) or as sample points of a distribution (sample).
		
		References:
		  [1] Sample Central Moment -- from Wolfram MathWorld,
		  http://mathworld.wolfram.com/SampleCentralMoment.html
		
		  [2] Central Moment - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Central_moment

		Args:
			mode (str): compute central moments considering array values as a probability density function over array index or as sample points of a distribution. Defaults to 'pdf'. Range {pdf,sample}
			range (float): the range of the input array, used for normalizing the results in the 'pdf' mode. Defaults to 1.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, array:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			array (NDArray[np.float32]): the input array. Defaults to None. 
		Returns:
			centralMoments (NDArray[np.float32]): the central moments of the input array
		""" 
		... 


class Centroid(_essentia.Algorithm): 
	def __init__(self, range:float=1.0) -> None:
		"""Computes the centroid of an array.
		
		The centroid is normalized to a specified range. This algorithm can be used to compute spectral centroid or temporal centroid.
		
		The spectral centroid is a measure that indicates where the "center of mass" of the spectrum is. Perceptually, it has a robust connection with the impression of "brightness" of a sound, and therefore is used to characterise musical timbre. It is calculated as the weighted mean of the frequencies present in the signal, with their magnitudes as the weights.
		
		The temporal centroid is the point in time in a signal that is a temporal balancing point of the sound event energy. It can be computed from the envelope of the signal across audio samples [3] (see Envelope algorithm) or over the RMS level of signal across frames [4] (see RMS algorithm).
		
		Note:
		- For a spectral centroid [hz], frequency range should be equal to samplerate/2
		- For a temporal envelope centroid [s], range should be equal to (audio_size_in_samples-1) / samplerate
		- Exceptions are thrown when input array contains less than 2 elements.
		
		References:
		  [1] Function Centroid -- from Wolfram MathWorld,
		  http://mathworld.wolfram.com/FunctionCentroid.html
		  [2] Spectral centroid - Wikipedia, the free encyclopedia,
		  https://en.wikipedia.org/wiki/Spectral_centroid
		  [3] G. Peeters, "A large set of audio features for sound description
		  (similarity and classification) in the CUIDADO project," CUIDADO I.S.T.
		  Project Report, 2004.
		  [4] Klapuri, A., & Davy, M. (Eds.). (2007). Signal processing methods for
		  music transcription. Springer Science & Business Media.

		Args:
			range (float): the range of the input array, used for normalizing the results. Defaults to 1.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, array:NDArray[np.float32]) -> float:
		"""compute
		Args:
			array (NDArray[np.float32]): the input array. Defaults to None. 
		Returns:
			centroid (float): the centroid of the array
		""" 
		... 


class ChordsDescriptors(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Chord progression this algorithm describes it by means of key, scale, histogram, and rate of change.
		
		Note:
		  - chordsHistogram indexes follow the circle of fifths order, while being shifted to the input key and scale
		  - key and scale are taken from the most frequent chord. In the case where multiple chords are equally frequent, the chord is hierarchically chosen from the circle of fifths.
		  - chords should follow this name convention `<A-G>[<#/b><m>]` (i.e. C, C# or C#m are valid chords). Chord names not fitting this convention will throw an exception.
		
		Input chords vector may not be empty, otherwise an exception is thrown.
		
		References:
		  [1] Chord progression - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Chord_progression
		
		  [2] Circle of fifths - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Circle_of_fifths		""" 
		... 
	def __call__(self, chords:list[str], key:str, scale:str) -> tuple[NDArray[np.float32], float, float, str, str]:
		"""compute
		Args:
			chords (list[str]): the chord progression. Defaults to None. 
			key (str): the key of the whole song, from A to G. Defaults to None. 
			scale (str): the scale of the whole song (major or minor). Defaults to None. 
		Returns:
			chordsHistogram (NDArray[np.float32]): the normalized histogram of chords
			chordsNumberRate (float): the ratio of different chords from the total number of chords in the progression
			chordsChangesRate (float): the rate at which chords change in the progression
			chordsKey (str): the most frequent chord of the progression
			chordsScale (str): the scale of the most frequent chord of the progression (either 'major' or 'minor')
		""" 
		... 


class ChordsDetection(_essentia.Algorithm): 
	def __init__(self, hopSize:int=2048, sampleRate:float=44100.0, windowSize:float=2.0) -> None:
		"""Estimates chords given an input sequence of harmonic pitch class profiles (HPCPs).
		
		It finds the best matching major or minor triad and outputs the result as a string (e.g. A#, Bm, G#m, C). The following note names are used in the output:
		"A", "Bb", "B", "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab".
		Note:
		  - The algorithm assumes that the sequence of the input HPCP frames has been computed with framesize = 2*hopsize
		  - The algorithm estimates a sequence of chord values corresponding to the input HPCP frames (one chord value for each frame, estimated using a temporal window of HPCPs centered at that frame).
		
		Quality: experimental (prone to errors, algorithm needs improvement)
		
		References:
		  [1] E. Gómez, "Tonal Description of Polyphonic Audio for Music Content
		  Processing," INFORMS Journal on Computing, vol. 18, no. 3, pp. 294–304,
		  2006.
		
		  [2] D. Temperley, "What's key for key? The Krumhansl-Schmuckler
		  key-finding algorithm reconsidered", Music Perception vol. 17, no. 1,
		  pp. 65-100, 1999.

		Args:
			hopSize (int): the hop size with which the input PCPs were computed. Defaults to 2048. Range (0,inf)
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			windowSize (float): the size of the window on which to estimate the chords [s]. Defaults to 2.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, pcp:np.ndarray) -> tuple[list[str], NDArray[np.float32]]:
		"""compute
		Args:
			pcp (np.ndarray): the pitch class profile from which to detect the chord. Defaults to None. 
		Returns:
			chords (list[str]): the resulting chords, from A to G
			strength (NDArray[np.float32]): the strength of the chord
		""" 
		... 


class ChordsDetectionBeats(_essentia.Algorithm): 
	def __init__(self, chromaPick:str='interbeat_median', hopSize:int=2048, sampleRate:float=44100.0) -> None:
		"""Estimates chords using pitch profile classes on segments between beats.
		
		It is similar to ChordsDetection algorithm, but the chords are estimated on audio segments between each pair of consecutive beats. For each segment the estimation is done based on a chroma (HPCP) vector characterizing it, which can be computed by two methods:
		  - 'interbeat_median', each resulting chroma vector component is a median of all the component values in the segment
		  - 'starting_beat', chroma vector is sampled from the start of the segment (that is, its starting beat position) using its first frame. It makes sense if chroma is preliminary smoothed.
		
		Quality: experimental (algorithm needs evaluation)
		
		References:
		  [1] E. Gómez, "Tonal Description of Polyphonic Audio for Music Content
		  Processing," INFORMS Journal on Computing, vol. 18, no. 3, pp. 294–304,
		  2006.
		
		  [2] D. Temperley, "What's key for key? The Krumhansl-Schmuckler
		  key-finding algorithm reconsidered", Music Perception vol. 17, no. 1,
		  pp. 65-100, 1999.

		Args:
			chromaPick (str): method of calculating singleton chroma for interbeat interval. Defaults to 'interbeat_median'. Range {starting_beat,interbeat_median}
			hopSize (int): the hop size with which the input PCPs were computed. Defaults to 2048. Range (0,inf)
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, pcp:np.ndarray, ticks:NDArray[np.float32]) -> tuple[list[str], NDArray[np.float32]]:
		"""compute
		Args:
			pcp (np.ndarray): the pitch class profile from which to detect the chord. Defaults to None. 
			ticks (NDArray[np.float32]): the list of beat positions (in seconds). One chord will be outputted for each segment between two adjacent ticks. If number of ticks is smaller than 2, exception will be thrown. Those ticks that exceeded the pcp time length will be ignored.. Defaults to None. 
		Returns:
			chords (list[str]): the resulting chords, from A to G
			strength (NDArray[np.float32]): the strength of the chords
		""" 
		... 


class ChromaCrossSimilarity(_essentia.Algorithm): 
	def __init__(self, binarizePercentile:float=0.095, frameStackSize:int=9, frameStackStride:int=1, noti:int=12, oti:bool=True, otiBinary:bool=False, streaming:bool=False) -> None:
		"""Computes a binary cross similarity matrix from two chromagam feature vectors of a query and reference song.
		
		With default parameters, this algorithm computes cross-similarity of two given input chromagrams as described in [2].
		
		Use HPCP algorithm for computing the chromagram with default parameters of this algorithm for the best results.
		
		If parameter 'oti=True', the algorithm transpose the reference song chromagram by optimal transposition index as described in [1].
		
		If parameter 'otiBinary=True', the algorithm computes the binary cross-similarity matrix based on optimal transposition index between each feature pairs instead of euclidean distance as described in [3].
		
		The input chromagram should be in the shape (n_frames, numbins), where 'n_frames' is number of frames and 'numbins' for the number of bins in the chromagram. An exception is thrown otherwise.
		
		An exception is also thrown if either one of the input chromagrams are empty.
		
		While param 'streaming=True', the algorithm accumulates the input 'queryFeature' in the pairwise similarity matrix calculation on each call of compute() method. You can reset it using the reset() method.
		
		References:
		
		[1] Serra, J., Gómez, E., & Herrera, P. (2008). Transposing chroma representations to a common key, IEEE Conference on The Use of Symbols to Represent Music and Multimedia Objects.
		
		[2] Serra, J., Serra, X., & Andrzejak, R. G. (2009). Cross recurrence quantification for cover song identification.New Journal of Physics.
		
		[3] Serra, Joan, et al. Chroma binary similarity and local alignment applied to cover song identification. IEEE Transactions on Audio, Speech, and Language Processing 16.6 (2008).
		

		Args:
			binarizePercentile (float): maximum percent of distance values to consider as similar in each row and each column. Defaults to 0.095. Range [0,1]
			frameStackSize (int): number of input frames to stack together and treat as a feature vector for similarity computation. Choose 'frameStackSize=1' to use the original input frames without stacking. Defaults to 9. Range [0,inf)
			frameStackStride (int): stride size to form a stack of frames (e.g., 'frameStackStride'=1 to use consecutive frames; 'frameStackStride'=2 for using every second frame). Defaults to 1. Range [1,inf)
			noti (int): number of circular shifts to be checked for Optimal Transposition Index [1]. Defaults to 12. Range [0,inf)
			oti (bool): whether to transpose the key of the reference song to the query song by Optimal Transposition Index [1]. Defaults to True. Range {true,false}
			otiBinary (bool): whether to use the OTI-based chroma binary similarity method [3]. Defaults to False. Range {true,false}
			streaming (bool): whether to accumulate the input 'queryFeature' in the euclidean similarity matrix calculation on each compute() method call. Defaults to False. Range {true,false} 
		""" 
		... 
	def __call__(self, queryFeature:np.ndarray, referenceFeature:np.ndarray) -> np.ndarray:
		"""compute
		Args:
			queryFeature (np.ndarray): frame-wise chromagram of the query song (e.g., a HPCP). Defaults to None. 
			referenceFeature (np.ndarray): frame-wise chromagram of the reference song (e.g., a HPCP). Defaults to None. 
		Returns:
			csm (np.ndarray): 2D binary cross-similarity matrix of the query and reference features
		""" 
		... 


class Chromagram(_essentia.Algorithm): 
	def __init__(self, binsPerOctave:int=12, minFrequency:float=32.7, minimumKernelSize:int=4, normalizeType:str='unit_max', numberBins:int=84, sampleRate:float=44100.0, scale:float=1.0, threshold:float=0.01, windowType:str='hann', zeroPhase:bool=True) -> None:
		"""Computes the Constant-Q chromagram using FFT.
		
		See ConstantQ algorithm for more details.
		

		Args:
			binsPerOctave (int): number of bins per octave. Defaults to 12. Range [1,inf)
			minFrequency (float): minimum frequency [Hz]. Defaults to 32.7. Range [1,inf)
			minimumKernelSize (int): minimum size allowed for frequency kernels. Defaults to 4. Range [2,inf)
			normalizeType (str): normalize type. Defaults to 'unit_max'. Range {none,unit_sum,unit_max}
			numberBins (int): number of frequency bins, starting at minFrequency. Defaults to 84. Range [1,inf)
			sampleRate (float): FFT sampling rate [Hz]. Defaults to 44100.0. Range [0,inf)
			scale (float): filters scale. Larger values use longer windows. Defaults to 1.0. Range [0,inf)
			threshold (float): bins whose magnitude is below this quantile are discarded. Defaults to 0.01. Range [0,1)
			windowType (str): the window type. Defaults to 'hann'. Range {hamming,hann,hannnsgcq,triangular,square,blackmanharris62,blackmanharris70,blackmanharris74,blackmanharris92}
			zeroPhase (bool): a boolean value that enables zero-phase windowing. Input audio frames should be windowed with the same phase mode. Defaults to True. Range {true,false} 
		""" 
		... 
	def __call__(self, frame:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			frame (NDArray[np.float32]): the input audio frame. Defaults to None. 
		Returns:
			chromagram (NDArray[np.float32]): the magnitude constant-Q chromagram
		""" 
		... 


class ClickDetector(_essentia.Algorithm): 
	def __init__(self, detectionThreshold:float=30.0, frameSize:int=512, hopSize:int=256, order:int=12, powerEstimationThreshold:int=10, sampleRate:float=44100.0, silenceThreshold:int=-50) -> None:
		"""Detects the locations of impulsive noises (clicks and pops) on the input audio frame.
		
		It relies on LPC coefficients to inverse-filter the audio in order to attenuate the stationary part and enhance the prediction error (or excitation noise)[1]. After this, a matched filter is used to further enhance the impulsive peaks. The detection threshold is obtained from a robust estimate of the excitation noise power [2] plus a parametric gain value.
		
		References:
		[1] Vaseghi, S. V., & Rayner, P. J. W. (1990). Detection and suppression of impulsive noise in speech communication systems. IEE Proceedings I (Communications, Speech and Vision), 137(1), 38-46.
		[2] Vaseghi, S. V. (2008). Advanced digital signal processing and noise reduction. John Wiley & Sons. Page 355

		Args:
			detectionThreshold (float): 'detectionThreshold' the threshold is based on the instant power of the noisy excitation signal plus detectionThreshold dBs. Defaults to 30.0. Range (-inf,inf)
			frameSize (int): the expected size of the input audio signal (this is an optional parameter to optimize memory allocation). Defaults to 512. Range (0,inf)
			hopSize (int): hop size used for the analysis. This parameter must be set correctly as it cannot be obtained from the input data. Defaults to 256. Range (0,inf)
			order (int): scalar giving the number of LPCs to use. Defaults to 12. Range [1,inf)
			powerEstimationThreshold (int): the noisy excitation is clipped to 'powerEstimationThreshold' times its median.. Defaults to 10. Range (0,inf)
			sampleRate (float): sample rate used for the analysis. Defaults to 44100.0. Range (0,inf)
			silenceThreshold (int): threshold to skip silent frames. Defaults to -50. Range (-inf,0) 
		""" 
		... 
	def __call__(self, frame:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			frame (NDArray[np.float32]): the input frame (must be non-empty). Defaults to None. 
		Returns:
			starts (NDArray[np.float32]): starting indexes of the clicks
			ends (NDArray[np.float32]): ending indexes of the clicks
		""" 
		... 


class Clipper(_essentia.Algorithm): 
	def __init__(self, max:float=1.0, min:float=-1.0) -> None:
		"""Clips the input signal to fit its values into a specified interval.
		
		References:
		  [1] Clipping - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Clipping_%28audio%29

		Args:
			max (float): the maximum value above which the signal will be clipped. Defaults to 1.0. Range (-inf,inf)
			min (float): the minimum value below which the signal will be clipped. Defaults to -1.0. Range (-inf,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input signal. Defaults to None. 
		Returns:
			signal (NDArray[np.float32]): the output signal with the added noise
		""" 
		... 


class ConstantQ(_essentia.Algorithm): 
	def __init__(self, binsPerOctave:int=12, minFrequency:float=32.7, minimumKernelSize:int=4, numberBins:int=84, sampleRate:float=44100.0, scale:float=1.0, threshold:float=0.01, windowType:str='hann', zeroPhase:bool=True) -> None:
		"""Computes Constant Q Transform using the FFT for fast calculation.
		
		It transforms a windowed audio frame into the log frequency domain.
		
		References:
		  [1] Constant Q transform - Wikipedia, the free encyclopedia,
		  https://en.wikipedia.org/wiki/Constant_Q_transform
		  [2] Brown, J. C., & Puckette, M. S. (1992). An efficient algorithm for the
		  calculation of a constant Q transform. The Journal of the Acoustical Society
		  of America, 92(5), 2698-2701.
		  [3] Schörkhuber, C., & Klapuri, A. (2010). Constant-Q transform toolbox for
		  music processing. In 7th Sound and Music Computing Conference, Barcelona,
		  Spain (pp. 3-64).

		Args:
			binsPerOctave (int): number of bins per octave. Defaults to 12. Range [1,inf)
			minFrequency (float): minimum frequency [Hz]. Defaults to 32.7. Range [1,inf)
			minimumKernelSize (int): minimum size allowed for frequency kernels. Defaults to 4. Range [2,inf)
			numberBins (int): number of frequency bins, starting at minFrequency. Defaults to 84. Range [1,inf)
			sampleRate (float): FFT sampling rate [Hz]. Defaults to 44100.0. Range [0,inf)
			scale (float): filters scale. Larger values use longer windows. Defaults to 1.0. Range [0,inf)
			threshold (float): bins whose magnitude is below this quantile are discarded. Defaults to 0.01. Range [0,1)
			windowType (str): the window type. Defaults to 'hann'. Range {hamming,hann,hannnsgcq,triangular,square,blackmanharris62,blackmanharris70,blackmanharris74,blackmanharris92}
			zeroPhase (bool): a boolean value that enables zero-phase windowing. Input audio frames should be windowed with the same phase mode. Defaults to True. Range {true,false} 
		""" 
		... 
	def __call__(self, frame:NDArray[np.float32]) -> np.ndarray:
		"""compute
		Args:
			frame (NDArray[np.float32]): the windowed input audio frame. Defaults to None. 
		Returns:
			constantq (np.ndarray): the Constant Q transform
		""" 
		... 


class CoverSongSimilarity(_essentia.Algorithm): 
	def __init__(self, alignmentType:str='serra09', disExtension:float=0.5, disOnset:float=0.5, distanceType:str='asymmetric') -> None:
		"""Computes a cover song similiarity measure from a binary cross similarity matrix input between two chroma vectors of a query and reference song using various alignment constraints of smith-waterman local-alignment algorithm.
		
		This algorithm expects to recieve the binary similarity matrix input from essentia 'ChromaCrossSimilarity' algorithm or essentia 'CrossSimilarityMatrix' with parameter 'binarize=True'.
		
		The algorithm provides two different allignment contraints for computing the smith-waterman score matrix (check references).
		
		Exceptions are thrown if the input similarity matrix is not binary or empty.
		
		References:
		
		[1] Smith-Waterman algorithm (Wikipedia, https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm).
		
		[2] Serra, J., Serra, X., & Andrzejak, R. G. (2009). Cross recurrence quantification for cover song identification.New Journal of Physics.
		
		[3] Chen, N., Li, W., & Xiao, H. (2017). Fusing similarity functions for cover song identification. Multimedia Tools and Applications.
		

		Args:
			alignmentType (str): choose either one of the given local-alignment constraints for smith-waterman algorithm as described in [2] or [3] respectively.. Defaults to 'serra09'. Range {serra09,chen17}
			disExtension (float): penalty for disruption extension. Defaults to 0.5. Range [0,inf)
			disOnset (float): penalty for disruption onset. Defaults to 0.5. Range [0,inf)
			distanceType (str): choose the type of distance. By default the algorithm outputs a asymmetric distance which is obtained by normalising the maximum score in the alignment score matrix with length of reference song. Defaults to 'asymmetric'. Range {asymmetric,symmetric} 
		""" 
		... 
	def __call__(self, inputArray:np.ndarray) -> tuple[np.ndarray, float]:
		"""compute
		Args:
			inputArray (np.ndarray):  a 2D binary cross-similarity matrix between two audio chroma vectors (query vs reference song) (refer 'ChromaCrossSimilarity' algorithm').. Defaults to None. 
		Returns:
			scoreMatrix (np.ndarray): a 2D smith-waterman alignment score matrix from the input binary cross-similarity matrix
			distance (float): cover song similarity distance between the query and reference song from the input similarity matrix. Either 'asymmetric' (as described in [2]) or 'symmetric' (maximum score in the alignment score matrix).
		""" 
		... 


class Crest(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Computes the crest of an array.
		
		The crest is defined as the ratio between the maximum value and the arithmetic mean of an array. Typically it is used on the magnitude spectrum.
		
		Crest cannot be computed neither on empty arrays nor arrays which contain negative values. In such cases, exceptions will be thrown.
		
		References:
		  [1] G. Peeters, "A large set of audio features for sound description
		  (similarity and classification) in the CUIDADO project," CUIDADO I.S.T.
		  Project Report, 2004		""" 
		... 
	def __call__(self, array:NDArray[np.float32]) -> float:
		"""compute
		Args:
			array (NDArray[np.float32]): the input array (cannot contain negative values, and must be non-empty). Defaults to None. 
		Returns:
			crest (float): the crest of the input array
		""" 
		... 


class CrossCorrelation(_essentia.Algorithm): 
	def __init__(self, maxLag:int=1, minLag:int=0) -> None:
		"""Computes the cross-correlation vector of two signals.
		
		It accepts 2 parameters, minLag and maxLag which define the range of the computation of the innerproduct.
		
		An exception is thrown if "minLag" is larger than "maxLag". An exception is also thrown if the input vectors are empty.
		
		References:
		  [1] Cross-correlation - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Cross-correlation

		Args:
			maxLag (int): the maximum lag to be computed between the two vectors. Defaults to 1. Range (-inf,inf)
			minLag (int): the minimum lag to be computed between the two vectors. Defaults to 0. Range (-inf,inf) 
		""" 
		... 
	def __call__(self, arrayX:NDArray[np.float32], arrayY:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			arrayX (NDArray[np.float32]): the first input array. Defaults to None. 
			arrayY (NDArray[np.float32]): the second input array. Defaults to None. 
		Returns:
			crossCorrelation (NDArray[np.float32]): the cross-correlation vector between the two input arrays (its size is equal to maxLag - minLag + 1)
		""" 
		... 


class CrossSimilarityMatrix(_essentia.Algorithm): 
	def __init__(self, binarize:bool=False, binarizePercentile:float=0.095, frameStackSize:int=1, frameStackStride:int=1) -> None:
		"""Computes a euclidean cross-similarity matrix of two sequences of frame features.
		
		Similarity values can be optionally binarized
		
		The default parameters for binarizing are optimized according to [1] for cover song identification using chroma features. 
		
		The input feature arrays are vectors of frames of features in the shape (n_frames, n_features), where 'n_frames' is the number frames, 'n_features' is the number of frame features.
		
		An exception is also thrown if either one of the input feature arrays are empty or if the output similarity matrix is empty.
		
		References:
		
		[1] Serra, J., Serra, X., & Andrzejak, R. G. (2009). Cross recurrence quantification for cover song identification. New Journal of Physics.
		
		

		Args:
			binarize (bool): whether to binarize the euclidean cross-similarity matrix. Defaults to False. Range {true,false}
			binarizePercentile (float): maximum percent of distance values to consider as similar in each row and each column. Defaults to 0.095. Range [0,1]
			frameStackSize (int): number of input frames to stack together and treat as a feature vector for similarity computation. Choose 'frameStackSize=1' to use the original input frames without stacking. Defaults to 1. Range [0,inf)
			frameStackStride (int): stride size to form a stack of frames (e.g., 'frameStackStride'=1 to use consecutive frames; 'frameStackStride'=2 for using every second frame). Defaults to 1. Range [1,inf) 
		""" 
		... 
	def __call__(self, queryFeature:np.ndarray, referenceFeature:np.ndarray) -> np.ndarray:
		"""compute
		Args:
			queryFeature (np.ndarray): input frame features of the query song (e.g., a chromagram). Defaults to None. 
			referenceFeature (np.ndarray): input frame features of the reference song (e.g., a chromagram). Defaults to None. 
		Returns:
			csm (np.ndarray): 2D cross-similarity matrix of two input frame sequences (query vs reference)
		""" 
		... 


class CubicSpline(_essentia.Algorithm): 
	def __init__(self, leftBoundaryFlag:int=0, leftBoundaryValue:float=0.0, rightBoundaryFlag:int=0, rightBoundaryValue:float=0.0, xPoints:NDArray[np.float32]=np.array([0, 1]), yPoints:NDArray[np.float32]=np.array([0, 1])) -> None:
		"""Second derivatives of a piecewise cubic spline.
		
		The input value, i.e. the point at which the spline is to be evaluated typically should be between xPoints[0] and xPoints[size-1]. If the value lies outside this range, extrapolation is used.
		Regarding [left/right] boundary condition flag parameters:
		  - 0: the cubic spline should be a quadratic over the first interval
		  - 1: the first derivative at the [left/right] endpoint should be [left/right]BoundaryFlag
		  - 2: the second derivative at the [left/right] endpoint should be [left/right]BoundaryFlag
		References:
		  [1] Spline interpolation - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Spline_interpolation

		Args:
			leftBoundaryFlag (int): type of boundary condition for the left boundary. Defaults to 0. Range {0,1,2}
			leftBoundaryValue (float): the value to be used in the left boundary, when leftBoundaryFlag is 1 or 2. Defaults to 0.0. Range (-inf,inf)
			rightBoundaryFlag (int): type of boundary condition for the right boundary. Defaults to 0. Range {0,1,2}
			rightBoundaryValue (float): the value to be used in the right boundary, when rightBoundaryFlag is 1 or 2. Defaults to 0.0. Range (-inf,inf)
			xPoints (NDArray[np.float32]): the x-coordinates where data is specified (the points must be arranged in ascending order and cannot contain duplicates). Defaults to np.array([0, 1]). Range None
			yPoints (NDArray[np.float32]): the y-coordinates to be interpolated (i.e. the known data). Defaults to np.array([0, 1]). Range None 
		""" 
		... 
	def __call__(self, x:float) -> tuple[float, float, float]:
		"""compute
		Args:
			x (float): the input coordinate (x-axis). Defaults to None. 
		Returns:
			y (float): the value of the spline at x
			dy (float): the first derivative of the spline at x
			ddy (float): the second derivative of the spline at x
		""" 
		... 


class DCRemoval(_essentia.Algorithm): 
	def __init__(self, cutoffFrequency:float=40.0, sampleRate:float=44100.0) -> None:
		"""Removes the DC offset from a signal using a 1st order IIR highpass filter.
		
		Because of its dependence on IIR, IIR's requirements are inherited.
		
		References:
		  [1] Smith, J.O.  Introduction to Digital Filters with Audio Applications,
		  http://ccrma-www.stanford.edu/~jos/filters/DC_Blocker.html

		Args:
			cutoffFrequency (float): the cutoff frequency for the filter [Hz]. Defaults to 40.0. Range (0,inf)
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input audio signal. Defaults to None. 
		Returns:
			signal (NDArray[np.float32]): the filtered signal, with the DC component removed
		""" 
		... 


class DCT(_essentia.Algorithm): 
	def __init__(self, dctType:int=2, inputSize:int=10, liftering:int=0, outputSize:int=10) -> None:
		"""Computes the Discrete Cosine Transform of an array.
		
		It uses the DCT-II form, with the 1/sqrt(2) scaling factor for the first coefficient.
		
		Note: The 'inputSize' parameter is only used as an optimization when the algorithm is configured. The DCT will automatically adjust to the size of any input.
		
		References:
		  [1] Discrete cosine transform - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Discrete_cosine_transform

		Args:
			dctType (int): the DCT type. Defaults to 2. Range [2,3]
			inputSize (int): the size of the input array. Defaults to 10. Range [1,inf)
			liftering (int): the liftering coefficient. Use '0' to bypass it. Defaults to 0. Range [0,inf)
			outputSize (int): the number of output coefficients. Defaults to 10. Range [1,inf) 
		""" 
		... 
	def __call__(self, array:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			array (NDArray[np.float32]): the input array. Defaults to None. 
		Returns:
			dct (NDArray[np.float32]): the discrete cosine transform of the input array
		""" 
		... 


class Danceability(_essentia.Algorithm): 
	def __init__(self, maxTau:float=8800.0, minTau:float=310.0, sampleRate:float=44100.0, tauMultiplier:float=1.1) -> None:
		"""Estimates danceability of a given audio signal.
		
		The algorithm is derived from Detrended Fluctuation Analysis (DFA) described in [1]. The parameters minTau and maxTau are used to define the range of time over which DFA will be performed. The output of this algorithm is the danceability of the audio signal. These values usually range from 0 to 3 (higher values meaning more danceable).
		
		Exception is thrown when minTau is greater than maxTau.
		
		References:
		  [1] Streich, S. and Herrera, P., Detrended Fluctuation Analysis of Music
		  Signals: Danceability Estimation and further Semantic Characterization,
		  Proceedings of the AES 118th Convention, Barcelona, Spain, 2005

		Args:
			maxTau (float): maximum segment length to consider [ms]. Defaults to 8800.0. Range (0,inf)
			minTau (float): minimum segment length to consider [ms]. Defaults to 310.0. Range (0,inf)
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			tauMultiplier (float): multiplier to increment from min to max tau. Defaults to 1.1. Range (1,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> tuple[float, NDArray[np.float32]]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input signal. Defaults to None. 
		Returns:
			danceability (float): the danceability value. Normal values range from 0 to ~3. The higher, the more danceable.
			dfa (NDArray[np.float32]): the DFA exponent vector for considered segment length (tau) values
		""" 
		... 


class Decrease(_essentia.Algorithm): 
	def __init__(self, range:float=1.0) -> None:
		"""Computes the decrease of an array defined as the linear regression coefficient.
		
		The range parameter is used to normalize the result. For a spectral centroid, the range should be equal to Nyquist and for an audio centroid the range should be equal to (audiosize - 1) / samplerate.
		The size of the input array must be at least two elements for "decrease" to be computed, otherwise an exception is thrown.
		References:
		  [1] Least Squares Fitting -- from Wolfram MathWorld,
		  http://mathworld.wolfram.com/LeastSquaresFitting.html

		Args:
			range (float): the range of the input array, used for normalizing the results. Defaults to 1.0. Range (-inf,inf) 
		""" 
		... 
	def __call__(self, array:NDArray[np.float32]) -> float:
		"""compute
		Args:
			array (NDArray[np.float32]): the input array. Defaults to None. 
		Returns:
			decrease (float): the decrease of the input array
		""" 
		... 


class Derivative(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Returns the first-order derivative of an input signal.
		
		That is, for each input value it returns the value minus the previous one.		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input signal. Defaults to None. 
		Returns:
			signal (NDArray[np.float32]): the derivative of the input signal
		""" 
		... 


class DerivativeSFX(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Computes two descriptors that are based on the derivative of a signal envelope.
		
		The first descriptor is calculated after the maximum value of the input signal occurred. It is the average of the signal's derivative weighted by its amplitude. This coefficient helps discriminating impulsive sounds, which have a steep release phase, from non-impulsive sounds. The smaller the value the more impulsive.
		
		The second descriptor is the maximum derivative, before the maximum value of the input signal occurred. This coefficient helps discriminating sounds that have a smooth attack phase, and therefore a smaller value than sounds with a fast attack.
		
		This algorithm is meant to be fed by the outputs of the Envelope algorithm. If used in streaming mode, RealAccumulator should be connected in between.
		An exception is thrown if the input signal is empty.		""" 
		... 
	def __call__(self, envelope:NDArray[np.float32]) -> tuple[float, float]:
		"""compute
		Args:
			envelope (NDArray[np.float32]): the envelope of the signal. Defaults to None. 
		Returns:
			derAvAfterMax (float): the weighted average of the derivative after the maximum amplitude
			maxDerBeforeMax (float): the maximum derivative before the maximum amplitude
		""" 
		... 


class DiscontinuityDetector(_essentia.Algorithm): 
	def __init__(self, detectionThreshold:float=8.0, energyThreshold:float=-60.0, frameSize:int=512, hopSize:int=256, kernelSize:int=7, order:int=3, silenceThreshold:int=-50, subFrameSize:int=32) -> None:
		"""Uses LPC and some heuristics to detect discontinuities in an audio signal.
		
		[1].
		
		References:
		  [1] Mühlbauer, R. (2010). Automatic Audio Defect Detection.
		

		Args:
			detectionThreshold (float): 'detectionThreshold' times the standard deviation plus the median of the frame is used as detection threshold. Defaults to 8.0. Range [1,inf)
			energyThreshold (float): threshold in dB to detect silent subframes. Defaults to -60.0. Range (-inf,inf)
			frameSize (int): the expected size of the input audio signal (this is an optional parameter to optimize memory allocation). Defaults to 512. Range (0,inf)
			hopSize (int): hop size used for the analysis. This parameter must be set correctly as it cannot be obtained from the input data. Defaults to 256. Range [0,inf)
			kernelSize (int): scalar giving the size of the median filter window. Must be odd. Defaults to 7. Range [1,inf)
			order (int): scalar giving the number of LPCs to use. Defaults to 3. Range [1,inf)
			silenceThreshold (int): threshold to skip silent frames. Defaults to -50. Range (-inf,0)
			subFrameSize (int): size of the window used to compute silent subframes. Defaults to 32. Range [1,inf) 
		""" 
		... 
	def __call__(self, frame:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			frame (NDArray[np.float32]): the input frame (must be non-empty). Defaults to None. 
		Returns:
			discontinuityLocations (NDArray[np.float32]): the index of the detected discontinuities (if any)
			discontinuityAmplitudes (NDArray[np.float32]): the peak values of the prediction error for the discontinuities (if any)
		""" 
		... 


class Dissonance(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Computes the sensory dissonance of an audio signal given its spectral peaks.
		
		Sensory dissonance (to be distinguished from musical or theoretical dissonance) measures perceptual roughness of the sound and is based on the roughness of its spectral peaks. Given the spectral peaks, the algorithm estimates total dissonance by summing up the normalized dissonance values for each pair of peaks. These values are computed using dissonance curves, which define dissonace between two spectral peaks according to their frequency and amplitude relations. The dissonance curves are based on perceptual experiments conducted in [1].
		Exceptions are thrown when the size of the input vectors are not equal or if input frequencies are not ordered ascendantly
		References:
		  [1] R. Plomp and W. J. M. Levelt, "Tonal Consonance and Critical
		  Bandwidth," The Journal of the Acoustical Society of America, vol. 38,
		  no. 4, pp. 548–560, 1965.
		
		  [2] Critical Band - Handbook for Acoustic Ecology
		  http://www.sfu.ca/sonic-studio/handbook/Critical_Band.html
		
		  [3] Bark Scale -  Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Bark_scale		""" 
		... 
	def __call__(self, frequencies:NDArray[np.float32], magnitudes:NDArray[np.float32]) -> float:
		"""compute
		Args:
			frequencies (NDArray[np.float32]): the frequencies of the spectral peaks (must be sorted by frequency). Defaults to None. 
			magnitudes (NDArray[np.float32]): the magnitudes of the spectral peaks (must be sorted by frequency. Defaults to None. 
		Returns:
			dissonance (float): the dissonance of the audio signal (0 meaning completely consonant, and 1 meaning completely dissonant)
		""" 
		... 


class DistributionShape(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Computes the spread (variance), skewness and kurtosis of an array given its central moments.
		
		The extracted features are good indicators of the shape of the distribution. For the required input see CentralMoments algorithm.
		The size of the input array must be at least 5. An exception will be thrown otherwise.
		
		References:
		  [1] G. Peeters, "A large set of audio features for sound description
		  (similarity and classification) in the CUIDADO project," CUIDADO I.S.T.
		  Project Report, 2004.
		
		  [2] Variance - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Variance
		
		  [3] Skewness - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Skewness
		
		  [4] Kurtosis - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Kurtosis		""" 
		... 
	def __call__(self, centralMoments:NDArray[np.float32]) -> tuple[float, float, float]:
		"""compute
		Args:
			centralMoments (NDArray[np.float32]): the central moments of a distribution. Defaults to None. 
		Returns:
			spread (float): the spread (variance) of the distribution
			skewness (float): the skewness of the distribution
			kurtosis (float): the kurtosis of the distribution
		""" 
		... 


class Duration(_essentia.Algorithm): 
	def __init__(self, sampleRate:float=44100.0) -> None:
		"""Outputs the total duration of an audio signal.

		Args:
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> float:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input signal. Defaults to None. 
		Returns:
			duration (float): the duration of the signal [s]
		""" 
		... 


class DynamicComplexity(_essentia.Algorithm): 
	def __init__(self, frameSize:float=0.2, sampleRate:float=44100.0) -> None:
		"""Computes the dynamic complexity defined as the average absolute deviation from the global loudness level estimate on the dB scale.
		
		It is related to the dynamic range and to the amount of fluctuation in loudness present in a recording. Silence at the beginning and at the end of a track are ignored in the computation in order not to deteriorate the results.
		
		References:
		  [1] S. Streich, Music complexity: a multi-faceted description of audio
		  content, UPF, Barcelona, Spain, 2007.

		Args:
			frameSize (float): the frame size [s]. Defaults to 0.2. Range (0,inf)
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> tuple[float, float]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input audio signal. Defaults to None. 
		Returns:
			dynamicComplexity (float): the dynamic complexity coefficient
			loudness (float): an estimate of the loudness [dB]
		""" 
		... 


class ERBBands(_essentia.Algorithm): 
	def __init__(self, highFrequencyBound:float=22050.0, inputSize:int=1025, lowFrequencyBound:float=50.0, numberBands:int=40, sampleRate:float=44100.0, type:str='power', width:float=1.0) -> None:
		"""Computes energies/magnitudes in ERB bands of a spectrum.
		
		The Equivalent Rectangular Bandwidth (ERB) scale is used. The algorithm applies a frequency domain filterbank using gammatone filters. Adapted from matlab code in:  D. P. W. Ellis (2009). 'Gammatone-like spectrograms', web resource [1].
		
		References:
		  [1] http://www.ee.columbia.edu/~dpwe/resources/matlab/gammatonegram/
		
		  [2] B. C. Moore and B. R. Glasberg, "Suggested formulae for calculating
		  auditory-filter bandwidths and excitation patterns," Journal of the
		  Acoustical Society of America, vol. 74, no. 3, pp. 750–753, 1983.

		Args:
			highFrequencyBound (float): an upper-bound limit for the frequencies to be included in the bands. Defaults to 22050.0. Range [0,inf)
			inputSize (int): the size of the spectrum. Defaults to 1025. Range (1,inf)
			lowFrequencyBound (float): a lower-bound limit for the frequencies to be included in the bands. Defaults to 50.0. Range [0,inf)
			numberBands (int): the number of output bands. Defaults to 40. Range (1,inf)
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			type (str): use magnitude or power spectrum. Defaults to 'power'. Range {magnitude,power}
			width (float): filter width with respect to ERB. Defaults to 1.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, spectrum:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			spectrum (NDArray[np.float32]): the audio spectrum. Defaults to None. 
		Returns:
			bands (NDArray[np.float32]): the energies/magnitudes of each band
		""" 
		... 


class EasyLoader(_essentia.Algorithm): 
	def __init__(self, filename:str, audioStream:int=0, downmix:str='mix', endTime:float=1000000.0, replayGain:float=-6.0, sampleRate:float=44100.0, startTime:float=0.0) -> None:
		"""Loads the raw audio data from an audio file, downmixes it to mono and normalizes using replayGain.
		
		The audio is resampled in case the given sampling rate does not match the sampling rate of the input signal and is normalized by the given replayGain value.
		
		This algorithm uses MonoLoader and therefore inherits all of its input requirements and exceptions.
		
		References:
		  [1] Replay Gain - A Proposed Standard,
		  http://replaygain.hydrogenaudio.org

		Args:
			audioStream (int): audio stream index to be loaded. Other streams are no taken into account (e.g. if stream 0 is video and 1 is audio use index 0 to access it.). Defaults to 0. Range [0,inf)
			downmix (str): the mixing type for stereo files. Defaults to 'mix'. Range {left,right,mix}
			endTime (float): the end time of the slice to be extracted [s]. Defaults to 1000000.0. Range [0,inf)
			filename (str): the name of the file from which to read. Defaults to None. Range None
			replayGain (float): the value of the replayGain that should be used to normalize the signal [dB]. Defaults to -6.0. Range (-inf,inf)
			sampleRate (float): the output sampling rate [Hz]. Defaults to 44100.0. Range (0,inf)
			startTime (float): the start time of the slice to be extracted [s]. Defaults to 0.0. Range [0,inf) 
		""" 
		... 
	def __call__(self, ) -> NDArray[np.float32]:
		"""compute
		Returns:
			audio (NDArray[np.float32]): the audio signal
		""" 
		... 


class EffectiveDuration(_essentia.Algorithm): 
	def __init__(self, sampleRate:float=44100.0, thresholdRatio:float=0.4) -> None:
		"""Computes the effective duration of an envelope signal.
		
		The effective duration is a measure of the time the signal is perceptually meaningful. This is approximated by the time the envelope is above or equal to a given threshold and is above the -90db noise floor. This measure allows to distinguish percussive sounds from sustained sounds but depends on the signal length.
		By default, this algorithm uses 40% of the envelope maximum as the threshold which is suited for short sounds. Note, that the 0% thresold corresponds to the duration of signal above -90db noise floor, while the 100% thresold corresponds to the number of times the envelope takes its maximum value.
		References:
		  [1] G. Peeters, "A large set of audio features for sound description
		  (similarity and classification) in the CUIDADO project," CUIDADO I.S.T.
		  Project Report, 2004

		Args:
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			thresholdRatio (float): the ratio of the envelope maximum to be used as the threshold. Defaults to 0.4. Range [0,1] 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> float:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input signal. Defaults to None. 
		Returns:
			effectiveDuration (float): the effective duration of the signal [s]
		""" 
		... 


class Energy(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Computes the energy of an array.
		
		The input array should not be empty or an exception will be thrown.
		
		References:
		  [1] Energy (signal processing) - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Energy_%28signal_processing%29		""" 
		... 
	def __call__(self, array:NDArray[np.float32]) -> float:
		"""compute
		Args:
			array (NDArray[np.float32]): the input array. Defaults to None. 
		Returns:
			energy (float): the energy of the input array
		""" 
		... 


class EnergyBand(_essentia.Algorithm): 
	def __init__(self, sampleRate:float=44100.0, startCutoffFrequency:float=0.0, stopCutoffFrequency:float=100.0) -> None:
		"""Computes energy in a given frequency band of a spectrum including both start and stop cutoff frequencies.
		
		Note that exceptions will be thrown when input spectrum is empty and if startCutoffFrequency is greater than stopCutoffFrequency.
		
		References:
		  [1] Energy (signal processing) - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Energy_(signal_processing)

		Args:
			sampleRate (float): the audio sampling rate [Hz]. Defaults to 44100.0. Range (0,inf)
			startCutoffFrequency (float): the start frequency from which to sum the energy [Hz]. Defaults to 0.0. Range [0,inf)
			stopCutoffFrequency (float): the stop frequency to which to sum the energy [Hz]. Defaults to 100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, spectrum:NDArray[np.float32]) -> float:
		"""compute
		Args:
			spectrum (NDArray[np.float32]): the input frequency spectrum. Defaults to None. 
		Returns:
			energyBand (float): the energy in the frequency band
		""" 
		... 


class EnergyBandRatio(_essentia.Algorithm): 
	def __init__(self, sampleRate:float=44100.0, startFrequency:float=0.0, stopFrequency:float=100.0) -> None:
		"""Computes the ratio of the spectral energy in the range [startFrequency, stopFrequency] over the total energy.
		
		An exception is thrown when startFrequency is larger than stopFrequency
		or the input spectrum is empty.
		
		References:
		  [1] Energy (signal processing) - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Energy_%28signal_processing%29

		Args:
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			startFrequency (float): the frequency from which to start summing the energy [Hz]. Defaults to 0.0. Range [0,inf)
			stopFrequency (float): the frequency up to which to sum the energy [Hz]. Defaults to 100.0. Range [0,inf) 
		""" 
		... 
	def __call__(self, spectrum:NDArray[np.float32]) -> float:
		"""compute
		Args:
			spectrum (NDArray[np.float32]): the input audio spectrum. Defaults to None. 
		Returns:
			energyBandRatio (float): the energy ratio of the specified band over the total energy
		""" 
		... 


class Entropy(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Computes the Shannon entropy of an array.
		
		Entropy can be used to quantify the peakiness of a distribution. This has been used for voiced/unvoiced decision in automatic speech recognition. 
		
		Entropy cannot be computed neither on empty arrays nor arrays which contain negative values. In such cases, exceptions will be thrown.
		
		References:
		  [1] H. Misra, S. Ikbal, H. Bourlard and H. Hermansky, "Spectral entropy
		  based feature for robust ASR," in IEEE International Conference on
		  Acoustics, Speech, and Signal Processing (ICASSP'04).		""" 
		... 
	def __call__(self, array:NDArray[np.float32]) -> float:
		"""compute
		Args:
			array (NDArray[np.float32]): the input array (cannot contain negative values, and must be non-empty). Defaults to None. 
		Returns:
			entropy (float): the entropy of the input array
		""" 
		... 


class Envelope(_essentia.Algorithm): 
	def __init__(self, applyRectification:bool=True, attackTime:float=10.0, releaseTime:float=1500.0, sampleRate:float=44100.0) -> None:
		"""Computes the envelope of a signal by applying a non-symmetric lowpass filter on a signal.
		
		By default it rectifies the signal, but that is optional.
		
		References:
		  [1] U. Zölzer, Digital Audio Signal Processing,
		  John Wiley & Sons Ltd, 1997, ch.7

		Args:
			applyRectification (bool): whether to apply rectification (envelope based on the absolute value of signal). Defaults to True. Range {true,false}
			attackTime (float): the attack time of the first order lowpass in the attack phase [ms]. Defaults to 10.0. Range [0,inf)
			releaseTime (float): the release time of the first order lowpass in the release phase [ms]. Defaults to 1500.0. Range [0,inf)
			sampleRate (float): the audio sampling rate [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input signal. Defaults to None. 
		Returns:
			signal (NDArray[np.float32]): the resulting envelope of the signal
		""" 
		... 


class EqloudLoader(_essentia.Algorithm): 
	def __init__(self, filename:str, downmix:str='mix', endTime:float=1000000.0, replayGain:float=-6.0, sampleRate:float=44100.0, startTime:float=0.0) -> None:
		"""Loads the raw audio data from an audio file, downmixes it to mono and normalizes using replayGain and equal-loudness filter.
		
		Audio is resampled in case the given sampling rate does not match the sampling rate of the input signal and normalized by the given replayGain gain. In addition, audio data is filtered through an equal-loudness filter.
		
		This algorithm uses MonoLoader and thus inherits all of its input requirements and exceptions.
		
		References:
		  [1] Replay Gain - A Proposed Standard,
		  http://replaygain.hydrogenaudio.org
		  [2] Replay Gain - Equal Loudness Filter,
		  http://replaygain.hydrogenaudio.org/proposal/equal_loudness.html

		Args:
			downmix (str): the mixing type for stereo files. Defaults to 'mix'. Range {left,right,mix}
			endTime (float): the end time of the slice to be extracted [s]. Defaults to 1000000.0. Range [0,inf)
			filename (str): the name of the file from which to read. Defaults to None. Range None
			replayGain (float): the value of the replayGain [dB] that should be used to normalize the signal [dB]. Defaults to -6.0. Range (-inf,inf)
			sampleRate (float): the output sampling rate [Hz]. Defaults to 44100.0. Range {32000,44100,48000}
			startTime (float): the start time of the slice to be extracted [s]. Defaults to 0.0. Range [0,inf) 
		""" 
		... 
	def __call__(self, ) -> NDArray[np.float32]:
		"""compute
		Returns:
			audio (NDArray[np.float32]): the audio signal
		""" 
		... 


class EqualLoudness(_essentia.Algorithm): 
	def __init__(self, sampleRate:float=44100.0) -> None:
		"""Implements an equal-loudness filter.
		
		The human ear does not perceive sounds of all frequencies as having equal loudness, and to account for this, the signal is filtered by an inverted approximation of the equal-loudness curves. Technically, the filter is a cascade of a 10th order Yulewalk filter with a 2nd order Butterworth high pass filter.
		
		This algorithm depends on the IIR algorithm. Any requirements of the IIR algorithm are imposed for this algorithm. This algorithm is only defined for the sampling rates specified in parameters. It will throw an exception if attempting to configure with any other sampling rate.
		
		References:
		  [1] Replay Gain - Equal Loudness Filter,
		  http://replaygain.hydrogenaud.io/proposal/equal_loudness.html

		Args:
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range {8000,16000,32000,44100,48000} 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input signal. Defaults to None. 
		Returns:
			signal (NDArray[np.float32]): the filtered signal
		""" 
		... 


class Extractor(_essentia.Algorithm): 
	def __init__(self, dynamics:bool=True, dynamicsFrameSize:int=88200, dynamicsHopSize:int=44100, highLevel:bool=True, lowLevel:bool=True, lowLevelFrameSize:int=2048, lowLevelHopSize:int=1024, midLevel:bool=True, namespace:str='', relativeIoi:bool=False, rhythm:bool=True, sampleRate:float=44100.0, tonalFrameSize:int=4096, tonalHopSize:int=2048, tuning:bool=True) -> None:
		"""Extracts all low-level, mid-level and high-level features from an audio signal and stores them in a pool.

		Args:
			dynamics (bool): compute dynamics' features. Defaults to True. Range {true,false}
			dynamicsFrameSize (int): the frame size for level dynamics. Defaults to 88200. Range (0,inf)
			dynamicsHopSize (int): the hop size for level dynamics. Defaults to 44100. Range (0,inf)
			highLevel (bool): compute high level features. Defaults to True. Range {true,false}
			lowLevel (bool): compute low level features. Defaults to True. Range {true,false}
			lowLevelFrameSize (int): the frame size for computing low level features. Defaults to 2048. Range (0,inf)
			lowLevelHopSize (int): the hop size for computing low level features. Defaults to 1024. Range (0,inf)
			midLevel (bool): compute mid level features. Defaults to True. Range {true,false}
			namespace (str): the main namespace under which to store the results. Defaults to ''. Range None
			relativeIoi (bool): compute relative inter onset intervals. Defaults to False. Range {true,false}
			rhythm (bool): compute rhythm features. Defaults to True. Range {true,false}
			sampleRate (float): the audio sampling rate. Defaults to 44100.0. Range (0,inf)
			tonalFrameSize (int): the frame size for low level tonal features. Defaults to 4096. Range (0,inf)
			tonalHopSize (int): the hop size for low level tonal features. Defaults to 2048. Range (0,inf)
			tuning (bool): compute tuning frequency. Defaults to True. Range {true,false} 
		""" 
		... 
	def __call__(self, audio:NDArray[np.float32]) -> Pool:
		"""compute
		Args:
			audio (NDArray[np.float32]): the input audio signal. Defaults to None. 
		Returns:
			pool (Pool): the pool where to store the results
		""" 
		... 


class FFT(_essentia.Algorithm): 
	def __init__(self, size:int=1024) -> None:
		"""Computes the positive complex short-term Fourier transform (STFT) of an array using the FFT algorithm.
		
		The resulting fft has a size of (s/2)+1, where s is the size of the input frame.
		At the moment FFT can only be computed on frames which size is even and non zero, otherwise an exception is thrown.
		
		References:
		  [1] Fast Fourier transform - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Fft
		
		  [2] Fast Fourier Transform -- from Wolfram MathWorld,
		  http://mathworld.wolfram.com/FastFourierTransform.html

		Args:
			size (int): the expected size of the input frame. This is purely optional and only targeted at optimizing the creation time of the FFT object. Defaults to 1024. Range [1,inf) 
		""" 
		... 
	def __call__(self, frame:NDArray[np.float32]) -> np.ndarray:
		"""compute
		Args:
			frame (NDArray[np.float32]): the input audio frame. Defaults to None. 
		Returns:
			fft (np.ndarray): the FFT of the input frame
		""" 
		... 


class FFTC(_essentia.Algorithm): 
	def __init__(self, negativeFrequencies:bool=False, size:int=1024) -> None:
		"""Computes the complex short-term Fourier transform (STFT) of a complex array using the FFT algorithm.
		
		If the `negativeFrequencies` flag is set on, the resulting fft has a size of (s/2)+1, where s is the size of the input frame. Otherwise, output matches the input size.
		At the moment FFT can only be computed on frames which size is even and non zero, otherwise an exception is thrown.
		
		References:
		  [1] Fast Fourier transform - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Fft
		
		  [2] Fast Fourier Transform -- from Wolfram MathWorld,
		  http://mathworld.wolfram.com/FastFourierTransform.html

		Args:
			negativeFrequencies (bool): returns the full spectrum or just the positive frequencies. Defaults to False. Range {true,false}
			size (int): the expected size of the input frame. This is purely optional and only targeted at optimizing the creation time of the FFT object. Defaults to 1024. Range [1,inf) 
		""" 
		... 
	def __call__(self, frame:np.ndarray) -> np.ndarray:
		"""compute
		Args:
			frame (np.ndarray): the input frame (complex). Defaults to None. 
		Returns:
			fft (np.ndarray): the FFT of the input frame
		""" 
		... 


class FadeDetection(_essentia.Algorithm): 
	def __init__(self, cutoffHigh:float=0.85, cutoffLow:float=0.2, frameRate:float=4.0, minLength:float=3.0) -> None:
		"""Detects fade-in and fade-outs time positions in an audio signal given a sequence of RMS values.
		
		It outputs two arrays containing the start/stop points of fade-ins and fade-outs. The main hypothesis for the detection is that an increase or decrease of the RMS over time in an audio file corresponds to a fade-in or fade-out, repectively. Minimum and maximum mean-RMS-thresholds are used to define where fade-in and fade-outs occur.
		
		An exception is thrown if the input "rms" is empty.
		
		References:
		  [1] Fade (audio engineering) - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Fade-in

		Args:
			cutoffHigh (float): fraction of the average RMS to define the maximum threshold. Defaults to 0.85. Range (0,1]
			cutoffLow (float): fraction of the average RMS to define the minimum threshold. Defaults to 0.2. Range [0,1)
			frameRate (float): the rate of frames used in calculation of the RMS [frames/s]. Defaults to 4.0. Range (0,inf)
			minLength (float): the minimum length to consider a fade-in/out [s]. Defaults to 3.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, rms:NDArray[np.float32]) -> tuple[np.ndarray, np.ndarray]:
		"""compute
		Args:
			rms (NDArray[np.float32]): rms values array. Defaults to None. 
		Returns:
			fadeIn (np.ndarray): 2D-array containing start/stop timestamps corresponding to fade-ins [s] (ordered chronologically)
			fadeOut (np.ndarray): 2D-array containing start/stop timestamps corresponding to fade-outs [s] (ordered chronologically)
		""" 
		... 


class FalseStereoDetector(_essentia.Algorithm): 
	def __init__(self, correlationThreshold:float=0.9995, silenceThreshold:int=-70) -> None:
		"""Detects if a stereo track has duplicated channels (false stereo).It is based on the Pearson linear correlation coefficient and thus it is robust scaling and shifting between channels.

		Args:
			correlationThreshold (float): threshold to activate the isFalseStereo flag. Defaults to 0.9995. Range [-1,1]
			silenceThreshold (int): Silent frames will be skkiped.. Defaults to -70. Range (-inf,0) 
		""" 
		... 
	def __call__(self, frame:NDArray[np.float32]) -> tuple[int, float]:
		"""compute
		Args:
			frame (NDArray[np.float32]): the input frame (must be non-empty). Defaults to None. 
		Returns:
			isFalseStereo (int): a flag indicating if the frame channes are simmilar
			correlation (float): correlation betweeen the input channels
		""" 
		... 


class Flatness(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Computes the flatness of an array, which is defined as the ratio between the geometric mean and the arithmetic mean.
		
		Flatness is undefined for empty input and negative values, therefore an exception is thrown in any both cases.
		
		References:
		  [1] G. Peeters, "A large set of audio features for sound description
		  (similarity and classification) in the CUIDADO project," CUIDADO I.S.T.
		  Project Report, 2004		""" 
		... 
	def __call__(self, array:NDArray[np.float32]) -> float:
		"""compute
		Args:
			array (NDArray[np.float32]): the input array. Defaults to None. 
		Returns:
			flatness (float): the flatness (ratio between the geometric and the arithmetic mean of the input array)
		""" 
		... 


class FlatnessDB(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Computes the flatness of an array, which is defined as the ratio between the geometric mean and the arithmetic mean converted to dB scale.
		
		Specifically, it can be used to compute spectral flatness [1,2], which is a measure of how noise-like a sound is, as opposed to being tone-like. The meaning of tonal in this context is in the sense of the amount of peaks or resonant structure in a power spectrum, as opposed to flat spectrum of a white noise. A high spectral flatness (approaching 1.0 for white noise) indicates that the spectrum has a similar amount of power in all spectral bands — this would sound similar to white noise, and the graph of the spectrum would appear relatively flat and smooth. A low spectral flatness (approaching 0.0 for a pure tone) indicates that the spectral power is concentrated in a relatively small number of bands — this would typically sound like a mixture of sine waves, and the spectrum would appear "spiky"
		
		The size of the input array must be greater than 0. If the input array is empty an exception will be thrown. This algorithm uses the Flatness algorithm and thus inherits its input requirements and exceptions.
		
		References:
		  [1] G. Peeters, "A large set of audio features for sound description
		  (similarity and classification) in the CUIDADO project," CUIDADO I.S.T.
		  Project Report, 2004
		
		  [2] Spectral flatness -  Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Spectral_flatness		""" 
		... 
	def __call__(self, array:NDArray[np.float32]) -> float:
		"""compute
		Args:
			array (NDArray[np.float32]): the input array. Defaults to None. 
		Returns:
			flatnessDB (float): the flatness dB
		""" 
		... 


class FlatnessSFX(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Calculates the flatness coefficient of a signal envelope.
		
		There are two thresholds defined: a lower one at 20% and an upper one at 95%. The thresholds yield two values: one value which has 20% of the total values underneath, and one value which has 95% of the total values underneath. The flatness coefficient is then calculated as the ratio of these two values. This algorithm is meant to be plugged after Envelope algorithm, however in streaming mode a RealAccumulator algorithm should be connected in between the two.
		In the current form the algorithm can't be calculated in streaming mode, since it would violate the streaming mode policy of having low memory consumption.
		
		An exception is thrown if the input envelope is empty.		""" 
		... 
	def __call__(self, envelope:NDArray[np.float32]) -> float:
		"""compute
		Args:
			envelope (NDArray[np.float32]): the envelope of the signal. Defaults to None. 
		Returns:
			flatness (float): the flatness coefficient
		""" 
		... 


class Flux(_essentia.Algorithm): 
	def __init__(self, halfRectify:bool=False, norm:str='L2') -> None:
		"""Computes the spectral flux of a spectrum.
		
		Flux is defined as the L2-norm [1] or L1-norm [2] of the difference between two consecutive frames of the magnitude spectrum. The frames have to be of the same size in order to yield a meaningful result. The default L2-norm is used more commonly.
		
		An exception is thrown if the size of the input spectrum does not equal the previous input spectrum's size.
		
		References:
		  [1] Tzanetakis, G., Cook, P., "Multifeature Audio Segmentation for
		  Browsing and Annotation", Proceedings of the 1999 IEEE Workshop on
		  Applications of Signal Processing to Audio and Acoustics, New Paltz,
		  NY, USA, 1999, W99 1-4
		
		  [2] S. Dixon, "Onset detection revisited", in International Conference on
		  Digital Audio Effects (DAFx'06), 2006, vol. 120, pp. 133-137.
		
		  [3] http://en.wikipedia.org/wiki/Spectral_flux
		

		Args:
			halfRectify (bool): half-rectify the differences in each spectrum bin. Defaults to False. Range {true,false}
			norm (str): the norm to use for difference computation. Defaults to 'L2'. Range {L1,L2} 
		""" 
		... 
	def __call__(self, spectrum:NDArray[np.float32]) -> float:
		"""compute
		Args:
			spectrum (NDArray[np.float32]): the input spectrum. Defaults to None. 
		Returns:
			flux (float): the spectral flux of the input spectrum
		""" 
		... 


class FrameBuffer(_essentia.Algorithm): 
	def __init__(self, bufferSize:int=2048, zeroPadding:bool=True) -> None:
		"""Buffers input non-overlapping audio frames into longer overlapping frames with a hop sizes equal to input frame size.
		
		In standard mode, each compute() call updates and outputs the gathered buffer.
		
		Input frames can be of variate length. Input frames longer than the buffer size will be cropped. Empty input frames will raise an exception.

		Args:
			bufferSize (int): the buffer size. Defaults to 2048. Range (0,inf)
			zeroPadding (bool): initialize the buffer with zeros (output zero-padded buffer frames if `true`, otherwise output empty frames until a full buffer is accumulated). Defaults to True. Range {true,false} 
		""" 
		... 
	def __call__(self, frame:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			frame (NDArray[np.float32]): the input audio frame. Defaults to None. 
		Returns:
			frame (NDArray[np.float32]): the buffered audio frame
		""" 
		... 


class FrameCutter(_essentia.Algorithm): 
	def __init__(self, frameSize:int=1024, hopSize:int=512, lastFrameToEndOfFile:bool=False, startFromZero:bool=False, validFrameThresholdRatio:float=0.0) -> None:
		"""Slices the input buffer into frames.
		
		It returns a frame of a constant size and jumps a constant amount of samples forward in the buffer on every compute() call until no more frames can be extracted; empty frame vectors are returned afterwards. Incomplete frames (frames starting before the beginning of the input buffer or going past its end) are zero-padded or dropped according to the "validFrameThresholdRatio" parameter.
		
		The algorithm outputs as many frames as needed to consume all the information contained in the input buffer. Depending on the "startFromZero" parameter:
		  - startFromZero = true: a frame is the last one if its end position is at or beyond the end of the stream. The last frame will be zero-padded if its size is less than "frameSize"
		  - startFromZero = false: a frame is the last one if its center position is at or beyond the end of the stream
		In both cases the start time of the last frame is never beyond the end of the stream.
		

		Args:
			frameSize (int): the output frame size. Defaults to 1024. Range [1,inf)
			hopSize (int): the hop size between frames. Defaults to 512. Range [1,inf)
			lastFrameToEndOfFile (bool): whether the beginning of the last frame should reach the end of file. Only applicable if startFromZero is true. Defaults to False. Range {true,false}
			startFromZero (bool): whether to start the first frame at time 0 (centered at frameSize/2) if true, or -frameSize/2 otherwise (zero-centered). Defaults to False. Range {true,false}
			validFrameThresholdRatio (float): frames smaller than this ratio will be discarded, those larger will be zero-padded to a full frame (i.e. a value of 0 will never discard frames and a value of 1 will only keep frames that are of length 'frameSize'). Defaults to 0.0. Range [0,1] 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the buffer from which to read data. Defaults to None. 
		Returns:
			frame (NDArray[np.float32]): the frame to write to
		""" 
		... 


class FrameGenerator(_essentia.Algorithm): 
	def __init__(self, frameSize:int=1024, hopSize:int=512, lastFrameToEndOfFile:bool=False, startFromZero:bool=False, validFrameThresholdRatio:float=0.0) -> None:
		"""Is a Python generator for the FrameCutter algorithm.
		
		It is not available in C++.
		
		FrameGenerator inherits all the parameters of the FrameCutter. The way to use it in Python is the following:
		
		  for frame in FrameGenerator(audio, frameSize, hopSize):
		      do_something()
		
		

		Args:
			frameSize (int): the output frame size. Defaults to 1024. Range [1,inf)
			hopSize (int): the hop size between frames. Defaults to 512. Range [1,inf)
			lastFrameToEndOfFile (bool): whether the beginning of the last frame should reach the end of file. Only applicable if startFromZero is true. Defaults to False. Range {true,false}
			startFromZero (bool): whether to start the first frame at time 0 (centered at frameSize/2) if true, or -frameSize/2 otherwise (zero-centered). Defaults to False. Range {true,false}
			validFrameThresholdRatio (float): frames smaller than this ratio will be discarded, those larger will be zero-padded to a full frame (i.e. a value of 0 will never discard frames and a value of 1 will only keep frames that are of length 'frameSize'). Defaults to 0.0. Range [0,1] 
		""" 
		... 
	def __call__(self, ) -> None:
		"""compute
		""" 
		... 


class FrameToReal(_essentia.Algorithm): 
	def __init__(self, frameSize:int=2048, hopSize:int=128) -> None:
		"""Converts a sequence of input audio signal frames into a sequence of audio samples.
		
		Empty input signals will raise an exception.

		Args:
			frameSize (int): the frame size for computing the overlap-add process. Defaults to 2048. Range (0,inf)
			hopSize (int): the hop size with which the overlap-add function is computed. Defaults to 128. Range (0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input audio frame. Defaults to None. 
		Returns:
			signal (NDArray[np.float32]): the output audio samples
		""" 
		... 


class FreesoundExtractor(_essentia.Algorithm): 
	def __init__(self, profile:str, analysisSampleRate:float=44100.0, endTime:float=1000000.0, gfccStats:list[str]=['mean', 'var', 'stdev', 'median', 'min', 'max', 'dmean', 'dmean2', 'dvar', 'dvar2'], lowlevelFrameSize:int=2048, lowlevelHopSize:int=1024, lowlevelSilentFrames:str='noise', lowlevelStats:list[str]=['mean', 'var', 'stdev', 'median', 'min', 'max', 'dmean', 'dmean2', 'dvar', 'dvar2'], lowlevelWindowType:str='blackmanharris62', lowlevelZeroPadding:int=0, mfccStats:list[str]=['mean', 'var', 'stdev', 'median', 'min', 'max', 'dmean', 'dmean2', 'dvar', 'dvar2'], rhythmMaxTempo:int=210, rhythmMethod:str='multifeature', rhythmMinTempo:int=40, rhythmStats:list[str]=['mean', 'var', 'stdev', 'median', 'min', 'max', 'dmean', 'dmean2', 'dvar', 'dvar2'], startTime:float=0.0, tonalFrameSize:int=4096, tonalHopSize:int=2048, tonalSilentFrames:str='noise', tonalStats:list[str]=['mean', 'var', 'stdev', 'median', 'min', 'max', 'dmean', 'dmean2', 'dvar', 'dvar2'], tonalWindowType:str='blackmanharris62', tonalZeroPadding:int=0) -> None:
		"""Is a wrapper for Freesound Extractor.
		
		See documentation for 'essentia_streaming_extractor_freesound'.

		Args:
			analysisSampleRate (float): the analysis sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			endTime (float): the end time of the slice you want to extract [s]. Defaults to 1000000.0. Range [0,inf)
			gfccStats (list[str]): the statistics to compute for GFCC features. Defaults to ['mean', 'var', 'stdev', 'median', 'min', 'max', 'dmean', 'dmean2', 'dvar', 'dvar2']. Range None
			lowlevelFrameSize (int): the frame size for computing low-level features. Defaults to 2048. Range (0,inf)
			lowlevelHopSize (int): the hop size for computing low-level features. Defaults to 1024. Range (0,inf)
			lowlevelSilentFrames (str): whether to [keep/drop/add noise to] silent frames for computing low-level features. Defaults to 'noise'. Range {drop,keep,noise}
			lowlevelStats (list[str]): the statistics to compute for low-level features. Defaults to ['mean', 'var', 'stdev', 'median', 'min', 'max', 'dmean', 'dmean2', 'dvar', 'dvar2']. Range None
			lowlevelWindowType (str): the window type for computing low-level features. Defaults to 'blackmanharris62'. Range {hamming,hann,triangular,square,blackmanharris62,blackmanharris70,blackmanharris74,blackmanharris92}
			lowlevelZeroPadding (int): zero padding factor for computing low-level features. Defaults to 0. Range [0,inf)
			mfccStats (list[str]): the statistics to compute for MFCC features. Defaults to ['mean', 'var', 'stdev', 'median', 'min', 'max', 'dmean', 'dmean2', 'dvar', 'dvar2']. Range None
			profile (str): profile filename. If specified, default parameter values are overwritten by values in the profile yaml file. If not specified (empty string), use values configured by user like in other normal algorithms. Defaults to None. Range None
			rhythmMaxTempo (int): the fastest tempo to detect [bpm]. Defaults to 210. Range [60,250]
			rhythmMethod (str): the method used for beat tracking. Defaults to 'multifeature'. Range {multifeature,degara}
			rhythmMinTempo (int): the slowest tempo to detect [bpm]. Defaults to 40. Range [40,180]
			rhythmStats (list[str]): the statistics to compute for rhythm features. Defaults to ['mean', 'var', 'stdev', 'median', 'min', 'max', 'dmean', 'dmean2', 'dvar', 'dvar2']. Range None
			startTime (float): the start time of the slice you want to extract [s]. Defaults to 0.0. Range [0,inf)
			tonalFrameSize (int): the frame size for computing tonal features. Defaults to 4096. Range (0,inf)
			tonalHopSize (int): the hop size for computing tonal features. Defaults to 2048. Range (0,inf)
			tonalSilentFrames (str): whether to [keep/drop/add noise to] silent frames for computing tonal features. Defaults to 'noise'. Range {drop,keep,noise}
			tonalStats (list[str]): the statistics to compute for tonal features. Defaults to ['mean', 'var', 'stdev', 'median', 'min', 'max', 'dmean', 'dmean2', 'dvar', 'dvar2']. Range None
			tonalWindowType (str): the window type for computing tonal features. Defaults to 'blackmanharris62'. Range {hamming,hann,triangular,square,blackmanharris62,blackmanharris70,blackmanharris74,blackmanharris92}
			tonalZeroPadding (int): zero padding factor for computing tonal features. Defaults to 0. Range [0,inf) 
		""" 
		... 
	def __call__(self, filename:str) -> tuple[Pool, Pool]:
		"""compute
		Args:
			filename (str): the input audiofile. Defaults to None. 
		Returns:
			results (Pool): Analysis results pool with across-frames statistics
			resultsFrames (Pool): Analysis results pool with computed frame values
		""" 
		... 


class FrequencyBands(_essentia.Algorithm): 
	def __init__(self, frequencyBands:NDArray[np.float32]=np.array([0, 50, 100, 150, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500, 20500, 27000]), sampleRate:float=44100.0) -> None:
		"""Computes energy in rectangular frequency bands of a spectrum.
		
		The bands are non-overlapping. For each band the power-spectrum (mag-squared) is summed.
		
		Parameter "frequencyBands" must contain at least 2 frequencies, they all must be positive and must be ordered ascentdantly, otherwise an exception will be thrown. FrequencyBands is only defined for spectra, which size is greater than 1.
		
		References:
		  [1] Frequency Range - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Frequency_band
		
		  [2] Band - Handbook For Acoustic Ecology,
		  http://www.sfu.ca/sonic-studio/handbook/Band.html

		Args:
			frequencyBands (NDArray[np.float32]): list of frequency ranges in to which the spectrum is divided (these must be in ascending order and connot contain duplicates). Defaults to np.array([0, 50, 100, 150, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500, 20500, 27000]). Range None
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, spectrum:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			spectrum (NDArray[np.float32]): the input spectrum (must be greater than size one). Defaults to None. 
		Returns:
			bands (NDArray[np.float32]): the energy in each band
		""" 
		... 


class GFCC(_essentia.Algorithm): 
	def __init__(self, dctType:int=2, highFrequencyBound:float=22050.0, inputSize:int=1025, logType:str='dbamp', lowFrequencyBound:float=40.0, numberBands:int=40, numberCoefficients:int=13, sampleRate:float=44100.0, silenceThreshold:float=1e-10, type:str='power') -> None:
		"""Computes the Gammatone-frequency cepstral coefficients of a spectrum.
		
		This is an equivalent of MFCCs, but using a gammatone filterbank (ERBBands) scaled on an Equivalent Rectangular Bandwidth (ERB) scale.
		
		References:
		  [1] Y. Shao, Z. Jin, D. Wang, and S. Srinivasan, "An auditory-based feature
		  for robust speech recognition," in IEEE International Conference on
		  Acoustics, Speech, and Signal Processing (ICASSP’09), 2009,
		  pp. 4625-4628.

		Args:
			dctType (int): the DCT type. Defaults to 2. Range [2,3]
			highFrequencyBound (float): the upper bound of the frequency range [Hz]. Defaults to 22050.0. Range (0,inf)
			inputSize (int): the size of input spectrum. Defaults to 1025. Range (1,inf)
			logType (str): logarithmic compression type. Use 'dbpow' if working with power and 'dbamp' if working with magnitudes. Defaults to 'dbamp'. Range {natural,dbpow,dbamp,log}
			lowFrequencyBound (float): the lower bound of the frequency range [Hz]. Defaults to 40.0. Range [0,inf)
			numberBands (int): the number of bands in the filter. Defaults to 40. Range [1,inf)
			numberCoefficients (int): the number of output cepstrum coefficients. Defaults to 13. Range [1,inf)
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			silenceThreshold (float): silence threshold for computing log-energy bands. Defaults to 1e-10. Range (0,inf)
			type (str): use magnitude or power spectrum. Defaults to 'power'. Range {magnitude,power} 
		""" 
		... 
	def __call__(self, spectrum:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			spectrum (NDArray[np.float32]): the audio spectrum. Defaults to None. 
		Returns:
			bands (NDArray[np.float32]): the energies in ERB bands
			gfcc (NDArray[np.float32]): the gammatone feature cepstrum coefficients
		""" 
		... 


class GapsDetector(_essentia.Algorithm): 
	def __init__(self, attackTime:float=0.05, frameSize:int=2048, hopSize:int=1024, kernelSize:int=11, maximumTime:float=3500.0, minimumTime:float=10.0, postpowerTime:float=40.0, prepowerThreshold:float=-30.0, prepowerTime:float=40.0, releaseTime:float=0.05, sampleRate:float=44100.0, silenceThreshold:float=-50.0) -> None:
		"""Uses energy and time thresholds to detect gaps in the waveform.
		
		A median filter is used to remove spurious silent samples. The power of a small audio region before the detected gaps (prepower) is thresholded to detect intentional pauses as described in [1]. This technique is extended to the region after the gap.
		The algorithm was designed for a framewise use and returns the start and end timestamps related to the first frame processed. Call configure() or reset() in order to restart the count.
		
		References:
		  [1] Mühlbauer, R. (2010). Automatic Audio Defect Detection.
		

		Args:
			attackTime (float): the attack time of the first order lowpass in the attack phase [ms]. Defaults to 0.05. Range [0,inf)
			frameSize (int): frame size used for the analysis. Should match the input frame size. Otherwise, an exception will be thrown. Defaults to 2048. Range [0,inf)
			hopSize (int): hop size used for the analysis. Defaults to 1024. Range [0,inf)
			kernelSize (int): scalar giving the size of the median filter window. Must be odd. Defaults to 11. Range [1,inf)
			maximumTime (float): time of the maximum gap duration [ms]. Defaults to 3500.0. Range (0,inf)
			minimumTime (float): time of the minimum gap duration [ms]. Defaults to 10.0. Range (0,inf)
			postpowerTime (float): time for the postpower calculation [ms]. Defaults to 40.0. Range (0,inf)
			prepowerThreshold (float): prepower threshold [dB]. . Defaults to -30.0. Range (-inf,inf)
			prepowerTime (float): time for the prepower calculation [ms]. Defaults to 40.0. Range (0,inf)
			releaseTime (float): the release time of the first order lowpass in the release phase [ms]. Defaults to 0.05. Range [0,inf)
			sampleRate (float): sample rate used for the analysis. Defaults to 44100.0. Range (0,inf)
			silenceThreshold (float): silence threshold [dB]. Defaults to -50.0. Range (-inf,inf) 
		""" 
		... 
	def __call__(self, frame:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			frame (NDArray[np.float32]): the input frame (must be non-empty). Defaults to None. 
		Returns:
			starts (NDArray[np.float32]): the start indexes of the detected gaps (if any) in seconds
			ends (NDArray[np.float32]): the end indexes of the detected gaps (if any) in seconds
		""" 
		... 


class GeometricMean(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Computes the geometric mean of an array of positive values.
		
		An exception is thrown if the input array does not contain strictly positive numbers or the array is empty.
		
		References:
		  [1] Energy (signal processing) - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Energy_%28signal_processing%29
		
		  [2] Geometric Mean -- from Wolfram MathWorld,
		  http://mathworld.wolfram.com/GeometricMean.html		""" 
		... 
	def __call__(self, array:NDArray[np.float32]) -> float:
		"""compute
		Args:
			array (NDArray[np.float32]): the input array. Defaults to None. 
		Returns:
			geometricMean (float): the geometric mean of the input array
		""" 
		... 


class HFC(_essentia.Algorithm): 
	def __init__(self, sampleRate:float=44100.0, type:str='Masri') -> None:
		"""Computes the High Frequency Content of a spectrum.
		
		It can be computed according to the following techniques:
		  - 'Masri' (default) which does: sum |X(n)|^2*k,
		  - 'Jensen' which does: sum |X(n)|*k^2
		  - 'Brossier' which does: sum |X(n)|*k
		
		Exception is thrown for empty input spectra.
		
		References:
		  [1] P. Masri and A. Bateman, “Improved modelling of attack transients in
		  music analysis-resynthesis,” in Proceedings of the International
		  Computer Music Conference, 1996, pp. 100–103.
		
		  [2] K. Jensen and T. H. Andersen, “Beat estimation on the beat,” in
		  Applications of Signal Processing to Audio and Acoustics, 2003 IEEE
		  Workshop on., 2003, pp. 87–90.
		
		  [3] High frequency content measure - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/High_Frequency_Content_measure
		

		Args:
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf]
			type (str): the type of HFC coefficient to be computed. Defaults to 'Masri'. Range {Masri,Jensen,Brossier} 
		""" 
		... 
	def __call__(self, spectrum:NDArray[np.float32]) -> float:
		"""compute
		Args:
			spectrum (NDArray[np.float32]): the input audio spectrum. Defaults to None. 
		Returns:
			hfc (float): the high-frequency coefficient
		""" 
		... 


class HPCP(_essentia.Algorithm): 
	def __init__(self, bandPreset:bool=True, bandSplitFrequency:float=500.0, harmonics:int=0, maxFrequency:float=5000.0, maxShifted:bool=False, minFrequency:float=40.0, nonLinear:bool=False, normalized:str='unitMax', referenceFrequency:float=440.0, sampleRate:float=44100.0, size:int=12, weightType:str='squaredCosine', windowSize:float=1.0) -> None:
		"""Harmonic Pitch Class Profile (HPCP) from the spectral peaks of a signal.
		
		HPCP is a k*12 dimensional vector which represents the intensities of the twelve (k==1) semitone pitch classes (corresponsing to notes from A to G#), or subdivisions of these (k>1).
		
		Exceptions are thrown if "minFrequency", "bandSplitFrequency" and "maxFrequency" are not separated by at least 200Hz from each other, requiring that "maxFrequency" be greater than "bandSplitFrequency" and "bandSplitFrequency" be greater than "minFrequency". Other exceptions are thrown if input vectors have different size, if parameter "size" is not a positive non-zero multiple of 12 or if "windowSize" is less than one hpcp bin (12/size).
		
		References:
		  [1] T. Fujishima, "Realtime Chord Recognition of Musical Sound: A System
		  Using Common Lisp Music," in International Computer Music Conference
		  (ICMC'99), pp. 464-467, 1999.
		
		  [2] E. Gómez, "Tonal Description of Polyphonic Audio for Music Content
		  Processing," INFORMS Journal on Computing, vol. 18, no. 3, pp. 294–304,
		  2006.
		
		  [3] Harmonic pitch class profiles - Wikipedia, the free encyclopedia,
		  https://en.wikipedia.org/wiki/Harmonic_pitch_class_profiles

		Args:
			bandPreset (bool): enables whether to use a band preset. Defaults to True. Range {true,false}
			bandSplitFrequency (float): the split frequency for low and high bands, not used if bandPreset is false [Hz]. Defaults to 500.0. Range (0,inf)
			harmonics (int): number of harmonics for frequency contribution, 0 indicates exclusive fundamental frequency contribution. Defaults to 0. Range [0,inf)
			maxFrequency (float): the maximum frequency that contributes to the HPCP [Hz] (the difference between the max and split frequencies must not be less than 200.0 Hz). Defaults to 5000.0. Range (0,inf)
			maxShifted (bool): whether to shift the HPCP vector so that the maximum peak is at index 0. Defaults to False. Range {true,false}
			minFrequency (float): the minimum frequency that contributes to the HPCP [Hz] (the difference between the min and split frequencies must not be less than 200.0 Hz). Defaults to 40.0. Range (0,inf)
			nonLinear (bool): apply non-linear post-processing to the output (use with normalized='unitMax'). Boosts values close to 1, decreases values close to 0.. Defaults to False. Range {true,false}
			normalized (str): whether to normalize the HPCP vector. Defaults to 'unitMax'. Range {none,unitSum,unitMax}
			referenceFrequency (float): the reference frequency for semitone index calculation, corresponding to A3 [Hz]. Defaults to 440.0. Range (0,inf)
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			size (int): the size of the output HPCP (defines bin resolution, must be a positive nonzero multiple of 12). Defaults to 12. Range [12,inf)
			weightType (str): type of weighting function for determining frequency contribution. Defaults to 'squaredCosine'. Range {none,cosine,squaredCosine}
			windowSize (float): the size, in semitones, of the window used for the weighting. Defaults to 1.0. Range (0,12] 
		""" 
		... 
	def __call__(self, frequencies:NDArray[np.float32], magnitudes:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			frequencies (NDArray[np.float32]): the frequencies of the spectral peaks [Hz]. Defaults to None. 
			magnitudes (NDArray[np.float32]): the magnitudes of the spectral peaks. Defaults to None. 
		Returns:
			hpcp (NDArray[np.float32]): the resulting harmonic pitch class profile
		""" 
		... 


class HarmonicBpm(_essentia.Algorithm): 
	def __init__(self, bpm:float=60.0, threshold:float=20.0, tolerance:float=5.0) -> None:
		"""Extracts bpms that are harmonically related to the tempo given by the 'bpm' parameter.
		
		The algorithm assumes a certain bpm is harmonically related to parameter bpm, when the greatest common divisor between both bpms is greater than threshold.
		The 'tolerance' parameter is needed in order to consider if two bpms are related. For instance, 120, 122 and 236 may be related or not depending on how much tolerance is given
		
		References:
		  [1] Greatest common divisor - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Greatest_common_divisor

		Args:
			bpm (float): the bpm used to find its harmonics. Defaults to 60.0. Range [1,inf)
			threshold (float): bpm threshold below which greatest common divisors are discarded. Defaults to 20.0. Range [1,inf)
			tolerance (float): percentage tolerance to consider two bpms are equal or equal to a harmonic. Defaults to 5.0. Range [0,inf) 
		""" 
		... 
	def __call__(self, bpms:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			bpms (NDArray[np.float32]): list of bpm candidates. Defaults to None. 
		Returns:
			harmonicBpms (NDArray[np.float32]): a list of bpms which are harmonically related to the bpm parameter 
		""" 
		... 


class HarmonicMask(_essentia.Algorithm): 
	def __init__(self, attenuation:float=-200.0, binWidth:int=4, sampleRate:float=44100.0) -> None:
		"""Applies a spectral mask to remove a pitched source component from the signal.
		
		It computes first an harmonic mask corresponding to the input pitch and applies the mask to the input FFT to remove that pitch. The bin width determines how many spectral bins are masked per harmonic partial. 
		An attenuation value in dB determines the amount of suppression of the pitched component w.r.t the background for the case of muting. A negative attenuation value allows soloing the pitched component. 
		
		References:
		 

		Args:
			attenuation (float): attenuation in dB's of the muted pitched component. If value is positive the pitched component is attenuated (muted), if the value is negative the pitched component is soloed (i.e. background component is attenuated).. Defaults to -200.0. Range [-inf,inf)
			binWidth (int): number of bins per harmonic partials applied to the mask. This will depend on the internal FFT size. Defaults to 4. Range [0,inf)
			sampleRate (float): the audio sampling rate [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, fft:np.ndarray, pitch:float) -> np.ndarray:
		"""compute
		Args:
			fft (np.ndarray): the input frame. Defaults to None. 
			pitch (float): an estimate of the fundamental frequency of the signal [Hz]. Defaults to None. 
		Returns:
			fft (np.ndarray): the output frame
		""" 
		... 


class HarmonicModelAnal(_essentia.Algorithm): 
	def __init__(self, freqDevOffset:float=20.0, freqDevSlope:float=0.01, harmDevSlope:float=0.01, hopSize:int=512, magnitudeThreshold:float=-74.0, maxFrequency:float=5000.0, maxPeaks:int=100, maxnSines:int=100, minFrequency:float=20.0, nHarmonics:int=100, orderBy:str='frequency', sampleRate:float=44100.0) -> None:
		"""Computes the harmonic model analysis.
		
		This algorithm uses SineModelAnal and keeps only the harmonic partials. It receives an external pitch value as input. You can use PitchYinFft algorithm to compute the pitch per frame.
		
		References:
		  https://github.com/MTG/sms-tools
		  http://mtg.upf.edu/technologies/sms
		

		Args:
			freqDevOffset (float): minimum frequency deviation at 0Hz. Defaults to 20.0. Range (0,inf)
			freqDevSlope (float): slope increase of minimum frequency deviation. Defaults to 0.01. Range (-inf,inf)
			harmDevSlope (float): slope increase of minimum frequency deviation. Defaults to 0.01. Range (-inf,inf)
			hopSize (int): the hop size between frames. Defaults to 512. Range [1,inf)
			magnitudeThreshold (float): peaks below this given threshold are not outputted. Defaults to -74.0. Range (-inf,inf)
			maxFrequency (float): the maximum frequency of the F0 [Hz]. Defaults to 5000.0. Range (0,inf)
			maxPeaks (int): the maximum number of returned peaks. Defaults to 100. Range [1,inf)
			maxnSines (int): maximum number of sines per frame. Defaults to 100. Range (0,inf)
			minFrequency (float): the minimum frequency of the F0 [Hz]. Defaults to 20.0. Range (0,inf)
			nHarmonics (int): maximum number of harmonics per frame. Defaults to 100. Range (0,inf)
			orderBy (str): the ordering type of the outputted peaks (ascending by frequency or descending by magnitude). Defaults to 'frequency'. Range {frequency,magnitude}
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, fft:np.ndarray, pitch:float) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			fft (np.ndarray): the input fft. Defaults to None. 
			pitch (float): external pitch input [Hz].. Defaults to None. 
		Returns:
			frequencies (NDArray[np.float32]): the frequencies of the sinusoidal peaks [Hz]
			magnitudes (NDArray[np.float32]): the magnitudes of the sinusoidal peaks
			phases (NDArray[np.float32]): the phases of the sinusoidal peaks
		""" 
		... 


class HarmonicPeaks(_essentia.Algorithm): 
	def __init__(self, maxHarmonics:int=20, tolerance:float=0.2) -> None:
		"""Finds the harmonic peaks of a signal given its spectral peaks and its fundamental frequency.
		
		Note:
		  - "tolerance" parameter defines the allowed fixed deviation from ideal harmonics, being a percentage over the F0. For example: if the F0 is 100Hz you may decide to allow a deviation of 20%, that is a fixed deviation of 20Hz; for the harmonic series it is: [180-220], [280-320], [380-420], etc.
		  - If "pitch" is zero, it means its value is unknown, or the sound is unpitched, and in that case the HarmonicPeaks algorithm returns an empty vector.
		  - The output frequency and magnitude vectors are of size "maxHarmonics". If a particular harmonic was not found among spectral peaks, its ideal frequency value is output together with 0 magnitude.
		This algorithm is intended to receive its "frequencies" and "magnitudes" inputs from the SpectralPeaks algorithm.
		  - When input vectors differ in size or are empty, an exception is thrown. Input vectors must be ordered by ascending frequency excluding DC components and not contain duplicates, otherwise an exception is thrown.
		
		References:
		  [1] Harmonic Spectrum - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Harmonic_spectrum

		Args:
			maxHarmonics (int): the number of harmonics to return including F0. Defaults to 20. Range [1,inf)
			tolerance (float): the allowed ratio deviation from ideal harmonics. Defaults to 0.2. Range (0,0.5) 
		""" 
		... 
	def __call__(self, frequencies:NDArray[np.float32], magnitudes:NDArray[np.float32], pitch:float) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			frequencies (NDArray[np.float32]): the frequencies of the spectral peaks [Hz] (ascending order). Defaults to None. 
			magnitudes (NDArray[np.float32]): the magnitudes of the spectral peaks (ascending frequency order). Defaults to None. 
			pitch (float): an estimate of the fundamental frequency of the signal [Hz]. Defaults to None. 
		Returns:
			harmonicFrequencies (NDArray[np.float32]): the frequencies of harmonic peaks [Hz]
			harmonicMagnitudes (NDArray[np.float32]): the magnitudes of harmonic peaks
		""" 
		... 


class HighPass(_essentia.Algorithm): 
	def __init__(self, cutoffFrequency:float=1500.0, sampleRate:float=44100.0) -> None:
		"""Implements a 1st order IIR high-pass filter.
		
		Because of its dependence on IIR, IIR's requirements are inherited.
		
		References:
		  [1] U. Zölzer, DAFX - Digital Audio Effects, p. 40,
		  John Wiley & Sons, 2002

		Args:
			cutoffFrequency (float): the cutoff frequency for the filter [Hz]. Defaults to 1500.0. Range (0,inf)
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input audio signal. Defaults to None. 
		Returns:
			signal (NDArray[np.float32]): the filtered signal
		""" 
		... 


class HighResolutionFeatures(_essentia.Algorithm): 
	def __init__(self, maxPeaks:int=24) -> None:
		"""Computes high-resolution chroma features from an HPCP vector.
		
		The vector's size must be a multiple of 12 and it is recommended that it be larger than 120. In otherwords, the HPCP's resolution should be 10 Cents or more.
		The high-resolution features being computed are:
		
		  - Equal-temperament deviation: a measure of the deviation of HPCP local maxima with respect to equal-tempered bins. This is done by:
		    a) Computing local maxima of HPCP vector
		    b) Computing the deviations from equal-tempered (abs) bins and their average
		
		  - Non-tempered energy ratio: the ratio betwen the energy on non-tempered bins and the total energy, computed from the HPCP average
		
		  - Non-tempered peak energy ratio: the ratio betwen the energy on non tempered peaks and the total energy, computed from the HPCP average
		
		HighFrequencyFeatures is intended to be used in conjunction with HPCP algorithm. Any input vector which size is not a positive multiple of 12, will raise an exception.
		
		References:
		  [1] E. Gómez and P. Herrera, "Comparative Analysis of Music Recordings
		  from Western and Non-Western traditions by Automatic Tonal Feature
		  Extraction," Empirical Musicology Review, vol. 3, pp. 140–156, 2008.
		
		  [2] https://en.wikipedia.org/wiki/Equal_temperament

		Args:
			maxPeaks (int): maximum number of HPCP peaks to consider when calculating outputs. Defaults to 24. Range [1,inf) 
		""" 
		... 
	def __call__(self, hpcp:NDArray[np.float32]) -> tuple[float, float, float]:
		"""compute
		Args:
			hpcp (NDArray[np.float32]): the HPCPs, preferably of size >= 120. Defaults to None. 
		Returns:
			equalTemperedDeviation (float): measure of the deviation of HPCP local maxima with respect to equal-tempered bins
			nonTemperedEnergyRatio (float): ratio between the energy on non-tempered bins and the total energy
			nonTemperedPeaksEnergyRatio (float): ratio between the energy on non-tempered peaks and the total energy
		""" 
		... 


class Histogram(_essentia.Algorithm): 
	def __init__(self, maxValue:float=1.0, minValue:float=0.0, normalize:str='none', numberBins:int=10) -> None:
		"""Computes a histogram.
		
		Values outside the range are ignored

		Args:
			maxValue (float): the max value of the histogram. Defaults to 1.0. Range [0,Inf)
			minValue (float): the min value of the histogram. Defaults to 0.0. Range [0,Inf)
			normalize (str): the normalization setting.. Defaults to 'none'. Range {none,unit_sum,unit_max}
			numberBins (int): the number of bins. Defaults to 10. Range (0,Inf) 
		""" 
		... 
	def __call__(self, array:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			array (NDArray[np.float32]): the input array. Defaults to None. 
		Returns:
			histogram (NDArray[np.float32]): the values in the equally-spaced bins
			binEdges (NDArray[np.float32]): the edges of the equally-spaced bins. Size is _histogram.size() + 1
		""" 
		... 


class HprModelAnal(_essentia.Algorithm): 
	def __init__(self, fftSize:int=2048, freqDevOffset:int=20, freqDevSlope:float=0.01, harmDevSlope:float=0.01, hopSize:int=512, magnitudeThreshold:float=0.0, maxFrequency:float=5000.0, maxPeaks:int=100, maxnSines:int=100, minFrequency:float=20.0, nHarmonics:int=100, orderBy:str='frequency', sampleRate:float=44100.0, stocf:float=0.2) -> None:
		"""Computes the harmonic plus residual model analysis.
		
		It uses the algorithms HarmonicModelAnal and SineSubtraction .
		
		References:
		  https://github.com/MTG/sms-tools
		  http://mtg.upf.edu/technologies/sms
		

		Args:
			fftSize (int): the size of the internal FFT size (full spectrum size). Defaults to 2048. Range [1,inf)
			freqDevOffset (int): minimum frequency deviation at 0Hz. Defaults to 20. Range (0,inf)
			freqDevSlope (float): slope increase of minimum frequency deviation. Defaults to 0.01. Range (-inf,inf)
			harmDevSlope (float): slope increase of minimum frequency deviation. Defaults to 0.01. Range (-inf,inf)
			hopSize (int): the hop size between frames. Defaults to 512. Range [1,inf)
			magnitudeThreshold (float): peaks below this given threshold are not outputted. Defaults to 0.0. Range (-inf,inf)
			maxFrequency (float): the maximum frequency of the range to evaluate [Hz]. Defaults to 5000.0. Range (0,inf)
			maxPeaks (int): the maximum number of returned peaks. Defaults to 100. Range [1,inf)
			maxnSines (int): maximum number of sines per frame. Defaults to 100. Range (0,inf)
			minFrequency (float): the minimum frequency of the range to evaluate [Hz]. Defaults to 20.0. Range (0,inf)
			nHarmonics (int): maximum number of harmonics per frame. Defaults to 100. Range (0,inf)
			orderBy (str): the ordering type of the outputted peaks (ascending by frequency or descending by magnitude). Defaults to 'frequency'. Range {frequency,magnitude}
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			stocf (float): decimation factor used for the stochastic approximation. Defaults to 0.2. Range (0,1] 
		""" 
		... 
	def __call__(self, frame:NDArray[np.float32], pitch:float) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			frame (NDArray[np.float32]): the input frame. Defaults to None. 
			pitch (float): external pitch input [Hz].. Defaults to None. 
		Returns:
			frequencies (NDArray[np.float32]): the frequencies of the sinusoidal peaks [Hz]
			magnitudes (NDArray[np.float32]): the magnitudes of the sinusoidal peaks
			phases (NDArray[np.float32]): the phases of the sinusoidal peaks
			res (NDArray[np.float32]): output residual frame
		""" 
		... 


class HpsModelAnal(_essentia.Algorithm): 
	def __init__(self, fftSize:int=2048, freqDevOffset:int=20, freqDevSlope:float=0.01, harmDevSlope:float=0.01, hopSize:int=512, magnitudeThreshold:float=0.0, maxFrequency:float=5000.0, maxPeaks:int=100, maxnSines:int=100, minFrequency:float=20.0, nHarmonics:int=100, orderBy:str='frequency', sampleRate:float=44100.0, stocf:float=0.2) -> None:
		"""Computes the harmonic plus stochastic model analysis.
		
		It uses the algorithms HarmonicModelAnal and StochasticModelAnal .
		
		References:
		  https://github.com/MTG/sms-tools
		  http://mtg.upf.edu/technologies/sms
		

		Args:
			fftSize (int): the size of the internal FFT size (full spectrum size). Defaults to 2048. Range [1,inf)
			freqDevOffset (int): minimum frequency deviation at 0Hz. Defaults to 20. Range (0,inf)
			freqDevSlope (float): slope increase of minimum frequency deviation. Defaults to 0.01. Range (-inf,inf)
			harmDevSlope (float): slope increase of minimum frequency deviation. Defaults to 0.01. Range (-inf,inf)
			hopSize (int): the hop size between frames. Defaults to 512. Range [1,inf)
			magnitudeThreshold (float): peaks below this given threshold are not outputted. Defaults to 0.0. Range (-inf,inf)
			maxFrequency (float): the maximum frequency of the range to evaluate [Hz]. Defaults to 5000.0. Range (0,inf)
			maxPeaks (int): the maximum number of returned peaks. Defaults to 100. Range [1,inf)
			maxnSines (int): maximum number of sines per frame. Defaults to 100. Range (0,inf)
			minFrequency (float): the minimum frequency of the range to evaluate [Hz]. Defaults to 20.0. Range (0,inf)
			nHarmonics (int): maximum number of harmonics per frame. Defaults to 100. Range (0,inf)
			orderBy (str): the ordering type of the outputted peaks (ascending by frequency or descending by magnitude). Defaults to 'frequency'. Range {frequency,magnitude}
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			stocf (float): decimation factor used for the stochastic approximation. Defaults to 0.2. Range (0,1] 
		""" 
		... 
	def __call__(self, frame:NDArray[np.float32], pitch:float) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			frame (NDArray[np.float32]): the input frame. Defaults to None. 
			pitch (float): external pitch input [Hz].. Defaults to None. 
		Returns:
			frequencies (NDArray[np.float32]): the frequencies of the sinusoidal peaks [Hz]
			magnitudes (NDArray[np.float32]): the magnitudes of the sinusoidal peaks
			phases (NDArray[np.float32]): the phases of the sinusoidal peaks
			stocenv (NDArray[np.float32]): the stochastic envelope
		""" 
		... 


class HumDetector(_essentia.Algorithm): 
	def __init__(self, Q0:float=0.1, Q1:float=0.55, detectionThreshold:float=5.0, frameSize:float=0.4, hopSize:float=0.2, maximumFrequency:float=400.0, minimumDuration:float=2.0, minimumFrequency:float=22.5, numberHarmonics:int=1, sampleRate:float=44100.0, timeContinuity:float=10.0, timeWindow:float=10.0) -> None:
		"""Detects low frequency tonal noises in the audio signal.
		
		First, the steadiness of the Power Spectral Density (PSD) of the signal is computed by measuring the quantile ratios as described in [1]. After this, the PitchContours algorithm is used to keep track of the humming tones [2].
		
		References:
		  [1] Brandt, M., & Bitzer, J. (2014). Automatic Detection of Hum in Audio
		  Signals. Journal of the Audio Engineering Society, 62(9), 584-595.
		
		  [2] J. Salamon and E. Gómez, Melody extraction from polyphonic music signals
		  using pitch contour characteristics, IEEE Transactions on Audio, Speech,
		  and Language Processing, vol. 20, no. 6, pp. 1759–1770, 2012.
		

		Args:
			Q0 (float): low quantile. Defaults to 0.1. Range (0,1)
			Q1 (float): high quatile. Defaults to 0.55. Range (0,1)
			detectionThreshold (float): the detection threshold for the peaks of the r matrix. Defaults to 5.0. Range (0,inf)
			frameSize (float): the frame size with which the loudness is computed [s]. Defaults to 0.4. Range (0,inf)
			hopSize (float): the hop size with which the loudness is computed [s]. Defaults to 0.2. Range (0,inf)
			maximumFrequency (float): maximum frequency to consider [Hz]. Defaults to 400.0. Range (0,inf)
			minimumDuration (float): minimun duration of the humming tones [s]. Defaults to 2.0. Range (0,inf)
			minimumFrequency (float): minimum frequency to consider [Hz]. Defaults to 22.5. Range (0,inf)
			numberHarmonics (int): number of considered harmonics. Defaults to 1. Range (0,inf)
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			timeContinuity (float): time continuity cue (the maximum allowed gap duration for a pitch contour) [s]. Defaults to 10.0. Range (0,inf)
			timeWindow (float): analysis time to use for the hum estimation [s]. Defaults to 10.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> tuple[np.ndarray, NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input audio signal. Defaults to None. 
		Returns:
			r (np.ndarray): the quantile ratios matrix
			frequencies (NDArray[np.float32]): humming tones frequencies
			saliences (NDArray[np.float32]): humming tones saliences
			starts (NDArray[np.float32]): humming tones starts
			ends (NDArray[np.float32]): humming tones ends
		""" 
		... 


class IDCT(_essentia.Algorithm): 
	def __init__(self, dctType:int=2, inputSize:int=10, liftering:int=0, outputSize:int=10) -> None:
		"""Computes the Inverse Discrete Cosine Transform of an array.
		
		It can be configured to perform the inverse DCT-II form, with the 1/sqrt(2) scaling factor for the first coefficient or the inverse DCT-III form based on the HTK implementation.
		
		IDCT can be used to compute smoothed Mel Bands. In order to do this:
		  - compute MFCC
		  - smoothedMelBands = 10^(IDCT(MFCC)/20)
		Note: The second step assumes that 'logType' = 'dbamp' was used to compute MFCCs, otherwise that formula should be changed in order to be consistent.
		
		Note: The 'inputSize' parameter is only used as an optimization when the algorithm is configured. The IDCT will automatically adjust to the size of any input.
		
		References:
		  [1] Discrete cosine transform - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Discrete_cosine_transform 
		  [2] HTK book, chapter 5.6 ,
		  http://speech.ee.ntu.edu.tw/homework/DSP_HW2-1/htkbook.pdf

		Args:
			dctType (int): the DCT type. Defaults to 2. Range [2,3]
			inputSize (int): the size of the input array. Defaults to 10. Range [1,inf)
			liftering (int): the liftering coefficient. Use '0' to bypass it. Defaults to 0. Range [0,inf)
			outputSize (int): the number of output coefficients. Defaults to 10. Range [1,inf) 
		""" 
		... 
	def __call__(self, dct:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			dct (NDArray[np.float32]): the discrete cosine transform. Defaults to None. 
		Returns:
			idct (NDArray[np.float32]): the inverse cosine transform of the input array
		""" 
		... 


class IFFT(_essentia.Algorithm): 
	def __init__(self, normalize:bool=True, size:int=1024) -> None:
		"""Calculates the inverse short-term Fourier transform (STFT) of an array of complex values using the FFT algorithm.
		
		The resulting frame has a size of (s-1)*2, where s is the size of the input fft frame. The inverse Fourier transform is not defined for frames which size is less than 2 samples. Otherwise an exception is thrown.
		
		An exception is thrown if the input's size is not larger than 1.
		
		References:
		  [1] Fast Fourier transform - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Fft
		
		  [2] Fast Fourier Transform -- from Wolfram MathWorld,
		  http://mathworld.wolfram.com/FastFourierTransform.html

		Args:
			normalize (bool): whether to normalize the output by the FFT length.. Defaults to True. Range {true,false}
			size (int): the expected size of the input frame. This is purely optional and only targeted at optimizing the creation time of the FFT object. Defaults to 1024. Range [1,inf) 
		""" 
		... 
	def __call__(self, fft:np.ndarray) -> NDArray[np.float32]:
		"""compute
		Args:
			fft (np.ndarray): the input frame. Defaults to None. 
		Returns:
			frame (NDArray[np.float32]): the IFFT of the input frame
		""" 
		... 


class IFFTC(_essentia.Algorithm): 
	def __init__(self, normalize:bool=True, size:int=1024) -> None:
		"""Calculates the inverse short-term Fourier transform (STFT) of an array of complex values using the FFT algorithm.
		
		The resulting frame has a size equal to the input fft frame size. The inverse Fourier transform is not defined for frames which size is less than 2 samples. Otherwise an exception is thrown.
		
		An exception is thrown if the input's size is not larger than 1.
		
		References:
		  [1] Fast Fourier transform - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Fft
		
		  [2] Fast Fourier Transform -- from Wolfram MathWorld,
		  http://mathworld.wolfram.com/FastFourierTransform.html

		Args:
			normalize (bool): whether to normalize the output by the FFT length.. Defaults to True. Range {true,false}
			size (int): the expected size of the input frame. This is purely optional and only targeted at optimizing the creation time of the FFT object. Defaults to 1024. Range [1,inf) 
		""" 
		... 
	def __call__(self, fft:np.ndarray) -> np.ndarray:
		"""compute
		Args:
			fft (np.ndarray): the input frame. Defaults to None. 
		Returns:
			frame (np.ndarray): the complex IFFT of the input frame
		""" 
		... 


class IIR(_essentia.Algorithm): 
	def __init__(self, denominator:NDArray[np.float32]=np.array([1]), numerator:NDArray[np.float32]=np.array([1])) -> None:
		"""Implements a standard IIR filter.
		
		It filters the data in the input vector with the filter described by parameter vectors 'numerator' and 'denominator' to create the output filtered vector. In the literature, the numerator is often referred to as the 'B' coefficients and the denominator as the 'A' coefficients.
		
		The filter is a Direct Form II Transposed implementation of the standard difference equation:
		  a(0)*y(n) = b(0)*x(n) + b(1)*x(n-1) + ... + b(nb-1)*x(n-nb+1) - a(1)*y(n-1) - ... - a(nb-1)*y(n-na+1)
		
		This algorithm maintains a state which is the state of the delays. One should call the reset() method to reinitialize the state to all zeros.
		
		An exception is thrown if the "numerator" or "denominator" parameters are empty. An exception is also thrown if the first coefficient of the "denominator" parameter is 0.
		
		References:
		  [1] Smith, J.O.  Introduction to Digital Filters with Audio Applications,
		  http://ccrma-www.stanford.edu/~jos/filters/
		
		  [2] Infinite Impulse Response - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/IIR

		Args:
			denominator (NDArray[np.float32]): the list of coefficients of the denominator. Often referred to as the A coefficient vector.. Defaults to np.array([1]). Range None
			numerator (NDArray[np.float32]): the list of coefficients of the numerator. Often referred to as the B coefficient vector.. Defaults to np.array([1]). Range None 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input signal. Defaults to None. 
		Returns:
			signal (NDArray[np.float32]): the filtered signal
		""" 
		... 


class Inharmonicity(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Calculates the inharmonicity of a signal given its spectral peaks.
		
		The inharmonicity value is computed as an energy weighted divergence of the spectral components from their closest multiple of the fundamental frequency. The fundamental frequency is taken as the first spectral peak from the input. The inharmonicity value ranges from 0 (purely harmonic signal) to 1 (inharmonic signal).
		
		Inharmonicity was designed to be fed by the output from the HarmonicPeaks algorithm. Note that DC components should be removed from the signal before obtaining its peaks. An exception is thrown if a peak is given at 0Hz.
		
		An exception is thrown if frequency vector is not sorted in ascendently, if it contains duplicates or if any input vector is empty.
		
		References:
		  [1] G. Peeters, "A large set of audio features for sound description
		  (similarity and classification) in the CUIDADO project," CUIDADO I.S.T.
		  Project Report, 2004.
		
		  [2] Inharmonicity - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Inharmonicity		""" 
		... 
	def __call__(self, frequencies:NDArray[np.float32], magnitudes:NDArray[np.float32]) -> float:
		"""compute
		Args:
			frequencies (NDArray[np.float32]): the frequencies of the harmonic peaks [Hz] (in ascending order). Defaults to None. 
			magnitudes (NDArray[np.float32]): the magnitudes of the harmonic peaks (in frequency ascending order. Defaults to None. 
		Returns:
			inharmonicity (float): the inharmonicity of the audio signal
		""" 
		... 


class InstantPower(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Computes the instant power of an array.
		
		That is, the energy of the array over its size.
		
		An exception is thrown when input array is empty.
		
		References:
		  [1] Energy (signal processing) - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Energy_%28signal_processing%29		""" 
		... 
	def __call__(self, array:NDArray[np.float32]) -> float:
		"""compute
		Args:
			array (NDArray[np.float32]): the input array. Defaults to None. 
		Returns:
			power (float): the instant power of the input array
		""" 
		... 


class Intensity(_essentia.Algorithm): 
	def __init__(self, sampleRate:float=44100.0) -> None:
		"""Classifies the input audio signal as either relaxed (-1), moderate (0), or aggressive (1).
		
		Quality: outdated (non-reliable, poor accuracy).
		
		An exception is thrown if empty input is provided because the "intensity" is not defined for that case.

		Args:
			sampleRate (float): the input audio sampling rate [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> int:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input audio signal. Defaults to None. 
		Returns:
			intensity (int): the intensity value
		""" 
		... 


class Key(_essentia.Algorithm): 
	def __init__(self, profileType:str='bgate', numHarmonics:int=4, pcpSize:int=36, slope:float=0.6, useMajMin:bool=False, usePolyphony:bool=True, useThreeChords:bool=True) -> None:
		"""Computes key estimate given a pitch class profile (HPCP).
		
		The algorithm was severely adapted and changed from the original implementation for readability and speed.
		
		Key will throw exceptions either when the input pcp size is not a positive multiple of 12 or if the key could not be found. Also if parameter "scale" is set to "minor" and the profile type is set to "weichai"
		
		  Abouth the Key Profiles:
		  - 'Diatonic' - binary profile with diatonic notes of both modes. Could be useful for ambient music or diatonic music which is not strictly 'tonal functional'.
		  - 'Tonic Triad' - just the notes of the major and minor chords. Exclusively for testing.
		  - 'Krumhansl' - reference key profiles after cognitive experiments with users. They should work generally fine for pop music.
		  - 'Temperley' - key profiles extracted from corpus analysis of euroclassical music. Therefore, they perform best on this repertoire (especially in minor).
		  - 'Shaath' -  profiles based on Krumhansl's specifically tuned to popular and electronic music.
		  - 'Noland' - profiles from Bach's 'Well Tempered Klavier'.
		  - 'edma' - automatic profiles extracted from corpus analysis of electronic dance music [3]. They normally perform better that Shaath's
		  - 'edmm' - automatic profiles extracted from corpus analysis of electronic dance music and manually tweaked according to heuristic observation. It will report major modes (which are poorly represented in EDM) as minor, but improve performance otherwise [3].
		  - 'braw' - profiles obtained by calculating the median profile for each mode from a subset of BeatPort dataset. There is an extra profile obtained from ambiguous tracks that are reported as minor [4]
		  - 'bgate' - same as braw but zeroing the 4 less relevant elements of each profile [4]
		
		The standard mode of the algorithm estimates key/scale for a given HPCP vector.
		
		The streaming mode first accumulates a stream of HPCP vectors and computes its mean to provide the estimation. In this mode, the algorithm can apply a tuning correction, based on peaks in the bins of the accumulated HPCP distribution [3] (the `averageDetuningCorrection` parameter). This detuning approach requires a high resolution of the input HPCP vectors (`pcpSize` larger than 12).
		
		References:
		  [1] E. Gómez, "Tonal Description of Polyphonic Audio for Music Content
		  Processing," INFORMS Journal on Computing, vol. 18, no. 3, pp. 294–304,
		  2006.
		
		  [2] D. Temperley, "What's key for key? The Krumhansl-Schmuckler
		  key-finding algorithm reconsidered", Music Perception vol. 17, no. 1,
		  pp. 65-100, 1999.
		
		  [3] Á. Faraldo, E. Gómez, S. Jordà, P.Herrera, "Key Estimation in Electronic
		  Dance Music. Proceedings of the 38th International Conference on information
		  Retrieval, pp. 335-347, 2016.
		
		  [4] Faraldo, Á., Jordà, S., & Herrera, P. (2017, June). A multi-profile method
		  for key estimation in edm. In Audio Engineering Society Conference: 2017 AES
		  International Conference on Semantic Audio. Audio Engineering Society.

		Args:
			numHarmonics (int): number of harmonics that should contribute to the polyphonic profile (1 only considers the fundamental harmonic). Defaults to 4. Range [1,inf)
			pcpSize (int): number of array elements used to represent a semitone times 12 (this parameter is only a hint, during computation, the size of the input PCP is used instead). Defaults to 36. Range [12,inf)
			profileType (str): the type of polyphic profile to use for correlation calculation. Defaults to 'bgate'. Range {diatonic,krumhansl,temperley,weichai,tonictriad,temperley2005,thpcp,shaath,gomez,noland,edmm,edma,bgate,braw}
			slope (float): value of the slope of the exponential harmonic contribution to the polyphonic profile. Defaults to 0.6. Range [0,inf)
			useMajMin (bool): use a third profile called 'majmin' for ambiguous tracks [4]. Only avalable for the edma, bgate and braw profiles. Defaults to False. Range {true,false}
			usePolyphony (bool): enables the use of polyphonic profiles to define key profiles (this includes the contributions from triads as well as pitch harmonics). Defaults to True. Range {true,false}
			useThreeChords (bool): consider only the 3 main triad chords of the key (T, D, SD) to build the polyphonic profiles. Defaults to True. Range {true,false} 
		""" 
		... 
	def __call__(self, pcp:NDArray[np.float32]) -> tuple[str, str, float, float]:
		"""compute
		Args:
			pcp (NDArray[np.float32]): the input pitch class profile. Defaults to None. 
		Returns:
			key (str): the estimated key, from A to G
			scale (str): the scale of the key (major or minor)
			strength (float): the strength of the estimated key
			firstToSecondRelativeStrength (float): the relative strength difference between the best estimate and second best estimate of the key
		""" 
		... 


class KeyExtractor(_essentia.Algorithm): 
	def __init__(self, profileType:str='bgate', averageDetuningCorrection:bool=True, frameSize:int=4096, hopSize:int=4096, hpcpSize:int=12, maxFrequency:float=3500.0, maximumSpectralPeaks:int=60, minFrequency:float=25.0, pcpThreshold:float=0.2, sampleRate:float=44100.0, spectralPeaksThreshold:float=0.0001, tuningFrequency:float=440.0, weightType:str='cosine', windowType:str='hann') -> None:
		"""Extracts key/scale for an audio signal.
		
		It computes HPCP frames for the input signal and applies key estimation using the Key algorithm.
		
		The algorithm allows tuning correction using two complementary methods:
		  - Specify the expected `tuningFrequency` for the HPCP computation. The algorithm will adapt the semitone crossover frequencies for computing the HPCPs accordingly. If not specified, the default tuning is used. Tuning frequency can be estimated in advance using TuningFrequency algorithm.
		  - Apply tuning correction posterior to HPCP computation, based on peaks in the HPCP distribution (`averageDetuningCorrection`). This is possible when hpcpSize > 12.
		
		For more information, see the HPCP and Key algorithms.

		Args:
			averageDetuningCorrection (bool): shifts a pcp to the nearest tempered bin. Defaults to True. Range {true,false}
			frameSize (int): the framesize for computing tonal features. Defaults to 4096. Range (0,inf)
			hopSize (int): the hopsize for computing tonal features. Defaults to 4096. Range (0,inf)
			hpcpSize (int): the size of the output HPCP (must be a positive nonzero multiple of 12). Defaults to 12. Range [12,inf)
			maxFrequency (float): max frequency to apply whitening to [Hz]. Defaults to 3500.0. Range (0,inf)
			maximumSpectralPeaks (int): the maximum number of spectral peaks. Defaults to 60. Range (0,inf)
			minFrequency (float): min frequency to apply whitening to [Hz]. Defaults to 25.0. Range (0,inf)
			pcpThreshold (float): pcp bins below this value are set to 0. Defaults to 0.2. Range [0,1]
			profileType (str): the type of polyphic profile to use for correlation calculation. Defaults to 'bgate'. Range {diatonic,krumhansl,temperley,weichai,tonictriad,temperley2005,thpcp,shaath,gomez,noland,faraldo,pentatonic,edmm,edma,bgate,braw}
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			spectralPeaksThreshold (float): the threshold for the spectral peaks. Defaults to 0.0001. Range (0,inf)
			tuningFrequency (float): the tuning frequency of the input signal. Defaults to 440.0. Range (0,inf)
			weightType (str): type of weighting function for determining frequency contribution. Defaults to 'cosine'. Range {none,cosine,squaredCosine}
			windowType (str): the window type. Defaults to 'hann'. Range {hamming,hann,hannnsgcq,triangular,square,blackmanharris62,blackmanharris70,blackmanharris74,blackmanharris92} 
		""" 
		... 
	def __call__(self, audio:NDArray[np.float32]) -> tuple[str, str, float]:
		"""compute
		Args:
			audio (NDArray[np.float32]): the audio input signal. Defaults to None. 
		Returns:
			key (str): See Key algorithm documentation
			scale (str): See Key algorithm documentation
			strength (float): See Key algorithm documentation
		""" 
		... 


class LPC(_essentia.Algorithm): 
	def __init__(self, order:int=10, sampleRate:float=44100.0, type:str='regular') -> None:
		"""Computes Linear Predictive Coefficients and associated reflection coefficients of a signal.
		
		An exception is thrown if the "order" provided is larger than the size of the input signal.
		
		References:
		  [1] Linear predictive coding - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Linear_predictive_coding
		
		  [2] J. Makhoul, "Spectral analysis of speech by linear prediction," IEEE
		  Transactions on Audio and Electroacoustics, vol. 21, no. 3, pp. 140–148,
		  1973.
		

		Args:
			order (int): the order of the LPC analysis (typically [8,14]). Defaults to 10. Range [2,inf)
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			type (str): the type of LPC (regular or warped). Defaults to 'regular'. Range {regular,warped} 
		""" 
		... 
	def __call__(self, frame:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			frame (NDArray[np.float32]): the input audio frame. Defaults to None. 
		Returns:
			lpc (NDArray[np.float32]): the LPC coefficients
			reflection (NDArray[np.float32]): the reflection coefficients
		""" 
		... 


class Larm(_essentia.Algorithm): 
	def __init__(self, attackTime:float=10.0, power:float=1.5, releaseTime:float=1500.0, sampleRate:float=44100.0) -> None:
		"""Estimates the long-term loudness of an audio signal.
		
		The LARM model is based on the asymmetrical low-pass filtering of the Peak Program Meter (PPM), combined with Revised Low-frequency B-weighting (RLB) and power mean calculations. LARM has shown to be a reliable and objective loudness estimate of music and speech.
		
		It accepts a power parameter to define the exponential for computing the power mean. Note that if the parameter's value is 2, this algorithm would be equivalent to RMS and if 1, this algorithm would be the mean of the absolute value.
		
		References:
		 [1] E. Skovenborg and S. H. Nielsen, "Evaluation of different loudness
		 models with music and speech material,” in The 117th AES Convention, 2004.

		Args:
			attackTime (float): the attack time of the first order lowpass in the attack phase [ms]. Defaults to 10.0. Range [0,inf)
			power (float): the power used for averaging. Defaults to 1.5. Range (-inf,inf)
			releaseTime (float): the release time of the first order lowpass in the release phase [ms]. Defaults to 1500.0. Range [0,inf)
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> float:
		"""compute
		Args:
			signal (NDArray[np.float32]): the audio input signal. Defaults to None. 
		Returns:
			larm (float): the LARM loudness estimate [dB]
		""" 
		... 


class Leq(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Computes the Equivalent sound level (Leq) of an audio signal.
		
		The Leq measure can be derived from the Revised Low-frequency B-weighting (RLB) or from the raw signal as described in [1]. If the signal contains no energy, Leq defaults to essentias definition of silence which is -90dB.
		This algorithm will throw an exception on empty input.
		
		References:
		  [1] G. A. Soulodre, "Evaluation of Objective Loudness Meters," in
		  The 116th AES Convention, 2004.		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> float:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input signal (must be non-empty). Defaults to None. 
		Returns:
			leq (float): the equivalent sound level estimate [dB]
		""" 
		... 


class LevelExtractor(_essentia.Algorithm): 
	def __init__(self, frameSize:int=88200, hopSize:int=44100) -> None:
		"""Extracts the loudness of an audio signal in frames using Loudness algorithm.

		Args:
			frameSize (int): frame size to compute loudness. Defaults to 88200. Range (0,inf)
			hopSize (int): hop size to compute loudness. Defaults to 44100. Range (0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the audio input signal. Defaults to None. 
		Returns:
			loudness (NDArray[np.float32]): the loudness values
		""" 
		... 


class LogAttackTime(_essentia.Algorithm): 
	def __init__(self, sampleRate:float=44100.0, startAttackThreshold:float=0.2, stopAttackThreshold:float=0.9) -> None:
		"""Computes the log (base 10) of the attack time of a signal envelope.
		
		The attack time is defined as the time duration from when the sound becomes perceptually audible to when it reaches its maximum intensity. By default, the start of the attack is estimated as the point where the signal envelope reaches 20% of its maximum value in order to account for possible noise presence. Also by default, the end of the attack is estimated as as the point where the signal envelope has reached 90% of its maximum value, in order to account for the possibility that the max value occurres after the logAttack, as in trumpet sounds.
		
		With this said, LogAttackTime's input is intended to be fed by the output of the Envelope algorithm. In streaming mode, the RealAccumulator algorithm should be connected between Envelope and LogAttackTime.
		
		Note that startAttackThreshold cannot be greater than stopAttackThreshold and the input signal should not be empty. In any of these cases an exception will be thrown.
		

		Args:
			sampleRate (float): the audio sampling rate [Hz]. Defaults to 44100.0. Range (0,inf)
			startAttackThreshold (float): the percentage of the input signal envelope at which the starting point of the attack is considered. Defaults to 0.2. Range [0,1]
			stopAttackThreshold (float): the percentage of the input signal envelope at which the ending point of the attack is considered. Defaults to 0.9. Range [0,1] 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> tuple[float, float, float]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input signal envelope (must be non-empty). Defaults to None. 
		Returns:
			logAttackTime (float): the log (base 10) of the attack time [log10(s)]
			attackStart (float): the attack start time [s]
			attackStop (float): the attack end time [s]
		""" 
		... 


class LogSpectrum(_essentia.Algorithm): 
	def __init__(self, binsPerSemitone:float=3.0, frameSize:int=1025, nOctave:int=7, rollOn:float=0.0, sampleRate:float=44100.0) -> None:
		"""Computes spectrum with logarithmically distributed frequency bins.
		
		This code is ported from NNLS Chroma [1, 2].This algorithm also returns a local tuning that is retrieved for input frame and a global tuning that is updated with a moving average.
		
		Note: As the algorithm uses moving averages that are updated every frame it should be reset before  processing a new audio file. To do this call reset() (or configure())
		
		References:
		  [1] Mauch, M., & Dixon, S. (2010, August). Approximate Note Transcription
		  for the Improved Identification of Difficult Chords. In ISMIR (pp. 135-140).
		  [2] Chordino and NNLS Chroma,
		  http://www.isophonics.net/nnls-chroma

		Args:
			binsPerSemitone (float):  bins per semitone. Defaults to 3.0. Range (0,inf)
			frameSize (int): the input frame size of the spectrum vector. Defaults to 1025. Range (1,inf)
			nOctave (int): the number of octave of the output vector. Defaults to 7. Range (0,10)
			rollOn (float): this removes low-frequency noise - useful in quiet recordings. Defaults to 0.0. Range [0,5]
			sampleRate (float): the input sample rate. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, spectrum:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32], float]:
		"""compute
		Args:
			spectrum (NDArray[np.float32]): spectrum frame. Defaults to None. 
		Returns:
			logFreqSpectrum (NDArray[np.float32]): log frequency spectrum frame
			meanTuning (NDArray[np.float32]): normalized mean tuning frequency
			localTuning (float): normalized local tuning frequency
		""" 
		... 


class LoopBpmConfidence(_essentia.Algorithm): 
	def __init__(self, sampleRate:float=44100.0) -> None:
		"""Takes an audio signal and a BPM estimate for that signal and predicts the reliability of the BPM estimate in a value from 0 to 1.
		
		The audio signal is assumed to be a musical loop with constant tempo. The confidence returned is based on comparing the duration of the signal with multiples of the BPM estimate (see [1] for more details).
		
		References:
		  [1] Font, F., & Serra, X. (2016). Tempo Estimation for Music Loops and a Simple Confidence Measure.
		  Proceedings of the International Society for Music Information Retrieval Conference (ISMIR).
		
		

		Args:
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32], bpmEstimate:float) -> float:
		"""compute
		Args:
			signal (NDArray[np.float32]): loop audio signal. Defaults to None. 
			bpmEstimate (float): estimated BPM for the audio signal (will be rounded to nearest integer). Defaults to None. 
		Returns:
			confidence (float): confidence value for the BPM estimation
		""" 
		... 


class LoopBpmEstimator(_essentia.Algorithm): 
	def __init__(self, confidenceThreshold:float=0.95) -> None:
		"""Estimates the BPM of audio loops.
		
		It internally uses PercivalBpmEstimator algorithm to produce a BPM estimate and LoopBpmConfidence to asses the reliability of the estimate. If the provided estimate is below the given confidenceThreshold, the algorithm outputs a BPM 0.0, otherwise it outputs the estimated BPM. For more details on the BPM estimation method and the confidence measure please check the used algorithms.

		Args:
			confidenceThreshold (float): confidence threshold below which bpm estimate will be considered unreliable. Defaults to 0.95. Range [0,1] 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> float:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input signal. Defaults to None. 
		Returns:
			bpm (float): the estimated bpm (will be 0 if unsure)
		""" 
		... 


class Loudness(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Computes the loudness of an audio signal defined by Steven's power law.
		
		It computes loudness as the energy of the signal raised to the power of 0.67.
		
		References:
		  [1] Energy (signal processing) - Wikipedia, the free encyclopedia
		  http://en.wikipedia.org/wiki/Energy_%28signal_processing%29
		
		  [2] Stevens' power law - Wikipedia, the free encyclopedia
		  http://en.wikipedia.org/wiki/Stevens%27_power_law
		
		  [3] S. S. Stevens, Psychophysics. Transaction Publishers, 1975.		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> float:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input signal. Defaults to None. 
		Returns:
			loudness (float): the loudness of the input signal
		""" 
		... 


class LoudnessEBUR128(_essentia.Algorithm): 
	def __init__(self, hopSize:float=0.1, sampleRate:float=44100.0, startAtZero:bool=False) -> None:
		"""Computes the EBU R128 loudness descriptors of an audio signal.
		
		- The input stereo signal is preprocessed with a K-weighting filter [2] (see LoudnessEBUR128Filter algorithm), composed of two stages: a shelving filter and a high-pass filter (RLB-weighting curve).
		- Momentary loudness is computed by integrating the sum of powers over a sliding rectangular window of 400 ms. The measurement is not gated.
		- Short-term loudness is computed by integrating the sum of powers over a sliding rectangular window of 3 seconds. The measurement is not gated.
		- Integrated loudness is a loudness value averaged over an arbitrary long time interval with gating of 400 ms blocks with two thresholds [2].
		  - Absolute 'silence' gating threshold at -70 LUFS for the computation of the absolute-gated loudness level.
		  - Relative gating threshold, 10 LU below the absolute-gated loudness level.
		- Loudness range is computed from short-term loudness values. It is defined as the difference between the estimates of the 10th and 95th percentiles of the distribution of the loudness values with applied gating [3].
		  - Absolute 'silence' gating threshold at -70 LUFS for the computation of the absolute-gated loudness level.
		  - Relative gating threshold, -20 LU below the absolute-gated loudness level.
		
		References:
		  [1] EBU Tech 3341-2011. "Loudness Metering: 'EBU Mode' metering to supplement
		  loudness normalisation in accordance with EBU R 128"
		
		  [2] ITU-R BS.1770-2. "Algorithms to measure audio programme loudness and true-peak audio level"
		
		  [3] EBU Tech Doc 3342-2011. "Loudness Range: A measure to supplement loudness
		  normalisation in accordance with EBU R 128"
		
		  [4] https://tech.ebu.ch/loudness
		
		  [5] https://en.wikipedia.org/wiki/EBU_R_128
		
		  [6] https://en.wikipedia.org/wiki/LKFS
		

		Args:
			hopSize (float): the hop size with which the loudness is computed [s]. Defaults to 0.1. Range (0,0.1]
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			startAtZero (bool): start momentary/short-term loudness estimation at time 0 (zero-centered loudness estimation windows) if true; otherwise start both windows at time 0 (time positions for momentary and short-term values will not be syncronized). Defaults to False. Range {true,false} 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32], float, float]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input stereo audio signal. Defaults to None. 
		Returns:
			momentaryLoudness (NDArray[np.float32]): momentary loudness (over 400ms) (LUFS)
			shortTermLoudness (NDArray[np.float32]): short-term loudness (over 3 seconds) (LUFS)
			integratedLoudness (float): integrated loudness (overall) (LUFS)
			loudnessRange (float): loudness range over an arbitrary long time interval [3] (dB, LU)
		""" 
		... 


class LoudnessVickers(_essentia.Algorithm): 
	def __init__(self, sampleRate:float=44100.0) -> None:
		"""Computes Vickers's loudness of an audio signal.
		
		Currently, this algorithm only works for signals with a 44100Hz sampling rate. This algorithm is meant to be given frames of audio as input (not entire audio signals). The algorithm described in the paper performs a weighted average of the loudness value computed for each of the given frames, this step is left as a post processing step and is not performed by this algorithm.
		
		References:
		  [1] E. Vickers, "Automatic Long-term Loudness and Dynamics Matching," in
		  The 111th AES Convention, 2001.

		Args:
			sampleRate (float): the audio sampling rate of the input signal which is used to create the weight vector [Hz] (currently, this algorithm only works on signals with a sampling rate of 44100Hz). Defaults to 44100.0. Range [44100,44100] 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> float:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input signal. Defaults to None. 
		Returns:
			loudness (float): the Vickers loudness [dB]
		""" 
		... 


class LowLevelSpectralEqloudExtractor(_essentia.Algorithm): 
	def __init__(self, frameSize:int=2048, hopSize:int=1024, sampleRate:float=44100.0) -> None:
		"""Extracts a set of level spectral features for which it is recommended to apply a preliminary equal-loudness filter over an input audio signal (according to the internal evaluations conducted at Music Technology Group).
		
		To this end, you are expected to provide the output of EqualLoudness algorithm as an input for this algorithm. Still, you are free to provide an unprocessed audio input in the case you want to compute these features without equal-loudness filter.
		
		Note that at present we do not dispose any reference to justify the necessity of equal-loudness filter. Our recommendation is grounded on internal evaluations conducted at Music Technology Group that have shown the increase in numeric robustness as a function of the audio encoders used (mp3, ogg, ...) for these features.

		Args:
			frameSize (int): the frame size for computing low level features. Defaults to 2048. Range (0,inf)
			hopSize (int): the hop size for computing low level features. Defaults to 1024. Range (0,inf)
			sampleRate (float): the audio sampling rate. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> tuple[NDArray[np.float32], np.ndarray, np.ndarray, NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input audio signal. Defaults to None. 
		Returns:
			dissonance (NDArray[np.float32]): See Dissonance algorithm documentation
			sccoeffs (np.ndarray): See SpectralContrast algorithm documentation
			scvalleys (np.ndarray): See SpectralContrast algorithm documentation
			spectral_centroid (NDArray[np.float32]): See Centroid algorithm documentation
			spectral_kurtosis (NDArray[np.float32]): See DistributionShape algorithm documentation
			spectral_skewness (NDArray[np.float32]): See DistributionShape algorithm documentation
			spectral_spread (NDArray[np.float32]): See DistributionShape algorithm documentation
		""" 
		... 


class LowLevelSpectralExtractor(_essentia.Algorithm): 
	def __init__(self, frameSize:int=2048, hopSize:int=1024, sampleRate:float=44100.0) -> None:
		"""Extracts all low-level spectral features, which do not require an equal-loudness filter for their computation, from an audio signal

		Args:
			frameSize (int): the frame size for computing low level features. Defaults to 2048. Range (0,inf)
			hopSize (int): the hop size for computing low level features. Defaults to 1024. Range (0,inf)
			sampleRate (float): the audio sampling rate. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> tuple[np.ndarray, NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], np.ndarray, NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], np.ndarray, NDArray[np.float32]]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the audio input signal. Defaults to None. 
		Returns:
			barkbands (np.ndarray): spectral energy at each bark band. See BarkBands alogithm
			barkbands_kurtosis (NDArray[np.float32]): kurtosis from bark bands. See DistributionShape algorithm documentation
			barkbands_skewness (NDArray[np.float32]): skewness from bark bands. See DistributionShape algorithm documentation
			barkbands_spread (NDArray[np.float32]): spread from barkbands. See DistributionShape algorithm documentation
			hfc (NDArray[np.float32]): See HFC algorithm documentation
			mfcc (np.ndarray): See MFCC algorithm documentation
			pitch (NDArray[np.float32]): See PitchYinFFT algorithm documentation
			pitch_instantaneous_confidence (NDArray[np.float32]): See PitchYinFFT algorithm documentation
			pitch_salience (NDArray[np.float32]): See PitchSalience algorithm documentation
			silence_rate_20dB (NDArray[np.float32]): See SilenceRate algorithm documentation
			silence_rate_30dB (NDArray[np.float32]): See SilenceRate algorithm documentation
			silence_rate_60dB (NDArray[np.float32]): See SilenceRate algorithm documentation
			spectral_complexity (NDArray[np.float32]): See Spectral algorithm documentation
			spectral_crest (NDArray[np.float32]): See Crest algorithm documentation
			spectral_decrease (NDArray[np.float32]): See Decrease algorithm documentation
			spectral_energy (NDArray[np.float32]): See Energy algorithm documentation
			spectral_energyband_low (NDArray[np.float32]): Energy in band (20,150] Hz. See EnergyBand algorithm documentation
			spectral_energyband_middle_low (NDArray[np.float32]): Energy in band (150,800] Hz.See EnergyBand algorithm documentation
			spectral_energyband_middle_high (NDArray[np.float32]): Energy in band (800,4000] Hz. See EnergyBand algorithm documentation
			spectral_energyband_high (NDArray[np.float32]): Energy in band (4000,20000] Hz. See EnergyBand algorithm documentation
			spectral_flatness_db (NDArray[np.float32]): See flatnessDB algorithm documentation
			spectral_flux (NDArray[np.float32]): See Flux algorithm documentation
			spectral_rms (NDArray[np.float32]): See RMS algorithm documentation
			spectral_rolloff (NDArray[np.float32]): See RollOff algorithm documentation
			spectral_strongpeak (NDArray[np.float32]): See StrongPeak algorithm documentation
			zerocrossingrate (NDArray[np.float32]): See ZeroCrossingRate algorithm documentation
			inharmonicity (NDArray[np.float32]): See Inharmonicity algorithm documentation
			tristimulus (np.ndarray): See Tristimulus algorithm documentation
			oddtoevenharmonicenergyratio (NDArray[np.float32]): See OddToEvenHarmonicEnergyRatio algorithm documentation
		""" 
		... 


class LowPass(_essentia.Algorithm): 
	def __init__(self, cutoffFrequency:float=1500.0, sampleRate:float=44100.0) -> None:
		"""Implements a 1st order IIR low-pass filter.
		
		Because of its dependence on IIR, IIR's requirements are inherited.
		References:
		  [1] U. Zölzer, DAFX - Digital Audio Effects, p. 40,
		  John Wiley & Sons, 2002

		Args:
			cutoffFrequency (float): the cutoff frequency for the filter [Hz]. Defaults to 1500.0. Range (0,inf)
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input audio signal. Defaults to None. 
		Returns:
			signal (NDArray[np.float32]): the filtered signal
		""" 
		... 


class MFCC(_essentia.Algorithm): 
	def __init__(self, dctType:int=2, highFrequencyBound:float=11000.0, inputSize:int=1025, liftering:int=0, logType:str='dbamp', lowFrequencyBound:float=0.0, normalize:str='unit_sum', numberBands:int=40, numberCoefficients:int=13, sampleRate:float=44100.0, silenceThreshold:float=1e-10, type:str='power', warpingFormula:str='htkMel', weighting:str='warping') -> None:
		"""Computes the mel-frequency cepstrum coefficients of a spectrum.
		
		As there is no standard implementation, the MFCC-FB40 is used by default:
		  - filterbank of 40 bands from 0 to 11000Hz
		  - take the log value of the spectrum energy in each mel band. Bands energy values below silence threshold will be clipped to its value before computing log-energies
		  - DCT of the 40 bands down to 13 mel coefficients
		There is a paper describing various MFCC implementations [1].
		
		The parameters of this algorithm can be configured in order to behave like HTK [3] as follows:
		  - type = 'magnitude'
		  - warpingFormula = 'htkMel'
		  - weighting = 'linear'
		  - highFrequencyBound = 8000
		  - numberBands = 26
		  - numberCoefficients = 13
		  - normalize = 'unit_max'
		  - dctType = 3
		  - logType = 'log'
		  - liftering = 22
		
		In order to completely behave like HTK the audio signal has to be scaled by 2^15 before the processing and if the Windowing and FrameCutter algorithms are used they should also be configured as follows. 
		
		FrameGenerator:
		  - frameSize = 1102
		  - hopSize = 441
		  - startFromZero = True
		  - validFrameThresholdRatio = 1
		
		Windowing:
		  - type = 'hamming'
		  - size = 1102
		  - zeroPadding = 946
		  - normalized = False
		
		This algorithm depends on the algorithms MelBands and DCT and therefore inherits their parameter restrictions. An exception is thrown if any of these restrictions are not met. The input "spectrum" is passed to the MelBands algorithm and thus imposes MelBands' input requirements. Exceptions are inherited by MelBands as well as by DCT.
		
		IDCT can be used to compute smoothed Mel Bands. In order to do this:
		  - compute MFCC
		  - smoothedMelBands = 10^(IDCT(MFCC)/20)
		
		Note: The second step assumes that 'logType' = 'dbamp' was used to compute MFCCs, otherwise that formula should be changed in order to be consistent.
		
		References:
		  [1] T. Ganchev, N. Fakotakis, and G. Kokkinakis, "Comparative evaluation
		  of various MFCC implementations on the speaker verification task," in
		  International Conference on Speach and Computer (SPECOM’05), 2005,
		  vol. 1, pp. 191–194.
		
		  [2] Mel-frequency cepstrum - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Mel_frequency_cepstral_coefficient
		
		  [3] Young, S. J., Evermann, G., Gales, M. J. F., Hain, T., Kershaw, D.,
		  Liu, X., … Woodland, P. C. (2009). The HTK Book (for HTK Version 3.4).
		  Construction, (July 2000), 384, https://doi.org/http://htk.eng.cam.ac.uk
		
		  [4] Slaney, M. Auditory Toolbox: A MATLAB Toolbox for Auditory Modeling Work.
		  Technical Report, version 2, Interval Research Corporation, 1998.

		Args:
			dctType (int): the DCT type. Defaults to 2. Range [2,3]
			highFrequencyBound (float): the upper bound of the frequency range [Hz]. Defaults to 11000.0. Range (0,inf)
			inputSize (int): the size of input spectrum. Defaults to 1025. Range (1,inf)
			liftering (int): the liftering coefficient. Use '0' to bypass it. Defaults to 0. Range [0,inf)
			logType (str): logarithmic compression type. Use 'dbpow' if working with power and 'dbamp' if working with magnitudes. Defaults to 'dbamp'. Range {natural,dbpow,dbamp,log}
			lowFrequencyBound (float): the lower bound of the frequency range [Hz]. Defaults to 0.0. Range [0,inf)
			normalize (str): spectrum bin weights to use for each mel band: 'unit_max' to make each mel band vertex equal to 1, 'unit_sum' to make each mel band area equal to 1 summing the actual weights of spectrum bins, 'unit_area' to make each triangle mel band area equal to 1 normalizing the weights of each triangle by its bandwidth. Defaults to 'unit_sum'. Range {unit_sum,unit_tri,unit_max}
			numberBands (int): the number of mel-bands in the filter. Defaults to 40. Range [1,inf)
			numberCoefficients (int): the number of output mel coefficients. Defaults to 13. Range [1,inf)
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			silenceThreshold (float): silence threshold for computing log-energy bands. Defaults to 1e-10. Range (0,inf)
			type (str): use magnitude or power spectrum. Defaults to 'power'. Range {magnitude,power}
			warpingFormula (str): The scale implementation type: 'htkMel' scale from the HTK toolkit [2, 3] (default) or 'slaneyMel' scale from the Auditory toolbox [4]. Defaults to 'htkMel'. Range {slaneyMel,htkMel}
			weighting (str): type of weighting function for determining triangle area. Defaults to 'warping'. Range {warping,linear} 
		""" 
		... 
	def __call__(self, spectrum:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			spectrum (NDArray[np.float32]): the audio spectrum. Defaults to None. 
		Returns:
			bands (NDArray[np.float32]): the energies in mel bands
			mfcc (NDArray[np.float32]): the mel frequency cepstrum coefficients
		""" 
		... 


class Magnitude(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Computes the absolute value of each element in a vector of complex numbers.
		
		References:
		  [1] Complex Modulus -- from Wolfram MathWorld,
		  http://mathworld.wolfram.com/ComplexModulus.html
		
		  [2] Complex number - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Complex_numbers#Absolute_value.2C_conjugation_and_distance.		""" 
		... 
	def __call__(self, complex:np.ndarray) -> NDArray[np.float32]:
		"""compute
		Args:
			complex (np.ndarray): the input vector of complex numbers. Defaults to None. 
		Returns:
			magnitude (NDArray[np.float32]): the magnitudes of the input vector
		""" 
		... 


class MaxFilter(_essentia.Algorithm): 
	def __init__(self, causal:bool=True, width:int=3) -> None:
		"""Implements a maximum filter for 1d signal using van Herk/Gil-Werman (HGW) algorithm.
		
		References:
		  [1] Kutil, R., and Mraz, E., Short vector SIMD parallelization of maximum filter,
		  Parallel Numerics 11: 70

		Args:
			causal (bool): use casual filter (window is behind current element otherwise it is centered around). Defaults to True. Range {true,false}
			width (int): the window size, even size is auto-resized to the next odd value in the non-casual case. Defaults to 3. Range [2,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			signal (NDArray[np.float32]): signal to be filtered. Defaults to None. 
		Returns:
			signal (NDArray[np.float32]): filtered output
		""" 
		... 


class MaxMagFreq(_essentia.Algorithm): 
	def __init__(self, sampleRate:float=44100.0) -> None:
		"""Computes the frequency with the largest magnitude in a spectrum.
		
		Note that a spectrum must contain at least two elements otherwise an exception is thrown

		Args:
			sampleRate (float): the audio sampling rate [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, spectrum:NDArray[np.float32]) -> float:
		"""compute
		Args:
			spectrum (NDArray[np.float32]): the input spectrum (must have more than 1 element). Defaults to None. 
		Returns:
			maxMagFreq (float): the frequency with the largest magnitude [Hz]
		""" 
		... 


class MaxToTotal(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Computes the ratio between the index of the maximum value of the envelope of a signal and the total length of the envelope.
		
		This ratio shows how much the maximum amplitude is off-center. Its value is close to 0 if the maximum is close to the beginning (e.g. Decrescendo or Impulsive sounds), close to 0.5 if it is close to the middle (e.g. Delta sounds) and close to 1 if it is close to the end of the sound (e.g. Crescendo sounds). This algorithm is intended to be fed by the output of the Envelope algorithm
		
		MaxToTotal will throw an exception if the input envelope is empty.		""" 
		... 
	def __call__(self, envelope:NDArray[np.float32]) -> float:
		"""compute
		Args:
			envelope (NDArray[np.float32]): the envelope of the signal. Defaults to None. 
		Returns:
			maxToTotal (float): the maximum amplitude position to total length ratio
		""" 
		... 


class Mean(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Computes the mean of an array.		""" 
		... 
	def __call__(self, array:NDArray[np.float32]) -> float:
		"""compute
		Args:
			array (NDArray[np.float32]): the input array. Defaults to None. 
		Returns:
			mean (float): the mean of the input array
		""" 
		... 


class Median(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Computes the median of an array.
		
		When there is an odd number of numbers, the median is simply the middle number. For example, the median of 2, 4, and 7 is 4. When there is an even number of numbers, the median is the mean of the two middle numbers. Thus, the median of the numbers 2, 4, 7, 12 is (4+7)/2 = 5.5. See [1] for more info.
		
		References:
		  [1] Statistical Median -- from Wolfram MathWorld,
		  http://mathworld.wolfram.com/StatisticalMedian.html		""" 
		... 
	def __call__(self, array:NDArray[np.float32]) -> float:
		"""compute
		Args:
			array (NDArray[np.float32]): the input array (must be non-empty). Defaults to None. 
		Returns:
			median (float): the median of the input array
		""" 
		... 


class MedianFilter(_essentia.Algorithm): 
	def __init__(self, kernelSize:int=11) -> None:
		"""Computes the median filtered version of the input signal giving the kernel size as detailed in [1].
		
		References:
		  [1] Median Filter -- from Wikipedia.org, 
		  https://en.wikipedia.org/wiki/Median_filter

		Args:
			kernelSize (int): scalar giving the size of the median filter window. Must be odd. Defaults to 11. Range [1,inf) 
		""" 
		... 
	def __call__(self, array:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			array (NDArray[np.float32]): the input array (must be non-empty). Defaults to None. 
		Returns:
			filteredArray (NDArray[np.float32]): the median-filtered input array
		""" 
		... 


class MelBands(_essentia.Algorithm): 
	def __init__(self, highFrequencyBound:float=22050.0, inputSize:int=1025, log:bool=False, lowFrequencyBound:float=0.0, normalize:str='unit_sum', numberBands:int=24, sampleRate:float=44100.0, type:str='power', warpingFormula:str='htkMel', weighting:str='warping') -> None:
		"""Computes energy in mel bands of a spectrum.
		
		It applies a frequency-domain filterbank (MFCC FB-40, [1]), which consists of equal area triangular filters spaced according to the mel scale. The filterbank is normalized in such a way that the sum of coefficients for every filter equals one. It is recommended that the input "spectrum" be calculated by the Spectrum algorithm.
		
		It is required that parameter "highMelFrequencyBound" not be larger than the Nyquist frequency, but must be larger than the parameter, "lowMelFrequencyBound". Also, The input spectrum must contain at least two elements. If any of these requirements are violated, an exception is thrown.
		
		Note: an exception will be thrown in the case when the number of spectrum bins (FFT size) is insufficient to compute the specified number of mel bands: in such cases the start and end bin of a band can be the same bin or adjacent bins, which will result in zero energy when summing bins for that band. Use zero padding to increase the number of spectrum bins in these cases.
		
		References:
		  [1] T. Ganchev, N. Fakotakis, and G. Kokkinakis, "Comparative evaluation
		  of various MFCC implementations on the speaker verification task," in
		  International Conference on Speach and Computer (SPECOM’05), 2005,
		  vol. 1, pp. 191–194.
		
		  [2] Mel-frequency cepstrum - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Mel_frequency_cepstral_coefficient
		
		  [3] Young, S. J., Evermann, G., Gales, M. J. F., Hain, T., Kershaw, D.,
		  Liu, X., … Woodland, P. C. (2009). The HTK Book (for HTK Version 3.4).
		  Construction, (July 2000), 384, https://doi.org/http://htk.eng.cam.ac.uk
		
		  [4] Slaney, M. Auditory Toolbox: A MATLAB Toolbox for Auditory Modeling Work.
		  Technical Report, version 2, Interval Research Corporation, 1998.

		Args:
			highFrequencyBound (float): an upper-bound limit for the frequencies to be included in the bands. Defaults to 22050.0. Range [0,inf)
			inputSize (int): the size of the spectrum. Defaults to 1025. Range (1,inf)
			log (bool): compute log-energies (log2 (1 + energy)). Defaults to False. Range {true,false}
			lowFrequencyBound (float): a lower-bound limit for the frequencies to be included in the bands. Defaults to 0.0. Range [0,inf)
			normalize (str): spectrum bin weights to use for each mel band: 'unit_max' to make each mel band vertex equal to 1, 'unit_sum' to make each mel band area equal to 1 summing the actual weights of spectrum bins, 'unit_area' to make each triangle mel band area equal to 1 normalizing the weights of each triangle by its bandwidth. Defaults to 'unit_sum'. Range {unit_sum,unit_tri,unit_max}
			numberBands (int): the number of output bands. Defaults to 24. Range (1,inf)
			sampleRate (float): the sample rate. Defaults to 44100.0. Range (0,inf)
			type (str): 'power' to output squared units, 'magnitude' to keep it as the input. Defaults to 'power'. Range {magnitude,power}
			warpingFormula (str): The scale implementation type: 'htkMel' scale from the HTK toolkit [2, 3] (default) or 'slaneyMel' scale from the Auditory toolbox [4]. Defaults to 'htkMel'. Range {slaneyMel,htkMel}
			weighting (str): type of weighting function for determining triangle area. Defaults to 'warping'. Range {warping,linear} 
		""" 
		... 
	def __call__(self, spectrum:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			spectrum (NDArray[np.float32]): the audio spectrum. Defaults to None. 
		Returns:
			bands (NDArray[np.float32]): the energy in mel bands
		""" 
		... 


class MetadataReader(_essentia.Algorithm): 
	def __init__(self, filename:str, failOnError:bool=False, filterMetadata:bool=False, filterMetadataTags:list[str]=[], tagPoolName:str='metadata.tags') -> None:
		"""Loads the metadata tags from an audio file as well as outputs its audio properties.
		
		Supported audio file types are:
		  - mp3
		  - flac
		  - ogg
		An exception is thrown if unsupported filetype is given or if the file does not exist.
		Please observe that the .wav format is not supported. Also note that this algorithm incorrectly calculates the number of channels for a file in mp3 format only for versions less than 1.5 of taglib in Linux and less or equal to 1.5 in Mac OS X
		If using this algorithm on Windows, you must ensure that the filename is encoded as UTF-8.
		This algorithm also contains some heuristic to try to deal with encoding errors in the tags and tries to do the appropriate conversion if a problem was found (mostly twice latin1->utf8 conversion).
		
		MetadataReader reads all metadata tags found in audio and stores them in the pool tagPool. Standard metadata tags found in audio files include strings mentioned in [1,2]. Tag strings are case-sensitive and they are converted to lower-case when stored to the pool. It is possible to filter these tags by using 'filterMetadataTags' parameter. This parameter should specify a white-list of tag strings as they are found in the audio file (e.g., "ARTIST").
		
		References:
		  [1] https://taglib.github.io/api/classTagLib_1_1PropertyMap.html#details
		
		  [2] https://picard.musicbrainz.org/docs/mappings/

		Args:
			failOnError (bool): if true, the algorithm throws an exception when encountering an error (e.g. trying to open an unsupported file format), otherwise the algorithm leaves all fields blank. Defaults to False. Range {true,false}
			filename (str): the name of the file from which to read the tags. Defaults to None. Range None
			filterMetadata (bool): if true, only add tags from filterMetadataTags to the pool. Defaults to False. Range None
			filterMetadataTags (list[str]): the list of tags to whitelist (original taglib names). Defaults to []. Range None
			tagPoolName (str): common prefix for tag descriptor names to use in tagPool. Defaults to 'metadata.tags'. Range None 
		""" 
		... 
	def __call__(self, ) -> tuple[str, str, str, str, str, str, str, Pool, int, int, int, int]:
		"""compute
		Returns:
			title (str): the title of the track
			artist (str): the artist of the track
			album (str): the album on which this track appears
			comment (str): the comment field stored in the tags
			genre (str): the genre as stored in the tags
			tracknumber (str): the track number
			date (str): the date of publication
			tagPool (Pool): the pool with all tags that were found
			duration (int): the duration of the track, in seconds
			bitrate (int): the bitrate of the track [kb/s]
			sampleRate (int): the sample rate [Hz]
			channels (int): the number of channels
		""" 
		... 


class Meter(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Estimates the time signature of a given beatogram by finding the highest correlation between beats.
		
		Quality: experimental (not evaluated, do not use)		""" 
		... 
	def __call__(self, beatogram:np.ndarray) -> float:
		"""compute
		Args:
			beatogram (np.ndarray): filtered matrix loudness. Defaults to None. 
		Returns:
			meter (float): the time signature
		""" 
		... 


class MinMax(_essentia.Algorithm): 
	def __init__(self, type:str='min') -> None:
		"""Calculates the minimum or maximum value of an array.
		
		If the array has more than one minimum or maximum value, the index of the first one is returned

		Args:
			type (str): the type of the operation. Defaults to 'min'. Range {min,max} 
		""" 
		... 
	def __call__(self, array:NDArray[np.float32]) -> tuple[float, int]:
		"""compute
		Args:
			array (NDArray[np.float32]): the input array. Defaults to None. 
		Returns:
			real (float): the minimum or maximum of the input array, according to the type parameter
			int (int): the index of the value
		""" 
		... 


class MinToTotal(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Computes the ratio between the index of the minimum value of the envelope of a signal and the total length of the envelope.
		
		An exception is thrown if the input envelop is empty.		""" 
		... 
	def __call__(self, envelope:NDArray[np.float32]) -> float:
		"""compute
		Args:
			envelope (NDArray[np.float32]): the envelope of the signal. Defaults to None. 
		Returns:
			minToTotal (float): the minimum amplitude position to total length ratio
		""" 
		... 


class MonoLoader(_essentia.Algorithm): 
	def __init__(self, filename:str, audioStream:int=0, downmix:str='mix', resampleQuality:int=1, sampleRate:float=44100.0) -> None:
		"""Loads the raw audio data from an audio file and downmixes it to mono.
		
		Audio is resampled using Resample in case the given sampling rate does not match the sampling rate of the input signal.
		
		This algorithm uses AudioLoader and thus inherits all of its input requirements and exceptions.

		Args:
			audioStream (int): audio stream index to be loaded. Other streams are no taken into account (e.g. if stream 0 is video and 1 is audio use index 0 to access it.). Defaults to 0. Range [0,inf)
			downmix (str): the mixing type for stereo files. Defaults to 'mix'. Range {left,right,mix}
			filename (str): the name of the file from which to read. Defaults to None. Range None
			resampleQuality (int): the resampling quality, 0 for best quality, 4 for fast linear approximation. Defaults to 1. Range [0,4]
			sampleRate (float): the desired output sampling rate [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, ) -> NDArray[np.float32]:
		"""compute
		Returns:
			audio (NDArray[np.float32]): the audio signal
		""" 
		... 


class MonoMixer(_essentia.Algorithm): 
	def __init__(self, type:str='mix') -> None:
		"""Downmixes the signal into a single channel given a stereo signal.
		
		If the signal was already a monoaural, it is left unchanged.
		
		References:
		  [1] downmixing - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Downmixing
		

		Args:
			type (str): the type of downmixing performed. Defaults to 'mix'. Range {left,right,mix} 
		""" 
		... 
	def __call__(self, audio:NDArray[np.float32], numberChannels:int) -> NDArray[np.float32]:
		"""compute
		Args:
			audio (NDArray[np.float32]): the input stereo signal. Defaults to None. 
			numberChannels (int): the number of channels of the input signal. Defaults to None. 
		Returns:
			audio (NDArray[np.float32]): the downmixed signal
		""" 
		... 


class MonoWriter(_essentia.Algorithm): 
	def __init__(self, filename:str, bitrate:int=192, format:str='wav', sampleRate:float=44100.0) -> None:
		"""Writes a mono audio stream to a file.
		
		The algorithm uses FFmpeg. Supported formats are wav, aiff, mp3, flac and ogg. An exception is thrown when other extensions are given. The default FFmpeg encoders are used for each format. Note that to encode in mp3 format it is mandatory that FFmpeg was configured with mp3 enabled.
		
		If the file specified by filename could not be opened or the header of the file omits channel's information, an exception is thrown.

		Args:
			bitrate (int): the audio bit rate for compressed formats [kbps]. Defaults to 192. Range {32,40,48,56,64,80,96,112,128,144,160,192,224,256,320}
			filename (str): the name of the encoded file. Defaults to None. Range None
			format (str): the audio output format. Defaults to 'wav'. Range {wav,aiff,mp3,ogg,flac}
			sampleRate (float): the audio sampling rate [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, audio:NDArray[np.float32]) -> None:
		"""compute
		Args:
			audio (NDArray[np.float32]): the audio signal. Defaults to None. 
		""" 
		... 


class MovingAverage(_essentia.Algorithm): 
	def __init__(self, size:int=6) -> None:
		"""Implements a FIR Moving Average filter.
		
		Because of its dependece on IIR, IIR's requirements are inherited.
		
		References:
		  [1] Moving Average Filters, http://www.dspguide.com/ch15.htm

		Args:
			size (int): the size of the window [audio samples]. Defaults to 6. Range (1,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input audio signal. Defaults to None. 
		Returns:
			signal (NDArray[np.float32]): the filtered signal
		""" 
		... 


class MultiPitchKlapuri(_essentia.Algorithm): 
	def __init__(self, binResolution:float=10.0, frameSize:int=2048, harmonicWeight:float=0.8, hopSize:int=128, magnitudeCompression:float=1.0, magnitudeThreshold:int=40, maxFrequency:float=1760.0, minFrequency:float=80.0, numberHarmonics:int=10, referenceFrequency:float=55.0, sampleRate:float=44100.0) -> None:
		"""Estimates multiple pitch values corresponding to the melodic lines present in a polyphonic music signal (for example, string quartet, piano).
		
		This implementation is based on the algorithm in [1]: In each frame, a set of possible fundamental frequency candidates is extracted based on the principle of harmonic summation. In an optimization stage, the number of harmonic sources (polyphony) is estimated and the final set of fundamental frequencies determined. In contrast to the pich salience function proposed in [2], this implementation uses the pitch salience function described in [1].
		The output is a vector for each frame containing the estimated melody pitch values.
		
		References:
		  [1] A. Klapuri, "Multiple Fundamental Frequency Estimation by Summing Harmonic
		  Amplitudes ", International Society for Music Information Retrieval Conference
		  (2006).
		  [2] J. Salamon and E. Gómez, "Melody extraction from polyphonic music
		  signals using pitch contour characteristics," IEEE Transactions on Audio,
		  Speech, and Language Processing, vol. 20, no. 6, pp. 1759–1770, 2012.
		
		

		Args:
			binResolution (float): salience function bin resolution [cents]. Defaults to 10.0. Range (0,inf)
			frameSize (int): the frame size for computing pitch saliecnce. Defaults to 2048. Range (0,inf)
			harmonicWeight (float): harmonic weighting parameter (weight decay ratio between two consequent harmonics, =1 for no decay). Defaults to 0.8. Range (0,1)
			hopSize (int): the hop size with which the pitch salience function was computed. Defaults to 128. Range (0,inf)
			magnitudeCompression (float): magnitude compression parameter for the salience function (=0 for maximum compression, =1 for no compression). Defaults to 1.0. Range (0,1]
			magnitudeThreshold (int): spectral peak magnitude threshold (maximum allowed difference from the highest peak in dBs). Defaults to 40. Range [0,inf)
			maxFrequency (float): the maximum allowed frequency for salience function peaks (ignore peaks above) [Hz]. Defaults to 1760.0. Range [0,inf)
			minFrequency (float): the minimum allowed frequency for salience function peaks (ignore peaks below) [Hz]. Defaults to 80.0. Range [0,inf)
			numberHarmonics (int): number of considered harmonics. Defaults to 10. Range [1,inf)
			referenceFrequency (float): the reference frequency for Hertz to cent convertion [Hz], corresponding to the 0th cent bin. Defaults to 55.0. Range (0,inf)
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> np.ndarray:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input signal. Defaults to None. 
		Returns:
			pitch (np.ndarray): the estimated pitch values [Hz]
		""" 
		... 


class MultiPitchMelodia(_essentia.Algorithm): 
	def __init__(self, binResolution:float=10.0, filterIterations:int=3, frameSize:int=2048, guessUnvoiced:bool=False, harmonicWeight:float=0.8, hopSize:int=128, magnitudeCompression:float=1.0, magnitudeThreshold:int=40, maxFrequency:float=20000.0, minDuration:int=100, minFrequency:float=40.0, numberHarmonics:int=20, peakDistributionThreshold:float=0.9, peakFrameThreshold:float=0.9, pitchContinuity:float=27.5625, referenceFrequency:float=55.0, sampleRate:float=44100.0, timeContinuity:int=100) -> None:
		"""Estimates multiple fundamental frequency contours from an audio signal.
		
		It is a multi pitch version of the MELODIA algorithm described in [1]. While the algorithm is originally designed to extract melody in polyphonic music, this implementation is adapted for multiple sources. The approach is based on the creation and characterization of pitch contours, time continuous sequences of pitch candidates grouped using auditory streaming cues. To this end, PitchSalienceFunction, PitchSalienceFunctionPeaks, PitchContours, and PitchContoursMultiMelody algorithms are employed. It is strongly advised to use the default parameter values which are optimized according to [1] (where further details are provided) except for minFrequency, maxFrequency, and voicingTolerance, which will depend on your application.
		
		The output is a vector of vectors of estimated pitch values for each frame.
		
		References:
		  [1] J. Salamon and E. Gómez, "Melody extraction from polyphonic music
		  signals using pitch contour characteristics," IEEE Transactions on Audio,
		  Speech, and Language Processing, vol. 20, no. 6, pp. 1759–1770, 2012.
		
		  [2] http://mtg.upf.edu/technologies/melodia
		
		  [3] http://www.justinsalamon.com/melody-extraction
		

		Args:
			binResolution (float): salience function bin resolution [cents]. Defaults to 10.0. Range (0,inf)
			filterIterations (int): number of iterations for the octave errors / pitch outlier filtering process. Defaults to 3. Range [1,inf)
			frameSize (int): the frame size for computing pitch saliecnce. Defaults to 2048. Range (0,inf)
			guessUnvoiced (bool): estimate pitch for non-voiced segments by using non-salient contours when no salient ones are present in a frame. Defaults to False. Range {false,true}
			harmonicWeight (float): harmonic weighting parameter (weight decay ratio between two consequent harmonics, =1 for no decay). Defaults to 0.8. Range (0,1)
			hopSize (int): the hop size with which the pitch salience function was computed. Defaults to 128. Range (0,inf)
			magnitudeCompression (float): magnitude compression parameter for the salience function (=0 for maximum compression, =1 for no compression). Defaults to 1.0. Range (0,1]
			magnitudeThreshold (int): spectral peak magnitude threshold (maximum allowed difference from the highest peak in dBs). Defaults to 40. Range [0,inf)
			maxFrequency (float): the maximum allowed frequency for salience function peaks (ignore contours with peaks above) [Hz]. Defaults to 20000.0. Range [0,inf)
			minDuration (int): the minimum allowed contour duration [ms]. Defaults to 100. Range (0,inf)
			minFrequency (float): the minimum allowed frequency for salience function peaks (ignore contours with peaks below) [Hz]. Defaults to 40.0. Range [0,inf)
			numberHarmonics (int): number of considered harmonics. Defaults to 20. Range [1,inf)
			peakDistributionThreshold (float): allowed deviation below the peak salience mean over all frames (fraction of the standard deviation). Defaults to 0.9. Range [0,2]
			peakFrameThreshold (float): per-frame salience threshold factor (fraction of the highest peak salience in a frame). Defaults to 0.9. Range [0,1]
			pitchContinuity (float): pitch continuity cue (maximum allowed pitch change during 1 ms time period) [cents]. Defaults to 27.5625. Range [0,inf)
			referenceFrequency (float): the reference frequency for Hertz to cent convertion [Hz], corresponding to the 0th cent bin. Defaults to 55.0. Range (0,inf)
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			timeContinuity (int): time continuity cue (the maximum allowed gap duration for a pitch contour) [ms]. Defaults to 100. Range (0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> np.ndarray:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input signal. Defaults to None. 
		Returns:
			pitch (np.ndarray): the estimated pitch values [Hz]
		""" 
		... 


class Multiplexer(_essentia.Algorithm): 
	def __init__(self, numberRealInputs:int=0, numberVectorRealInputs:int=0) -> None:
		"""Returns a single vector from a given number of real values and/or frames.
		
		Frames from different inputs are multiplexed onto a single stream in an alternating fashion.
		
		This algorithm throws an exception if the number of input reals (or vector<real>) is less than the number specified in configuration parameters or if the user tries to acces an input which has not been specified.
		
		References:
		  [1] Multiplexer - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Multiplexer
		

		Args:
			numberRealInputs (int): the number of inputs of type Real to multiplex. Defaults to 0. Range [0,inf)
			numberVectorRealInputs (int): the number of inputs of type vector<Real> to multiplex. Defaults to 0. Range [0,inf) 
		""" 
		... 
	def __call__(self, ) -> np.ndarray:
		"""compute
		Returns:
			data (np.ndarray): the frame containing the input values and/or input frames
		""" 
		... 


class MusicExtractor(_essentia.Algorithm): 
	def __init__(self, profile:str, analysisSampleRate:float=44100.0, endTime:float=1000000.0, gfccStats:list[str]=['mean', 'cov', 'icov'], loudnessFrameSize:int=88200, loudnessHopSize:int=44100, lowlevelFrameSize:int=2048, lowlevelHopSize:int=1024, lowlevelSilentFrames:str='noise', lowlevelStats:list[str]=['mean', 'var', 'stdev', 'median', 'min', 'max', 'dmean', 'dmean2', 'dvar', 'dvar2'], lowlevelWindowType:str='blackmanharris62', lowlevelZeroPadding:int=0, mfccStats:list[str]=['mean', 'cov', 'icov'], requireMbid:bool=False, rhythmMaxTempo:int=208, rhythmMethod:str='degara', rhythmMinTempo:int=40, rhythmStats:list[str]=['mean', 'var', 'stdev', 'median', 'min', 'max', 'dmean', 'dmean2', 'dvar', 'dvar2'], startTime:float=0.0, tonalFrameSize:int=4096, tonalHopSize:int=2048, tonalSilentFrames:str='noise', tonalStats:list[str]=['mean', 'var', 'stdev', 'median', 'min', 'max', 'dmean', 'dmean2', 'dvar', 'dvar2'], tonalWindowType:str='blackmanharris62', tonalZeroPadding:int=0) -> None:
		"""Is a wrapper for Music Extractor.
		
		See documentation for 'essentia_streaming_extractor_music'.

		Args:
			analysisSampleRate (float): the analysis sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			endTime (float): the end time of the slice you want to extract [s]. Defaults to 1000000.0. Range [0,inf)
			gfccStats (list[str]): the statistics to compute for GFCC features. Defaults to ['mean', 'cov', 'icov']. Range None
			loudnessFrameSize (int): the frame size for computing average loudness. Defaults to 88200. Range (0,inf)
			loudnessHopSize (int): the hop size for computing average loudness. Defaults to 44100. Range (0,inf)
			lowlevelFrameSize (int): the frame size for computing low-level features. Defaults to 2048. Range (0,inf)
			lowlevelHopSize (int): the hop size for computing low-level features. Defaults to 1024. Range (0,inf)
			lowlevelSilentFrames (str): whether to [keep/drop/add noise to] silent frames for computing low-level features. Defaults to 'noise'. Range {drop,keep,noise}
			lowlevelStats (list[str]): the statistics to compute for low-level features. Defaults to ['mean', 'var', 'stdev', 'median', 'min', 'max', 'dmean', 'dmean2', 'dvar', 'dvar2']. Range None
			lowlevelWindowType (str): the window type for computing low-level features. Defaults to 'blackmanharris62'. Range {hamming,hann,triangular,square,blackmanharris62,blackmanharris70,blackmanharris74,blackmanharris92}
			lowlevelZeroPadding (int): zero padding factor for computing low-level features. Defaults to 0. Range [0,inf)
			mfccStats (list[str]): the statistics to compute for MFCC features. Defaults to ['mean', 'cov', 'icov']. Range None
			profile (str): profile filename. If specified, default parameter values are overwritten by values in the profile yaml file. If not specified (empty string), use values configured by user like in other normal algorithms. Defaults to None. Range None
			requireMbid (bool): ignore audio files without musicbrainz recording id tag (throw exception). Defaults to False. Range {true,false}
			rhythmMaxTempo (int): the fastest tempo to detect [bpm]. Defaults to 208. Range [60,250]
			rhythmMethod (str): the method used for beat tracking. Defaults to 'degara'. Range {multifeature,degara}
			rhythmMinTempo (int): the slowest tempo to detect [bpm]. Defaults to 40. Range [40,180]
			rhythmStats (list[str]): the statistics to compute for rhythm features. Defaults to ['mean', 'var', 'stdev', 'median', 'min', 'max', 'dmean', 'dmean2', 'dvar', 'dvar2']. Range None
			startTime (float): the start time of the slice you want to extract [s]. Defaults to 0.0. Range [0,inf)
			tonalFrameSize (int): the frame size for computing tonal features. Defaults to 4096. Range (0,inf)
			tonalHopSize (int): the hop size for computing tonal features. Defaults to 2048. Range (0,inf)
			tonalSilentFrames (str): whether to [keep/drop/add noise to] silent frames for computing tonal features. Defaults to 'noise'. Range {drop,keep,noise}
			tonalStats (list[str]): the statistics to compute for tonal features. Defaults to ['mean', 'var', 'stdev', 'median', 'min', 'max', 'dmean', 'dmean2', 'dvar', 'dvar2']. Range None
			tonalWindowType (str): the window type for computing tonal features. Defaults to 'blackmanharris62'. Range {hamming,hann,triangular,square,blackmanharris62,blackmanharris70,blackmanharris74,blackmanharris92}
			tonalZeroPadding (int): zero padding factor for computing tonal features. Defaults to 0. Range [0,inf) 
		""" 
		... 
	def __call__(self, filename:str) -> tuple[Pool, Pool]:
		"""compute
		Args:
			filename (str): the input audiofile. Defaults to None. 
		Returns:
			results (Pool): Analysis results pool with across-frames statistics
			resultsFrames (Pool): Analysis results pool with computed frame values
		""" 
		... 


class NNLSChroma(_essentia.Algorithm): 
	def __init__(self, chromaNormalization:str='none', frameSize:int=1025, sampleRate:float=44100.0, spectralShape:float=0.7, spectralWhitening:float=1.0, tuningMode:str='global', useNNLS:bool=True) -> None:
		"""Extracts treble and bass chromagrams from a sequence of log-frequency spectrum frames.
		
		On this representation, two processing steps are performed:
		  -tuning, after which each centre bin (i.e. bin 2, 5, 8, ...) corresponds to a semitone, even if the tuning of the piece deviates from 440 Hz standard pitch.
		  -running standardisation: subtraction of the running mean, division by the running standard deviation. This has a spectral whitening effect.
		This code is ported from NNLS Chroma [1, 2]. To achieve similar results follow this processing chain:
		frame slicing with sample rate = 44100, frame size = 16384, hop size = 2048 -> Windowing with Hann and no normalization -> Spectrum -> LogSpectrum.
		
		References:
		  [1] Mauch, M., & Dixon, S. (2010, August). Approximate Note Transcription
		  for the Improved Identification of Difficult Chords. In ISMIR (pp. 135-140).
		  [2] Chordino and NNLS Chroma,
		  http://www.isophonics.net/nnls-chroma

		Args:
			chromaNormalization (str): determines whether or how the chromagrams are normalised. Defaults to 'none'. Range {none,maximum,L1,L2}
			frameSize (int): the input frame size of the spectrum vector. Defaults to 1025. Range (1,inf)
			sampleRate (float): the input sample rate. Defaults to 44100.0. Range (0,inf)
			spectralShape (float):  the shape of the notes in the NNLS dictionary. Defaults to 0.7. Range (0.5,0.9)
			spectralWhitening (float): determines how much the log-frequency spectrum is whitened. Defaults to 1.0. Range [0,1.0]
			tuningMode (str): local uses a local average for tuning, global uses all audio frames. Local tuning is only advisable when the tuning is likely to change over the audio. Defaults to 'global'. Range {global,local}
			useNNLS (bool): toggle between NNLS approximate transcription and linear spectral mapping. Defaults to True. Range {true,false} 
		""" 
		... 
	def __call__(self, logSpectrogram:np.ndarray, meanTuning:NDArray[np.float32], localTuning:NDArray[np.float32]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		"""compute
		Args:
			logSpectrogram (np.ndarray): log spectrum frames. Defaults to None. 
			meanTuning (NDArray[np.float32]): mean tuning frames. Defaults to None. 
			localTuning (NDArray[np.float32]): local tuning frames. Defaults to None. 
		Returns:
			tunedLogfreqSpectrum (np.ndarray): Log frequency spectrum after tuning
			semitoneSpectrum (np.ndarray): a spectral representation with one bin per semitone
			bassChromagram (np.ndarray):  a 12-dimensional chromagram, restricted to the bass range
			chromagram (np.ndarray): a 12-dimensional chromagram, restricted with mid-range emphasis
		""" 
		... 


class NSGConstantQ(_essentia.Algorithm): 
	def __init__(self, binsPerOctave:int=48, gamma:int=0, inputSize:int=4096, maxFrequency:float=7040.0, minFrequency:float=27.5, minimumWindow:int=4, normalize:str='none', phaseMode:str='global', rasterize:str='full', sampleRate:float=44100.0, window:str='hannnsgcq', windowSizeFactor:int=1) -> None:
		"""Computes a constant Q transform using non stationary Gabor frames and returns a complex time-frequency representation of the input vector.
		
		The implementation is inspired by the toolbox described in [1].
		References:
		  [1] Schörkhuber, C., Klapuri, A., Holighaus, N., & Dörfler, M. (n.d.). A Matlab Toolbox for Efficient Perfect Reconstruction Time-Frequency Transforms with Log-Frequency Resolution.

		Args:
			binsPerOctave (int): the number of bins per octave. Defaults to 48. Range [1,inf)
			gamma (int): The bandwidth of each filter is given by Bk = 1/Q * fk + gamma. Defaults to 0. Range [0,inf)
			inputSize (int): the size of the input. Defaults to 4096. Range (0,inf)
			maxFrequency (float): the maximum frequency. Defaults to 7040.0. Range (0,inf)
			minFrequency (float): the minimum frequency. Defaults to 27.5. Range (0,inf)
			minimumWindow (int): minimum size allowed for the windows. Defaults to 4. Range [2,inf)
			normalize (str): coefficient normalization. Defaults to 'none'. Range {sine,impulse,none}
			phaseMode (str): 'local' to use zero-centered filters. 'global' to use a phase mapping function as described in [1]. Defaults to 'global'. Range {local,global}
			rasterize (str): hop sizes for each frequency channel. With 'none' each frequency channel is distinct. 'full' sets the hop sizes of all the channels to the smallest. 'piecewise' rounds down the hop size to a power of two. Defaults to 'full'. Range {none,full,piecewise}
			sampleRate (float): the desired sampling rate [Hz]. Defaults to 44100.0. Range [0,inf)
			window (str): the type of window for the frequency filters. It is not recommended to change the default window.. Defaults to 'hannnsgcq'. Range {hamming,hann,hannnsgcq,triangular,square,blackmanharris62,blackmanharris70,blackmanharris74,blackmanharris92}
			windowSizeFactor (int): window sizes are rounded to multiples of this. Defaults to 1. Range [1,inf) 
		""" 
		... 
	def __call__(self, frame:NDArray[np.float32]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""compute
		Args:
			frame (NDArray[np.float32]): the input frame (vector). Defaults to None. 
		Returns:
			constantq (np.ndarray): the constant Q transform of the input frame
			constantqdc (np.ndarray): the DC band transform of the input frame. Only needed for the inverse transform
			constantqnf (np.ndarray): the Nyquist band transform of the input frame. Only needed for the inverse transform
		""" 
		... 


class NSGIConstantQ(_essentia.Algorithm): 
	def __init__(self, binsPerOctave:int=48, gamma:int=0, inputSize:int=4096, maxFrequency:float=7040.0, minFrequency:float=27.5, minimumWindow:int=4, normalize:str='none', phaseMode:str='global', rasterize:str='full', sampleRate:float=44100.0, window:str='hannnsgcq', windowSizeFactor:int=1) -> None:
		"""Computes an inverse constant Q transform using non stationary Gabor frames and returns a complex time-frequency representation of the input vector.
		
		The implementation is inspired by the toolbox described in [1].
		References:
		  [1] Schörkhuber, C., Klapuri, A., Holighaus, N., & Dörfler, M. (n.d.). A Matlab Toolbox for Efficient Perfect Reconstruction Time-Frequency Transforms with Log-Frequency Resolution.

		Args:
			binsPerOctave (int): the number of bins per octave. Defaults to 48. Range [1,inf)
			gamma (int): The bandwidth of each filter is given by Bk = 1/Q * fk + gamma. Defaults to 0. Range [0,inf)
			inputSize (int): the size of the input. Defaults to 4096. Range (0,inf)
			maxFrequency (float): the maximum frequency. Defaults to 7040.0. Range (0,inf)
			minFrequency (float): the minimum frequency. Defaults to 27.5. Range (0,inf)
			minimumWindow (int): minimum size allowed for the windows. Defaults to 4. Range [2,inf)
			normalize (str): coefficient normalization. Defaults to 'none'. Range {sine,impulse,none}
			phaseMode (str): 'local' to use zero-centered filters. 'global' to use a phase mapping function as described in [1]. Defaults to 'global'. Range {local,global}
			rasterize (str): hop sizes for each frequency channel. With 'none' each frequency channel is distinct. 'full' sets the hop sizes of all the channels to the smallest. 'piecewise' rounds down the hop size to a power of two. Defaults to 'full'. Range {none,full,piecewise}
			sampleRate (float): the desired sampling rate [Hz]. Defaults to 44100.0. Range [0,inf)
			window (str): the type of window for the frequency filters. It is not recommended to change the default window.. Defaults to 'hannnsgcq'. Range {hamming,hann,hannnsgcq,triangular,square,blackmanharris62,blackmanharris70,blackmanharris74,blackmanharris92}
			windowSizeFactor (int): window sizes are rounded to multiples of this. Defaults to 1. Range [1,inf) 
		""" 
		... 
	def __call__(self, constantq:np.ndarray, constantqdc:np.ndarray, constantqnf:np.ndarray) -> NDArray[np.float32]:
		"""compute
		Args:
			constantq (np.ndarray): the constant Q transform of the input frame. Defaults to None. 
			constantqdc (np.ndarray): the DC band transform of the input frame. Defaults to None. 
			constantqnf (np.ndarray): the Nyquist band transform of the input frame. Defaults to None. 
		Returns:
			frame (NDArray[np.float32]): the input frame (vector)
		""" 
		... 


class NoiseAdder(_essentia.Algorithm): 
	def __init__(self, fixSeed:bool=False, level:int=-100) -> None:
		"""Adds noise to an input signal.
		
		The average energy of the noise in dB is defined by the level parameter, and is generated using the Mersenne Twister random number generator.
		
		References:
		  [1] Mersenne Twister: A random number generator (since 1997/10),
		  http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
		
		  [2] Mersenne twister - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Mersenne_twister

		Args:
			fixSeed (bool): if true, 0 is used as the seed for generating random values. Defaults to False. Range {true,false}
			level (int): power level of the noise generator [dB]. Defaults to -100. Range (-inf,0] 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input signal. Defaults to None. 
		Returns:
			signal (NDArray[np.float32]): the output signal with the added noise
		""" 
		... 


class NoiseBurstDetector(_essentia.Algorithm): 
	def __init__(self, alpha:float=0.9, silenceThreshold:int=-50, threshold:int=8) -> None:
		"""Detects noise bursts in the waveform by thresholding  the peaks of the second derivative.
		
		The threshold is computed using an Exponential Moving Average filter over the RMS of the second derivative of the input frame.

		Args:
			alpha (float): alpha coefficient for the Exponential Moving Average threshold estimation.. Defaults to 0.9. Range (0,1)
			silenceThreshold (int): threshold to skip silent frames. Defaults to -50. Range (-inf,0)
			threshold (int): factor to control the dynamic theshold. Defaults to 8. Range (-inf,inf) 
		""" 
		... 
	def __call__(self, frame:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			frame (NDArray[np.float32]): the input frame (must be non-empty). Defaults to None. 
		Returns:
			indexes (NDArray[np.float32]): indexes of the noisy samples
		""" 
		... 


class NoveltyCurve(_essentia.Algorithm): 
	def __init__(self, frameRate:float=344.531, normalize:bool=False, weightCurve:NDArray[np.float32]=np.array([]), weightCurveType:str='hybrid') -> None:
		"""Computes the "novelty curve" (Grosche & Müller, 2009) onset detection function.
		
		The algorithm expects as an input a frame-wise sequence of frequency-bands energies or spectrum magnitudes as originally proposed in [1] (see FrequencyBands and Spectrum algorithms). Novelty in each band (or frequency bin) is computed as a derivative between log-compressed energy (magnitude) values in consequent frames. The overall novelty value is then computed as a weighted sum that can be configured using 'weightCurve' parameter. The resulting novelty curve can be used for beat tracking and onset detection (see BpmHistogram and Onsets).
		
		Notes:
		
		- Recommended frame/hop size for spectrum computation is 2048/1024 samples (44.1 kHz sampling rate) [2].
		- Log compression is applied with C=1000 as in [1].
		- Frequency bands energies (see FrequencyBands) as well as bin magnitudes for the whole spectrum can be used as an input. The implementation for the original algorithm [2] works with spectrum bin magnitudes for which novelty functions are computed separately and are then summarized into bands.
		- In the case if 'weightCurve' is set to 'hybrid' a complex combination of flat, quadratic, linear and inverse quadratic weight curves is used. It was reported to improve performance of beat tracking in some informal in-house experiments (Note: this information is probably outdated).
		
		References:
		
		1. Grosche, P. & Müller, M. (2009). A mid-level representation for capturing dominant tempo and pulse information in music recordings. International Society for Music Information Retrieval Conference (ISMIR 2009).
		
		2. Tempogram Toolbox (Matlab implementation), http://resources.mpi%2Dinf.mpg.de/MIR/tempogramtoolbox
		
		

		Args:
			frameRate (float): the sampling rate of the input audio. Defaults to 344.531. Range [1,inf)
			normalize (bool): whether to normalize each band's energy. Defaults to False. Range {true,false}
			weightCurve (NDArray[np.float32]): vector containing the weights for each frequency band. Only if weightCurveType==supplied. Defaults to np.array([]). Range None
			weightCurveType (str): the type of weighting to be used for the bands novelty. Defaults to 'hybrid'. Range {flat,triangle,inverse_triangle,parabola,inverse_parabola,linear,quadratic,inverse_quadratic,hybrid,supplied} 
		""" 
		... 
	def __call__(self, frequencyBands:np.ndarray) -> NDArray[np.float32]:
		"""compute
		Args:
			frequencyBands (np.ndarray): the frequency bands. Defaults to None. 
		Returns:
			novelty (NDArray[np.float32]): the novelty curve as a single vector
		""" 
		... 


class NoveltyCurveFixedBpmEstimator(_essentia.Algorithm): 
	def __init__(self, hopSize:int=512, maxBpm:float=560.0, minBpm:float=30.0, sampleRate:float=44100.0, tolerance:float=3.0) -> None:
		"""Outputs a histogram of the most probable bpms assuming the signal has constant tempo given the novelty curve.
		
		This algorithm is based on the autocorrelation of the novelty curve (see NoveltyCurve algorithm) and should only be used for signals that have a constant tempo or as a first tempo estimator to be used in conjunction with other algorithms such as BpmHistogram.It is a simplified version of the algorithm described in [1] as, in order to predict the best BPM candidate,  it computes autocorrelation of the entire novelty curve instead of analyzing it on frames and histogramming the peaks over frames.
		
		References:
		  [1] E. Aylon and N. Wack, "Beat detection using plp," in Music Information
		  Retrieval Evaluation Exchange (MIREX’10), 2010.
		

		Args:
			hopSize (int): the hopSize used to computeh the novelty curve from the original signal. Defaults to 512. Range None
			maxBpm (float): the maximum bpm to look for. Defaults to 560.0. Range (0,inf)
			minBpm (float): the minimum bpm to look for. Defaults to 30.0. Range (0,inf)
			sampleRate (float): the sampling rate original audio signal [Hz]. Defaults to 44100.0. Range [1,inf)
			tolerance (float): tolerance (in percentage) for considering bpms to be equal. Defaults to 3.0. Range (0,100] 
		""" 
		... 
	def __call__(self, novelty:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			novelty (NDArray[np.float32]): the novelty curve of the audio signal. Defaults to None. 
		Returns:
			bpms (NDArray[np.float32]): the bpm candidates sorted by magnitude
			amplitudes (NDArray[np.float32]): the magnitude of each bpm candidate
		""" 
		... 


class OddToEvenHarmonicEnergyRatio(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Computes the ratio between a signal's odd and even harmonic energy given the signal's harmonic peaks.
		
		The odd to even harmonic energy ratio is a measure allowing to distinguish odd-harmonic-energy predominant sounds (such as from a clarinet) from equally important even-harmonic-energy sounds (such as from a trumpet). The required harmonic frequencies and magnitudes can be computed by the HarmonicPeaks algorithm.
		In the case when the even energy is zero, which may happen when only even harmonics where found or when only one peak was found, the algorithm outputs the maximum real number possible. Therefore, this algorithm should be used in conjunction with the harmonic peaks algorithm.
		If no peaks are supplied, the algorithm outputs a value of one, assuming either the spectrum was flat or it was silent.
		
		An exception is thrown if the input frequency and magnitude vectors have different size. Finally, an exception is thrown if the frequency and magnitude vectors are not ordered by ascending frequency.
		
		References:
		  [1] K. D. Martin and Y. E. Kim, "Musical instrument identification:
		  A pattern-recognition approach," The Journal of the Acoustical Society of
		  America, vol. 104, no. 3, pp. 1768–1768, 1998.
		
		  [2] K. Ringgenberg et al., "Musical Instrument Recognition,"
		  http://cnx.org/content/col10313/1.3/pdf		""" 
		... 
	def __call__(self, frequencies:NDArray[np.float32], magnitudes:NDArray[np.float32]) -> float:
		"""compute
		Args:
			frequencies (NDArray[np.float32]): the frequencies of the harmonic peaks (at least two frequencies in frequency ascending order). Defaults to None. 
			magnitudes (NDArray[np.float32]): the magnitudes of the harmonic peaks (at least two magnitudes in frequency ascending order). Defaults to None. 
		Returns:
			oddToEvenHarmonicEnergyRatio (float): the ratio between the odd and even harmonic energies of the given harmonic peaks
		""" 
		... 


class OnsetDetection(_essentia.Algorithm): 
	def __init__(self, method:str='hfc', sampleRate:float=44100.0) -> None:
		"""Computes various onset detection functions.
		
		The output of this algorithm should be post-processed in order to determine whether the frame contains an onset or not. Namely, it could be fed to the Onsets algorithm. It is recommended that the input "spectrum" is generated by the Spectrum algorithm.
		Four methods are available:
		  - 'HFC', the High Frequency Content detection function which accurately detects percussive events (see HFC algorithm for details).
		  - 'complex', the Complex-Domain spectral difference function [1] taking into account changes in magnitude and phase. It emphasizes note onsets either as a result of significant change in energy in the magnitude spectrum, and/or a deviation from the expected phase values in the phase spectrum, caused by a change in pitch.
		  - 'complex_phase', the simplified Complex-Domain spectral difference function [2] taking into account phase changes, weighted by magnitude. TODO:It reacts better on tonal sounds such as bowed string, but tends to over-detect percussive events.
		  - 'flux', the Spectral Flux detection function which characterizes changes in magnitude spectrum. See Flux algorithm for details.
		  - 'melflux', the spectral difference function, similar to spectral flux, but using half-rectified energy changes in Mel-frequency bands of the spectrum [3].
		  - 'rms', the difference function, measuring the half-rectified change of the RMS of the magnitude spectrum (i.e., measuring overall energy flux) [4].
		
		If using the 'HFC' detection function, make sure to adhere to HFC's input requirements when providing an input spectrum. Input vectors of different size or empty input spectra will raise exceptions.
		If using the 'complex' detection function, suggested parameters for computation of "spectrum" and "phase" are 44100Hz sample rate, frame size of 1024 and hopSize of 512 samples, which results in a resolution of 11.6ms, and a Hann window.
		
		References:
		  [1] Bello, Juan P., Chris Duxbury, Mike Davies, and Mark Sandler, On the
		  use of phase and energy for musical onset detection in the complex domain,
		  Signal Processing Letters, IEEE 11, no. 6 (2004): 553-556.
		
		  [2] P. Brossier, J. P. Bello, and M. D. Plumbley, "Fast labelling of notes
		  in music signals," in International Symposium on Music Information
		  Retrieval (ISMIR’04), 2004, pp. 331–336.
		
		  [3] D. P. W. Ellis, "Beat Tracking by Dynamic Programming," Journal of
		  New Music Research, vol. 36, no. 1, pp. 51–60, 2007.
		
		  [4] J. Laroche, "Efficient Tempo and Beat Tracking in Audio Recordings,"
		  JAES, vol. 51, no. 4, pp. 226–233, 2003.
		

		Args:
			method (str): the method used for onset detection. Defaults to 'hfc'. Range {hfc,complex,complex_phase,flux,melflux,rms}
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, spectrum:NDArray[np.float32], phase:NDArray[np.float32]) -> float:
		"""compute
		Args:
			spectrum (NDArray[np.float32]): the input spectrum. Defaults to None. 
			phase (NDArray[np.float32]): the phase vector corresponding to this spectrum (used only by the "complex" method). Defaults to None. 
		Returns:
			onsetDetection (float): the value of the detection function in the current frame
		""" 
		... 


class OnsetDetectionGlobal(_essentia.Algorithm): 
	def __init__(self, frameSize:int=2048, hopSize:int=512, method:str='infogain', sampleRate:float=44100.0) -> None:
		"""Computes various onset detection functions.
		
		Detection values are computed frame-wisely given an input signal. The output of this algorithm should be post-processed in order to determine whether the frame contains an onset or not. Namely, it could be fed to the Onsets algorithm.
		The following method are available:
		  - 'infogain', the spectral difference measured by the modified information gain [1]. For each frame, it accounts for energy change in between preceding and consecutive frames, histogrammed together, in order to suppress short-term variations on frame-by-frame basis.
		  - 'beat_emphasis', the beat emphasis function [1]. This function is a linear combination of onset detection functions (complex spectral differences) in a number of sub-bands, weighted by their beat strength computed over the entire input signal.
		Note:
		  - 'infogain' onset detection has been optimized for the default sampleRate=44100Hz, frameSize=2048, hopSize=512.
		  - 'beat_emphasis' is optimized for a fixed resolution of 11.6ms, which corresponds to the default sampleRate=44100Hz, frameSize=1024, hopSize=512.
		  Optimal performance of beat detection with TempoTapDegara is not guaranteed for other settings.
		
		References:
		  [1] S. Hainsworth and M. Macleod, "Onset detection in musical audio
		  signals," in International Computer Music Conference (ICMC’03), 2003,
		  pp. 163–6.
		
		  [2] M. E. P. Davies, M. D. Plumbley, and D. Eck, "Towards a musical beat
		  emphasis function," in IEEE Workshop on Applications of Signal Processing
		  to Audio and Acoustics, 2009. WASPAA  ’09, 2009, pp. 61–64.

		Args:
			frameSize (int): the frame size for computing onset detection function. Defaults to 2048. Range (0,inf)
			hopSize (int): the hop size for computing onset detection function. Defaults to 512. Range (0,inf)
			method (str): the method used for onset detection. Defaults to 'infogain'. Range {infogain,beat_emphasis}
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input signal. Defaults to None. 
		Returns:
			onsetDetections (NDArray[np.float32]): the frame-wise values of the detection function
		""" 
		... 


class OnsetRate(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Computes the number of onsets per second and their position in time for an audio signal.
		
		Onset detection functions are computed using both high frequency content and complex-domain methods available in OnsetDetection algorithm. See OnsetDetection for more information.
		Please note that due to a dependence on the Onsets algorithm, this algorithm is only valid for audio signals with a sampling rate of 44100Hz.
		This algorithm throws an exception if the input signal is empty.		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> tuple[NDArray[np.float32], float]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input signal. Defaults to None. 
		Returns:
			onsets (NDArray[np.float32]): the positions of detected onsets [s]
			onsetRate (float): the number of onsets per second
		""" 
		... 


class Onsets(_essentia.Algorithm): 
	def __init__(self, alpha:float=0.1, delay:int=5, frameRate:float=86.1328, silenceThreshold:float=0.02) -> None:
		"""Computes onset positions given various onset detection functions.
		
		The main operations are:
		  - normalizing detection functions,
		  - summing detection functions into a global detection function,
		  - smoothing the global detection function,
		  - thresholding the global detection function for silence,
		  - finding the possible onsets using an adaptative threshold,
		  - cleaning operations on the vector of possible onsets,
		  - onsets time conversion.
		
		Note:
		  - This algorithm has been optimized for a frameRate of 44100.0/512.0.
		  - At least one Detection function must be supplied at input.
		  - The number of weights must match the number of detection functions.
		
		As mentioned above, the "frameRate" parameter expects a value of 44100/512 (the default), but will work with other values, although the quality of the results is not guaranteed then. An exception is also thrown if the input "detections" matrix is empty. Finally, an exception is thrown if the size of the "weights" input does not equal the first dimension of the "detections" matrix.
		
		References:
		  [1] P. Brossier, J. P. Bello, and M. D. Plumbley, "Fast labelling of notes
		  in music signals,” in International Symposium on Music Information
		  Retrieval (ISMIR’04), 2004, pp. 331–336.

		Args:
			alpha (float): the proportion of the mean included to reject smaller peaks--filters very short onsets. Defaults to 0.1. Range [0,1]
			delay (int): the number of frames used to compute the threshold--size of short-onset filter. Defaults to 5. Range (0,inf)
			frameRate (float): frames per second. Defaults to 86.1328. Range (0,inf)
			silenceThreshold (float): the threshold for silence. Defaults to 0.02. Range [0,1] 
		""" 
		... 
	def __call__(self, detections:np.ndarray, weights:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			detections (np.ndarray): matrix containing onset detection functions--rows represent the values of different detection functions and columns represent different frames of audio (i.e. detections[i][j] represents the value of the ith detection function for the jth frame of audio). Defaults to None. 
			weights (NDArray[np.float32]): the weighting coefficicients for each detection function, must be the same as the first dimension of "detections". Defaults to None. 
		Returns:
			onsets (NDArray[np.float32]): the onset positions [s]
		""" 
		... 


class OverlapAdd(_essentia.Algorithm): 
	def __init__(self, frameSize:int=2048, gain:float=1.0, hopSize:int=128) -> None:
		"""Returns the output of an overlap-add process for a sequence of frames of an audio signal.
		
		It considers that the input audio frames are windowed audio signals. Giving the size of the frame and the hop size, overlapping and adding consecutive frames will produce a continuous signal. A normalization gain can be passed as a parameter.
		
		Empty input signals will raise an exception.
		
		References:
		  [1] Overlap–add method - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Overlap-add_method

		Args:
			frameSize (int): the frame size for computing the overlap-add process. Defaults to 2048. Range (0,inf)
			gain (float): the normalization gain that scales the output signal. Useful for IFFT output. Defaults to 1.0. Range (0.,inf)
			hopSize (int): the hop size with which the overlap-add function is computed. Defaults to 128. Range (0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the windowed input audio frame. Defaults to None. 
		Returns:
			signal (NDArray[np.float32]): the output overlap-add audio signal frame
		""" 
		... 


class PCA(_essentia.Algorithm): 
	def __init__(self, dimensions:int=0, namespaceIn:str='spectral contrast', namespaceOut:str='spectral contrast pca') -> None:
		"""Applies Principal Component Analysis based on the covariance matrix of the signal.
		
		References:
		  [1] Principal component analysis - Wikipedia, the free enciclopedia
		  http://en.wikipedia.org/wiki/Principal_component_analysis

		Args:
			dimensions (int): number of dimension to reduce the input to. Defaults to 0. Range [0,inf)
			namespaceIn (str): will look for this namespace in poolIn. Defaults to 'spectral contrast'. Range None
			namespaceOut (str): will save to this namespace in poolOut. Defaults to 'spectral contrast pca'. Range None 
		""" 
		... 
	def __call__(self, poolIn:Pool) -> Pool:
		"""compute
		Args:
			poolIn (Pool): the pool where to get the spectral contrast feature vectors. Defaults to None. 
		Returns:
			poolOut (Pool): the pool where to store the transformed feature vectors
		""" 
		... 


class Panning(_essentia.Algorithm): 
	def __init__(self, averageFrames:int=43, numBands:int=1, numCoeffs:int=20, panningBins:int=512, sampleRate:float=44100.0, warpedPanorama:bool=True) -> None:
		"""Characterizes panorama distribution by comparing spectra from the left and right channels.
		
		The panning coefficients are extracted by:
		
		- determining the spatial location of frequency bins given left and right channel spectra;
		
		- computing panorama histogram weighted by the energy of frequency bins, averaging it across frames and normalizing;
		
		- converting the normalized histogram into panning coefficients (IFFT of the log-histogram).
		
		The resulting coefficients will show peaks on the initial bins for left panned audio, and right panning will appear as peaks in the upper bins.
		
		Since panning can vary very rapidly from one frame to the next, the coefficients can be averaged over a time window of several frames by specifying "averageFrames" parameter. If a single vector of panning coefficients for the whole audio input is required, "averageFrames" should correspond to the length of audio input. In standard mode, sequential runs of compute() method on each frame are required for averaging across frames.
		
		Application: music classification, in particular genre classification [2].
		
		Note: At present time, the original algorithm has not been tested in multi-band mode. That is, numBands must remain 1.
		References:
		  [1] E. Gómez, P. Herrera, P. Cano, J. Janer, J. Serrà, J. Bonada,
		  S. El-Hajj, T. Aussenac, and G. Holmberg, "Music similarity systems and
		  methods using descriptors,” U.S. Patent WO 2009/0012022009.
		
		  [2] Guaus, E. (2009). Audio content processing for automatic music genre
		  classification: descriptors, databases, and classifiers. PhD Thesis.

		Args:
			averageFrames (int): number of frames to take into account for averaging. Defaults to 43. Range [0,inf)
			numBands (int): number of mel bands. Defaults to 1. Range [1,inf)
			numCoeffs (int): number of coefficients used to define the panning curve at each frame. Defaults to 20. Range (0,inf)
			panningBins (int): size of panorama histogram (in bins). Defaults to 512. Range (1,inf)
			sampleRate (float): audio sampling rate [Hz]. Defaults to 44100.0. Range (0,inf)
			warpedPanorama (bool): if true, warped panorama is applied, having more resolution in the center area. Defaults to True. Range {false,true} 
		""" 
		... 
	def __call__(self, spectrumLeft:NDArray[np.float32], spectrumRight:NDArray[np.float32]) -> np.ndarray:
		"""compute
		Args:
			spectrumLeft (NDArray[np.float32]): left channel's spectrum. Defaults to None. 
			spectrumRight (NDArray[np.float32]): right channel's spectrum. Defaults to None. 
		Returns:
			panningCoeffs (np.ndarray): parameters that define the panning curve at each frame
		""" 
		... 


class PeakDetection(_essentia.Algorithm): 
	def __init__(self, interpolate:bool=True, maxPeaks:int=100, maxPosition:float=1.0, minPeakDistance:float=0.0, minPosition:float=0.0, orderBy:str='position', range:float=1.0, threshold:float=-1000000.0) -> None:
		"""Detects local maxima (peaks) in an array.
		
		The algorithm finds positive slopes and detects a peak when the slope changes sign and the peak is above the threshold.
		It optionally interpolates using parabolic curve fitting.
		When two consecutive peaks are closer than the `minPeakDistance` parameter, the smallest one is discarded. A value of 0 bypasses this feature.
		
		Exceptions are thrown if parameter "minPosition" is greater than parameter "maxPosition", also if the size of the input array is less than 2 elements.
		
		References:
		  [1] Peak Detection,
		  http://ccrma.stanford.edu/~jos/parshl/Peak_Detection_Steps_3.html

		Args:
			interpolate (bool): boolean flag to enable interpolation. Defaults to True. Range {true,false}
			maxPeaks (int): the maximum number of returned peaks. Defaults to 100. Range [1,inf)
			maxPosition (float): the maximum value of the range to evaluate. Defaults to 1.0. Range (0,inf)
			minPeakDistance (float): minimum distance between consecutive peaks (0 to bypass this feature). Defaults to 0.0. Range [0,inf)
			minPosition (float): the minimum value of the range to evaluate. Defaults to 0.0. Range [0,inf)
			orderBy (str): the ordering type of the output peaks (ascending by position or descending by value). Defaults to 'position'. Range {position,amplitude}
			range (float): the input range. Defaults to 1.0. Range (0,inf)
			threshold (float): peaks below this given threshold are not output. Defaults to -1000000.0. Range (-inf,inf) 
		""" 
		... 
	def __call__(self, array:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			array (NDArray[np.float32]): the input array. Defaults to None. 
		Returns:
			positions (NDArray[np.float32]): the positions of the peaks
			amplitudes (NDArray[np.float32]): the amplitudes of the peaks
		""" 
		... 


class PercivalBpmEstimator(_essentia.Algorithm): 
	def __init__(self, frameSize:int=1024, frameSizeOSS:int=2048, hopSize:int=128, hopSizeOSS:int=128, maxBPM:int=210, minBPM:int=50, sampleRate:int=44100) -> None:
		"""Estimates the tempo in beats per minute (BPM) from an input signal as described in [1].
		
		References:
		  [1] Percival, G., & Tzanetakis, G. (2014). Streamlined tempo estimation based on autocorrelation and cross-correlation with pulses.
		  IEEE/ACM Transactions on Audio, Speech, and Language Processing, 22(12), 1765–1776.
		
		

		Args:
			frameSize (int): frame size for the analysis of the input signal. Defaults to 1024. Range (0,inf)
			frameSizeOSS (int): frame size for the analysis of the Onset Strength Signal. Defaults to 2048. Range (0,inf)
			hopSize (int): hop size for the analysis of the input signal. Defaults to 128. Range (0,inf)
			hopSizeOSS (int): hop size for the analysis of the Onset Strength Signal. Defaults to 128. Range (0,inf)
			maxBPM (int): maximum BPM to detect. Defaults to 210. Range (0,inf)
			minBPM (int): minimum BPM to detect. Defaults to 50. Range (0,inf)
			sampleRate (int): the sampling rate of the audio signal [Hz]. Defaults to 44100. Range (0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> float:
		"""compute
		Args:
			signal (NDArray[np.float32]): input signal. Defaults to None. 
		Returns:
			bpm (float): the tempo estimation [bpm]
		""" 
		... 


class PercivalEnhanceHarmonics(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Implements the 'Enhance Harmonics' step as described in [1].Given an input autocorrelation signal, two time-stretched versions of it scaled by factors of 2 and 4 are added to the original.For more details check the referenced paper.
		
		References:
		  [1] Percival, G., & Tzanetakis, G. (2014). Streamlined tempo estimation based on autocorrelation and cross-correlation with pulses.
		  IEEE/ACM Transactions on Audio, Speech, and Language Processing, 22(12), 1765–1776.
		
				""" 
		... 
	def __call__(self, array:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			array (NDArray[np.float32]): the input signal. Defaults to None. 
		Returns:
			array (NDArray[np.float32]): the input signal with enhanced harmonics
		""" 
		... 


class PercivalEvaluatePulseTrains(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Implements the 'Evaluate Pulse Trains' step as described in [1].Given an input onset detection function (ODF, called "onset strength signal", OSS, in the original paper) and a number of candidate BPM peak positions, the ODF is correlated with ideal expected pulse trains (for each candidate tempo lag) shifted in time by different amounts.The candidate tempo lag that generates a periodic pulse train with the best correlation to the ODF is returned as the best tempo estimate.
		
		For more details check the referenced paper.Please note that in the original paper, the term OSS (Onset Strength Signal) is used instead of ODF.
		
		References:
		  [1] Percival, G., & Tzanetakis, G. (2014). Streamlined tempo estimation based on autocorrelation and cross-correlation with pulses.
		  IEEE/ACM Transactions on Audio, Speech, and Language Processing, 22(12), 1765–1776.
		
				""" 
		... 
	def __call__(self, oss:NDArray[np.float32], positions:NDArray[np.float32]) -> float:
		"""compute
		Args:
			oss (NDArray[np.float32]): onset strength signal (or other novelty curve). Defaults to None. 
			positions (NDArray[np.float32]): peak positions of BPM candidates. Defaults to None. 
		Returns:
			lag (float): best tempo lag estimate
		""" 
		... 


class Pitch2Midi(_essentia.Algorithm): 
	def __init__(self, applyTimeCompensation:bool=True, hopSize:int=128, midiBufferDuration:float=0.015, minFrequency:float=60.0, minNoteChangePeriod:float=0.03, minOccurrenceRate:float=0.5, minOffsetCheckPeriod:float=0.2, minOnsetCheckPeriod:float=0.075, sampleRate:int=44100, transpositionAmount:int=0, tuningFrequency:int=440) -> None:
		"""Estimates the midi note ON/OFF detection from raw pitch and voiced values, using midi buffer and uncertainty checkers.

		Args:
			applyTimeCompensation (bool): whether to apply time compensation in the timestamp of the note toggle messages.. Defaults to True. Range {true,false}
			hopSize (int): Pitch Detection analysis hop size in samples, equivalent to I/O buffer size. Defaults to 128. Range [1,inf)
			midiBufferDuration (float): duration in seconds of buffer used for voting in the note toggle detection algorithm. Defaults to 0.015. Range [0.005,0.5]
			minFrequency (float): minimum detectable frequency. Defaults to 60.0. Range [20,20000]
			minNoteChangePeriod (float): minimum time to wait until a note change is detected (s). Defaults to 0.03. Range (0,1]
			minOccurrenceRate (float): minimum number of times a midi note has to ocur compared to total capacity. Defaults to 0.5. Range [0,1]
			minOffsetCheckPeriod (float): minimum time to wait until an offset is detected (s). Defaults to 0.2. Range (0,1]
			minOnsetCheckPeriod (float): minimum time to wait until an onset is detected (s). Defaults to 0.075. Range (0,1]
			sampleRate (int): Audio sample rate. Defaults to 44100. Range [8000,inf)
			transpositionAmount (int): Apply transposition (in semitones) to the detected MIDI notes.. Defaults to 0. Range (-69,50)
			tuningFrequency (int): reference tuning frequency in Hz. Defaults to 440. Range {432,440} 
		""" 
		... 
	def __call__(self, pitch:float, voiced:int) -> tuple[list[str], NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			pitch (float): pitch given in Hz for conversion. Defaults to None. 
			voiced (int): whether the frame is voiced or not, (0, 1). Defaults to None. 
		Returns:
			messageType (list[str]): the output of MIDI message type, as string, {noteoff, noteon, noteoff-noteon}
			midiNoteNumber (NDArray[np.float32]): the output of detected MIDI note number, as integer, in range [0,127]
			timeCompensation (NDArray[np.float32]): time to be compensated in the messages
		""" 
		... 


class PitchContourSegmentation(_essentia.Algorithm): 
	def __init__(self, hopSize:int=128, minDuration:float=0.1, pitchDistanceThreshold:int=60, rmsThreshold:int=-2, sampleRate:int=44100, tuningFrequency:int=440) -> None:
		"""Converts a pitch sequence estimated from an audio signal into a set of discrete note events.
		
		Each note is defined by its onset time, duration and MIDI pitch value, quantized to the equal tempered scale.
		
		Note segmentation is performed based on pitch contour characteristics (island building) and signal RMS. Notes below an adjustable minimum duration are rejected.
		
		References:
		  [1] R. J. McNab et al., "Signal processing for melody transcription," in Proc. 
		  Proc. 19th Australasian Computer Science Conf., 1996

		Args:
			hopSize (int): hop size of the extracted pitch. Defaults to 128. Range (0,inf)
			minDuration (float): minimum note duration [s]. Defaults to 0.1. Range (0,inf)
			pitchDistanceThreshold (int): pitch threshold for note segmentation [cents]. Defaults to 60. Range (0,inf)
			rmsThreshold (int): zscore threshold for note segmentation. Defaults to -2. Range (-inf,0)
			sampleRate (int): sample rate of the audio signal. Defaults to 44100. Range (0,inf)
			tuningFrequency (int): tuning reference frequency  [Hz]. Defaults to 440. Range (0,22000) 
		""" 
		... 
	def __call__(self, pitch:NDArray[np.float32], signal:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			pitch (NDArray[np.float32]): estimated pitch contour [Hz]. Defaults to None. 
			signal (NDArray[np.float32]): input audio signal. Defaults to None. 
		Returns:
			onset (NDArray[np.float32]): note onset times [s]
			duration (NDArray[np.float32]): note durations [s]
			MIDIpitch (NDArray[np.float32]): quantized MIDI pitch value
		""" 
		... 


class PitchContours(_essentia.Algorithm): 
	def __init__(self, binResolution:float=10.0, hopSize:int=128, minDuration:float=100.0, peakDistributionThreshold:float=0.9, peakFrameThreshold:float=0.9, pitchContinuity:float=27.5625, sampleRate:float=44100.0, timeContinuity:float=100.0) -> None:
		"""Tracks a set of predominant pitch contours of an audio signal.
		
		This algorithm is intended to receive its "frequencies" and "magnitudes" inputs from the PitchSalienceFunctionPeaks algorithm outputs aggregated over all frames in the sequence. The output is a vector of estimated melody pitch values.
		
		When input vectors differ in size, an exception is thrown. Input vectors must not contain negative salience values otherwise an exception is thrown. Avoiding erroneous peak duplicates (peaks of the same cent bin) is up to the user's own control and is highly recommended, but no exception will be thrown.
		
		Recommended processing chain: (see [1]): EqualLoudness -> frame slicing with sample rate = 44100, frame size = 2048, hop size = 128 -> Windowing with Hann, x4 zero padding -> Spectrum -> SpectralPeaks -> PitchSalienceFunction (10 cents bin resolution) -> PitchSalienceFunctionPeaks.
		
		References:
		  [1] J. Salamon and E. Gómez, "Melody extraction from polyphonic music
		  signals using pitch contour characteristics," IEEE Transactions on Audio,
		  Speech, and Language Processing, vol. 20, no. 6, pp. 1759–1770, 2012.
		

		Args:
			binResolution (float): salience function bin resolution [cents]. Defaults to 10.0. Range (0,inf)
			hopSize (int): the hop size with which the pitch salience function was computed. Defaults to 128. Range (0,inf)
			minDuration (float): the minimum allowed contour duration [ms]. Defaults to 100.0. Range (0,inf)
			peakDistributionThreshold (float): allowed deviation below the peak salience mean over all frames (fraction of the standard deviation). Defaults to 0.9. Range [0,2]
			peakFrameThreshold (float): per-frame salience threshold factor (fraction of the highest peak salience in a frame). Defaults to 0.9. Range [0,1]
			pitchContinuity (float): pitch continuity cue (maximum allowed pitch change durig 1 ms time period) [cents]. Defaults to 27.5625. Range [0,inf)
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			timeContinuity (float): time continuity cue (the maximum allowed gap duration for a pitch contour) [ms]. Defaults to 100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, peakBins:np.ndarray, peakSaliences:np.ndarray) -> tuple[np.ndarray, np.ndarray, NDArray[np.float32], float]:
		"""compute
		Args:
			peakBins (np.ndarray): frame-wise array of cent bins corresponding to pitch salience function peaks. Defaults to None. 
			peakSaliences (np.ndarray): frame-wise array of values of salience function peaks. Defaults to None. 
		Returns:
			contoursBins (np.ndarray): array of frame-wise vectors of cent bin values representing each contour
			contoursSaliences (np.ndarray): array of frame-wise vectors of pitch saliences representing each contour
			contoursStartTimes (NDArray[np.float32]): array of start times of each contour [s]
			duration (float): time duration of the input signal [s]
		""" 
		... 


class PitchContoursMelody(_essentia.Algorithm): 
	def __init__(self, binResolution:float=10.0, filterIterations:int=3, guessUnvoiced:bool=False, hopSize:int=128, maxFrequency:float=20000.0, minFrequency:float=80.0, referenceFrequency:float=55.0, sampleRate:float=44100.0, voiceVibrato:bool=False, voicingTolerance:float=0.2) -> None:
		"""Converts a set of pitch contours into a sequence of predominant f0 values in Hz by taking the value of the most predominant contour in each frame.
		
		This algorithm is intended to receive its "contoursBins", "contoursSaliences", and "contoursStartTimes" inputs from the PitchContours algorithm. The "duration" input corresponds to the time duration of the input signal. The output is a vector of estimated pitch values and a vector of confidence values.
		
		Note that "pitchConfidence" can be negative in the case of "guessUnvoiced"=True: the absolute values represent the confidence, negative values correspond to segments for which non-salient contours where selected, zero values correspond to non-voiced segments.
		
		When input vectors differ in size, or "numberFrames" is negative, an exception is thrown. Input vectors must not contain negative start indices nor negative bin and salience values otherwise an exception is thrown.
		
		Recommended processing chain: (see [1]): EqualLoudness -> frame slicing with sample rate = 44100, frame size = 2048, hop size = 128 -> Windowing with Hann, x4 zero padding -> Spectrum -> SpectralPeaks -> PitchSalienceFunction -> PitchSalienceFunctionPeaks -> PitchContours.
		
		References:
		  [1] J. Salamon and E. Gómez, "Melody extraction from polyphonic music
		  signals using pitch contour characteristics," IEEE Transactions on Audio,
		  Speech, and Language Processing, vol. 20, no. 6, pp. 1759–1770, 2012.
		

		Args:
			binResolution (float): salience function bin resolution [cents]. Defaults to 10.0. Range (0,inf)
			filterIterations (int): number of interations for the octave errors / pitch outlier filtering process. Defaults to 3. Range [1,inf)
			guessUnvoiced (bool): Estimate pitch for non-voiced segments by using non-salient contours when no salient ones are present in a frame. Defaults to False. Range {false,true}
			hopSize (int): the hop size with which the pitch salience function was computed. Defaults to 128. Range (0,inf)
			maxFrequency (float): the maximum allowed frequency for salience function peaks (ignore contours with peaks above) [Hz]. Defaults to 20000.0. Range [0,inf)
			minFrequency (float): the minimum allowed frequency for salience function peaks (ignore contours with peaks below) [Hz]. Defaults to 80.0. Range [0,inf)
			referenceFrequency (float): the reference frequency for Hertz to cent convertion [Hz], corresponding to the 0th cent bin. Defaults to 55.0. Range (0,inf)
			sampleRate (float): the sampling rate of the audio signal (Hz). Defaults to 44100.0. Range (0,inf)
			voiceVibrato (bool): detect voice vibrato. Defaults to False. Range {true,false}
			voicingTolerance (float): allowed deviation below the average contour mean salience of all contours (fraction of the standard deviation). Defaults to 0.2. Range [-1.0,1.4] 
		""" 
		... 
	def __call__(self, contoursBins:np.ndarray, contoursSaliences:np.ndarray, contoursStartTimes:NDArray[np.float32], duration:float) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			contoursBins (np.ndarray): array of frame-wise vectors of cent bin values representing each contour. Defaults to None. 
			contoursSaliences (np.ndarray): array of frame-wise vectors of pitch saliences representing each contour. Defaults to None. 
			contoursStartTimes (NDArray[np.float32]): array of the start times of each contour [s]. Defaults to None. 
			duration (float): time duration of the input signal [s]. Defaults to None. 
		Returns:
			pitch (NDArray[np.float32]): vector of estimated pitch values (i.e., melody) [Hz]
			pitchConfidence (NDArray[np.float32]): confidence with which the pitch was detected
		""" 
		... 


class PitchContoursMonoMelody(_essentia.Algorithm): 
	def __init__(self, binResolution:float=10.0, filterIterations:int=3, guessUnvoiced:bool=False, hopSize:int=128, maxFrequency:float=20000.0, minFrequency:float=80.0, referenceFrequency:float=55.0, sampleRate:float=44100.0) -> None:
		"""Converts a set of pitch contours into a sequence of f0 values in Hz by taking the value of the most salient contour in each frame.
		
		In contrast to pitchContoursMelody, it assumes a single source. 
		This algorithm is intended to receive its "contoursBins", "contoursSaliences", and "contoursStartTimes" inputs from the PitchContours algorithm. The "duration" input corresponds to the time duration of the input signal. The output is a vector of estimated pitch values and a vector of confidence values.
		
		Note that "pitchConfidence" can be negative in the case of "guessUnvoiced"=True: the absolute values represent the confidence, negative values correspond to segments for which non-salient contours where selected, zero values correspond to non-voiced segments.
		
		When input vectors differ in size, or "numberFrames" is negative, an exception is thrown. Input vectors must not contain negative start indices nor negative bin and salience values otherwise an exception is thrown.
		
		Recommended processing chain: (see [1]): EqualLoudness -> frame slicing with sample rate = 44100, frame size = 2048, hop size = 128 -> Windowing with Hann, x4 zero padding -> Spectrum -> SpectralPeaks -> PitchSalienceFunction -> PitchSalienceFunctionPeaks -> PitchContours.
		
		References:
		  [1] J. Salamon and E. Gómez, "Melody extraction from polyphonic music
		  signals using pitch contour characteristics," IEEE Transactions on Audio,
		  Speech, and Language Processing, vol. 20, no. 6, pp. 1759–1770, 2012.
		

		Args:
			binResolution (float): salience function bin resolution [cents]. Defaults to 10.0. Range (0,inf)
			filterIterations (int): number of interations for the octave errors / pitch outlier filtering process. Defaults to 3. Range [1,inf)
			guessUnvoiced (bool): Estimate pitch for non-voiced segments by using non-salient contours when no salient ones are present in a frame. Defaults to False. Range {false,true}
			hopSize (int): the hop size with which the pitch salience function was computed. Defaults to 128. Range (0,inf)
			maxFrequency (float): the maximum allowed frequency for salience function peaks (ignore contours with peaks above) [Hz]. Defaults to 20000.0. Range [0,inf)
			minFrequency (float): the minimum allowed frequency for salience function peaks (ignore contours with peaks below) [Hz]. Defaults to 80.0. Range [0,inf)
			referenceFrequency (float): the reference frequency for Hertz to cent convertion [Hz], corresponding to the 0th cent bin. Defaults to 55.0. Range (0,inf)
			sampleRate (float): the sampling rate of the audio signal (Hz). Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, contoursBins:np.ndarray, contoursSaliences:np.ndarray, contoursStartTimes:NDArray[np.float32], duration:float) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			contoursBins (np.ndarray): array of frame-wise vectors of cent bin values representing each contour. Defaults to None. 
			contoursSaliences (np.ndarray): array of frame-wise vectors of pitch saliences representing each contour. Defaults to None. 
			contoursStartTimes (NDArray[np.float32]): array of the start times of each contour [s]. Defaults to None. 
			duration (float): time duration of the input signal [s]. Defaults to None. 
		Returns:
			pitch (NDArray[np.float32]): vector of estimated pitch values (i.e., melody) [Hz]
			pitchConfidence (NDArray[np.float32]): confidence with which the pitch was detected
		""" 
		... 


class PitchContoursMultiMelody(_essentia.Algorithm): 
	def __init__(self, binResolution:float=10.0, filterIterations:int=3, guessUnvoiced:bool=False, hopSize:int=128, maxFrequency:float=20000.0, minFrequency:float=80.0, referenceFrequency:float=55.0, sampleRate:float=44100.0) -> None:
		"""Post-processes a set of pitch contours into a sequence of mutliple f0 values in Hz.
		
		This algorithm is intended to receive its "contoursBins", "contoursSaliences", and "contoursStartTimes" inputs from the PitchContours algorithm. The "duration" input corresponds to the time duration of the input signal. The output is a vector of vectors of estimated pitch values for each frame.
		
		When input vectors differ in size, or "numberFrames" is negative, an exception is thrown. Input vectors must not contain negative start indices nor negative bin and salience values otherwise an exception is thrown.
		
		References:
		  [1] J. Salamon and E. Gómez, "Melody extraction from polyphonic music
		  signals using pitch contour characteristics," IEEE Transactions on Audio,
		  Speech, and Language Processing, vol. 20, no. 6, pp. 1759–1770, 2012.
		

		Args:
			binResolution (float): salience function bin resolution [cents]. Defaults to 10.0. Range (0,inf)
			filterIterations (int): number of interations for the octave errors / pitch outlier filtering process. Defaults to 3. Range [1,inf)
			guessUnvoiced (bool): Estimate pitch for non-voiced segments by using non-salient contours when no salient ones are present in a frame. Defaults to False. Range {false,true}
			hopSize (int): the hop size with which the pitch salience function was computed. Defaults to 128. Range (0,inf)
			maxFrequency (float): the maximum allowed frequency for salience function peaks (ignore contours with peaks above) [Hz]. Defaults to 20000.0. Range [0,inf)
			minFrequency (float): the minimum allowed frequency for salience function peaks (ignore contours with peaks below) [Hz]. Defaults to 80.0. Range [0,inf)
			referenceFrequency (float): the reference frequency for Hertz to cent convertion [Hz], corresponding to the 0th cent bin. Defaults to 55.0. Range (0,inf)
			sampleRate (float): the sampling rate of the audio signal (Hz). Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, contoursBins:np.ndarray, contoursSaliences:np.ndarray, contoursStartTimes:NDArray[np.float32], duration:float) -> np.ndarray:
		"""compute
		Args:
			contoursBins (np.ndarray): array of frame-wise vectors of cent bin values representing each contour. Defaults to None. 
			contoursSaliences (np.ndarray): array of frame-wise vectors of pitch saliences representing each contour. Defaults to None. 
			contoursStartTimes (NDArray[np.float32]): array of the start times of each contour [s]. Defaults to None. 
			duration (float): time duration of the input signal [s]. Defaults to None. 
		Returns:
			pitch (np.ndarray): vector of estimated pitch values (i.e., melody) [Hz]
		""" 
		... 


class PitchFilter(_essentia.Algorithm): 
	def __init__(self, confidenceThreshold:int=36, minChunkSize:int=30, useAbsolutePitchConfidence:bool=False) -> None:
		"""Corrects the fundamental frequency estimations for a sequence of frames given pitch values together with their confidence values.
		
		In particular, it removes non-confident parts and spurious jumps in pitch and applies octave corrections.
		
		They can be computed with the PitchYinFFT, PitchYin, or PredominantPitchMelodia algorithms.
		If you use PredominantPitchMelodia with guessUnvoiced=True, set useAbsolutePitchConfidence=True.
		
		The algorithm can be used for any type of monophonic and heterophonic music.
		
		The original algorithm [1] was proposed to be used for Makam music and employs signal"energy" of frames instead of pitch confidence.
		
		References:
		  [1] B. Bozkurt, "An Automatic Pitch Analysis Method for Turkish Maqam
		  Music," Journal of New Music Research. 37(1), 1-13.
		

		Args:
			confidenceThreshold (int): ratio between the average confidence of the most confident chunk and the minimum allowed average confidence of a chunk. Defaults to 36. Range [0,inf)
			minChunkSize (int): minumum number of frames in non-zero pitch chunks. Defaults to 30. Range [0,inf)
			useAbsolutePitchConfidence (bool): treat negative pitch confidence values as positive (use with melodia guessUnvoiced=True). Defaults to False. Range {true,false} 
		""" 
		... 
	def __call__(self, pitch:NDArray[np.float32], pitchConfidence:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			pitch (NDArray[np.float32]): vector of pitch values for the input frames [Hz]. Defaults to None. 
			pitchConfidence (NDArray[np.float32]): vector of pitch confidence values for the input frames. Defaults to None. 
		Returns:
			pitchFiltered (NDArray[np.float32]): vector of corrected pitch values [Hz]
		""" 
		... 


class PitchMelodia(_essentia.Algorithm): 
	def __init__(self, binResolution:float=10.0, filterIterations:int=3, frameSize:int=2048, guessUnvoiced:bool=False, harmonicWeight:float=0.8, hopSize:int=128, magnitudeCompression:float=1.0, magnitudeThreshold:int=40, maxFrequency:float=20000.0, minDuration:int=100, minFrequency:float=40.0, numberHarmonics:int=20, peakDistributionThreshold:float=0.9, peakFrameThreshold:float=0.9, pitchContinuity:float=27.5625, referenceFrequency:float=55.0, sampleRate:float=44100.0, timeContinuity:int=100) -> None:
		"""Estimates the fundamental frequency corresponding to the melody of a monophonic music signal based on the MELODIA algorithm.
		
		While the algorithm is originally designed to extract the predominant melody from polyphonic music [1], this implementation is adapted for monophonic signals. The approach is based on the creation and characterization of pitch contours, time continuous sequences of pitch candidates grouped using auditory streaming cues. To this end, PitchSalienceFunction, PitchSalienceFunctionPeaks, PitchContours, and PitchContoursMonoMelody algorithms are employed. It is strongly advised to use the default parameter values which are optimized according to [1] (where further details are provided) except for minFrequency and maxFrequency, which will depend on your application.
		
		The output is a vector of estimated melody pitch values and a vector of confidence values.
		
		It is recommended to apply EqualLoudness on the input signal (see [1]) as a pre-processing stage before running this algorithm.
		
		References:
		  [1] J. Salamon and E. Gómez, "Melody extraction from polyphonic music
		  signals using pitch contour characteristics," IEEE Transactions on Audio,
		  Speech, and Language Processing, vol. 20, no. 6, pp. 1759–1770, 2012.
		
		  [2] http://mtg.upf.edu/technologies/melodia
		
		  [3] http://www.justinsalamon.com/melody-extraction
		

		Args:
			binResolution (float): salience function bin resolution [cents]. Defaults to 10.0. Range (0,inf)
			filterIterations (int): number of iterations for the octave errors / pitch outlier filtering process. Defaults to 3. Range [1,inf)
			frameSize (int): the frame size for computing pitch saliecnce. Defaults to 2048. Range (0,inf)
			guessUnvoiced (bool): estimate pitch for non-voiced segments by using non-salient contours when no salient ones are present in a frame. Defaults to False. Range {false,true}
			harmonicWeight (float): harmonic weighting parameter (weight decay ratio between two consequent harmonics, =1 for no decay). Defaults to 0.8. Range (0,1)
			hopSize (int): the hop size with which the pitch salience function was computed. Defaults to 128. Range (0,inf)
			magnitudeCompression (float): magnitude compression parameter for the salience function (=0 for maximum compression, =1 for no compression). Defaults to 1.0. Range (0,1]
			magnitudeThreshold (int): spectral peak magnitude threshold (maximum allowed difference from the highest peak in dBs). Defaults to 40. Range [0,inf)
			maxFrequency (float): the maximum allowed frequency for salience function peaks (ignore contours with peaks above) [Hz]. Defaults to 20000.0. Range [0,inf)
			minDuration (int): the minimum allowed contour duration [ms]. Defaults to 100. Range (0,inf)
			minFrequency (float): the minimum allowed frequency for salience function peaks (ignore contours with peaks below) [Hz]. Defaults to 40.0. Range [0,inf)
			numberHarmonics (int): number of considered harmonics. Defaults to 20. Range [1,inf)
			peakDistributionThreshold (float): allowed deviation below the peak salience mean over all frames (fraction of the standard deviation). Defaults to 0.9. Range [0,2]
			peakFrameThreshold (float): per-frame salience threshold factor (fraction of the highest peak salience in a frame). Defaults to 0.9. Range [0,1]
			pitchContinuity (float): pitch continuity cue (maximum allowed pitch change during 1 ms time period) [cents]. Defaults to 27.5625. Range [0,inf)
			referenceFrequency (float): the reference frequency for Hertz to cent convertion [Hz], corresponding to the 0th cent bin. Defaults to 55.0. Range (0,inf)
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			timeContinuity (int): time continuity cue (the maximum allowed gap duration for a pitch contour) [ms]. Defaults to 100. Range (0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input signal. Defaults to None. 
		Returns:
			pitch (NDArray[np.float32]): the estimated pitch values [Hz]
			pitchConfidence (NDArray[np.float32]): confidence with which the pitch was detected
		""" 
		... 


class PitchSalience(_essentia.Algorithm): 
	def __init__(self, highBoundary:float=5000.0, lowBoundary:float=100.0, sampleRate:float=44100.0) -> None:
		"""Computes the pitch salience of a spectrum.
		
		The pitch salience is given by the ratio of the highest auto correlation value of the spectrum to the non-shifted auto correlation value. Pitch salience was designed as quick measure of tone sensation. Unpitched sounds (non-musical sound effects) and pure tones have an average pitch salience value close to 0 whereas sounds containing several harmonics in the spectrum tend to have a higher value.
		
		Note that this algorithm may give better results when used with low sampling rates (i.e. 8000) as the information in the bands musically meaningful will have more relevance.
		
		This algorithm uses AutoCorrelation on the input "spectrum" and thus inherits its input requirements and exceptions. An exception is thrown at configuration time if "lowBoundary" is larger than "highBoundary" and/or if "highBoundary" is not smaller than half "sampleRate". At computation time, an exception is thrown if the input spectrum is empty. Also note that feeding silence to this algorithm will return zero.
		
		Application: characterizing percussive sounds.
		
		References:
		  [1] J. Ricard "Towards computational morphological description of sound.
		  DEA pre-thesis research work, Universitat Pompeu Fabra, Barcelona, 2004.

		Args:
			highBoundary (float): until which frequency we are looking for the minimum (must be smaller than half sampleRate) [Hz]. Defaults to 5000.0. Range (0,inf)
			lowBoundary (float): from which frequency we are looking for the maximum (must not be larger than highBoundary) [Hz]. Defaults to 100.0. Range (0,inf)
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, spectrum:NDArray[np.float32]) -> float:
		"""compute
		Args:
			spectrum (NDArray[np.float32]): the input audio spectrum. Defaults to None. 
		Returns:
			pitchSalience (float): the pitch salience (normalized from 0 to 1)
		""" 
		... 


class PitchSalienceFunction(_essentia.Algorithm): 
	def __init__(self, binResolution:float=10.0, harmonicWeight:float=0.8, magnitudeCompression:float=1.0, magnitudeThreshold:float=40.0, numberHarmonics:int=20, referenceFrequency:float=55.0) -> None:
		"""Computes the pitch salience function of a signal frame given its spectral peaks.
		
		The salience function covers a pitch range of nearly five octaves (i.e., 6000 cents), starting from the "referenceFrequency", and is quantized into cent bins according to the specified "binResolution". The salience of a given frequency is computed as the sum of the weighted energies found at integer multiples (harmonics) of that frequency. 
		
		This algorithm is intended to receive its "frequencies" and "magnitudes" inputs from the SpectralPeaks algorithm. The output is a vector of salience values computed for the cent bins. The 0th bin corresponds to the specified "referenceFrequency".
		
		If both input vectors are empty (i.e., no spectral peaks are provided), a zero salience function is returned. Input vectors must contain positive frequencies, must not contain negative magnitudes and these input vectors must be of the same size, otherwise an exception is thrown. It is highly recommended to avoid erroneous peak duplicates (peaks of the same frequency occurring more than once), but it is up to the user's own control and no exception will be thrown.
		
		References:
		  [1] J. Salamon and E. Gómez, "Melody extraction from polyphonic music
		  signals using pitch contour characteristics," IEEE Transactions on Audio,
		  Speech, and Language Processing, vol. 20, no. 6, pp. 1759–1770, 2012.
		

		Args:
			binResolution (float): salience function bin resolution [cents]. Defaults to 10.0. Range (0,100]
			harmonicWeight (float): harmonic weighting parameter (weight decay ratio between two consequent harmonics, =1 for no decay). Defaults to 0.8. Range [0,1]
			magnitudeCompression (float): magnitude compression parameter (=0 for maximum compression, =1 for no compression). Defaults to 1.0. Range (0,1]
			magnitudeThreshold (float): peak magnitude threshold (maximum allowed difference from the highest peak in dBs). Defaults to 40.0. Range [0,inf)
			numberHarmonics (int): number of considered harmonics. Defaults to 20. Range [1,inf)
			referenceFrequency (float): the reference frequency for Hertz to cent convertion [Hz], corresponding to the 0th cent bin. Defaults to 55.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, frequencies:NDArray[np.float32], magnitudes:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			frequencies (NDArray[np.float32]): the frequencies of the spectral peaks [Hz]. Defaults to None. 
			magnitudes (NDArray[np.float32]): the magnitudes of the spectral peaks. Defaults to None. 
		Returns:
			salienceFunction (NDArray[np.float32]): array of the quantized pitch salience values
		""" 
		... 


class PitchSalienceFunctionPeaks(_essentia.Algorithm): 
	def __init__(self, binResolution:float=10.0, maxFrequency:float=1760.0, minFrequency:float=55.0, referenceFrequency:float=55.0) -> None:
		"""Computes the peaks of a given pitch salience function.
		
		This algorithm is intended to receive its "salienceFunction" input from the PitchSalienceFunction algorithm. The peaks are detected using PeakDetection algorithm. The outputs are two arrays of bin numbers and salience values corresponding to the peaks.
		
		References:
		  [1] Salamon, J., & Gómez E. (2012).  Melody Extraction from Polyphonic Music Signals using Pitch Contour Characteristics.
		      IEEE Transactions on Audio, Speech and Language Processing. 20(6), 1759-1770.
		

		Args:
			binResolution (float): salience function bin resolution [cents]. Defaults to 10.0. Range (0,inf)
			maxFrequency (float): the maximum frequency to evaluate (ignore peaks above) [Hz]. Defaults to 1760.0. Range [0,inf)
			minFrequency (float): the minimum frequency to evaluate (ignore peaks below) [Hz]. Defaults to 55.0. Range [0,inf)
			referenceFrequency (float): the reference frequency for Hertz to cent convertion [Hz], corresponding to the 0th cent bin. Defaults to 55.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, salienceFunction:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			salienceFunction (NDArray[np.float32]): the array of salience function values corresponding to cent frequency bins. Defaults to None. 
		Returns:
			salienceBins (NDArray[np.float32]): the cent bins corresponding to salience function peaks
			salienceValues (NDArray[np.float32]): the values of salience function peaks
		""" 
		... 


class PitchYin(_essentia.Algorithm): 
	def __init__(self, frameSize:int=2048, interpolate:bool=True, maxFrequency:float=22050.0, minFrequency:float=20.0, sampleRate:float=44100.0, tolerance:float=0.15) -> None:
		"""Estimates the fundamental frequency given the frame of a monophonic music signal.
		
		It is an implementation of the Yin algorithm [1] for computations in the time domain.
		
		An exception is thrown if an empty signal is provided.
		
		Please note that if "pitchConfidence" is zero, "pitch" is undefined and should not be used for other algorithms.
		Also note that a null "pitch" is never ouput by the algorithm and that "pitchConfidence" must always be checked out.
		
		References:
		  [1] De Cheveigné, A., & Kawahara, H. "YIN, a fundamental frequency estimator
		  for speech and music. The Journal of the Acoustical Society of America,
		  111(4), 1917-1930, 2002.
		
		  [2] Pitch detection algorithm - Wikipedia, the free encyclopedia
		  http://en.wikipedia.org/wiki/Pitch_detection_algorithm

		Args:
			frameSize (int): number of samples in the input frame (this is an optional parameter to optimize memory allocation). Defaults to 2048. Range [2,inf)
			interpolate (bool): enable interpolation. Defaults to True. Range {true,false}
			maxFrequency (float): the maximum allowed frequency [Hz]. Defaults to 22050.0. Range (0,inf)
			minFrequency (float): the minimum allowed frequency [Hz]. Defaults to 20.0. Range (0,inf)
			sampleRate (float): sampling rate of the input audio [Hz]. Defaults to 44100.0. Range (0,inf)
			tolerance (float): tolerance for peak detection. Defaults to 0.15. Range [0,1] 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> tuple[float, float]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input signal frame. Defaults to None. 
		Returns:
			pitch (float): detected pitch [Hz]
			pitchConfidence (float): confidence with which the pitch was detected [0,1]
		""" 
		... 


class PitchYinFFT(_essentia.Algorithm): 
	def __init__(self, frameSize:int=2048, interpolate:bool=True, maxFrequency:float=22050.0, minFrequency:float=20.0, sampleRate:float=44100.0, tolerance:float=1.0, weighting:str='custom') -> None:
		"""Estimates the fundamental frequency given the spectrum of a monophonic music signal.
		
		It is an implementation of YinFFT algorithm [1], which is an optimized version of Yin algorithm for computation in the frequency domain. It is recommended to window the input spectrum with a Hann window. The raw spectrum can be computed with the Spectrum algorithm.
		
		An exception is thrown if an empty spectrum is provided.
		
		Please note that if "pitchConfidence" is zero, "pitch" is undefined and should not be used for other algorithms.
		Also note that a null "pitch" is never ouput by the algorithm and that "pitchConfidence" must always be checked out.
		
		References:
		  [1] P. M. Brossier, "Automatic Annotation of Musical Audio for Interactive
		  Applications,” QMUL, London, UK, 2007.
		
		  [2] Pitch detection algorithm - Wikipedia, the free encyclopedia
		  http://en.wikipedia.org/wiki/Pitch_detection_algorithm

		Args:
			frameSize (int): number of samples in the input spectrum. Defaults to 2048. Range [2,inf)
			interpolate (bool): boolean flag to enable interpolation. Defaults to True. Range {true,false}
			maxFrequency (float): the maximum allowed frequency [Hz]. Defaults to 22050.0. Range (0,inf)
			minFrequency (float): the minimum allowed frequency [Hz]. Defaults to 20.0. Range (0,inf)
			sampleRate (float): sampling rate of the input spectrum [Hz]. Defaults to 44100.0. Range (0,inf)
			tolerance (float): tolerance for peak detection. Defaults to 1.0. Range [0,1]
			weighting (str): string to assign a weighting function. Defaults to 'custom'. Range {custom,A,B,C,D,Z} 
		""" 
		... 
	def __call__(self, spectrum:NDArray[np.float32]) -> tuple[float, float]:
		"""compute
		Args:
			spectrum (NDArray[np.float32]): the input spectrum (preferably created with a hann window). Defaults to None. 
		Returns:
			pitch (float): detected pitch [Hz]
			pitchConfidence (float): confidence with which the pitch was detected [0,1]
		""" 
		... 


class PitchYinProbabilistic(_essentia.Algorithm): 
	def __init__(self, frameSize:int=2048, hopSize:int=256, lowRMSThreshold:float=0.1, outputUnvoiced:str='negative', preciseTime:bool=False, sampleRate:float=44100.0) -> None:
		"""Computes the pitch track of a mono audio signal using probabilistic Yin algorithm.
		
		- The input mono audio signal is preprocessed with a FrameCutter to segment into frameSize chunks with a overlap hopSize.
		- The pitch frequencies, probabilities and RMS values of the chunks are then calculated by PitchYinProbabilities algorithm. The results of all chunks are aggregated into a Essentia pool.
		- The pitch frequencies and probabilities are finally sent to PitchYinProbabilitiesHMM algorithm to get a smoothed pitch track and a voiced probability.
		
		References:
		  [1] M. Mauch and S. Dixon, "pYIN: A Fundamental Frequency Estimator
		  Using Probabilistic Threshold Distributions," in Proceedings of the
		  IEEE International Conference on Acoustics, Speech, and Signal Processing
		  (ICASSP 2014)Project Report, 2004

		Args:
			frameSize (int): the frame size of FFT. Defaults to 2048. Range (0,inf)
			hopSize (int): the hop size with which the pitch is computed. Defaults to 256. Range [1,inf)
			lowRMSThreshold (float): the low RMS amplitude threshold. Defaults to 0.1. Range (0,1]
			outputUnvoiced (str): whether output unvoiced frame, zero: output non-voiced pitch as 0.; abs: output non-voiced pitch as absolute values; negative: output non-voiced pitch as negative values. Defaults to 'negative'. Range {zero,abs,negative}
			preciseTime (bool): use non-standard precise YIN timing (slow).. Defaults to False. Range {true,false}
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input mono audio signal. Defaults to None. 
		Returns:
			pitch (NDArray[np.float32]): the output pitch estimations
			voicedProbabilities (NDArray[np.float32]): the voiced probabilities
		""" 
		... 


class PitchYinProbabilities(_essentia.Algorithm): 
	def __init__(self, frameSize:int=2048, lowAmp:float=0.1, preciseTime:bool=False, sampleRate:float=44100.0) -> None:
		"""Estimates the fundamental frequencies, their probabilities given the frame of a monophonic music signal.
		
		It is a part of the implementation of the probabilistic Yin algorithm [1].
		
		An exception is thrown if an empty signal is provided.
		
		References:
		  [1] M. Mauch and S. Dixon, "pYIN: A Fundamental Frequency Estimator
		  Using Probabilistic Threshold Distributions," in Proceedings of the
		  IEEE International Conference on Acoustics, Speech, and Signal Processing
		  (ICASSP 2014)Project Report, 2004

		Args:
			frameSize (int): number of samples in the input frame. Defaults to 2048. Range [2,inf)
			lowAmp (float): the low RMS amplitude threshold. Defaults to 0.1. Range (0,1]
			preciseTime (bool): use non-standard precise YIN timing (slow).. Defaults to False. Range {true,false}
			sampleRate (float): sampling rate of the input audio [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32], float]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input signal frame. Defaults to None. 
		Returns:
			pitch (NDArray[np.float32]): the output pitch candidate frequencies in cents
			probabilities (NDArray[np.float32]): the output pitch candidate probabilities
			RMS (float): the output RMS value
		""" 
		... 


class PitchYinProbabilitiesHMM(_essentia.Algorithm): 
	def __init__(self, minFrequency:float=61.735, numberBinsPerSemitone:int=5, selfTransition:float=0.99, yinTrust:float=0.5) -> None:
		"""Estimates the smoothed fundamental frequency given the pitch candidates and probabilities using hidden Markov models.
		
		It is a part of the implementation of the probabilistic Yin algorithm [1].
		
		An exception is thrown if an empty signal is provided.
		
		References:
		  [1] M. Mauch and S. Dixon, "pYIN: A Fundamental Frequency Estimator
		  Using Probabilistic Threshold Distributions," in Proceedings of the
		  IEEE International Conference on Acoustics, Speech, and Signal Processing
		  (ICASSP 2014)Project Report, 2004

		Args:
			minFrequency (float): minimum detected frequency. Defaults to 61.735. Range (0,inf)
			numberBinsPerSemitone (int): number of bins per semitone. Defaults to 5. Range (1,inf)
			selfTransition (float): the self transition probabilities. Defaults to 0.99. Range (0,1)
			yinTrust (float): the yin trust parameter. Defaults to 0.5. Range (0,1) 
		""" 
		... 
	def __call__(self, pitchCandidates:np.ndarray, probabilities:np.ndarray) -> NDArray[np.float32]:
		"""compute
		Args:
			pitchCandidates (np.ndarray): the pitch candidates. Defaults to None. 
			probabilities (np.ndarray): the pitch probabilities. Defaults to None. 
		Returns:
			pitch (NDArray[np.float32]): pitch frequencies in Hz
		""" 
		... 


class PolarToCartesian(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Converts an array of complex numbers from polar to cartesian form.
		
		It uses the Euler formula:
		  z = x + i*y = |z|(cos(α) + i sin(α))
		    where x = Real part, y = Imaginary part,
		    and |z| = modulus = magnitude, α = phase
		
		An exception is thrown if the size of the magnitude vector does not match the size of the phase vector.
		
		References:
		  [1] Polar coordinate system - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Polar_coordinates		""" 
		... 
	def __call__(self, magnitude:NDArray[np.float32], phase:NDArray[np.float32]) -> np.ndarray:
		"""compute
		Args:
			magnitude (NDArray[np.float32]): the magnitude vector. Defaults to None. 
			phase (NDArray[np.float32]): the phase vector. Defaults to None. 
		Returns:
			complex (np.ndarray): the resulting complex vector
		""" 
		... 


class PoolAggregator(_essentia.Algorithm): 
	def __init__(self, defaultStats:list[str]=['mean', 'stdev', 'min', 'max', 'median'], exceptions:dict[Any, Any]={}) -> None:
		"""Performs statistical aggregation on a Pool and places the results of the aggregation into a new Pool.
		
		Supported statistical units are:
		  - 'min' (minimum),
		  - 'max' (maximum),
		  - 'median',
		  - 'mean',
		  - 'var' (variance),
		  - 'stdev' (standard deviation),
		  - 'skew' (skewness),
		  - 'kurt' (kurtosis),
		  - 'dmean' (mean of the derivative),
		  - 'dvar' (variance of the derivative),
		  - 'dmean2' (mean of the second derivative),
		  - 'dvar2' (variance of the second derivative),
		  - 'cov' (covariance), and
		  - 'icov' (inverse covariance).
		  - 'value' (copy of descriptor, but the value is placed under the name '<descriptor name>.value')
		  - 'copy' (verbatim copy of descriptor, no aggregation; exclusive: cannot be performed with any other statistical units).
		  - 'last' (last value of descriptor placed under the name '<descriptor name>'; exclusive: cannot be performed with any other statistical units
		
		These statistics can be computed for single-dimensional vectors (vectors of Reals) and two-dimensional vectors (vectors of vectors of Reals) in the Pool. Statistics for two-dimensional vectors are computed by aggregating each column placing the result into a vector of the same size as the size of each vector in the input Pool under the given descriptor (which implies their equal size).
		
		In the case of 'cov' and 'icov', two-dimensional vectors are required, and each statistic returns a square matrix with the dimensions equal to the length of the vectors under the given descriptor. Computing 'icov' requires the corresponding covariance matrix to be invertible.
		
		Note that only the absolute values of the first and second derivatives are considered when computing their mean ('dmean' and 'dmean2') and variance ('dvar' and 'dvar2'). This is to avoid a trivial solution for the mean.
		
		For vectors, if the input pool value consists of only one vector, its aggregation will be skipped, and the vector itself will be added to the output.
		
		The 'value' and 'copy' are auxiliary aggregation methods that can be used to copy values in the input Pool to the output Pool without aggregation. In the case of 'last', the last value in the input vector of Reals (or input vector of vectors of Reals) will be taken and saved as a single Real (or single vector of Reals) in the output Pool.

		Args:
			defaultStats (list[str]): the default statistics to be computed for each descriptor in the input pool. Defaults to ['mean', 'stdev', 'min', 'max', 'median']. Range None
			exceptions (dict[Any, Any]): a mapping between descriptor names (no duplicates) and the types of statistics to be computed for those descriptors (e.g. { lowlevel.bpm : [min, max], lowlevel.gain : [var, min, dmean] }). Defaults to {}. Range None 
		""" 
		... 
	def __call__(self, input:Pool) -> Pool:
		"""compute
		Args:
			input (Pool): the input pool. Defaults to None. 
		Returns:
			output (Pool): a pool containing the aggregate values of the input pool
		""" 
		... 


class PowerMean(_essentia.Algorithm): 
	def __init__(self, power:float=1.0) -> None:
		"""Computes the power mean of an array.
		
		It accepts one parameter, p, which is the power (or order or degree) of the Power Mean. Note that if p=-1, the Power Mean is equal to the Harmonic Mean, if p=0, the Power Mean is equal to the Geometric Mean, if p=1, the Power Mean is equal to the Arithmetic Mean, if p=2, the Power Mean is equal to the Root Mean Square.
		
		Exceptions are thrown if input array either is empty or it contains non positive numbers.
		
		References:
		  [1] Power Mean -- from Wolfram MathWorld,
		  http://mathworld.wolfram.com/PowerMean.html
		  [2] Generalized mean - Wikipedia, the free encyclopedia,
		  https://en.wikipedia.org/wiki/Generalized_mean

		Args:
			power (float): the power to which to elevate each element before taking the mean. Defaults to 1.0. Range (-inf,inf) 
		""" 
		... 
	def __call__(self, array:NDArray[np.float32]) -> float:
		"""compute
		Args:
			array (NDArray[np.float32]): the input array (must contain only positive real numbers). Defaults to None. 
		Returns:
			powerMean (float): the power mean of the input array
		""" 
		... 


class PowerSpectrum(_essentia.Algorithm): 
	def __init__(self, size:int=2048) -> None:
		"""Computes the power spectrum of an array of Reals.
		
		The resulting power spectrum has a size which is half the size of the input array plus one. Bins contain squared magnitude values.
		
		References:
		  [1] Power Spectrum - from Wolfram MathWorld,
		  http://mathworld.wolfram.com/PowerSpectrum.html

		Args:
			size (int): the expected size of the input frame (this is purely optional and only targeted at optimizing the creation time of the FFT object). Defaults to 2048. Range [1,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input signal. Defaults to None. 
		Returns:
			powerSpectrum (NDArray[np.float32]): power spectrum of the input signal
		""" 
		... 


class PredominantPitchMelodia(_essentia.Algorithm): 
	def __init__(self, binResolution:float=10.0, filterIterations:int=3, frameSize:int=2048, guessUnvoiced:bool=False, harmonicWeight:float=0.8, hopSize:int=128, magnitudeCompression:float=1.0, magnitudeThreshold:int=40, maxFrequency:float=20000.0, minDuration:int=100, minFrequency:float=80.0, numberHarmonics:int=20, peakDistributionThreshold:float=0.9, peakFrameThreshold:float=0.9, pitchContinuity:float=27.5625, referenceFrequency:float=55.0, sampleRate:float=44100.0, timeContinuity:int=100, voiceVibrato:bool=False, voicingTolerance:float=0.2) -> None:
		"""Estimates the fundamental frequency of the predominant melody from polyphonic music signals using the MELODIA algorithm.
		
		It is specifically suited for music with a predominent melodic element, for example the singing voice melody in an accompanied singing recording. The approach [1] is based on the creation and characterization of pitch contours, time continuous sequences of pitch candidates grouped using auditory streaming cues. It furthermore determines for each frame, if the predominant melody is present or not. To this end, PitchSalienceFunction, PitchSalienceFunctionPeaks, PitchContours, and PitchContoursMelody algorithms are employed. It is strongly advised to use the default parameter values which are optimized according to [1] (where further details are provided) except for minFrequency, maxFrequency, and voicingTolerance, which will depend on your application.
		
		The output is a vector of estimated melody pitch values and a vector of confidence values. The first value corresponds to the beginning of the input signal (time 0).
		
		It is recommended to apply EqualLoudness on the input signal (see [1]) as a pre-processing stage before running this algorithm.
		
		Note that "pitchConfidence" can be negative in the case of "guessUnvoiced"=True: the absolute values represent the confidence, negative values correspond to segments for which non-salient contours where selected, zero values correspond to non-voiced segments.
		
		References:
		  [1] J. Salamon and E. Gómez, "Melody extraction from polyphonic music
		  signals using pitch contour characteristics," IEEE Transactions on Audio,
		  Speech, and Language Processing, vol. 20, no. 6, pp. 1759–1770, 2012.
		
		  [2] http://mtg.upf.edu/technologies/melodia
		
		  [3] http://www.justinsalamon.com/melody-extraction
		

		Args:
			binResolution (float): salience function bin resolution [cents]. Defaults to 10.0. Range (0,inf)
			filterIterations (int): number of iterations for the octave errors / pitch outlier filtering process. Defaults to 3. Range [1,inf)
			frameSize (int): the frame size for computing pitch salience. Defaults to 2048. Range (0,inf)
			guessUnvoiced (bool): estimate pitch for non-voiced segments by using non-salient contours when no salient ones are present in a frame. Defaults to False. Range {false,true}
			harmonicWeight (float): harmonic weighting parameter (weight decay ratio between two consequent harmonics, =1 for no decay). Defaults to 0.8. Range (0,1)
			hopSize (int): the hop size with which the pitch salience function was computed. Defaults to 128. Range (0,inf)
			magnitudeCompression (float): magnitude compression parameter for the salience function (=0 for maximum compression, =1 for no compression). Defaults to 1.0. Range (0,1]
			magnitudeThreshold (int): spectral peak magnitude threshold (maximum allowed difference from the highest peak in dBs). Defaults to 40. Range [0,inf)
			maxFrequency (float): the maximum allowed frequency for salience function peaks (ignore contours with peaks above) [Hz]. Defaults to 20000.0. Range [0,inf)
			minDuration (int): the minimum allowed contour duration [ms]. Defaults to 100. Range (0,inf)
			minFrequency (float): the minimum allowed frequency for salience function peaks (ignore contours with peaks below) [Hz]. Defaults to 80.0. Range [0,inf)
			numberHarmonics (int): number of considered harmonics. Defaults to 20. Range [1,inf)
			peakDistributionThreshold (float): allowed deviation below the peak salience mean over all frames (fraction of the standard deviation). Defaults to 0.9. Range [0,2]
			peakFrameThreshold (float): per-frame salience threshold factor (fraction of the highest peak salience in a frame). Defaults to 0.9. Range [0,1]
			pitchContinuity (float): pitch continuity cue (maximum allowed pitch change during 1 ms time period) [cents]. Defaults to 27.5625. Range [0,inf)
			referenceFrequency (float): the reference frequency for Hertz to cent conversion [Hz], corresponding to the 0th cent bin. Defaults to 55.0. Range (0,inf)
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			timeContinuity (int): time continuity cue (the maximum allowed gap duration for a pitch contour) [ms]. Defaults to 100. Range (0,inf)
			voiceVibrato (bool): detect voice vibrato. Defaults to False. Range {true,false}
			voicingTolerance (float): allowed deviation below the average contour mean salience of all contours (fraction of the standard deviation). Defaults to 0.2. Range [-1.0,1.4] 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input signal. Defaults to None. 
		Returns:
			pitch (NDArray[np.float32]): the estimated pitch values [Hz]
			pitchConfidence (NDArray[np.float32]): confidence with which the pitch was detected
		""" 
		... 


class RMS(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Computes the root mean square (quadratic mean) of an array.
		
		RMS is not defined for empty arrays. In such case, an exception will be thrown
		.
		References:
		  [1] Root mean square - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Root_mean_square		""" 
		... 
	def __call__(self, array:NDArray[np.float32]) -> float:
		"""compute
		Args:
			array (NDArray[np.float32]): the input array. Defaults to None. 
		Returns:
			rms (float): the root mean square of the input array
		""" 
		... 


class RawMoments(_essentia.Algorithm): 
	def __init__(self, range:float=22050.0) -> None:
		"""Computes the first 5 raw moments of an array.
		
		The output array is of size 6 because the zero-ith moment is used for padding so that the first moment corresponds to index 1.
		
		Note:
		  This algorithm has a range parameter, which usually represents a frequency (results will range from 0 to range). For a spectral centroid, the range should be equal to samplerate / 2. For an audio centroid, the frequency range should be equal to (audio_size-1) / samplerate.
		
		An exception is thrown if the input array's size is smaller than 2.
		
		References:
		  [1] Raw Moment -- from Wolfram MathWorld,
		  http://mathworld.wolfram.com/RawMoment.html

		Args:
			range (float): the range of the input array, used for normalizing the results. Defaults to 22050.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, array:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			array (NDArray[np.float32]): the input array. Defaults to None. 
		Returns:
			rawMoments (NDArray[np.float32]): the (raw) moments of the input array
		""" 
		... 


class ReplayGain(_essentia.Algorithm): 
	def __init__(self, sampleRate:float=44100.0) -> None:
		"""Computes the Replay Gain loudness value of an audio signal.
		
		The algorithm is described in detail in [1]. The value returned is the 'standard' ReplayGain value, not the value with 6dB preamplification as computed by lame, mp3gain, vorbisgain, and all widely used ReplayGain programs.
		
		This algorithm is only defined for input signals which size is larger than 0.05ms, otherwise an exception will be thrown.
		
		As a pre-processing step, the algorithm applies equal-loudness filtering to the input signal. This is always done in the standard mode, but it can be turned off in the streaming mode, which is useful in the case one already has an equal-loudness filtered signal.
		
		References:
		  [1] ReplayGain 1.0 specification, https://wiki.hydrogenaud.io/index.php?title=ReplayGain_1.0_specification
		

		Args:
			sampleRate (float): the sampling rate of the input audio signal [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> float:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input audio signal (must be longer than 0.05ms). Defaults to None. 
		Returns:
			replayGain (float): the distance to the suitable average replay level (~-31dbB) defined by SMPTE [dB]
		""" 
		... 


class Resample(_essentia.Algorithm): 
	def __init__(self, inputSampleRate:float=44100.0, outputSampleRate:float=44100.0, quality:int=1) -> None:
		"""Resamples the input signal to the desired sampling rate.
		
		The quality of conversion is documented in [3].
		
		This algorithm is only supported if essentia has been compiled with Real=float, otherwise it will throw an exception. It may also throw an exception if there is an internal error in the SRC library during conversion.
		
		References:
		  [1] Secret Rabbit Code, http://www.mega-nerd.com/SRC
		
		  [2] Resampling - Wikipedia, the free encyclopedia
		  http://en.wikipedia.org/wiki/Resampling
		
		  [3] http://www.mega-nerd.com/SRC/api_misc.html#Converters

		Args:
			inputSampleRate (float): the sampling rate of the input signal [Hz]. Defaults to 44100.0. Range (0,inf)
			outputSampleRate (float): the sampling rate of the output signal [Hz]. Defaults to 44100.0. Range (0,inf)
			quality (int): the quality of the conversion, 0 for best quality, 4 for fast linear approximation. Defaults to 1. Range [0,4] 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input signal. Defaults to None. 
		Returns:
			signal (NDArray[np.float32]): the resampled signal
		""" 
		... 


class ResampleFFT(_essentia.Algorithm): 
	def __init__(self, inSize:int=128, outSize:int=128) -> None:
		"""Resamples a sequence using FFT/IFFT.
		
		The input and output sizes must be an even number. The algorithm is a counterpart of the resample function in SciPy.

		Args:
			inSize (int): the size of the input sequence. It needs to be even-sized.. Defaults to 128. Range [1,inf)
			outSize (int): the size of the output sequence. It needs to be even-sized.. Defaults to 128. Range [1,inf) 
		""" 
		... 
	def __call__(self, input:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			input (NDArray[np.float32]): input array. Defaults to None. 
		Returns:
			output (NDArray[np.float32]): output resample array
		""" 
		... 


class RhythmDescriptors(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Computes rhythm features (bpm, beat positions, beat histogram peaks) for an audio signal.
		
		It combines RhythmExtractor2013 for beat tracking and BPM estimation with BpmHistogramDescriptors algorithms.		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> tuple[NDArray[np.float32], float, float, NDArray[np.float32], NDArray[np.float32], float, float, float, float, float, float, NDArray[np.float32]]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the audio input signal. Defaults to None. 
		Returns:
			beats_position (NDArray[np.float32]): See RhythmExtractor2013 algorithm documentation
			confidence (float): See RhythmExtractor2013 algorithm documentation
			bpm (float): See RhythmExtractor2013 algorithm documentation
			bpm_estimates (NDArray[np.float32]): See RhythmExtractor2013 algorithm documentation
			bpm_intervals (NDArray[np.float32]): See RhythmExtractor2013 algorithm documentation
			first_peak_bpm (float): See BpmHistogramDescriptors algorithm documentation
			first_peak_spread (float): See BpmHistogramDescriptors algorithm documentation
			first_peak_weight (float): See BpmHistogramDescriptors algorithm documentation
			second_peak_bpm (float): See BpmHistogramDescriptors algorithm documentation
			second_peak_spread (float): See BpmHistogramDescriptors algorithm documentation
			second_peak_weight (float): See BpmHistogramDescriptors algorithm documentation
			histogram (NDArray[np.float32]): bpm histogram [bpm]
		""" 
		... 


class RhythmExtractor(_essentia.Algorithm): 
	def __init__(self, frameHop:int=1024, frameSize:int=1024, hopSize:int=256, lastBeatInterval:float=0.1, maxTempo:int=208, minTempo:int=40, numberFrames:int=1024, sampleRate:float=44100.0, tempoHints:NDArray[np.float32]=np.array([]), tolerance:float=0.24, useBands:bool=True, useOnset:bool=True) -> None:
		"""Estimates the tempo in bpm and beat positions given an audio signal.
		
		The algorithm combines several periodicity functions and estimates beats using TempoTap and TempoTapTicks. It combines:
		- onset detection functions based on high-frequency content (see OnsetDetection)
		- complex-domain spectral difference function (see OnsetDetection)
		- periodicity function based on energy bands (see FrequencyBands, TempoScaleBands)
		
		Note that this algorithm is outdated in terms of beat tracking accuracy, and it is highly recommended to use RhythmExtractor2013 instead.
		
		Quality: outdated (use RhythmExtractor2013 instead).
		
		An exception is thrown if neither "useOnset" nor "useBands" are enabled (i.e. set to true).

		Args:
			frameHop (int): the number of feature frames separating two evaluations. Defaults to 1024. Range (0,inf)
			frameSize (int): the number audio samples used to compute a feature. Defaults to 1024. Range (0,inf)
			hopSize (int): the number of audio samples per features. Defaults to 256. Range (0,inf)
			lastBeatInterval (float): the minimum interval between last beat and end of file [s]. Defaults to 0.1. Range [0,inf)
			maxTempo (int): the fastest tempo to detect [bpm]. Defaults to 208. Range [60,250]
			minTempo (int): the slowest tempo to detect [bpm]. Defaults to 40. Range [40,180]
			numberFrames (int): the number of feature frames to buffer on. Defaults to 1024. Range (0,inf)
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			tempoHints (NDArray[np.float32]): the optional list of initial beat locations, to favor the detection of pre-determined tempo period and beats alignment [s]. Defaults to np.array([]). Range None
			tolerance (float): the minimum interval between two consecutive beats [s]. Defaults to 0.24. Range [0,inf)
			useBands (bool): whether or not to use band energy as periodicity function. Defaults to True. Range {true,false}
			useOnset (bool): whether or not to use onsets as periodicity function. Defaults to True. Range {true,false} 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> tuple[float, NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the audio input signal. Defaults to None. 
		Returns:
			bpm (float): the tempo estimation [bpm]
			ticks (NDArray[np.float32]):  the estimated tick locations [s]
			estimates (NDArray[np.float32]): the bpm estimation per frame [bpm]
			bpmIntervals (NDArray[np.float32]): list of beats interval [s]
		""" 
		... 


class RhythmExtractor2013(_essentia.Algorithm): 
	def __init__(self, maxTempo:int=208, method:str='multifeature', minTempo:int=40) -> None:
		"""Extracts the beat positions and estimates their confidence as well as tempo in bpm for an audio signal.
		
		The beat locations can be computed using:
		  - 'multifeature', the BeatTrackerMultiFeature algorithm
		  - 'degara', the BeatTrackerDegara algorithm (note that there is no confidence estimation for this method, the output confidence value is always 0)
		
		See BeatTrackerMultiFeature and BeatTrackerDegara algorithms for more details.
		
		Note that the algorithm requires the sample rate of the input signal to be 44100 Hz in order to work correctly.
		

		Args:
			maxTempo (int): the fastest tempo to detect [bpm]. Defaults to 208. Range [60,250]
			method (str): the method used for beat tracking. Defaults to 'multifeature'. Range {multifeature,degara}
			minTempo (int): the slowest tempo to detect [bpm]. Defaults to 40. Range [40,180] 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> tuple[float, NDArray[np.float32], float, NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the audio input signal. Defaults to None. 
		Returns:
			bpm (float): the tempo estimation [bpm]
			ticks (NDArray[np.float32]):  the estimated tick locations [s]
			confidence (float): confidence with which the ticks are detected (ignore this value if using 'degara' method)
			estimates (NDArray[np.float32]): the list of bpm estimates characterizing the bpm distribution for the signal [bpm]
			bpmIntervals (NDArray[np.float32]): list of beats interval [s]
		""" 
		... 


class RhythmTransform(_essentia.Algorithm): 
	def __init__(self, frameSize:int=256, hopSize:int=32) -> None:
		"""Implements the rhythm transform.
		
		It computes a tempogram, a representation of rhythmic periodicities in the input signal in the rhythm domain, by using FFT similarly to computation of spectrum in the frequency domain [1]. Additional features, including rhythmic centroid and a rhythmic counterpart of MFCCs, can be derived from this rhythmic representation.
		
		The algorithm relies on a time sequence of frames of Mel bands energies as an input (see MelBands), but other types of frequency bands can be used as well (see BarkBands, ERBBands, FrequencyBands). For each band, the derivative of the frame to frame energy evolution is computed, and the periodicity of the resulting signal is computed: the signal is cut into frames of "frameSize" size and is analyzed with FFT. For each frame, the obtained power spectrums are summed across all bands forming a frame of rhythm transform values.
		
		Quality: experimental (non-reliable, poor accuracy according to tests on simple loops, more tests are necessary)
		
		References:
		  [1] E. Guaus and P. Herrera, "The rhythm transform: towards a generic
		  rhythm description," in International Computer Music Conference (ICMC’05),
		  2005.

		Args:
			frameSize (int): the frame size to compute the rhythm trasform. Defaults to 256. Range (0,inf)
			hopSize (int): the hop size to compute the rhythm transform. Defaults to 32. Range (0,inf) 
		""" 
		... 
	def __call__(self, melBands:np.ndarray) -> np.ndarray:
		"""compute
		Args:
			melBands (np.ndarray): the energies in the mel bands. Defaults to None. 
		Returns:
			rhythm (np.ndarray): consecutive frames in the rhythm domain
		""" 
		... 


class RollOff(_essentia.Algorithm): 
	def __init__(self, cutoff:float=0.85, sampleRate:float=44100.0) -> None:
		"""Computes the roll-off frequency of a spectrum.
		
		The roll-off frequency is defined as the frequency under which some percentage (cutoff) of the total energy of the spectrum is contained. The roll-off frequency can be used to distinguish between harmonic (below roll-off) and noisy sounds (above roll-off).
		
		An exception is thrown if the input audio spectrum is smaller than 2.
		References:
		  [1] G. Peeters, "A large set of audio features for sound description
		  (similarity and classification) in the CUIDADO project," CUIDADO I.S.T.
		  Project Report, 2004

		Args:
			cutoff (float): the ratio of total energy to attain before yielding the roll-off frequency. Defaults to 0.85. Range (0,1)
			sampleRate (float): the sampling rate of the audio signal (used to normalize rollOff) [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, spectrum:NDArray[np.float32]) -> float:
		"""compute
		Args:
			spectrum (NDArray[np.float32]): the input audio spectrum (must have more than one elements). Defaults to None. 
		Returns:
			rollOff (float): the roll-off frequency [Hz]
		""" 
		... 


class SBic(_essentia.Algorithm): 
	def __init__(self, cpw:float=1.5, inc1:int=60, inc2:int=20, minLength:int=10, size1:int=300, size2:int=200) -> None:
		"""Segments audio using the Bayesian Information Criterion given a matrix of frame features.
		
		The algorithm searches homogeneous segments for which the feature vectors have the same probability distribution based on the implementation in [1]. The input matrix is assumed to have features along dim1 (horizontal) while frames along dim2 (vertical).
		
		The segmentation is done in three phases: coarse segmentation, fine segmentation and segment validation. The first phase uses parameters 'size1' and 'inc1' to perform BIC segmentation. The second phase uses parameters 'size2' and 'inc2' to perform a local search for segmentation around the segmentation done by the first phase. Finally, the validation phase verifies that BIC differentials at segmentation points are positive as well as filters out any segments that are smaller than 'minLength'.
		
		Because this algorithm takes as input feature vectors of frames, all units are in terms of frames. For example, if a 44100Hz audio signal is segmented as [0, 99, 199] with a frame size of 1024 and a hopsize of 512, this means, in the time domain, that the audio signal is segmented at [0s, 99*512/44100s, 199*512/44100s].
		
		An exception is thrown if the input only contains one frame of features (i.e. second dimension is less than 2).
		
		References:
		  [1] Audioseg, http://audioseg.gforge.inria.fr
		
		  [2] G. Gravier, M. Betser, and M. Ben, Audio Segmentation Toolkit,
		  release 1.2, 2010. Available online:
		  https://gforge.inria.fr/frs/download.php/25187/audioseg-1.2.pdf
		

		Args:
			cpw (float): complexity penalty weight. Defaults to 1.5. Range [0,inf)
			inc1 (int): first pass increment [frames]. Defaults to 60. Range [1,inf)
			inc2 (int): second pass increment [frames]. Defaults to 20. Range [1,inf)
			minLength (int): minimum length of a segment [frames]. Defaults to 10. Range [1,inf)
			size1 (int): first pass window size [frames]. Defaults to 300. Range [1,inf)
			size2 (int): second pass window size [frames]. Defaults to 200. Range [1,inf) 
		""" 
		... 
	def __call__(self, features:np.ndarray) -> NDArray[np.float32]:
		"""compute
		Args:
			features (np.ndarray): extracted features matrix (rows represent features, and columns represent frames of audio). Defaults to None. 
		Returns:
			segmentation (NDArray[np.float32]): a list of frame indices that indicate where a segment of audio begins/ends (the indices of the first and last frame are also added to the list at the beginning and end, respectively)
		""" 
		... 


class SNR(_essentia.Algorithm): 
	def __init__(self, MAAlpha:float=0.95, MMSEAlpha:float=0.98, NoiseAlpha:float=0.9, frameSize:int=512, noiseThreshold:float=-40.0, sampleRate:float=44100.0, useBroadbadNoiseCorrection:bool=True) -> None:
		"""Computes the SNR of the input audio in a frame-wise manner.
		
		The algorithm assumes that:
		
		- The noise is gaussian.
		- There is a region of noise (without signal) at the beginning of the stream in order to estimate the PSD of the noise [1].
		
		Once the noise PSD is estimated, the algorithm relies on the Ephraim-Malah [2] recursion to estimate the SNR for each frequency bin.
		
		The algorithm also returns an overall (a single value for the whole spectrum) SNR estimation and an averaged overall SNR estimation using Exponential Moving Average filtering.
		
		This algorithm throws a warning if less than 15 frames are used to estimate the noise PSD.
		
		References:
		
		1. Vaseghi, S. V. (2008). Advanced digital signal processing and noise reduction. John Wiley & Sons. Page 336.
		
		2. Ephraim, Y., & Malah, D. (1984). Speech enhancement using a minimum-mean square error short-time spectral amplitude estimator. IEEE Transactions on acoustics, speech, and signal processing, 32(6), 1109-1121.
		
		

		Args:
			MAAlpha (float): Alpha coefficient for the EMA SNR estimation [2]. Defaults to 0.95. Range [0,1]
			MMSEAlpha (float): Alpha coefficient for the MMSE estimation [1].. Defaults to 0.98. Range [0,1]
			NoiseAlpha (float): Alpha coefficient for the EMA noise estimation [2]. Defaults to 0.9. Range [0,1]
			frameSize (int): the size of the input frame. Defaults to 512. Range (1,inf)
			noiseThreshold (float): Threshold to detect frames without signal. Defaults to -40.0. Range (-inf,0]
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			useBroadbadNoiseCorrection (bool): flag to apply the -10 * log10(BW) broadband noise correction factor. Defaults to True. Range {true,false} 
		""" 
		... 
	def __call__(self, frame:NDArray[np.float32]) -> tuple[float, float, NDArray[np.float32]]:
		"""compute
		Args:
			frame (NDArray[np.float32]): the input audio frame. Defaults to None. 
		Returns:
			instantSNR (float): SNR value for the the current frame
			averagedSNR (float): averaged SNR through an Exponential Moving Average filter
			spectralSNR (NDArray[np.float32]): instant SNR for each frequency bin
		""" 
		... 


class SaturationDetector(_essentia.Algorithm): 
	def __init__(self, differentialThreshold:float=0.001, energyThreshold:float=-1.0, frameSize:int=512, hopSize:int=256, minimumDuration:float=0.005, sampleRate:float=44100.0) -> None:
		"""Outputs the staring/ending locations of the saturated regions in seconds.
		
		Saturated regions are found by means of a tripe criterion:
			 1. samples in a saturated region should have more energy than a given threshold.
			 2. the difference between the samples in a saturated region should be smaller than a given threshold.
			 3. the duration of the saturated region should be longer than a given threshold.
		
		note: The algorithm was designed for a framewise use and the returned timestamps are related to the first frame processed. Use reset() or configure() to restart the count.

		Args:
			differentialThreshold (float): minimum difference between contiguous samples of the salturated regions. Defaults to 0.001. Range [0,inf)
			energyThreshold (float): mininimum energy of the samples in the saturated regions [dB]. Defaults to -1.0. Range (-inf,0]
			frameSize (int): expected input frame size. Defaults to 512. Range (0,inf)
			hopSize (int): hop size used for the analysis. Defaults to 256. Range (0,inf)
			minimumDuration (float): minimum duration of the saturated regions [ms]. Defaults to 0.005. Range [0,inf)
			sampleRate (float): sample rate used for the analysis. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, frame:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			frame (NDArray[np.float32]): the input audio frame. Defaults to None. 
		Returns:
			starts (NDArray[np.float32]): starting times of the detected saturated regions [s]
			ends (NDArray[np.float32]): ending times of the detected saturated regions [s]
		""" 
		... 


class Scale(_essentia.Algorithm): 
	def __init__(self, clipping:bool=True, factor:float=10.0, maxAbsValue:float=1.0) -> None:
		"""Scales the audio by the specified factor using clipping if required.

		Args:
			clipping (bool): boolean flag whether to apply clipping or not. Defaults to True. Range {true,false}
			factor (float): the multiplication factor by which the audio will be scaled. Defaults to 10.0. Range [0,inf)
			maxAbsValue (float): the maximum value above which to apply clipping. Defaults to 1.0. Range [0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input audio signal. Defaults to None. 
		Returns:
			signal (NDArray[np.float32]): the output audio signal
		""" 
		... 


class SilenceRate(_essentia.Algorithm): 
	def __init__(self, thresholds:NDArray[np.float32]=np.array([])) -> None:
		"""Estimates if a frame is silent.
		
		Given a list of thresholds, this algorithm creates a equally-sized list of outputs and returns 1 on a given output whenever the instant power of the input frame is below the given output's respective threshold, and returns 0 otherwise. This is done for each frame with respect to all outputs. In other words, if a given frame's instant power is below several given thresholds, then each of the corresponding outputs will emit a 1.

		Args:
			thresholds (NDArray[np.float32]): the threshold values. Defaults to np.array([]). Range None 
		""" 
		... 
	def __call__(self, frame:NDArray[np.float32]) -> None:
		"""compute
		Args:
			frame (NDArray[np.float32]): the input frame. Defaults to None. 
		""" 
		... 


class SineModelAnal(_essentia.Algorithm): 
	def __init__(self, freqDevOffset:float=20.0, freqDevSlope:float=0.01, magnitudeThreshold:float=-74.0, maxFrequency:float=22050.0, maxPeaks:int=250, maxnSines:int=100, minFrequency:float=0.0, orderBy:str='frequency', sampleRate:float=44100.0) -> None:
		"""Computes the sine model analysis.
		
		It is recommended that the input "spectrum" be computed by the Spectrum algorithm. This algorithm uses PeakDetection. See documentation for possible exceptions and input requirements on input "spectrum".
		
		References:
		  [1] Peak Detection,
		  http://ccrma.stanford.edu/~jos/parshl/Peak_Detection_Steps_3.html

		Args:
			freqDevOffset (float): minimum frequency deviation at 0Hz. Defaults to 20.0. Range (0,inf)
			freqDevSlope (float): slope increase of minimum frequency deviation. Defaults to 0.01. Range (-inf,inf)
			magnitudeThreshold (float): peaks below this given threshold are not outputted. Defaults to -74.0. Range (-inf,inf)
			maxFrequency (float): the maximum frequency of the range to evaluate [Hz]. Defaults to 22050.0. Range (0,inf)
			maxPeaks (int): the maximum number of returned peaks. Defaults to 250. Range [1,inf)
			maxnSines (int): maximum number of sines per frame. Defaults to 100. Range (0,inf)
			minFrequency (float): the minimum frequency of the range to evaluate [Hz]. Defaults to 0.0. Range [0,inf)
			orderBy (str): the ordering type of the outputted peaks (ascending by frequency or descending by magnitude). Defaults to 'frequency'. Range {frequency,magnitude}
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, fft:np.ndarray) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			fft (np.ndarray): the input frame. Defaults to None. 
		Returns:
			frequencies (NDArray[np.float32]): the frequencies of the sinusoidal peaks [Hz]
			magnitudes (NDArray[np.float32]): the magnitudes of the sinusoidal peaks
			phases (NDArray[np.float32]): the phases of the sinusoidal peaks
		""" 
		... 


class SineModelSynth(_essentia.Algorithm): 
	def __init__(self, fftSize:int=2048, hopSize:int=512, sampleRate:float=44100.0) -> None:
		"""Computes the sine model synthesis from sine model analysis.

		Args:
			fftSize (int): the size of the output FFT frame (full spectrum size). Defaults to 2048. Range [1,inf)
			hopSize (int): the hop size between frames. Defaults to 512. Range [1,inf)
			sampleRate (float): the audio sampling rate [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, magnitudes:NDArray[np.float32], frequencies:NDArray[np.float32], phases:NDArray[np.float32]) -> np.ndarray:
		"""compute
		Args:
			magnitudes (NDArray[np.float32]): the magnitudes of the sinusoidal peaks. Defaults to None. 
			frequencies (NDArray[np.float32]): the frequencies of the sinusoidal peaks [Hz]. Defaults to None. 
			phases (NDArray[np.float32]): the phases of the sinusoidal peaks. Defaults to None. 
		Returns:
			fft (np.ndarray): the output FFT frame
		""" 
		... 


class SineSubtraction(_essentia.Algorithm): 
	def __init__(self, fftSize:int=512, hopSize:int=128, sampleRate:float=44100.0) -> None:
		"""Subtracts the sinusoids computed with the sine model analysis from an input audio signal.
		
		It ouputs an audio signal.

		Args:
			fftSize (int): the size of the FFT internal process (full spectrum size) and output frame. Minimum twice the hopsize.. Defaults to 512. Range [1,inf)
			hopSize (int): the hop size between frames. Defaults to 128. Range [1,inf)
			sampleRate (float): the audio sampling rate [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, frame:NDArray[np.float32], magnitudes:NDArray[np.float32], frequencies:NDArray[np.float32], phases:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			frame (NDArray[np.float32]): the input audio frame to subtract from. Defaults to None. 
			magnitudes (NDArray[np.float32]): the magnitudes of the sinusoidal peaks. Defaults to None. 
			frequencies (NDArray[np.float32]): the frequencies of the sinusoidal peaks [Hz]. Defaults to None. 
			phases (NDArray[np.float32]): the phases of the sinusoidal peaks. Defaults to None. 
		Returns:
			frame (NDArray[np.float32]): the output audio frame
		""" 
		... 


class SingleBeatLoudness(_essentia.Algorithm): 
	def __init__(self, beatDuration:float=0.05, beatWindowDuration:float=0.1, frequencyBands:NDArray[np.float32]=np.array([0, 200, 400, 800, 1600, 3200, 22000]), onsetStart:str='sumEnergy', sampleRate:float=44100.0) -> None:
		"""Computes the spectrum energy of a single beat across the whole frequency range and on each specified frequency band given an audio segment.
		
		It detects the onset of the beat within the input segment, computes spectrum on a window starting on this onset, and estimates energy (see Energy and EnergyBandRatio algorithms). The frequency bands used by default are: 0-200 Hz, 200-400 Hz, 400-800 Hz, 800-1600 Hz, 1600-3200 Hz, 3200-22000Hz, following E. Scheirer [1].
		
		This algorithm throws an exception either when parameter beatDuration is larger than beatWindowSize or when the size of the input beat segment is less than beatWindowSize plus beatDuration.
		
		References:
		  [1] E. D. Scheirer, "Tempo and beat analysis of acoustic musical signals,"
		  The Journal of the Acoustical Society of America, vol. 103, p. 588, 1998.
		

		Args:
			beatDuration (float): window size for the beat's energy computation (the window starts at the onset) [s]. Defaults to 0.05. Range (0,inf)
			beatWindowDuration (float): window size for the beat's onset detection [s]. Defaults to 0.1. Range (0,inf)
			frequencyBands (NDArray[np.float32]): frequency bands. Defaults to np.array([0, 200, 400, 800, 1600, 3200, 22000]). Range None
			onsetStart (str): criteria for finding the start of the beat. Defaults to 'sumEnergy'. Range {sumEnergy,peakEnergy}
			sampleRate (float): the audio sampling rate [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, beat:NDArray[np.float32]) -> tuple[float, NDArray[np.float32]]:
		"""compute
		Args:
			beat (NDArray[np.float32]): audio segement containing a beat. Defaults to None. 
		Returns:
			loudness (float): the beat's energy across the whole spectrum
			loudnessBandRatio (NDArray[np.float32]): the beat's energy ratio for each band
		""" 
		... 


class SingleGaussian(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Estimates the single gaussian distribution for a matrix of feature vectors.
		
		For example, using the single gaussian on descriptors like MFCC with the symmetric Kullback-Leibler divergence might be a much better option than just the mean and variance of the descriptors over a whole signal.
		
		An exception is thrown if the covariance of the input matrix is singular or if the input matrix is empty.
		
		References:
		  [1] E. Pampalk, "Computational models of music similarity and their
		  application in music information retrieval,” Vienna University of
		  Technology, 2006.		""" 
		... 
	def __call__(self, matrix:np.ndarray) -> tuple[NDArray[np.float32], np.ndarray, np.ndarray]:
		"""compute
		Args:
			matrix (np.ndarray): the input data matrix (e.g. the MFCC descriptor over frames). Defaults to None. 
		Returns:
			mean (NDArray[np.float32]): the mean of the values
			covariance (np.ndarray): the covariance matrix
			inverseCovariance (np.ndarray): the inverse of the covariance matrix
		""" 
		... 


class Slicer(_essentia.Algorithm): 
	def __init__(self, endTimes:NDArray[np.float32]=np.array([]), sampleRate:float=44100.0, startTimes:NDArray[np.float32]=np.array([]), timeUnits:str='seconds') -> None:
		"""Splits an audio signal into segments given their start and end times.
		
		The parameters, "startTimes" and "endTimes" must be coherent. If these parameters differ in size, an exception is thrown. If a particular startTime is larger than its corresponding endTime, an exception is thrown.

		Args:
			endTimes (NDArray[np.float32]): the list of end times for the slices you want to extract. Defaults to np.array([]). Range None
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			startTimes (NDArray[np.float32]): the list of start times for the slices you want to extract. Defaults to np.array([]). Range None
			timeUnits (str): the units of time of the start and end times. Defaults to 'seconds'. Range {samples,seconds} 
		""" 
		... 
	def __call__(self, audio:NDArray[np.float32]) -> np.ndarray:
		"""compute
		Args:
			audio (NDArray[np.float32]): the input audio signal. Defaults to None. 
		Returns:
			frame (np.ndarray): the frames of the sliced input signal
		""" 
		... 


class SpectralCentroidTime(_essentia.Algorithm): 
	def __init__(self, sampleRate:float=44100.0) -> None:
		"""Computes the spectral centroid of a signal in time domain.
		
		A first difference filter is applied to the input signal. Then the centroid is computed by dividing the norm of the resulting signal by the norm of the input signal. The centroid is given in hertz.
		References:
		 [1] Udo Zölzer (2002). DAFX Digital Audio Effects pag.364-365
		

		Args:
			sampleRate (float): sampling rate of the input spectrum [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, array:NDArray[np.float32]) -> float:
		"""compute
		Args:
			array (NDArray[np.float32]): the input array. Defaults to None. 
		Returns:
			centroid (float): the spectral centroid of the signal
		""" 
		... 


class SpectralComplexity(_essentia.Algorithm): 
	def __init__(self, magnitudeThreshold:float=0.005, sampleRate:float=44100.0) -> None:
		"""Computes the spectral complexity of a spectrum.
		
		The spectral complexity is based on the number of peaks in the input spectrum.
		
		It is recommended that the input "spectrum" be computed by the Spectrum algorithm. The input "spectrum" is passed to the SpectralPeaks algorithm and thus inherits its input requirements and exceptions.
		References:
		  [1] C. Laurier, O. Meyers, J. Serrà, M. Blech, P. Herrera, and X. Serra,
		  "Indexing music by mood: design and integration of an automatic
		  content-based annotator," Multimedia Tools and Applications, vol. 48,
		  no. 1, pp. 161–184, 2009.
		

		Args:
			magnitudeThreshold (float): the minimum spectral-peak magnitude that contributes to spectral complexity. Defaults to 0.005. Range [0,inf)
			sampleRate (float): the audio sampling rate [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, spectrum:NDArray[np.float32]) -> float:
		"""compute
		Args:
			spectrum (NDArray[np.float32]): the input spectrum. Defaults to None. 
		Returns:
			spectralComplexity (float): the spectral complexity of the input spectrum
		""" 
		... 


class SpectralContrast(_essentia.Algorithm): 
	def __init__(self, frameSize:int=2048, highFrequencyBound:float=11000.0, lowFrequencyBound:float=20.0, neighbourRatio:float=0.4, numberBands:int=6, sampleRate:float=22050.0, staticDistribution:float=0.15) -> None:
		"""Computes the Spectral Contrast feature of a spectrum.
		
		It is based on the Octave Based Spectral Contrast feature as described in [1]. The version implemented here is a modified version to improve discriminative power and robustness. The modifications are described in [2].
		
		References:
		  [1] D.-N. Jiang, L. Lu, H.-J. Zhang, J.-H. Tao, and L.-H. Cai, "Music type
		  classification by spectral contrast feature," in IEEE International
		  Conference on Multimedia and Expo (ICME’02), 2002, vol. 1, pp. 113–116.
		
		  [2] V. Akkermans, J. Serrà, and P. Herrera, "Shape-based spectral contrast
		  descriptor," in Sound and Music Computing Conference (SMC’09), 2009,
		  pp. 143–148.
		

		Args:
			frameSize (int): the size of the fft frames. Defaults to 2048. Range [2,inf)
			highFrequencyBound (float): the upper bound of the highest band. Defaults to 11000.0. Range (0,inf)
			lowFrequencyBound (float): the lower bound of the lowest band. Defaults to 20.0. Range (0,inf)
			neighbourRatio (float): the ratio of the bins in the sub band used to calculate the peak and valley. Defaults to 0.4. Range (0,1]
			numberBands (int): the number of bands in the filter. Defaults to 6. Range (0,inf)
			sampleRate (float): the sampling rate of the audio signal. Defaults to 22050.0. Range (0,inf)
			staticDistribution (float): the ratio of the bins to distribute equally. Defaults to 0.15. Range [0,1] 
		""" 
		... 
	def __call__(self, spectrum:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			spectrum (NDArray[np.float32]): the audio spectrum. Defaults to None. 
		Returns:
			spectralContrast (NDArray[np.float32]): the spectral contrast coefficients
			spectralValley (NDArray[np.float32]): the magnitudes of the valleys
		""" 
		... 


class SpectralPeaks(_essentia.Algorithm): 
	def __init__(self, magnitudeThreshold:float=0.0, maxFrequency:float=5000.0, maxPeaks:int=100, minFrequency:float=0.0, orderBy:str='frequency', sampleRate:float=44100.0) -> None:
		"""Extracts peaks from a spectrum.
		
		It is important to note that the peak algorithm is independent of an input that is linear or in dB, so one has to adapt the threshold to fit with the type of data fed to it. The algorithm relies on PeakDetection algorithm which is run with parabolic interpolation [1]. The exactness of the peak-searching depends heavily on the windowing type. It gives best results with dB input, a blackman-harris 92dB window and interpolation set to true. According to [1], spectral peak frequencies tend to be about twice as accurate when dB magnitude is used rather than just linear magnitude. For further information about the peak detection, see the description of the PeakDetection algorithm.
		
		It is recommended that the input "spectrum" be computed by the Spectrum algorithm. This algorithm uses PeakDetection. See documentation for possible exceptions and input requirements on input "spectrum".
		
		References:
		  [1] Peak Detection,
		  http://ccrma.stanford.edu/~jos/parshl/Peak_Detection_Steps_3.html

		Args:
			magnitudeThreshold (float): peaks below this given threshold are not outputted. Defaults to 0.0. Range (-inf,inf)
			maxFrequency (float): the maximum frequency of the range to evaluate [Hz]. Defaults to 5000.0. Range (0,inf)
			maxPeaks (int): the maximum number of returned peaks. Defaults to 100. Range [1,inf)
			minFrequency (float): the minimum frequency of the range to evaluate [Hz]. Defaults to 0.0. Range [0,inf)
			orderBy (str): the ordering type of the outputted peaks (ascending by frequency or descending by magnitude). Defaults to 'frequency'. Range {frequency,magnitude}
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, spectrum:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			spectrum (NDArray[np.float32]): the input spectrum. Defaults to None. 
		Returns:
			frequencies (NDArray[np.float32]): the frequencies of the spectral peaks [Hz]
			magnitudes (NDArray[np.float32]): the magnitudes of the spectral peaks
		""" 
		... 


class SpectralWhitening(_essentia.Algorithm): 
	def __init__(self, maxFrequency:float=5000.0, sampleRate:float=44100.0) -> None:
		"""Whitening of spectral peaks of a spectrum.
		
		The algorithm works in dB scale, but the conversion is done by the algorithm so input should be in linear scale. The concept of 'whitening' refers to 'white noise' or a non-zero flat spectrum. It first computes a spectral envelope similar to the 'true envelope' in [1], and then modifies the amplitude of each peak relative to the envelope. For example, the predominant peaks will have a value close to 0dB because they are very close to the envelope. On the other hand, minor peaks between significant peaks will have lower amplitudes such as -30dB.
		
		The input "frequencies" and "magnitudes" can be computed using the SpectralPeaks algorithm.
		
		An exception is thrown if the input frequency and magnitude input vectors are of different size.
		
		References:
		  [1] A. Röbel and X. Rodet, "Efficient spectral envelope estimation and its
		  application to pitch shifting and envelope preservation," in International
		  Conference on Digital Audio Effects (DAFx’05), 2005.

		Args:
			maxFrequency (float): max frequency to apply whitening to [Hz]. Defaults to 5000.0. Range (0,inf)
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, spectrum:NDArray[np.float32], frequencies:NDArray[np.float32], magnitudes:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			spectrum (NDArray[np.float32]): the audio linear spectrum. Defaults to None. 
			frequencies (NDArray[np.float32]): the spectral peaks' linear frequencies. Defaults to None. 
			magnitudes (NDArray[np.float32]): the spectral peaks' linear magnitudes. Defaults to None. 
		Returns:
			magnitudes (NDArray[np.float32]): the whitened spectral peaks' linear magnitudes
		""" 
		... 


class Spectrum(_essentia.Algorithm): 
	def __init__(self, size:int=2048) -> None:
		"""Computes the magnitude spectrum of an array of Reals.
		
		The resulting magnitude spectrum has a size which is half the size of the input array plus one. Bins contain raw (linear) magnitude values.
		
		References:
		  [1] Frequency spectrum - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Frequency_spectrum

		Args:
			size (int): the expected size of the input audio signal (this is an optional parameter to optimize memory allocation). Defaults to 2048. Range [1,inf) 
		""" 
		... 
	def __call__(self, frame:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			frame (NDArray[np.float32]): the input audio frame. Defaults to None. 
		Returns:
			spectrum (NDArray[np.float32]): magnitude spectrum of the input audio signal
		""" 
		... 


class SpectrumCQ(_essentia.Algorithm): 
	def __init__(self, binsPerOctave:int=12, minFrequency:float=32.7, minimumKernelSize:int=4, numberBins:int=84, sampleRate:float=44100.0, scale:float=1.0, threshold:float=0.01, windowType:str='hann', zeroPhase:bool=True) -> None:
		"""Computes the magnitude of the Constant-Q spectrum.
		
		See ConstantQ algorithm for more details.
		

		Args:
			binsPerOctave (int): number of bins per octave. Defaults to 12. Range [1,inf)
			minFrequency (float): minimum frequency [Hz]. Defaults to 32.7. Range [1,inf)
			minimumKernelSize (int): minimum size allowed for frequency kernels. Defaults to 4. Range [2,inf)
			numberBins (int): number of frequency bins, starting at minFrequency. Defaults to 84. Range [1,inf)
			sampleRate (float): FFT sampling rate [Hz]. Defaults to 44100.0. Range [0,inf)
			scale (float): filters scale. Larger values use longer windows. Defaults to 1.0. Range [0,inf)
			threshold (float): bins whose magnitude is below this quantile are discarded. Defaults to 0.01. Range [0,1)
			windowType (str): the window type. Defaults to 'hann'. Range {hamming,hann,hannnsgcq,triangular,square,blackmanharris62,blackmanharris70,blackmanharris74,blackmanharris92}
			zeroPhase (bool): a boolean value that enables zero-phase windowing. Input audio frames should be windowed with the same phase mode. Defaults to True. Range {true,false} 
		""" 
		... 
	def __call__(self, frame:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			frame (NDArray[np.float32]): the input audio frame. Defaults to None. 
		Returns:
			spectrumCQ (NDArray[np.float32]): the magnitude constant-Q spectrum
		""" 
		... 


class SpectrumToCent(_essentia.Algorithm): 
	def __init__(self, bands:int=720, centBinResolution:float=10.0, inputSize:int=32768, log:bool=True, minimumFrequency:float=164.0, normalize:str='unit_sum', sampleRate:float=44100.0, type:str='power') -> None:
		"""Computes energy in triangular frequency bands of a spectrum equally spaced on the cent scale.
		
		Each band is computed to have a constant wideness in the cent scale. For each band the power-spectrum (mag-squared) is summed.
		
		Parameter "centBinResolution" should be and integer greater than 1, otherwise an exception will be thrown. TriangularBands is only defined for spectrum, which size is greater than 1.
		

		Args:
			bands (int): number of bins to compute. Default is 720 (6 octaves with the default 'centBinResolution'). Defaults to 720. Range [1,inf)
			centBinResolution (float): Width of each band in cents. Default is 10 cents. Defaults to 10.0. Range (0,inf)
			inputSize (int): the size of the spectrum. Defaults to 32768. Range (1,inf)
			log (bool): compute log-energies (log2 (1 + energy)). Defaults to True. Range {true,false}
			minimumFrequency (float): central frequency of the first band of the bank [Hz]. Defaults to 164.0. Range (0,inf)
			normalize (str): use unit area or vertex equal to 1 triangles.. Defaults to 'unit_sum'. Range {unit_sum,unit_max}
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			type (str): use magnitude or power spectrum. Defaults to 'power'. Range {magnitude,power} 
		""" 
		... 
	def __call__(self, spectrum:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			spectrum (NDArray[np.float32]): the input spectrum (must be greater than size one). Defaults to None. 
		Returns:
			bands (NDArray[np.float32]): the energy in each band
			frequencies (NDArray[np.float32]): the central frequency of each band
		""" 
		... 


class Spline(_essentia.Algorithm): 
	def __init__(self, beta1:float=1.0, beta2:float=0.0, type:str='b', xPoints:NDArray[np.float32]=np.array([0, 1]), yPoints:NDArray[np.float32]=np.array([0, 1])) -> None:
		"""Piecewise spline of type b, beta or quadratic.
		
		The input value, i.e. the point at which the spline is to be evaluated typically should be between xPoins[0] and xPoinst[size-1]. If the value lies outside this range, extrapolation is used.
		Regarding spline types:
		  - B: evaluates a cubic B spline approximant.
		  - Beta: evaluates a cubic beta spline approximant. For beta splines parameters 'beta1' and 'beta2' can be supplied. For no bias set beta1 to 1 and for no tension set beta2 to 0. Note that if beta1=1 and beta2=0, the cubic beta becomes a cubic B spline. On the other hand if beta1=1 and beta2 is large the beta spline turns into a linear spline.
		  - Quadratic: evaluates a piecewise quadratic spline at a point. Note that size of input must be odd.
		
		References:
		  [1] Spline interpolation - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Spline_interpolation

		Args:
			beta1 (float): the skew or bias parameter (only available for type beta). Defaults to 1.0. Range [0,inf]
			beta2 (float): the tension parameter. Defaults to 0.0. Range [0,inf)
			type (str): the type of spline to be computed. Defaults to 'b'. Range {b,beta,quadratic}
			xPoints (NDArray[np.float32]): the x-coordinates where data is specified (the points must be arranged in ascending order and cannot contain duplicates). Defaults to np.array([0, 1]). Range None
			yPoints (NDArray[np.float32]): the y-coordinates to be interpolated (i.e. the known data). Defaults to np.array([0, 1]). Range None 
		""" 
		... 
	def __call__(self, x:float) -> float:
		"""compute
		Args:
			x (float): the input coordinate (x-axis). Defaults to None. 
		Returns:
			y (float): the value of the spline at x
		""" 
		... 


class SprModelAnal(_essentia.Algorithm): 
	def __init__(self, fftSize:int=2048, freqDevOffset:int=20, freqDevSlope:float=0.01, hopSize:int=512, magnitudeThreshold:float=0.0, maxFrequency:float=5000.0, maxPeaks:int=100, maxnSines:int=100, minFrequency:float=0.0, orderBy:str='frequency', sampleRate:float=44100.0) -> None:
		"""Computes the sinusoidal plus residual model analysis.
		
		It is recommended that the input "spectrum" be computed by the Spectrum algorithm. This algorithm uses SineModelAnal. See documentation for possible exceptions and input requirements on input "spectrum".
		
		References:
		  https://github.com/MTG/sms-tools
		  http://mtg.upf.edu/technologies/sms
		

		Args:
			fftSize (int): the size of the internal FFT size (full spectrum size). Defaults to 2048. Range [1,inf)
			freqDevOffset (int): minimum frequency deviation at 0Hz. Defaults to 20. Range (0,inf)
			freqDevSlope (float): slope increase of minimum frequency deviation. Defaults to 0.01. Range (-inf,inf)
			hopSize (int): the hop size between frames. Defaults to 512. Range [1,inf)
			magnitudeThreshold (float): peaks below this given threshold are not outputted. Defaults to 0.0. Range (-inf,inf)
			maxFrequency (float): the maximum frequency of the range to evaluate [Hz]. Defaults to 5000.0. Range (0,inf)
			maxPeaks (int): the maximum number of returned peaks. Defaults to 100. Range [1,inf)
			maxnSines (int): maximum number of sines per frame. Defaults to 100. Range (0,inf)
			minFrequency (float): the minimum frequency of the range to evaluate [Hz]. Defaults to 0.0. Range [0,inf)
			orderBy (str): the ordering type of the outputted peaks (ascending by frequency or descending by magnitude). Defaults to 'frequency'. Range {frequency,magnitude}
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, frame:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			frame (NDArray[np.float32]): the input frame. Defaults to None. 
		Returns:
			frequencies (NDArray[np.float32]): the frequencies of the sinusoidal peaks [Hz]
			magnitudes (NDArray[np.float32]): the magnitudes of the sinusoidal peaks
			phases (NDArray[np.float32]): the phases of the sinusoidal peaks
			res (NDArray[np.float32]): output residual frame
		""" 
		... 


class SprModelSynth(_essentia.Algorithm): 
	def __init__(self, fftSize:int=2048, hopSize:int=512, sampleRate:float=44100.0) -> None:
		"""Computes the sinusoidal plus residual model synthesis from SPS model analysis.

		Args:
			fftSize (int): the size of the output FFT frame (full spectrum size). Defaults to 2048. Range [1,inf)
			hopSize (int): the hop size between frames. Defaults to 512. Range [1,inf)
			sampleRate (float): the audio sampling rate [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, magnitudes:NDArray[np.float32], frequencies:NDArray[np.float32], phases:NDArray[np.float32], res:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			magnitudes (NDArray[np.float32]): the magnitudes of the sinusoidal peaks. Defaults to None. 
			frequencies (NDArray[np.float32]): the frequencies of the sinusoidal peaks [Hz]. Defaults to None. 
			phases (NDArray[np.float32]): the phases of the sinusoidal peaks. Defaults to None. 
			res (NDArray[np.float32]): the residual frame. Defaults to None. 
		Returns:
			frame (NDArray[np.float32]): the output audio frame of the Sinusoidal Plus Stochastic model
			sineframe (NDArray[np.float32]): the output audio frame for sinusoidal component 
			resframe (NDArray[np.float32]): the output audio frame for stochastic component 
		""" 
		... 


class SpsModelAnal(_essentia.Algorithm): 
	def __init__(self, fftSize:int=2048, freqDevOffset:int=20, freqDevSlope:float=0.01, hopSize:int=512, magnitudeThreshold:float=0.0, maxFrequency:float=5000.0, maxPeaks:int=100, maxnSines:int=100, minFrequency:float=0.0, orderBy:str='frequency', sampleRate:float=44100.0, stocf:float=0.2) -> None:
		"""Computes the stochastic model analysis.
		
		It is recommended that the input "spectrum" be computed by the Spectrum algorithm. This algorithm uses SineModelAnal. See documentation for possible exceptions and input requirements on input "spectrum".
		
		References:
		  https://github.com/MTG/sms-tools
		  http://mtg.upf.edu/technologies/sms
		

		Args:
			fftSize (int): the size of the internal FFT size (full spectrum size). Defaults to 2048. Range [1,inf)
			freqDevOffset (int): minimum frequency deviation at 0Hz. Defaults to 20. Range (0,inf)
			freqDevSlope (float): slope increase of minimum frequency deviation. Defaults to 0.01. Range (-inf,inf)
			hopSize (int): the hop size between frames. Defaults to 512. Range [1,inf)
			magnitudeThreshold (float): peaks below this given threshold are not outputted. Defaults to 0.0. Range (-inf,inf)
			maxFrequency (float): the maximum frequency of the range to evaluate [Hz]. Defaults to 5000.0. Range (0,inf)
			maxPeaks (int): the maximum number of returned peaks. Defaults to 100. Range [1,inf)
			maxnSines (int): maximum number of sines per frame. Defaults to 100. Range (0,inf)
			minFrequency (float): the minimum frequency of the range to evaluate [Hz]. Defaults to 0.0. Range [0,inf)
			orderBy (str): the ordering type of the outputted peaks (ascending by frequency or descending by magnitude). Defaults to 'frequency'. Range {frequency,magnitude}
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			stocf (float): decimation factor used for the stochastic approximation. Defaults to 0.2. Range (0,1] 
		""" 
		... 
	def __call__(self, frame:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			frame (NDArray[np.float32]): the input frame. Defaults to None. 
		Returns:
			frequencies (NDArray[np.float32]): the frequencies of the sinusoidal peaks [Hz]
			magnitudes (NDArray[np.float32]): the magnitudes of the sinusoidal peaks
			phases (NDArray[np.float32]): the phases of the sinusoidal peaks
			stocenv (NDArray[np.float32]): the stochastic envelope
		""" 
		... 


class SpsModelSynth(_essentia.Algorithm): 
	def __init__(self, fftSize:int=2048, hopSize:int=512, sampleRate:float=44100.0, stocf:float=0.2) -> None:
		"""Computes the sinusoidal plus stochastic model synthesis from SPS model analysis.

		Args:
			fftSize (int): the size of the output FFT frame (full spectrum size). Defaults to 2048. Range [1,inf)
			hopSize (int): the hop size between frames. Defaults to 512. Range [1,inf)
			sampleRate (float): the audio sampling rate [Hz]. Defaults to 44100.0. Range (0,inf)
			stocf (float): decimation factor used for the stochastic approximation. Defaults to 0.2. Range (0,1] 
		""" 
		... 
	def __call__(self, magnitudes:NDArray[np.float32], frequencies:NDArray[np.float32], phases:NDArray[np.float32], stocenv:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			magnitudes (NDArray[np.float32]): the magnitudes of the sinusoidal peaks. Defaults to None. 
			frequencies (NDArray[np.float32]): the frequencies of the sinusoidal peaks [Hz]. Defaults to None. 
			phases (NDArray[np.float32]): the phases of the sinusoidal peaks. Defaults to None. 
			stocenv (NDArray[np.float32]): the stochastic envelope. Defaults to None. 
		Returns:
			frame (NDArray[np.float32]): the output audio frame of the Sinusoidal Plus Stochastic model
			sineframe (NDArray[np.float32]): the output audio frame for sinusoidal component 
			stocframe (NDArray[np.float32]): the output audio frame for stochastic component 
		""" 
		... 


class StartStopCut(_essentia.Algorithm): 
	def __init__(self, frameSize:int=256, hopSize:int=256, maximumStartTime:float=10.0, maximumStopTime:float=10.0, sampleRate:float=44100.0, threshold:int=-60) -> None:
		"""Outputs if there is a cut at the beginning or at the end of the audio by locating the first and last non-silent frames and comparing their positions to the actual beginning and end of the audio.
		
		The input audio is considered to be cut at the beginning (or the end) and the corresponding flag is activated if the first (last) non-silent frame occurs before (after) the configurable time threshold.
		
		Notes: This algorithm is designed to operate on the entire (file) audio. In the streaming mode, use it in combination with RealAccumulator.
		The encoding/decoding process of lossy formats can introduce some padding at the beginning/end of the file. E.g. an MP3 file encoded and decoded with LAME using the default parameters will introduce a delay of 1104 samples [http://lame.sourceforge.net/tech-FAQ.txt]. In this case, the maximumStartTime can be increased by 1104 ÷ 44100 × 1000 = 25 ms to prevent misdetections.
		

		Args:
			frameSize (int): the frame size for the internal power analysis. Defaults to 256. Range (0,inf)
			hopSize (int): the hop size for the internal power analysis. Defaults to 256. Range (0,inf)
			maximumStartTime (float): if the first non-silent frame occurs before maximumStartTime startCut is activated [ms]. Defaults to 10.0. Range [0,inf)
			maximumStopTime (float): if the last non-silent frame occurs after maximumStopTime to the end stopCut is activated [ms]. Defaults to 10.0. Range [0,inf)
			sampleRate (float): the sample rate. Defaults to 44100.0. Range (0,inf)
			threshold (int): the threshold below which average energy is defined as silence [dB]. Defaults to -60. Range (-inf,0] 
		""" 
		... 
	def __call__(self, audio:NDArray[np.float32]) -> tuple[int, int]:
		"""compute
		Args:
			audio (NDArray[np.float32]): the input audio . Defaults to None. 
		Returns:
			startCut (int): 1 if there is a cut at the begining of the audio
			stopCut (int): 1 if there is a cut at the end of the audio
		""" 
		... 


class StartStopSilence(_essentia.Algorithm): 
	def __init__(self, threshold:int=-60) -> None:
		"""Outputs the frame at which sound begins and the frame at which sound ends.
		
		Note: In standard mode the algorithm is to be run iteratively on a sequence of frames. The outputs are updated on each iteration, and the final result is produced at the end of the sequence.

		Args:
			threshold (int): the threshold below which average energy is defined as silence [dB]. Defaults to -60. Range (-inf,0]) 
		""" 
		... 
	def __call__(self, frame:NDArray[np.float32]) -> tuple[int, int]:
		"""compute
		Args:
			frame (NDArray[np.float32]): the input audio frames. Defaults to None. 
		Returns:
			startFrame (int): number of the first non-silent frame
			stopFrame (int): number of the last non-silent frame
		""" 
		... 


class StereoDemuxer(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Outputs left and right channel separately given a stereo signal.
		
		If the signal is monophonic, it outputs a zero signal on the right channel.		""" 
		... 
	def __call__(self, audio:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			audio (NDArray[np.float32]): the audio signal. Defaults to None. 
		Returns:
			left (NDArray[np.float32]): the left channel of the audio signal
			right (NDArray[np.float32]): the right channel of the audio signal
		""" 
		... 


class StereoMuxer(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Outputs a stereo signal given left and right channel separately.		""" 
		... 
	def __call__(self, left:NDArray[np.float32], right:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			left (NDArray[np.float32]): the left channel of the audio signal. Defaults to None. 
			right (NDArray[np.float32]): the right channel of the audio signal. Defaults to None. 
		Returns:
			audio (NDArray[np.float32]): the audio signal
		""" 
		... 


class StereoTrimmer(_essentia.Algorithm): 
	def __init__(self, checkRange:bool=False, endTime:float=1000000.0, sampleRate:float=44100.0, startTime:float=0.0) -> None:
		"""Extracts a segment of a stereo audio signal given its start and end times.
		
		Giving "startTime" greater than "endTime" will raise an exception.

		Args:
			checkRange (bool): check whether the specified time range for a slice fits the size of input signal (throw exception if not). Defaults to False. Range {true,false}
			endTime (float): the end time of the slice you want to extract [s]. Defaults to 1000000.0. Range [0,inf)
			sampleRate (float): the sampling rate of the input audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			startTime (float): the start time of the slice you want to extract [s]. Defaults to 0.0. Range [0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input stereo signal. Defaults to None. 
		Returns:
			signal (NDArray[np.float32]): the trimmed stereo signal
		""" 
		... 


class StochasticModelAnal(_essentia.Algorithm): 
	def __init__(self, fftSize:int=2048, hopSize:int=512, sampleRate:float=44100.0, stocf:float=0.2) -> None:
		"""Computes the stochastic model analysis.
		
		It gets the resampled spectral envelope of the stochastic component.
		
		References:
		  https://github.com/MTG/sms-tools
		  http://mtg.upf.edu/technologies/sms
		

		Args:
			fftSize (int): the size of the internal FFT size (full spectrum size). Defaults to 2048. Range [1,inf)
			hopSize (int): the hop size between frames. Defaults to 512. Range [1,inf)
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			stocf (float): decimation factor used for the stochastic approximation. Defaults to 0.2. Range (0,1] 
		""" 
		... 
	def __call__(self, frame:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			frame (NDArray[np.float32]): the input frame. Defaults to None. 
		Returns:
			stocenv (NDArray[np.float32]): the stochastic envelope
		""" 
		... 


class StochasticModelSynth(_essentia.Algorithm): 
	def __init__(self, fftSize:int=2048, hopSize:int=512, sampleRate:float=44100.0, stocf:float=0.2) -> None:
		"""Computes the stochastic model synthesis.
		
		It generates the noisy spectrum from a resampled spectral envelope of the stochastic component.
		
		References:
		  https://github.com/MTG/sms-tools
		  http://mtg.upf.edu/technologies/sms
		

		Args:
			fftSize (int): the size of the internal FFT size (full spectrum size). Defaults to 2048. Range [1,inf)
			hopSize (int): the hop size between frames. Defaults to 512. Range [1,inf)
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			stocf (float): decimation factor used for the stochastic approximation. Defaults to 0.2. Range (0,1] 
		""" 
		... 
	def __call__(self, stocenv:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			stocenv (NDArray[np.float32]): the stochastic envelope input. Defaults to None. 
		Returns:
			frame (NDArray[np.float32]): the output frame
		""" 
		... 


class StrongDecay(_essentia.Algorithm): 
	def __init__(self, sampleRate:float=44100.0) -> None:
		"""Computes the Strong Decay of an audio signal.
		
		The Strong Decay is built from the non-linear combination of the signal energy and the signal temporal centroid, the latter being the balance of the absolute value of the signal. A signal containing a temporal centroid near its start boundary and a strong energy is said to have a strong decay.
		
		This algorithm returns 0.0 for zero signals (i.e. silence), and throws an exception when the signal's size is less than two as it can't compute its centroid.
		
		References:
		  [1] F. Gouyon and P. Herrera, "Exploration of techniques for automatic
		  labeling of audio drum tracks instruments," in MOSART: Workshop on Current
		  Directions in Computer Music, 2001.

		Args:
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> float:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input audio signal. Defaults to None. 
		Returns:
			strongDecay (float): the strong decay
		""" 
		... 


class StrongPeak(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Computes the Strong Peak of a spectrum.
		
		The Strong Peak is defined as the ratio between the spectrum's maximum peak's magnitude and the "bandwidth" of the peak above a threshold (half its amplitude). This ratio reveals whether the spectrum presents a very "pronounced" maximum peak (i.e. the thinner and the higher the maximum of the spectrum is, the higher the ratio value).
		
		Note that "bandwidth" is defined as the width of the peak in the log10-frequency domain. This is different than as implemented in [1]. Using the log10-frequency domain allows this algorithm to compare strong peaks at lower frequencies with those from higher frequencies.
		
		An exception is thrown if the input spectrum contains less than two elements.
		
		References:
		  [1] F. Gouyon and P. Herrera, "Exploration of techniques for automatic
		  labeling of audio drum tracks instruments,” in MOSART: Workshop on Current
		  Directions in Computer Music, 2001.		""" 
		... 
	def __call__(self, spectrum:NDArray[np.float32]) -> float:
		"""compute
		Args:
			spectrum (NDArray[np.float32]): the input spectrum (must be greater than one element and cannot contain negative values). Defaults to None. 
		Returns:
			strongPeak (float): the Strong Peak ratio
		""" 
		... 


class SuperFluxExtractor(_essentia.Algorithm): 
	def __init__(self, combine:float=20.0, frameSize:int=2048, hopSize:int=256, ratioThreshold:float=16.0, sampleRate:float=44100.0, threshold:float=0.05) -> None:
		"""Detects onsets given an audio signal using SuperFlux algorithm.
		
		This implementation is based on the available reference implementation in python [2]. The algorithm computes spectrum of the input signal, summarizes it into triangular band energies, and computes a onset detection function based on spectral flux tracking spectral trajectories with a maximum filter (SuperFluxNovelty). The peaks of the function are then detected (SuperFluxPeaks).
		
		References:
		  [1] Böck, S. and Widmer, G., Maximum Filter Vibrato Suppression for Onset
		  Detection, Proceedings of the 16th International Conference on Digital
		  Audio Effects (DAFx-13), 2013
		  [2] https://github.com/CPJKU/SuperFlux

		Args:
			combine (float): time threshold for double onsets detections (ms). Defaults to 20.0. Range (0,inf)
			frameSize (int): the frame size for computing low-level features. Defaults to 2048. Range (0,inf)
			hopSize (int): the hop size for computing low-level features. Defaults to 256. Range (0,inf)
			ratioThreshold (float): ratio threshold for peak picking with respect to novelty_signal/novelty_average rate, use 0 to disable it (for low-energy onsets). Defaults to 16.0. Range [0,inf)
			sampleRate (float): the audio sampling rate [Hz]. Defaults to 44100.0. Range (0,inf)
			threshold (float): threshold for peak peaking with respect to the difference between novelty_signal and average_signal (for onsets in ambient noise). Defaults to 0.05. Range [0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the audio input signal. Defaults to None. 
		Returns:
			onsets (NDArray[np.float32]): the onsets times
		""" 
		... 


class SuperFluxNovelty(_essentia.Algorithm): 
	def __init__(self, binWidth:int=3, frameWidth:int=2) -> None:
		"""Function for Superflux algorithm.
		
		See SuperFluxExtractor for more details.

		Args:
			binWidth (int): filter width (number of frequency bins). Defaults to 3. Range [3,inf)
			frameWidth (int): differentiation offset (compute the difference with the N-th previous frame). Defaults to 2. Range (0,inf) 
		""" 
		... 
	def __call__(self, bands:np.ndarray) -> float:
		"""compute
		Args:
			bands (np.ndarray): the input bands spectrogram. Defaults to None. 
		Returns:
			differences (float): SuperFlux novelty curve
		""" 
		... 


class SuperFluxPeaks(_essentia.Algorithm): 
	def __init__(self, combine:float=30.0, frameRate:float=172.0, pre_avg:float=100.0, pre_max:float=30.0, ratioThreshold:float=16.0, threshold:float=0.05) -> None:
		"""Detects peaks of an onset detection function computed by the SuperFluxNovelty algorithm.
		
		See SuperFluxExtractor for more details.

		Args:
			combine (float): time threshold for double onsets detections (ms). Defaults to 30.0. Range (0,inf)
			frameRate (float): frameRate. Defaults to 172.0. Range (0,inf)
			pre_avg (float): look back duration for moving average filter [ms]. Defaults to 100.0. Range (0,inf)
			pre_max (float): look back duration for moving maximum filter [ms]. Defaults to 30.0. Range (0,inf)
			ratioThreshold (float): ratio threshold for peak picking with respect to novelty_signal/novelty_average rate, use 0 to disable it (for low-energy onsets). Defaults to 16.0. Range [0,inf)
			threshold (float): threshold for peak peaking with respect to the difference between novelty_signal and average_signal (for onsets in ambient noise). Defaults to 0.05. Range [0,inf) 
		""" 
		... 
	def __call__(self, novelty:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			novelty (NDArray[np.float32]): the input onset detection function. Defaults to None. 
		Returns:
			peaks (NDArray[np.float32]): detected peaks' instants [s]
		""" 
		... 


class TCToTotal(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Calculates the ratio of the temporal centroid to the total length of a signal envelope.
		
		This ratio shows how the sound is 'balanced'. Its value is close to 0 if most of the energy lies at the beginning of the sound (e.g. decrescendo or impulsive sounds), close to 0.5 if the sound is symetric (e.g. 'delta unvarying' sounds), and close to 1 if most of the energy lies at the end of the sound (e.g. crescendo sounds).
		
		Please note that the TCToTotal ratio will return 0.5 for a zero signal (a signal consisting of only zeros) as 0.5 is the middle point of the signal. TCToTotal is not defined for a signal of less than 2 elements.An exception is thrown if the given envelope's size is not larger than 1.
		
		This algorithm is intended to be plugged after the Envelope algorithm		""" 
		... 
	def __call__(self, envelope:NDArray[np.float32]) -> float:
		"""compute
		Args:
			envelope (NDArray[np.float32]): the envelope of the signal (its length must be greater than 1. Defaults to None. 
		Returns:
			TCToTotal (float): the temporal centroid to total length ratio
		""" 
		... 


class TempoScaleBands(_essentia.Algorithm): 
	def __init__(self, bandsGain:NDArray[np.float32]=np.array([2, 3, 2, 1, 1.20000004768, 2, 3, 2.5]), frameTime:float=512.0) -> None:
		"""Computes features for tempo tracking to be used with the TempoTap algorithm.
		
		See standard_rhythmextractor_tempotap in examples folder.
		
		An exception is thrown if less than 1 band is given. An exception is also thrown if the there are not an equal number of bands given as band-gains given.
		
		Quality: outdated (the associated TempoTap algorithm is outdated, however it can be potentially used as an onset detection function for other tempo estimation algorithms although no evaluation has been done)
		
		References:
		  [1] Algorithm by Fabien Gouyon and Simon Dixon. There is no reference at
		  the time of this writing.
		

		Args:
			bandsGain (NDArray[np.float32]): gain for each bands. Defaults to np.array([2, 3, 2, 1, 1.20000004768, 2, 3, 2.5]). Range None
			frameTime (float): the frame rate in samples. Defaults to 512.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, bands:NDArray[np.float32]) -> tuple[NDArray[np.float32], float]:
		"""compute
		Args:
			bands (NDArray[np.float32]): the audio power spectrum divided into bands. Defaults to None. 
		Returns:
			scaledBands (NDArray[np.float32]): the output bands after scaling
			cumulativeBands (float): cumulative sum of the output bands before scaling
		""" 
		... 


class TempoTap(_essentia.Algorithm): 
	def __init__(self, frameHop:int=1024, frameSize:int=256, maxTempo:int=208, minTempo:int=40, numberFrames:int=1024, sampleRate:float=44100.0, tempoHints:NDArray[np.float32]=np.array([])) -> None:
		"""Estimates the periods and phases of a periodic signal, represented by a sequence of values of any number of detection functions, such as energy bands, onsets locations, etc.
		
		It requires to be sequentially run on a vector of such values ("featuresFrame") for each particular audio frame in order to get estimations related to that frames. The estimations are done for each detection function separately, utilizing the latest "frameHop" frames, including the present one, to compute autocorrelation. Empty estimations will be returned until enough frames are accumulated in the algorithm's buffer.
		The algorithm uses elements of the following beat-tracking methods:
		 - BeatIt, elaborated by Fabien Gouyon and Simon Dixon (input features) [1]
		 - Multi-comb filter with Rayleigh weighting, Mathew Davies [2]
		
		Parameter "maxTempo" should be 20bpm larger than "minTempo", otherwise an exception is thrown. The same applies for parameter "frameHop", which should not be greater than numberFrames. If the supplied "tempoHints" did not match any realistic bpm value, an exeception is thrown.
		
		This algorithm is thought to provide the input for TempoTapTicks algorithm. The "featureFrame" vectors can be formed by Multiplexer algorithm in the case of combining different features.
		
		Quality: outdated (use TempoTapDegara instead)
		
		References:
		  [1] F. Gouyon, "A computational approach to rhythm description: Audio
		  features for the computation of rhythm periodicity functions and their use
		  in tempo induction and music content processing," UPF, Barcelona, Spain,
		  2005.
		
		  [2] M. Davies and M. Plumbley, "Causal tempo tracking of audio," in
		  International Symposium on Music Information Retrieval (ISMIR'04), 2004.

		Args:
			frameHop (int): number of feature frames separating two evaluations. Defaults to 1024. Range (0,inf)
			frameSize (int): number of audio samples in a frame. Defaults to 256. Range (0,inf)
			maxTempo (int): fastest tempo allowed to be detected [bpm]. Defaults to 208. Range [60,250]
			minTempo (int): slowest tempo allowed to be detected [bpm]. Defaults to 40. Range [40,180]
			numberFrames (int): number of feature frames to buffer on. Defaults to 1024. Range (0,inf)
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			tempoHints (NDArray[np.float32]): optional list of initial beat locations, to favor the detection of pre-determined tempo period and beats alignment [s]. Defaults to np.array([]). Range None 
		""" 
		... 
	def __call__(self, featuresFrame:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			featuresFrame (NDArray[np.float32]): input temporal features of a frame. Defaults to None. 
		Returns:
			periods (NDArray[np.float32]): list of tempo estimates found for each input feature, in frames
			phases (NDArray[np.float32]): list of initial phase candidates found for each input feature, in frames
		""" 
		... 


class TempoTapDegara(_essentia.Algorithm): 
	def __init__(self, maxTempo:int=208, minTempo:int=40, resample:str='none', sampleRateODF:float=86.1328) -> None:
		"""Estimates beat positions given an onset detection function.
		
		The detection function is partitioned into 6-second frames with a 1.5-second increment, and the autocorrelation is computed for each frame, and is weighted by a tempo preference curve [2]. Periodicity estimations are done frame-wisely, searching for the best match with the Viterbi algorith [3]. The estimated periods are then passed to the probabilistic beat tracking algorithm [1], which computes beat positions.
		
		Note that the input values of the onset detection functions must be non-negative otherwise an exception is thrown. Parameter "maxTempo" should be 20bpm larger than "minTempo", otherwise an exception is thrown.
		
		References:
		  [1] Degara, N., Rua, E. A., Pena, A., Torres-Guijarro, S., Davies, M. E., & Plumbley, M. D. (2012). Reliability-informed beat tracking of musical signals. Audio, Speech, and Language Processing, IEEE Transactions on, 20(1), 290-301.
		  [2] Davies, M. E., & Plumbley, M. D. (2007). Context-dependent beat tracking of musical audio. Audio, Speech, and Language Processing, IEEE Transactions on, 15(3), 1009-1020.
		  [3] Stark, A. M., Davies, M. E., & Plumbley, M. D. (2009, September). Real-time beatsynchronous analysis of musical audio. In 12th International Conference on Digital Audio Effects (DAFx-09), Como, Italy.

		Args:
			maxTempo (int): fastest tempo allowed to be detected [bpm]. Defaults to 208. Range [60,250]
			minTempo (int): slowest tempo allowed to be detected [bpm]. Defaults to 40. Range [40,180]
			resample (str): use upsampling of the onset detection function (may increase accuracy). Defaults to 'none'. Range {none,x2,x3,x4}
			sampleRateODF (float): the sampling rate of the onset detection function [Hz]. Defaults to 86.1328. Range (0,inf) 
		""" 
		... 
	def __call__(self, onsetDetections:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			onsetDetections (NDArray[np.float32]): the input frame-wise vector of onset detection values. Defaults to None. 
		Returns:
			ticks (NDArray[np.float32]): the list of resulting ticks [s]
		""" 
		... 


class TempoTapMaxAgreement(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Outputs beat positions and confidence of their estimation based on the maximum mutual agreement between beat candidates estimated by different beat trackers (or using different features).
		
		Note that the input tick times should be in ascending order and that they cannot contain negative values otherwise an exception will be thrown.
		
		References:
		  [1] J. R. Zapata, A. Holzapfel, M. E. Davies, J. L. Oliveira, and
		  F. Gouyon, "Assigning a confidence threshold on automatic beat annotation
		  in large datasets," in International Society for Music Information
		  Retrieval Conference (ISMIR’12), 2012.
		
		  [2] A. Holzapfel, M. E. Davies, J. R. Zapata, J. L. Oliveira, and
		  F. Gouyon, "Selective sampling for beat tracking evaluation," IEEE
		  Transactions on Audio, Speech, and Language Processing, vol. 13, no. 9,
		  pp. 2539-2548, 2012.
				""" 
		... 
	def __call__(self, tickCandidates:np.ndarray) -> tuple[NDArray[np.float32], float]:
		"""compute
		Args:
			tickCandidates (np.ndarray): the tick candidates estimated using different beat trackers (or features) [s]. Defaults to None. 
		Returns:
			ticks (NDArray[np.float32]): the list of resulting ticks [s]
			confidence (float): confidence with which the ticks were detected [0, 5.32]
		""" 
		... 


class TempoTapTicks(_essentia.Algorithm): 
	def __init__(self, frameHop:int=512, hopSize:int=256, sampleRate:float=44100.0) -> None:
		"""Builds the list of ticks from the period and phase candidates given by the TempoTap algorithm.
		
		Quality: outdated (use TempoTapDegara instead)
		
		References:
		  [1] F. Gouyon, "A computational approach to rhythm description: Audio
		  features for the computation of rhythm periodicity functions and their use
		  in tempo induction and music content processing," UPF, Barcelona, Spain,
		  2005.
		

		Args:
			frameHop (int): number of feature frames separating two evaluations. Defaults to 512. Range (0,inf)
			hopSize (int): number of audio samples per features. Defaults to 256. Range (0,inf)
			sampleRate (float): sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, periods:NDArray[np.float32], phases:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			periods (NDArray[np.float32]): tempo period candidates for the current frame, in frames. Defaults to None. 
			phases (NDArray[np.float32]): tempo ticks phase candidates for the current frame, in frames. Defaults to None. 
		Returns:
			ticks (NDArray[np.float32]): the list of resulting ticks [s]
			matchingPeriods (NDArray[np.float32]): list of matching periods [s]
		""" 
		... 


class TensorNormalize(_essentia.Algorithm): 
	def __init__(self, axis:int=0, scaler:str='standard', skipConstantSlices:bool=True) -> None:
		"""Performs normalization over a tensor.
		
		When the axis parameter is set to -1 the input tensor is globally normalized. Any other value means that the tensor will be normalized along that axis.
		This algorithm supports Standard and MinMax normalizations.
		
		References:
		  [1] Feature scaling - Wikipedia, the free encyclopedia,
		  https://en.wikipedia.org/wiki/Feature_scaling

		Args:
			axis (int): Normalize along the given axis. -1 to normalize along all the dimensions. Defaults to 0. Range [-1,4)
			scaler (str): the type of the normalization to apply to input tensor. Defaults to 'standard'. Range {standard,minMax}
			skipConstantSlices (bool): Whether to prevent dividing by zero constant slices (zero standard deviation). Defaults to True. Range {false,true} 
		""" 
		... 
	def __call__(self, tensor:np.ndarray) -> np.ndarray:
		"""compute
		Args:
			tensor (np.ndarray): the input tensor. Defaults to None. 
		Returns:
			tensor (np.ndarray): the normalized output tensor
		""" 
		... 


class TensorTranspose(_essentia.Algorithm): 
	def __init__(self, permutation:np.ndarray) -> None:
		"""Performs transpositions over the axes of a tensor.
		
		

		Args:
			permutation (np.ndarray): permutation of [0,1,2,3]. The i'th dimension of the returned tensor will correspond to the dimension numbered permutation[i] of the input.. Defaults to None. Range None 
		""" 
		... 
	def __call__(self, tensor:np.ndarray) -> np.ndarray:
		"""compute
		Args:
			tensor (np.ndarray): the input tensor. Defaults to None. 
		Returns:
			tensor (np.ndarray): the transposed output tensor
		""" 
		... 


class TensorflowInputFSDSINet(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Computes mel bands from an audio frame with the specific parametrization required by the FSD-SINet models.
		
		References:
		  [1] Fonseca, E., Ferraro, A., & Serra, X. (2021). Improving sound event classification by increasing shift invariance in convolutional neural networks. arXiv preprint arXiv:2107.00623.
		  [2] https://github.com/edufonseca/shift_sec		""" 
		... 
	def __call__(self, frame:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			frame (NDArray[np.float32]): the audio frame. Defaults to None. 
		Returns:
			bands (NDArray[np.float32]): the log-compressed mel bands
		""" 
		... 


class TensorflowInputMusiCNN(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Computes mel-bands specific to the input of MusiCNN-based models.
		
		References:
		  [1] Pons, J., & Serra, X. (2019). musicnn: Pre-trained convolutional neural networks for music audio tagging. arXiv preprint arXiv:1909.06654.
		
		  [2] Supported models at https://essentia.upf.edu/models/		""" 
		... 
	def __call__(self, frame:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			frame (NDArray[np.float32]): the audio frame. Defaults to None. 
		Returns:
			bands (NDArray[np.float32]): the log compressed mel bands
		""" 
		... 


class TensorflowInputTempoCNN(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Computes mel-bands specific to the input of TempoCNN-based models.
		
		References:
		  [1] Hendrik Schreiber, Meinard Müller, A Single-Step Approach to Musical Tempo Estimation Using a Convolutional Neural Network Proceedings of the 19th International Society for Music Information Retrieval Conference (ISMIR), Paris, France, Sept. 2018.
		  [2] Hendrik Schreiber, Meinard Müller, Musical Tempo and Key Estimation using Convolutional Neural Networks with Directional Filters Proceedings of the Sound and Music Computing Conference (SMC), Málaga, Spain, 2019.
		  [3] Original models and code at https://github.com/hendriks73/tempo-cnn
		  [4] Supported models at https://essentia.upf.edu/models/		""" 
		... 
	def __call__(self, frame:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			frame (NDArray[np.float32]): the audio frame. Defaults to None. 
		Returns:
			bands (NDArray[np.float32]): the mel bands
		""" 
		... 


class TensorflowInputVGGish(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Computes mel-bands specific to the input of VGGish-based models.
		
		References:
		  [1] Gemmeke, J. et. al., AudioSet: An ontology and human-labelled dataset for audio events, ICASSP 2017
		
		  [2] Hershey, S. et. al., CNN Architectures for Large-Scale Audio Classification, ICASSP 2017
		
		  [3] Supported models at https://essentia.upf.edu/models/		""" 
		... 
	def __call__(self, frame:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			frame (NDArray[np.float32]): the audio frame. Defaults to None. 
		Returns:
			bands (NDArray[np.float32]): the log compressed mel bands
		""" 
		... 


class TonalExtractor(_essentia.Algorithm): 
	def __init__(self, frameSize:int=4096, hopSize:int=2048, tuningFrequency:float=440.0) -> None:
		"""Computes tonal features for an audio signal

		Args:
			frameSize (int): the framesize for computing tonal features. Defaults to 4096. Range (0,inf)
			hopSize (int): the hopsize for computing tonal features. Defaults to 2048. Range (0,inf)
			tuningFrequency (float): the tuning frequency of the input signal. Defaults to 440.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> tuple[float, NDArray[np.float32], str, float, list[str], str, NDArray[np.float32], np.ndarray, np.ndarray, str, str, float]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the audio input signal. Defaults to None. 
		Returns:
			chords_changes_rate (float): See ChordsDescriptors algorithm documentation
			chords_histogram (NDArray[np.float32]): See ChordsDescriptors algorithm documentation
			chords_key (str): See ChordsDescriptors algorithm documentation
			chords_number_rate (float): See ChordsDescriptors algorithm documentation
			chords_progression (list[str]): See ChordsDetection algorithm documentation
			chords_scale (str): See ChordsDetection algorithm documentation
			chords_strength (NDArray[np.float32]): See ChordsDetection algorithm documentation
			hpcp (np.ndarray): See HPCP algorithm documentation
			hpcp_highres (np.ndarray): See HPCP algorithm documentation
			key_key (str): See Key algorithm documentation
			key_scale (str): See Key algorithm documentation
			key_strength (float): See Key algorithm documentation
		""" 
		... 


class TonicIndianArtMusic(_essentia.Algorithm): 
	def __init__(self, binResolution:float=10.0, frameSize:int=2048, harmonicWeight:float=0.85, hopSize:int=512, magnitudeCompression:float=1.0, magnitudeThreshold:float=40.0, maxTonicFrequency:float=375.0, minTonicFrequency:float=100.0, numberHarmonics:int=20, numberSaliencePeaks:int=5, referenceFrequency:float=55.0, sampleRate:float=44100.0) -> None:
		"""Estimates the tonic frequency of the lead artist in Indian art music.
		
		It uses multipitch representation of the audio signal (pitch salience) to compute a histogram using which the tonic is identified as one of its peak. The decision is made based on the distance between the prominent peaks, the classification is done using a decision tree. An empty input signal will throw an exception. An exception will also be thrown if no predominant pitch salience peaks are detected within the maxTonicFrequency to minTonicFrequency range. 
		
		References:
		  [1] J. Salamon, S. Gulati, and X. Serra, "A Multipitch Approach to Tonic
		  Identification in Indian Classical Music," in International Society for
		  Music Information Retrieval Conference (ISMIR’12), 2012.

		Args:
			binResolution (float): salience function bin resolution [cents]. Defaults to 10.0. Range (0,inf)
			frameSize (int): the frame size for computing pitch saliecnce. Defaults to 2048. Range (0,inf)
			harmonicWeight (float): harmonic weighting parameter (weight decay ratio between two consequent harmonics, =1 for no decay). Defaults to 0.85. Range (0,1)
			hopSize (int): the hop size with which the pitch salience function was computed. Defaults to 512. Range (0,inf)
			magnitudeCompression (float): magnitude compression parameter (=0 for maximum compression, =1 for no compression). Defaults to 1.0. Range (0,1]
			magnitudeThreshold (float): peak magnitude threshold (maximum allowed difference from the highest peak in dBs). Defaults to 40.0. Range [0,inf)
			maxTonicFrequency (float): the maximum allowed tonic frequency [Hz]. Defaults to 375.0. Range [0,inf)
			minTonicFrequency (float): the minimum allowed tonic frequency [Hz]. Defaults to 100.0. Range [0,inf)
			numberHarmonics (int): number of considered hamonics. Defaults to 20. Range [1,inf)
			numberSaliencePeaks (int): number of top peaks of the salience function which should be considered for constructing histogram. Defaults to 5. Range [1,15]
			referenceFrequency (float): the reference frequency for Hertz to cent convertion [Hz], corresponding to the 0th cent bin. Defaults to 55.0. Range (0,inf)
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> float:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input signal. Defaults to None. 
		Returns:
			tonic (float): the estimated tonic frequency [Hz]
		""" 
		... 


class TriangularBands(_essentia.Algorithm): 
	def __init__(self, frequencyBands:NDArray[np.float32]=np.array([21.533203125, 43.06640625, 64.599609375, 86.1328125, 107.666015625, 129.19921875, 150.732421875, 172.265625, 193.798828125, 215.33203125, 236.865234375, 258.3984375, 279.931640625, 301.46484375, 322.998046875, 344.53125, 366.064453125, 387.59765625, 409.130859375, 430.6640625, 452.197265625, 473.73046875, 495.263671875, 516.796875, 538.330078125, 559.86328125, 581.396484375, 602.9296875, 624.462890625, 645.99609375, 667.529296875, 689.0625, 710.595703125, 732.12890625, 753.662109375, 775.1953125, 796.728515625, 839.794921875, 861.328125, 882.861328125, 904.39453125, 925.927734375, 968.994140625, 990.52734375, 1012.06054688, 1055.12695312, 1076.66015625, 1098.19335938, 1141.25976562, 1184.32617188, 1205.859375, 1248.92578125, 1270.45898438, 1313.52539062, 1356.59179688, 1399.65820312, 1442.72460938, 1485.79101562, 1528.85742188, 1571.92382812, 1614.99023438, 1658.05664062, 1701.12304688, 1765.72265625, 1808.7890625, 1873.38867188, 1916.45507812, 1981.0546875, 2024.12109375, 2088.72070312, 2153.3203125, 2217.91992188, 2282.51953125, 2347.11914062, 2411.71875, 2497.8515625, 2562.45117188, 2627.05078125, 2713.18359375, 2799.31640625, 2885.44921875, 2950.04882812, 3036.18164062, 3143.84765625, 3229.98046875, 3316.11328125, 3423.77929688, 3509.91210938, 3617.578125, 3725.24414062, 3832.91015625, 3940.57617188, 4069.77539062, 4177.44140625, 4306.640625, 4435.83984375, 4565.0390625, 4694.23828125, 4844.97070312, 4974.16992188, 5124.90234375, 5275.63476562, 5426.3671875, 5577.09960938, 5749.36523438, 5921.63085938, 6093.89648438, 6266.16210938, 6459.9609375, 6653.75976562, 6847.55859375, 7041.35742188, 7256.68945312, 7450.48828125, 7687.35351562, 7902.68554688, 8139.55078125, 8376.41601562, 8613.28125, 8871.6796875, 9130.078125, 9388.4765625, 9668.40820312, 9948.33984375, 10249.8046875, 10551.2695312, 10852.734375, 11175.7324219, 11498.7304688, 11843.2617188, 12187.7929688, 12553.8574219, 12919.921875, 13285.9863281, 13673.5839844, 14082.7148438, 14491.8457031, 14922.5097656, 15353.1738281, 15805.3710938, 16257.5683594]), inputSize:int=1025, log:bool=True, normalize:str='unit_sum', sampleRate:float=44100.0, type:str='power', weighting:str='linear') -> None:
		"""Computes energy in triangular frequency bands of a spectrum.
		
		The arbitrary number of overlapping bands can be specified. For each band the power-spectrum (mag-squared) is summed.
		
		Parameter "frequencyBands" must contain at least two frequencies, they all must be positive and must be ordered ascentdantly, otherwise an exception will be thrown. TriangularBands is only defined for spectrum, which size is greater than 1.
		

		Args:
			frequencyBands (NDArray[np.float32]): list of frequency ranges into which the spectrum is divided (these must be in ascending order and connot contain duplicates),each triangle is build as x(i-1)=0, x(i)=1, x(i+1)=0 over i, the resulting number of bands is size of input array - 2. Defaults to np.array([21.533203125, 43.06640625, 64.599609375, 86.1328125, 107.666015625, 129.19921875, 150.732421875, 172.265625, 193.798828125, 215.33203125, 236.865234375, 258.3984375, 279.931640625, 301.46484375, 322.998046875, 344.53125, 366.064453125, 387.59765625, 409.130859375, 430.6640625, 452.197265625, 473.73046875, 495.263671875, 516.796875, 538.330078125, 559.86328125, 581.396484375, 602.9296875, 624.462890625, 645.99609375, 667.529296875, 689.0625, 710.595703125, 732.12890625, 753.662109375, 775.1953125, 796.728515625, 839.794921875, 861.328125, 882.861328125, 904.39453125, 925.927734375, 968.994140625, 990.52734375, 1012.06054688, 1055.12695312, 1076.66015625, 1098.19335938, 1141.25976562, 1184.32617188, 1205.859375, 1248.92578125, 1270.45898438, 1313.52539062, 1356.59179688, 1399.65820312, 1442.72460938, 1485.79101562, 1528.85742188, 1571.92382812, 1614.99023438, 1658.05664062, 1701.12304688, 1765.72265625, 1808.7890625, 1873.38867188, 1916.45507812, 1981.0546875, 2024.12109375, 2088.72070312, 2153.3203125, 2217.91992188, 2282.51953125, 2347.11914062, 2411.71875, 2497.8515625, 2562.45117188, 2627.05078125, 2713.18359375, 2799.31640625, 2885.44921875, 2950.04882812, 3036.18164062, 3143.84765625, 3229.98046875, 3316.11328125, 3423.77929688, 3509.91210938, 3617.578125, 3725.24414062, 3832.91015625, 3940.57617188, 4069.77539062, 4177.44140625, 4306.640625, 4435.83984375, 4565.0390625, 4694.23828125, 4844.97070312, 4974.16992188, 5124.90234375, 5275.63476562, 5426.3671875, 5577.09960938, 5749.36523438, 5921.63085938, 6093.89648438, 6266.16210938, 6459.9609375, 6653.75976562, 6847.55859375, 7041.35742188, 7256.68945312, 7450.48828125, 7687.35351562, 7902.68554688, 8139.55078125, 8376.41601562, 8613.28125, 8871.6796875, 9130.078125, 9388.4765625, 9668.40820312, 9948.33984375, 10249.8046875, 10551.2695312, 10852.734375, 11175.7324219, 11498.7304688, 11843.2617188, 12187.7929688, 12553.8574219, 12919.921875, 13285.9863281, 13673.5839844, 14082.7148438, 14491.8457031, 14922.5097656, 15353.1738281, 15805.3710938, 16257.5683594]). Range None
			inputSize (int): the size of the spectrum. Defaults to 1025. Range (1,inf)
			log (bool): compute log-energies (log2 (1 + energy)). Defaults to True. Range {true,false}
			normalize (str): spectrum bin weights to use for each triangular band: 'unit_max' to make each triangle vertex equal to 1, 'unit_sum' to make each triangle area equal to 1 summing the actual weights of spectrum bins, 'unit_area' to make each triangle area equal to 1 normalizing the weights of each triangle by its bandwidth. Defaults to 'unit_sum'. Range {unit_sum,unit_tri,unit_max}
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			type (str): use magnitude or power spectrum. Defaults to 'power'. Range {magnitude,power}
			weighting (str): type of weighting function for determining triangle area. Defaults to 'linear'. Range {linear,slaneyMel,htkMel} 
		""" 
		... 
	def __call__(self, spectrum:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			spectrum (NDArray[np.float32]): the input spectrum (must be greater than size one). Defaults to None. 
		Returns:
			bands (NDArray[np.float32]): the energy in each band
		""" 
		... 


class TriangularBarkBands(_essentia.Algorithm): 
	def __init__(self, highFrequencyBound:float=22050.0, inputSize:int=1025, log:bool=False, lowFrequencyBound:float=0.0, normalize:str='unit_sum', numberBands:int=24, sampleRate:float=44100.0, type:str='power', weighting:str='warping') -> None:
		"""Computes energy in the bark bands of a spectrum.
		
		It is different to the regular BarkBands algorithm in that is more configurable so that it can be used in the BFCC algorithm to produce output similar to Rastamat (http://www.ee.columbia.edu/ln/rosa/matlab/rastamat/)
		See the BFCC algorithm documentation for more information as to why you might want to choose this over Mel frequency analysis
		It is recommended that the input "spectrum" be calculated by the Spectrum algorithm.
		
		

		Args:
			highFrequencyBound (float): an upper-bound limit for the frequencies to be included in the bands. Defaults to 22050.0. Range [0,inf)
			inputSize (int): the size of the spectrum. Defaults to 1025. Range (1,inf)
			log (bool): compute log-energies (log2 (1 + energy)). Defaults to False. Range {true,false}
			lowFrequencyBound (float): a lower-bound limit for the frequencies to be included in the bands. Defaults to 0.0. Range [0,inf)
			normalize (str): 'unit_max' makes the vertex of all the triangles equal to 1, 'unit_sum' makes the area of all the triangles equal to 1. Defaults to 'unit_sum'. Range {unit_sum,unit_max}
			numberBands (int): the number of output bands. Defaults to 24. Range (1,inf)
			sampleRate (float): the sample rate. Defaults to 44100.0. Range (0,inf)
			type (str): 'power' to output squared units, 'magnitude' to keep it as the input. Defaults to 'power'. Range {magnitude,power}
			weighting (str): type of weighting function for determining triangle area. Defaults to 'warping'. Range {warping,linear} 
		""" 
		... 
	def __call__(self, spectrum:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			spectrum (NDArray[np.float32]): the audio spectrum. Defaults to None. 
		Returns:
			bands (NDArray[np.float32]): the energy in bark bands
		""" 
		... 


class Trimmer(_essentia.Algorithm): 
	def __init__(self, checkRange:bool=False, endTime:float=1000000.0, sampleRate:float=44100.0, startTime:float=0.0) -> None:
		"""Extracts a segment of an audio signal given its start and end times.
		
		Giving "startTime" greater than "endTime" will raise an exception.

		Args:
			checkRange (bool): check whether the specified time range for a slice fits the size of input signal (throw exception if not). Defaults to False. Range {true,false}
			endTime (float): the end time of the slice you want to extract [s]. Defaults to 1000000.0. Range [0,inf)
			sampleRate (float): the sampling rate of the input audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			startTime (float): the start time of the slice you want to extract [s]. Defaults to 0.0. Range [0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input signal. Defaults to None. 
		Returns:
			signal (NDArray[np.float32]): the trimmed signal
		""" 
		... 


class Tristimulus(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Calculates the tristimulus of a signal given its harmonic peaks.
		
		The tristimulus has been introduced as a timbre equivalent to the color attributes in the vision. Tristimulus measures the mixture of harmonics in a given sound, grouped into three sections. The first tristimulus measures the relative weight of the first harmonic; the second tristimulus measures the relative weight of the second, third, and fourth harmonics taken together; and the third tristimulus measures the relative weight of all the remaining harmonics.
		
		Tristimulus is intended to be fed by the output of the HarmonicPeaks algorithm. The algorithm throws an exception when the input frequencies are not in ascending order and/or if the input vectors are of different sizes.
		
		References:
		  [1] Tristimulus (audio) - Wikipedia, the free encyclopedia
		  http://en.wikipedia.org/wiki/Tristimulus_%28audio%29
		
		  [2] G. Peeters, "A large set of audio features for sound description
		  (similarity and classification) in the CUIDADO project," CUIDADO I.S.T.
		  Project Report, 2004		""" 
		... 
	def __call__(self, frequencies:NDArray[np.float32], magnitudes:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			frequencies (NDArray[np.float32]): the frequencies of the harmonic peaks ordered by frequency. Defaults to None. 
			magnitudes (NDArray[np.float32]): the magnitudes of the harmonic peaks ordered by frequency. Defaults to None. 
		Returns:
			tristimulus (NDArray[np.float32]): a three-element vector that measures the mixture of harmonics of the given spectrum
		""" 
		... 


class TruePeakDetector(_essentia.Algorithm): 
	def __init__(self, blockDC:bool=False, emphasise:bool=False, oversamplingFactor:int=4, quality:int=1, sampleRate:float=44100.0, threshold:float=-0.0002, version:int=4) -> None:
		"""Implements a “true-peak” level meter for clipping detection.
		
		According to the ITU-R recommendations, “true-peak” values overcoming the full-scale range are potential sources of “clipping in subsequent processes, such as within particular D/A converters or during sample-rate conversion”.
		The ITU-R BS.1770-4[1] (by default) and the ITU-R BS.1770-2[2] signal-flows can be used. Go to the references for information about the differences.
		Only the peaks (if any) exceeding the configurable amplitude threshold are returned.
		Note: the parameters 'blockDC' and 'emphasise' work only when 'version' is set to 2.
		References:
		  [1] Series, B. S. (2011). Recommendation  ITU-R  BS.1770-4. Algorithms to measure audio programme loudness and true-peak audio level,
		  https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-4-201510-I!!PDF-E.pdf
		  [2] Series, B. S. (2011). Recommendation  ITU-R  BS.1770-2. Algorithms to measure audio programme loudness and true-peak audio level,
		  https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-2-201103-S!!PDF-E.pdf
		

		Args:
			blockDC (bool): flag to activate the optional DC blocker. Defaults to False. Range {true,false}
			emphasise (bool): flag to activate the optional emphasis filter. Defaults to False. Range {true,false}
			oversamplingFactor (int): times the signal is oversapled. Defaults to 4. Range [1,inf)
			quality (int): type of interpolation applied (see libresmple). Defaults to 1. Range [0,4]
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			threshold (float): threshold to detect peaks [dB]. Defaults to -0.0002. Range (-inf,inf)
			version (int): algorithm version. Defaults to 4. Range {2,4} 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input audio signal. Defaults to None. 
		Returns:
			peakLocations (NDArray[np.float32]): the peak locations in the ouput signal
			output (NDArray[np.float32]): the processed signal
		""" 
		... 


class TuningFrequency(_essentia.Algorithm): 
	def __init__(self, resolution:float=1.0) -> None:
		"""Estimates the tuning frequency give a sequence/set of spectral peaks.
		
		The result is the tuning frequency in Hz, and its distance from 440Hz in cents. This version is slightly adapted from the original algorithm [1], but gives the same results.
		
		Input vectors should have the same size, otherwise an exception is thrown. This algorithm should be given the outputs of the spectral peaks algorithm.
		
		Application: Western vs non-western music classification, key estimation, HPCP computation, tonal similarity.
		References:
		  [1] E. Gómez, "Key estimation from polyphonic audio," in Music Information
		  Retrieval Evaluation Exchange (MIREX’05), 2005.

		Args:
			resolution (float): resolution in cents (logarithmic scale, 100 cents = 1 semitone) for tuning frequency determination. Defaults to 1.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, frequencies:NDArray[np.float32], magnitudes:NDArray[np.float32]) -> tuple[float, float]:
		"""compute
		Args:
			frequencies (NDArray[np.float32]): the frequencies of the spectral peaks [Hz]. Defaults to None. 
			magnitudes (NDArray[np.float32]): the magnitudes of the spectral peaks. Defaults to None. 
		Returns:
			tuningFrequency (float): the tuning frequency [Hz]
			tuningCents (float): the deviation from 440 Hz (between -35 to 65 cents)
		""" 
		... 


class TuningFrequencyExtractor(_essentia.Algorithm): 
	def __init__(self, frameSize:int=4096, hopSize:int=2048) -> None:
		"""Extracts the tuning frequency of an audio signal

		Args:
			frameSize (int): the frameSize for computing tuning frequency. Defaults to 4096. Range (0,inf)
			hopSize (int): the hopsize for computing tuning frequency. Defaults to 2048. Range (0,inf) 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			signal (NDArray[np.float32]): the audio input signal. Defaults to None. 
		Returns:
			tuningFrequency (NDArray[np.float32]): the computed tuning frequency
		""" 
		... 


class UnaryOperator(_essentia.Algorithm): 
	def __init__(self, scale:float=1.0, shift:float=0.0, type:str='identity') -> None:
		"""Performs basic arithmetical operations element by element given an array.
		
		Note:
		  - log and ln are equivalent to the natural logarithm
		  - for log, ln, log10 and lin2db, x is clipped to 1e-30 for x<1e-30
		  - for x<0, sqrt(x) is invalid
		  - scale and shift parameters define linear transformation to be applied to the resulting elements

		Args:
			scale (float): multiply result by factor. Defaults to 1.0. Range (-inf,inf)
			shift (float): shift result by value (add value). Defaults to 0.0. Range (-inf,inf)
			type (str): the type of the unary operator to apply to input array. Defaults to 'identity'. Range {identity,abs,log10,log,ln,lin2db,db2lin,sin,cos,sqrt,square} 
		""" 
		... 
	def __call__(self, array:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			array (NDArray[np.float32]): the input array. Defaults to None. 
		Returns:
			array (NDArray[np.float32]): the input array transformed by unary operation
		""" 
		... 


class UnaryOperatorStream(_essentia.Algorithm): 
	def __init__(self, scale:float=1.0, shift:float=0.0, type:str='identity') -> None:
		"""Performs basic arithmetical operations element by element given an array.
		
		Note:
		  - log and ln are equivalent to the natural logarithm
		  - for log, ln, log10 and lin2db, x is clipped to 1e-30 for x<1e-30
		  - for x<0, sqrt(x) is invalid
		  - scale and shift parameters define linear transformation to be applied to the resulting elements

		Args:
			scale (float): multiply result by factor. Defaults to 1.0. Range (-inf,inf)
			shift (float): shift result by value (add value). Defaults to 0.0. Range (-inf,inf)
			type (str): the type of the unary operator to apply to input array. Defaults to 'identity'. Range {identity,abs,log10,log,ln,lin2db,db2lin,sin,cos,sqrt,square} 
		""" 
		... 
	def __call__(self, array:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			array (NDArray[np.float32]): the input array. Defaults to None. 
		Returns:
			array (NDArray[np.float32]): the input array transformed by unary operation
		""" 
		... 


class Variance(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Computes the variance of an array.		""" 
		... 
	def __call__(self, array:NDArray[np.float32]) -> float:
		"""compute
		Args:
			array (NDArray[np.float32]): the input array. Defaults to None. 
		Returns:
			variance (float): the variance of the input array
		""" 
		... 


class Vibrato(_essentia.Algorithm): 
	def __init__(self, maxExtend:float=250.0, maxFrequency:float=8.0, minExtend:float=50.0, minFrequency:float=4.0, sampleRate:float=344.531) -> None:
		"""Detects the presence of vibrato and estimates its parameters given a pitch contour [Hz].
		
		The result is the vibrato frequency in Hz and the extent (peak to peak) in cents. If no vibrato is detected in a frame, the output of both values is zero.
		
		This algorithm should be given the outputs of a pitch estimator, i.e. PredominantMelody, PitchYinFFT or PitchMelodia and the corresponding sample rate with which it was computed.
		
		The algorithm is an extended version of the vocal vibrato detection in PerdominantMelody.
		References:
		  [1] J. Salamon and E. Gómez, "Melody extraction from polyphonic music
		  signals using pitch contour characteristics," IEEE Transactions on Audio,
		  Speech, and Language Processing, vol. 20, no. 6, pp. 1759–1770, 2012.
		
		

		Args:
			maxExtend (float): maximum considered vibrato extent [cents]. Defaults to 250.0. Range (0,inf)
			maxFrequency (float): maximum considered vibrato frequency [Hz]. Defaults to 8.0. Range (0,inf)
			minExtend (float): minimum considered vibrato extent [cents]. Defaults to 50.0. Range (0,inf)
			minFrequency (float): minimum considered vibrato frequency [Hz]. Defaults to 4.0. Range (0,inf)
			sampleRate (float): sample rate of the input pitch contour. Defaults to 344.531. Range (0,inf) 
		""" 
		... 
	def __call__(self, pitch:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
		"""compute
		Args:
			pitch (NDArray[np.float32]): the pitch trajectory [Hz].. Defaults to None. 
		Returns:
			vibratoFrequency (NDArray[np.float32]): estimated vibrato frequency (or speed) [Hz]; zero if no vibrato was detected.
			vibratoExtend (NDArray[np.float32]): estimated vibrato extent (or depth) [cents]; zero if no vibrato was detected.
		""" 
		... 


class Viterbi(_essentia.Algorithm): 
	def __init__(self, ) -> None:
		"""Estimates the most-likely path by Viterbi algorithm.
		
		It is used in PitchYinProbabilistiesHMM algorithm.
		
		This Viterbi algorithm returns the most likely path. The internal variable calculation uses double for a better precision.
		
		References:
		  [1] M. Mauch and S. Dixon, "pYIN: A Fundamental Frequency Estimator
		  Using Probabilistic Threshold Distributions," in Proceedings of the
		  IEEE International Conference on Acoustics, Speech, and Signal Processing
		  (ICASSP 2014)Project Report, 2004		""" 
		... 
	def __call__(self, observationProbabilities:np.ndarray, initialization:NDArray[np.float32], fromIndex:np.ndarray, toIndex:np.ndarray, transitionProbabilities:NDArray[np.float32]) -> np.ndarray:
		"""compute
		Args:
			observationProbabilities (np.ndarray): the observation probabilities. Defaults to None. 
			initialization (NDArray[np.float32]): the initialization. Defaults to None. 
			fromIndex (np.ndarray): the transition matrix from index. Defaults to None. 
			toIndex (np.ndarray): the transition matrix to index. Defaults to None. 
			transitionProbabilities (NDArray[np.float32]): the transition probabilities matrix. Defaults to None. 
		Returns:
			path (np.ndarray): the decoded path
		""" 
		... 


class WarpedAutoCorrelation(_essentia.Algorithm): 
	def __init__(self, maxLag:int=1, sampleRate:float=44100.0) -> None:
		"""Computes the warped auto-correlation of an audio signal.
		
		The implementation is an adapted version of K. Schmidt's implementation of the matlab algorithm from the 'warped toolbox' by Aki Harma and Matti Karjalainen found [2]. For a detailed explanation of the algorithm, see [1].
		This algorithm is only defined for positive lambda = 1.0674*sqrt(2.0*atan(0.00006583*sampleRate)/PI) - 0.1916, thus it will throw an exception when the supplied sampling rate does not pass the requirements.
		If maxLag is larger than the size of the input array, an exception is thrown.
		
		References:
		  [1] A. Härmä, M. Karjalainen, L. Savioja, V. Välimäki, U. K. Laine, and
		  J. Huopaniemi, "Frequency-Warped Signal Processing for Audio Applications,"
		  JAES, vol. 48, no. 11, pp. 1011–1031, 2000.
		
		  [2] WarpTB - Matlab Toolbox for Warped DSP
		  http://www.acoustics.hut.fi/software/warp

		Args:
			maxLag (int): the maximum lag for which the auto-correlation is computed (inclusive) (must be smaller than signal size) . Defaults to 1. Range (0,inf)
			sampleRate (float): the audio sampling rate [Hz]. Defaults to 44100.0. Range (0,inf) 
		""" 
		... 
	def __call__(self, array:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			array (NDArray[np.float32]): the array to be analyzed. Defaults to None. 
		Returns:
			warpedAutoCorrelation (NDArray[np.float32]): the warped auto-correlation vector
		""" 
		... 


class Welch(_essentia.Algorithm): 
	def __init__(self, averagingFrames:int=10, fftSize:int=1024, frameSize:int=512, sampleRate:float=44100.0, scaling:str='density', windowType:str='hann') -> None:
		"""Estimates the Power Spectral Density of the input signal using the Welch's method [1].
		
		The input should be fed with the overlapped audio frames. The algorithm stores internally therequired past frames to compute each output. Call reset() to clear the buffers. This implentation is based on Scipy [2]
		
		References:
		  [1] The Welch's method - Wikipedia, the free encyclopedia,
		https://en.wikipedia.org/wiki/Welch%27s_method
		  [2] https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.welch.html

		Args:
			averagingFrames (int): amount of frames to average. Defaults to 10. Range (0,inf)
			fftSize (int): size of the FFT. Zero padding is added if this is larger the input frame size.. Defaults to 1024. Range (0,inf)
			frameSize (int): the expected size of the input audio signal (this is an optional parameter to optimize memory allocation). Defaults to 512. Range (0,inf)
			sampleRate (float): the sampling rate of the audio signal [Hz]. Defaults to 44100.0. Range (0,inf)
			scaling (str): 'density' normalizes the result to the bandwidth while 'power' outputs the unnormalized power spectrum. Defaults to 'density'. Range {density,power}
			windowType (str): the window type. Defaults to 'hann'. Range {hamming,hann,triangular,square,blackmanharris62,blackmanharris70,blackmanharris74,blackmanharris92} 
		""" 
		... 
	def __call__(self, frame:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			frame (NDArray[np.float32]): the input stereo audio signal. Defaults to None. 
		Returns:
			psd (NDArray[np.float32]): Power Spectral Density [dB] or [dB/Hz]
		""" 
		... 


class Windowing(_essentia.Algorithm): 
	def __init__(self, constantsDecimals:int=5, normalized:bool=True, size:int=1024, splitPadding:bool=False, symmetric:bool=True, type:str='hann', zeroPadding:int=0, zeroPhase:bool=True) -> None:
		"""Applies windowing to an audio signal.
		
		It optionally applies zero-phase windowing and optionally adds zero-padding. The resulting windowed frame size is equal to the incoming frame size plus the number of padded zeros. By default, the available windows are normalized (to have an area of 1) and then scaled by a factor of 2.
		
		The parameter constantsDecimals allows choosing the number of decimals used in the constants for the formulation of the Hamming and Blackman-Harris windows, which allows replicating alternative windowing implementations. For example, setting type='hamming', constantsDecimals=2, normalized=False, and zeroPhase=False results in a Hamming window similar to the default SciPy implementation [3].
		
		An exception is thrown if the size of the frame is less than 2.
		
		References:
		  [1] F. J. Harris, "On the use of windows for harmonic analysis with the
		  discrete Fourier transform, Proceedings of the IEEE, vol. 66, no. 1,
		  pp. 51-83, Jan. 1978
		
		  [2] Window function - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Window_function
		
		  [3] Hamming window - SciPy documentation,
		  https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.hamming.html

		Args:
			constantsDecimals (int): number of decimals considered in the constants for the formulation of the hamming and blackmanharris* windows . Defaults to 5. Range [1,5]
			normalized (bool): a boolean value to specify whether to normalize windows (to have an area of 1) and then scale by a factor of 2. Defaults to True. Range {true,false}
			size (int): the window size. Defaults to 1024. Range [2,inf)
			splitPadding (bool): whether to split the padding to the edges of the signal (_/\_) or to add it to the right (/\__). This option is ignored when zeroPhase (\__/) is true. Defaults to False. Range {true,false}
			symmetric (bool): whether to create a symmetric or asymmetric window as implemented in SciPy. Defaults to True. Range {true,false}
			type (str): the window type. Defaults to 'hann'. Range {hamming,hann,hannnsgcq,triangular,square,blackmanharris62,blackmanharris70,blackmanharris74,blackmanharris92}
			zeroPadding (int): the size of the zero-padding. Defaults to 0. Range [0,inf)
			zeroPhase (bool): a boolean value that enables zero-phase windowing. Defaults to True. Range {true,false} 
		""" 
		... 
	def __call__(self, frame:NDArray[np.float32]) -> NDArray[np.float32]:
		"""compute
		Args:
			frame (NDArray[np.float32]): the input audio frame. Defaults to None. 
		Returns:
			frame (NDArray[np.float32]): the windowed audio frame
		""" 
		... 


class YamlInput(_essentia.Algorithm): 
	def __init__(self, filename:str, format:str='yaml') -> None:
		"""Deserializes a file formatted in YAML to a Pool.
		
		This file can be serialized back into a YAML file using the YamlOutput algorithm. See the documentation for YamlOutput for more information on the specification of the YAML file.
		
		Note: If an empty sequence is encountered (i.e. "[]"), this algorithm will assume it was intended to be a sequence of Reals and will add it to the output pool accordingly. This only applies to sequences which contain empty sequences. Empty sequences (which are not subsequences) are not possible in a Pool and therefore will be ignored if encountered (i.e. foo: [] (ignored), but foo: [[]] (added as a vector of one empty vector of reals).

		Args:
			filename (str): Input filename. Defaults to None. Range None
			format (str): whether to the input file is in JSON or YAML format. Defaults to 'yaml'. Range {json,yaml} 
		""" 
		... 
	def __call__(self, ) -> Pool:
		"""compute
		Returns:
			pool (Pool): Pool of deserialized values
		""" 
		... 


class YamlOutput(_essentia.Algorithm): 
	def __init__(self, filename:str='-', doubleCheck:bool=False, format:str='yaml', indent:int=4, writeVersion:bool=True) -> None:
		"""Emits a YAML or JSON representation of a Pool.
		
		Each descriptor key in the Pool is decomposed into different nodes of the YAML (JSON) format by splitting on the '.' character. For example a Pool that looks like this:
		
		    foo.bar.some.thing: [23.1, 65.2, 21.3]
		
		will be emitted as:
		
		    metadata:
		        essentia:
		            version: <version-number>
		
		    foo:
		        bar:
		            some:
		                thing: [23.1, 65.2, 21.3]

		Args:
			doubleCheck (bool): whether to double-check if the file has been correctly written to the disk. Defaults to False. Range None
			filename (str): output filename (use '-' to emit to stdout). Defaults to '-'. Range None
			format (str): whether to output data in JSON or YAML format. Defaults to 'yaml'. Range {json,yaml}
			indent (int): (json only) how many characters to indent each line, or 0 for no newlines. Defaults to 4. Range None
			writeVersion (bool): whether to write the essentia version to the output file. Defaults to True. Range None 
		""" 
		... 
	def __call__(self, pool:Pool) -> None:
		"""compute
		Args:
			pool (Pool): Pool to serialize into a YAML formatted file. Defaults to None. 
		""" 
		... 


class ZeroCrossingRate(_essentia.Algorithm): 
	def __init__(self, threshold:float=0.0) -> None:
		"""Computes the zero-crossing rate of an audio signal.
		
		It is the number of sign changes between consecutive signal values divided by the total number of values. Noisy signals tend to have higher zero-crossing rate.
		In order to avoid small variations around zero caused by noise, a threshold around zero is given to consider a valid zerocrosing whenever the boundary is crossed.
		
		Empty input signals will raise an exception.
		
		References:
		  [1] Zero Crossing - Wikipedia, the free encyclopedia,
		  http://en.wikipedia.org/wiki/Zero-crossing_rate
		
		  [2] G. Peeters, "A large set of audio features for sound description
		  (similarity and classification) in the CUIDADO project," CUIDADO I.S.T.
		  Project Report, 2004

		Args:
			threshold (float): the threshold which will be taken as the zero axis in both positive and negative sign. Defaults to 0.0. Range [0,inf] 
		""" 
		... 
	def __call__(self, signal:NDArray[np.float32]) -> float:
		"""compute
		Args:
			signal (NDArray[np.float32]): the input signal. Defaults to None. 
		Returns:
			zeroCrossingRate (float): the zero-crossing rate
		""" 
		... 


