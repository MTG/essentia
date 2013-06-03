#ifndef STREAMING_EXTRACTOR_UTILS_H
#define STREAMING_EXTRACTOR_UTILS_H

#include "algorithmfactory.h"
#include "pool.h"
#include "types.h"
#include <string>

void pcmMetadata(essentia::streaming::AlgorithmFactory& factory,
                 const std::string& audioFilename, essentia::Pool& pool);
void readMetadata(const std::string& audioFilename, essentia::Pool& pool);
void LevelAverage(essentia::Pool& pool, const std::string& nspace="");
void TuningSystemFeatures(essentia::Pool& pool, const std::string& nspace="");
void SFXPitch(essentia::Pool& pool, const std::string& nspace="");
void TonalPoolCleaning(essentia::Pool& pool, const std::string& nspace="");

void PCA(essentia::Pool& pool, const std::string& nspace="");
void PostProcess(essentia::Pool& pool, const std::string& nspace="");
void getAnalysisData(const essentia::Pool& pool, essentia::Real& replayGain,
                     essentia::Real& sampleRate, std::string& downmix);

#endif // STREAMING_EXTRACTOR_UTILS_H
