#ifndef STREAMING_EXTRACTOR_METADATA_H
#define STREAMING_EXTRACTOR_METADATA_H

#include "algorithmfactory.h"
#include "pool.h"
#include "types.h"
#include <string>

void setDefaultOptions(essentia::Pool& pool);
void setOptions(essentia::Pool& options, const std::string& filename);
void mergeOptionsAndResults(essentia::Pool& results, const essentia::Pool& options);
void pcmMetadata(essentia::streaming::AlgorithmFactory& factory,
                 const std::string& audioFilename, essentia::Pool& pool);
void readMetadata(const std::string& audioFilename, essentia::Pool& pool);

#endif // STREAMING_EXTRACTOR_METADATA_H
