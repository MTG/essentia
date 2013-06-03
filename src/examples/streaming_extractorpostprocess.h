#ifndef STREAMING_EXTRACTOR_POSTPROCESS_H
#define STREAMING_EXTRACTOR_POSTPROCESS_H

#include "pool.h"
#include "types.h"
#include <string>

void PCA(essentia::Pool& pool, const std::string& nspace="");
void PostProcess(essentia::Pool& pool, const essentia::Pool& options, const std::string& nspace="");

#endif // STREAMING_EXTRACTOR_POSTPROCESS_H
