#ifndef STREAMING_EXTRACTORTONAL_H
#define STREAMING_EXTRACTORTONAL_H

#include "sourcebase.h"
#include "pool.h"
#include "types.h"

void TuningFrequency(essentia::streaming::SourceBase& input, essentia::Pool& pool,
                     const essentia::Pool& options, const std::string& nspace="");

void TonalDescriptors(essentia::streaming::SourceBase& input, essentia::Pool& pool,
                      const essentia::Pool& options, const std::string& nspace="");

void TuningSystemFeatures(essentia::Pool& pool, const std::string& nspace="");
void TonalPoolCleaning(essentia::Pool& pool, const std::string& nspace="");

#endif // STREAMING_EXTRACTORTONAL_H
