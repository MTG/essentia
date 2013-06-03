#ifndef STREAMING_EXTRACTORLOWLEVEL_H
#define STREAMING_EXTRACTORLOWLEVEL_H

#include "sourcebase.h"
#include "pool.h"
#include "types.h"

void LowLevelSpectral(essentia::streaming::SourceBase& input,
                      essentia::Pool& pool,
                      const essentia::Pool& options,
                      const std::string& nspace="");

void LowLevelSpectralEqLoud(essentia::streaming::SourceBase& input,
                            essentia::Pool& pool,
                            const essentia::Pool& options,
                            const std::string& nspace="");

void Level(essentia::streaming::SourceBase& input, essentia::Pool& pool,
           const essentia::Pool& options, const std::string& nspace="");

void LevelAverage(essentia::Pool& pool, const std::string& nspace="");

#endif // STREAMING_EXTRACTORLOWLEVEL_H
