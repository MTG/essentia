#ifndef STREAMING_EXTRACTORSFX_H
#define STREAMING_EXTRACTORSFX_H

#include "sourcebase.h"
#include "pool.h"

void SFX(essentia::streaming::SourceBase& input, essentia::Pool& pool, const std::string& nspace="");
void SFXPitch(essentia::Pool& pool, const std::string& nspace="");

#endif // STREAMING_EXTRACTORSFX_H
