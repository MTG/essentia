#ifndef STREAMING_EXTRACTORPANNING_H
#define STREAMING_EXTRACTORPANNING_H
#include "sourcebase.h"
#include "pool.h"
#include "types.h"

void Panning(essentia::streaming::SourceBase& input, essentia::Pool& pool,
             const essentia::Pool& options, const std::string& nspace="");

#endif // STREAMING_EXTRACTORLOWLEVEL_H
