#ifndef STREAMING_EXTRACTORBEATTRACK_H
#define STREAMING_EXTRACTORBEATTRACK_H

#include "sourcebase.h"
#include "pool.h"
#include "types.h"

// outdated beat tracker (2009), bad performance
void BeatTrack(essentia::Pool& pool,
               const essentia::Pool& options,
               const std::string& nspace);

#endif // STREAMING_EXTRACTORBEATTRACK_H
