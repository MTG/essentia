import sys
import os, fnmatch
from essentia.standard import MetadataReader, YamlOutput
from essentia import Pool

FILE_EXT = ('.mp3', '.flac', '.ogg')

def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if basename.lower().endswith(pattern):
                filename = os.path.join(root, basename)
                yield filename

try:
    indir = sys.argv[1]
    result_file = sys.argv[2]
except:
    print("usage: %s <input-directory> <result.json>" % sys.argv[0])
    sys.exit()


result = Pool()
files = [f for f in find_files(indir, FILE_EXT)]

print('Found %d audio files (%s)' % (len(files), '/'.join(FILE_EXT)))

i = 0
for filename in files:
    i += 1
    print('Extracting metadata: %s' % filename)
    namespace = 'track_' + str(i)
    try:
        meta = MetadataReader(filename=filename, failOnError=True, tagPoolName=namespace + '.metadata')()
        pool_meta, duration, bitrate, samplerate, channels = meta[7:]
        pool_meta.set(namespace + ".file_path", os.path.relpath(filename))
        pool_meta.set(namespace + ".duration", duration)
        pool_meta.set(namespace + ".bit_rate", bitrate)
        pool_meta.set(namespace + ".sample_rate", samplerate)
        pool_meta.set(namespace + ".channels", channels)
        result.merge(pool_meta)
    except Exception as e:
        print(str(e))

print("Saving results to %s" % result_file)
YamlOutput(filename=result_file, format='json', doubleCheck=True, writeVersion=False)(result)
