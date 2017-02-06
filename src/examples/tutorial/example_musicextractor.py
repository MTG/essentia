from essentia.standard import MusicExtractor
from essentia import Pool
from argparse import ArgumentParser
import numpy
import os
import json
import fnmatch


def isMatch(name, patterns):
    if not patterns:
        return False
    for pattern in patterns:
        if fnmatch.fnmatch(name, pattern):
            return True
    return False


def add_to_dict(dict, keys, value):
    for key in keys[:-1]:
        dict = dict.setdefault(key, {})
    dict[keys[-1]] = value


def pool_to_dict(pool, include_descs=None, ignore_descs=None):
    # a workaround to convert Pool to dict

    descs = pool.descriptorNames()
    if include_descs:
        descs = [d for d in descs if isMatch(d, include_descs)]
    if ignore_descs:
        descs = [d for d in descs if not isMatch(d, ignore_descs)]

    result = {}
    pool = Pool(pool)

    for d in descs:
        keys = d.split('.')
        value = pool[d]
        if type(value) is numpy.ndarray:
            value = value.tolist()
        add_to_dict(result, keys, value)

    return result


def analyze_dir(audio_dir, result_file, audio_types=None, profile=None, store_frames=False, include_descs=None, ignore_descs=None):

    """
    if args.include and args.ignore and not set(args.include).isdisjoint(args.ignore):
        print 'You cannot specify the same descriptor patterns in both 'include_descs' and --ignore flags'
        sys.exit() # TODO return None instead in this function
    """

    if not audio_types:
        audio_types = ['*.wav', '*.aiff', '*.flac', '*.mp3', '*.ogg']
        print "Audio files extensions considered by default: " + ' '.join(audio_types)
    else:
        print "Searching for audio files extensions: " + ' '.join(audio_types)
    print

    if profile:
        extractor = MusicExtractor(profile=profile)
    else:
        extractor = MusicExtractor()

    # find all audio files
    audio_files = []
    for root, dirnames, filenames in os.walk(audio_dir):
        for match in audio_types:
            for filename in fnmatch.filter(filenames, match):
                audio_files.append(os.path.join(root, filename))

    # analyze
    errors = 0
    results = {}
    for audio_file in audio_files:
        print "Analyzing", audio_file
        try:
            poolStats, poolFrames = extractor(audio_file)
            results[audio_file] = {}
            results[audio_file]['stats'] = pool_to_dict(poolStats, include_descs, ignore_descs)
            if store_frames:
                results[audio_file]['frames'] = pool_to_dict(poolFrames, include_descs, ignore_descs)
        except Exception, e:
            print "Error processing", audio_file, ":", str(e)
            errors += 1
            continue

    # save to json
    print
    print "Analysis done.", errors, "files have been skipped due to errors"
    print "Saving results to", result_file
    with open(result_file, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    parser = ArgumentParser(description = """
Analyzes all audio files found (recursively) in a folder using MusicExtractor.
""")

    parser.add_argument('-d', '--dir', help='input directory', required=True)
    parser.add_argument('-o', '--output', help='output json file with audio analysis results', required=True)
    parser.add_argument('-t', '--type', nargs='+', help='type of audio files to include (can use wildcards)', required=False)
    parser.add_argument('--profile', help='MusicExtractor profile', required=False)
    parser.add_argument('--frames', help='store frames data', action='store_true', required=False)
    parser.add_argument('--include', nargs='+', help='descriptors to include (can use wildcards)', required=False)
    parser.add_argument('--ignore', nargs='+', help='descriptors to ignore (can use wildcards)', required=False)

    args = parser.parse_args()

    analyze_dir(args.dir, args.output, args.type, args.profile, args.frames, args.include, args.ignore)
