from essentia.standard import MusicExtractor, YamlOutput
from essentia import Pool
from argparse import ArgumentParser
import numpy
import os
import json
import fnmatch
import sys


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

    for d in descs:
        keys = d.split('.')
        value = pool[d]
        if type(value) is numpy.ndarray:
            value = value.tolist()
        add_to_dict(result, keys, value)
    return result


def analyze_dir(audio_dir, output_json=None, output_dir=None, audio_types=None, profile=None, store_frames=False, include_descs=None, ignore_descs=None):

    """
    if args.include and args.ignore and not set(args.include).isdisjoint(args.ignore):
        print 'You cannot specify the same descriptor patterns in both 'include_descs' and --ignore flags'
        sys.exit() # TODO return None instead in this function
    """
    if not output_json and not output_dir:
        print "Error: Neither output json file nor output directory were specified."
        return

    if output_dir:
        output_dir = os.path.abspath(output_dir)

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
    os.chdir(audio_dir)
    audio_files = []
    for root, dirnames, filenames in os.walk("."):
        for match in audio_types:
            for filename in fnmatch.filter(filenames, match):
                audio_files.append(os.path.relpath(os.path.join(root, filename)))

    # analyze
    errors = 0
    results = {}
    for audio_file in audio_files:
        print "Analyzing", audio_file
        try:
            poolStats, poolFrames = extractor(audio_file)

        except Exception, e:
            print "Error processing", audio_file, ":", str(e)
            errors += 1
            continue

        if output_json:
            results[audio_file] = {}
            results[audio_file]['stats'] = pool_to_dict(poolStats, include_descs, ignore_descs)
            if store_frames:
                results[audio_file]['frames'] = pool_to_dict(poolFrames, include_descs, ignore_descs)

        if output_dir:
            audio_file = os.path.join(output_dir, audio_file)
            folder = os.path.dirname(audio_file)

            if not os.path.exists(folder):
                os.makedirs(folder)
            elif os.path.isfile(folder):
                print "Cannot create directory", folder
                print "There exist a file with the same name. Aborting analysis."
                sys.exit()

            output = YamlOutput(filename=audio_file+'.sig')
            output(poolStats)
            if store_frames:
                YamlOutput(filename=audio_file + '.frames.sig')(poolFrames)

    print
    print "Analysis done.", errors, "files have been skipped due to errors"

    # save to json
    if output_json:
        print "Saving results to", output_json
        with open(output_json, 'w') as f:
            json.dump(results, f)


if __name__ == '__main__':
    parser = ArgumentParser(description = """
Analyzes all audio files found (recursively) in a folder using MusicExtractor.
""")

    parser.add_argument('-d', '--dir', help='input directory', required=True)
    parser.add_argument('--output-json', help='output json file with audio analysis results or output directory to store separate analysis files for each track', required=False)
    parser.add_argument('--output-dir', help='output directory to store descriptor files (maintains input directory structure)', required=False)
    parser.add_argument('-t', '--type', nargs='+', help='type of audio files to include (can use wildcards)', required=False)
    parser.add_argument('--profile', help='MusicExtractor profile', required=False)
    parser.add_argument('--frames', help='store frames data', action='store_true', required=False)
    parser.add_argument('--include', nargs='+', help='descriptors to include (can use wildcards)', required=False)
    parser.add_argument('--ignore', nargs='+', help='descriptors to ignore (can use wildcards)', required=False)

    args = parser.parse_args()

    analyze_dir(args.dir, args.output_json, args.output_dir, args.type, args.profile, args.frames, args.include, args.ignore)
