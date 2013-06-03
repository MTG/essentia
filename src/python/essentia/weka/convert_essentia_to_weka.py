from wekafile import WekaFile
import yaml

# all the labels you will use
all_labels = ["label1", "label2", "label3"]

# all the files + ground-truth
files = [
	("c:/some_essentia_file.txt", "label1"),
	("c:/some_essentia_file.txt", "label2"),
	("c:/some_essentia_file.txt", "label2"),
	("c:/some_essentia_file.txt", "label2"),
	("c:/some_essentia_file.txt", "label3")
]

# the relation
relation_name = "intensity"

# the output weka file
output_file = "weka.arff"

# the descriptors you would like to use inside weka...
# if you want to use *all* descriptors, set
use_all_descriptors = False
 
# if the previous part was set to True, don't worry about this 
used_descriptors = ["mfcc",
                    "danceability",
                    "dissonance",
                    "dynamiccomplexity",
                    "dynamiccomplexity_loudness",
                    "energy",
                    "hfc",
                    "larm",
                    "leq",
                    "loudness",
                    "lowfreqenergyrelation",
                    "maxmagfreq",
                    "onsetrate",
                    "rolloff",
                    "spectral_centroid",
                    "spectral_crest",
                    "spectral_decrease",
                    "spectral_kurtosis",
                    "spectral_skewness",
                    "spectral_spread",
                    "strongpeak",
                    "tonality",
                    "zerocrossingrate"]

weka_file = WekaFile(relation_name)

first_time = True

def ns_to_single_name(descriptors):
    result = {}
    for namespace in descriptors:
        for descriptor in namespace:
            result[namespace + '.' + descriptor] = descriptors[namespace][descriptor]

    return result

def convert(inputFilenames, outputFilename):
    for (inputFilename, inClass) in inputFilenames:
        print "processing", inputFilename

        descriptors = yaml.load(open(inputFilename, 'r').read())

        values = []

        for descriptor in used_descriptors or use_all_descriptors:
            if descriptor in descriptors:
                if isinstance(descriptors[descriptor]['mean'],list):
                    for index in range(len(descriptors[descriptor]['mean'])):
                        values.append(descriptors[descriptor]['mean'][index])
                        if first_time:
                            weka_file.add_attribute('descriptors[-%s-][-mean-][%d]' % (descriptor, index), "numeric")
                    for index in range(len(descriptors[descriptor]['var'])):
                        values.append(descriptors[descriptor]['var'][index])
                        if first_time:
                            weka_file.add_attribute('descriptors[-%s-][-var-][%d]' % (descriptor, index), "numeric")
                else:
                    values.append(descriptors[descriptor]['mean'])
                    if first_time:
                        weka_file.add_attribute('descriptors[-%s-][-mean-]' % descriptor, "numeric")
                    values.append(descriptors[descriptor]['var'])
                    if first_time:
                        weka_file.add_attribute('descriptors[-%s-][-var-]' % descriptor, "numeric")

        weka_file.add_data(values, [inClass])
        
        if first_time:
            weka_file.add_attribute("ground_truth", all_labels)
        
        first_time = False        

    weka_file.write(outputFilename)

if __name__ == '__main__':
    convert(files, output_file)
