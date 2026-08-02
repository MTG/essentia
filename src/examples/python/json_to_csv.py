import sys, json, csv
from fnmatch import fnmatch
from argparse import ArgumentParser

JSON_FILENAME = 'json_file_name'

def isMatch(name, patterns):
    if not patterns:
        return False
    for pattern in patterns:
        if fnmatch(name, pattern):
            return True
    return False


def parse_descriptors(d, include=None, ignore=None):
    results = {}

    stack = [(k, k, v) for k, v in d.items()]
    while stack:
        name, k, v = stack.pop()
        if isinstance(v, dict):
            stack.extend([(name + '.' + k1, k1, v1) for k1, v1 in v.items()])
        elif isinstance(v, list):
            stack.extend([(name + '.' + str(i), i, v[i]) for i in range(len(v))])
        else:
            if include:
                # 'include' flag specified => apply both include and ignore
                if isMatch(name, include) and not isMatch(name, ignore):
                    results[name] = v
            else:
                # 'include' flag not specified => apply only ignore
                if not isMatch(name, ignore):
                    results[name] = v
    
    return results


def convert(json_file, include, ignore):
    print ('Converting %s' % json_file)
    data = json.load(open(json_file, 'r'))
    
    return parse_descriptors(data, include, ignore)

def convert_all(json_files, csv_file, include=None, ignore=None, add_filename=True):

    with open(csv_file, 'w') as f_csv:
        print("Writing to %s" % csv_file)
        writer = csv.writer(f_csv, 
                            delimiter=',',
                            quotechar='"', 
                            quoting=csv.QUOTE_NONNUMERIC)
        header = None

        for f_json in json_files:
            d = convert(f_json, include, ignore)

            if add_filename:
                if JSON_FILENAME in d:
                    print("Error appending json filename to the CSV: `%s` name is already used." % JSON_FILENAME)
                    sys.exit()
                else:
                    d[JSON_FILENAME] = f_json

            if not header:
                header = sorted(d.keys())
                if not len(header):
                    print("Error: no descriptors found to be written.")
                    sys.exit()
                writer.writerow(header)

            try:
                if len(d.keys()) != len(header):
                    raise Exception()
                raw = [d[h] for h in header]
            except Exception:
                print("Error: Incompatible descriptor layouts")
                print("Layout difference:")
                print(list(set(header).symmetric_difference(set(d.keys()))))
                sys.exit()
            
            writer.writerow(raw)


    # TODO: Currently, the same descriptor layout is required for all
    #       input files (after filtering)
    # Make alternative version that
    # - gathers a list of all descriptors found in input files
    # - creates a CSV based on such a list, so that files with
    #   different descriptor layouts can be merged into the same CSV

    return


if __name__ == '__main__':
    parser = ArgumentParser(description = """
Converts a bunch of descriptor files from json to csv format.
Descriptor trees are flattened, with additional indices added to descriptor
names in the case of lists or nested lists
(for example: {'group': {'name': [[1,2,3], [4,5,6]]}} will be mapped to descriptor names
'group.name.0.0', 'group.name.0.1', 'group.name.0.2', 'group.name.1.0', 'group.name 1.1', 'group.name 1.2').
Descriptors can then be included/ignored by their flattened names using wildcards.
After flattening and filtering, all inputs are expected to have exactly the same set
of descriptor names to be able to merge them into one csv.
""")

    parser.add_argument('-i', '--input', nargs='+', help='Input JSON files', required=True)
    parser.add_argument('-o', '--output', help='Output CSV file', required=True)

    parser.add_argument('--include', nargs='+', help='Descriptors to include (can use wildcards)', required=False)
    parser.add_argument('--ignore', nargs='+', help='Descriptors to ignore (can use wildcards)', required=False)

    parser.add_argument('--add-filename', help='Add input filenames to "%s" field in CSV' % JSON_FILENAME, action='store_true', required=False)

    args = parser.parse_args()

    if args.include and args.ignore and not set(args.include).isdisjoint(args.ignore):
        print('You cannot specify the same descriptor patterns in both --include and --ignore flags')
        sys.exit()

    convert_all(args.input, args.output, args.include, args.ignore, args.add_filename)
