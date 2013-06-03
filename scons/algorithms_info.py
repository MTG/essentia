#!/usr/bin/python
import os
import sys
import glob

# @TODO remove cgi.escpae here and put it in the HTML generation

def get_all_algorithms(algo_dir):
    algorithms = {}

    for filename in glob.glob(os.path.join(algo_dir, '*/*.h')):
        try:
            lines = [ line.strip() for line in file(filename).readlines() ]

            algo = ''
            salgo = '' # name of streaming algo
            has_standard  = False
            has_streaming = False
            parameters = {}
            inputs = {}
            outputs = {}

            for line in lines:
                if line.startswith('//'):
                    continue

                if line.find('namespace standard') >= 0:
                    has_standard = True
                    continue

                if line.find('namespace streaming') >= 0:
                    has_streaming = True
                    continue

                if (line.find('public Algorithm') >= 0 or
                    line.find('public AccumulatorAlgorithm') >= 0 or
                    line.find('public StreamingAlgorithmWrapper') >= 0
                    ):
                    name = line.split(' ')[1]

                    if has_standard and not has_streaming: algo = name

                    if not has_standard and has_streaming: salgo = name

                    if has_standard and has_streaming:
                        if algo == '': algo = name
                        else: salgo = name

                    # if we have both standard and streaming, stop here because we
                    # don't want to override the previous values
                    if has_standard and has_streaming: break

                if line.find('Input<') >= 0:
                    line = line[line.find('Input<')+6:]
                    var_type = line[:line.rfind('>')].strip()
                    var_name = line[line.rfind('>')+1:].replace(';', '').strip()
                    inputs[var_name] = { 'type': var_type }
                    continue

                if line.find('Output<') >= 0:
                    line = line[line.find('Output<')+7:]
                    var_type = line[:line.rfind('>')].strip()
                    var_name = line[line.rfind('>')+1:].replace(";","").strip()
                    outputs[var_name] = { 'type': var_type }
                    continue

                if line.find('Sink<') >= 0:
                    line = line[line.find('Sink<')+5:]
                    var_type = line[:line.rfind('>')].strip()
                    var_name = line[line.rfind('>')+1:].replace(';', '').strip()
                    inputs[var_name] = { 'type': var_type }
                    continue

                if line.find('SinkProxy<') >= 0:
                    line = line[line.find('SinkProxy<')+10:]
                    var_type = line[:line.rfind('>')].strip()
                    var_name = line[line.rfind('>')+1:].replace(';', '').strip()
                    inputs[var_name] = { 'type': var_type }
                    continue

                if line.find('Source<') >= 0:
                    line = line[line.find('Source<')+7:]
                    var_type = line[:line.rfind('>')].strip()
                    var_name = line[line.rfind('>')+1:].replace(";","").strip()
                    outputs[var_name] = { 'type': var_type }
                    continue

                if line.find('SourceProxy<') >= 0:
                    line = line[line.find('SourceProxy<')+12:]
                    var_type = line[:line.rfind('>')].strip()
                    var_name = line[line.rfind('>')+1:].replace(";","").strip()
                    outputs[var_name] = { 'type': var_type }
                    continue

                if line.find('declareParameter(') >= 0:
                    line = line[line.find("(")+1:]
                    line = line[:line.find(");")]
                    name = line.split(',',1)[0][1:-1]
                    line = line.split(',',1)[1].lstrip(" \"")
                    description = line[0:line.find("\"")]
                    rest = line[line.find("\""):]
                    default = rest.replace("\", ","")
                    parameters[name] = (description, default)
                    continue

                if line.find('declareInput') >= 0:
                    line = line[line.find('(')+1:]
                    line = line[0:line.find(');')]
                    parts = line.strip().split(',',3)
                    len_parts = len(parts)
                    if len_parts == 2:
                        parts.append("")

                    if len_parts == 4 :
                        input_name_idx = 3
                        input_desc_idx = -1
                    elif len_parts < 4 :
                        if not has_streaming:
                            input_name_idx = 1
                            input_desc_idx = 2
                        else:
                            input_name_idx = 2
                            input_desc_idx = -1
                    if input_name_idx >= 0:
                        input_name = parts[input_name_idx].strip(" ,\"")
                    else:
                        input_name = ''
                    if input_desc_idx >= 0:
                        input_desc = parts[input_desc_idx].strip(" ,\"")
                    else:
                        input_desc = input_name

                    if not has_streaming:
                        var_name = parts[0]
                    else:
                        var_name = parts[0]
                        # big hack to correctly parser stuff like declareInput(_algo->input("a"))
                        if var_name.find('-') >= 0:
                            inputs[var_name] = { 'type': 'unknown' }
                    inputs[var_name]['name'] = input_name
                    inputs[var_name]['description'] = input_desc
                    continue

                if line.find('declareOutput') >= 0:
                    #line = line[line.find('(')+1:]
                    #line = line[0:line.find(');')]
                    #parts = line.strip().split(',', 2) #[1:]
                    #if len(parts) == 2:
                    #    parts.append("")
                    #var_name = parts[0]
                    #output_name = parts[1].strip(" ,\"")
                    #output_desc = parts[2].strip(" ,\"")
                    #outputs[var_name]['name'] = output_name
                    #outputs[var_name]['description'] = output_desc
                    #continue

                    line = line[line.find('(')+1:]
                    line = line[0:line.find(');')]
                    parts = line.strip().split(',',3)
                    len_parts = len(parts)
                    if len_parts == 2:
                        parts.append("")

                    if len_parts == 4 :
                        input_name_idx = 3
                        input_desc_idx = -1
                    elif len_parts < 4 :
                        if not has_streaming:
                            input_name_idx = 1
                            input_desc_idx = 2
                        else:
                            input_name_idx = 2
                            input_desc_idx = -1
                    if input_name_idx >= 0:
                        input_name = parts[input_name_idx].strip(" ,\"")
                    else:
                        input_name = ''
                    if input_desc_idx >= 0:
                        input_desc = parts[input_desc_idx].strip(" ,\"")
                    else:
                        input_desc = input_name

                    if not has_streaming:
                        var_name = parts[0]
                    else:
                        var_name = parts[0]
                        # big hack to correctly parser stuff like declareInput(_algo->input("a"))
                        if var_name.find('-') >= 0:
                            inputs[var_name] = { 'type': 'unknown' }
                    outputs[var_name]['name'] = input_name
                    outputs[var_name]['description'] = input_desc
                    continue


            # Sanity checks

            # 1- if a file is in the algo dir, it *must* be an algorithm (now a warning)
            if not has_standard and not has_streaming:
                print 'WARNING: file "%s" does not seem to contain an algorithm.' % filename
                continue
                #sys.exit(1)

            # 2- if an algo has both standard & streaming form in the same file,
            #    they must have the same name
            if has_standard and has_streaming and algo != salgo:
                print 'ERROR: file "%s" contains both standard and streaming'\
                      'version of the algorithm, but they have different '\
                      'names: %s != %s.' %(filename, algo, salgo)
                sys.exit(1)

            # 3- if algo is only available as streaming, use salgo as algo name
            if has_streaming:
                algo = salgo

            # 4- if we couldn't determine some attributes of inputs/outputs, put
            #    placeholders
            for (input, attr) in inputs.items():
                if 'name' not in attr:
                    attr['name'] = 'Unknown'
                if 'description' not in attr:
                    attr['description'] = 'TODO'

            for (output, attr) in outputs.items():
                if 'name' not in attr:
                    attr['name'] = 'Unknown'
                if 'description' not in attr:
                    attr['description'] = 'TODO'


            algorithms[algo] = { 'header' : filename,
                                 'source' : filename.replace('.h', '.cpp'),
                                 'has_standard' : has_standard,
                                 'has_streaming' : has_streaming,
                                 'inputs': inputs,
                                 'outputs': outputs,
                                 'parameters': parameters
                                 }
        except:
            print 'Error while trying to parse file "%s"' % filename
            raise

    return algorithms


if __name__ == '__main__':
    algo_dir = sys.argv[1]
    algos = get_all_algorithms(algo_dir)

    for algo in algos:
        print algo, ':\n',
        for attr in algos[algo]:
            print '  %s:' % attr, algos[algo][attr]
        print
