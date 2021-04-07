# Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
#
# This file is part of Essentia
#
# Essentia is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation (FSF), either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the Affero GNU General Public License
# version 3 along with this program. If not, see http://www.gnu.org/licenses/

import inspect, types
import streaming
import _essentia
import common
from streaming import _reloadStreamingAlgorithms

# genetic marker used to track which composite parameters configure which inner algorithms
class MarkerObject(object):
    def __init__(self, default_value=None):
        self.default_value = default_value


edt_parameter_code = { common.Edt.STRING:             'String',
                       common.Edt.INTEGER:            'Int',
                       common.Edt.VECTOR_REAL:        'VectorReal',
                       common.Edt.VECTOR_VECTOR_REAL: 'VectorVectorReal',
                       common.Edt.REAL:               'Real',
                       common.Edt.BOOL:               'Bool'}


edt_cpp_code = { common.Edt.STRING:             'string',
                 common.Edt.INTEGER:            'int',
                 common.Edt.VECTOR_INTEGER:     'vector<int>',
                 common.Edt.VECTOR_STRING:      'vector<string>',
                 common.Edt.VECTOR_REAL:        'vector<Real>',
                 common.Edt.VECTOR_VECTOR_REAL: 'vector<vector<Real> >',
                 common.Edt.REAL:               'Real',
                 common.Edt.BOOL:               'bool'}


# html code for edt_dot_code:
# &lt; --> less than or <
# &gt; --> greater than or >
# #&#58; --> colon or :

edt_dot_code = { common.Edt.STRING:             'string',
                 common.Edt.INTEGER:            'int',
                 common.Edt.VECTOR_INTEGER:     'vector&lt;int&gt;',
                 common.Edt.VECTOR_STRING:      'vector&lt;string&gt;',
                 common.Edt.VECTOR_REAL:        'vector&lt;Real&gt;',
                 common.Edt.VECTOR_VECTOR_REAL: 'vector&lt;vector&lt;Real&gt; &gt;',
                 common.Edt.REAL:               'Real',
                 common.Edt.BOOL:               'bool',
                 common.Edt.STEREOSAMPLE:       'StereoSample',
                 common.Edt.MATRIX_REAL:        'TNT&#58;&#58;Array2D&lt;Real&gt;'} #&#58; -> colon

# finds the EDT of a parameter of an algorithm given the name of the parameter, the marker_obj used
# to configure that parameter, and the configure_log
def find_edt(composite_param_name, marker_obj, configure_log):
    # find inner algorithm and inner parameter name that this composite_param will configure
    for inner_algo_name, properties in configure_log.iteritems():
        for inner_param_name, value in properties['parameters'].iteritems():
            if marker_obj == value:
                return properties['instance'].paramType(inner_param_name)

    raise RuntimeError('Could not determine parameter type of composite algorithm\'s \''+composite_param_name+'\' parameter')


# given a reference to an inner algorithm and the configure_log, returns the name of the algo (use
# lower() if referring to the member var name)
def inner_algo_name(instance, configure_log):
    for algo_name, properties in configure_log.iteritems():
        if instance == properties['instance']:
            return algo_name

    raise RuntimeError('Could not find the name of the inner algorithm')


def generate_dot_algo(algo_name, algo_inst):
    '''declares an algorithm in dot'''
    dot_code = '_'+algo_name.lower()+' [shape="box", style="rounded,filled", fillcolor="grey50", color="transparent", \n'
    indent = ' '*len('_'+algo_name)
    dot_code += indent+'  label=<\n'
    indent += ' '*len('label=<')
    dot_code += generate_dot_algo_label(algo_inst, indent)
    dot_code += '>]\n\n'

    return dot_code


def generate_dot_algo_label(algo_inst, indent=''):
    ### each label of a node consists of algo inputs, algo name,
    ### configuration parameters and algo outputs

    dot_code =          indent+'<table border="0"><tr>\n'+\
                        indent+'    <td><table border="0" bgcolor="white">\n'

    # add inputs:
    if not len(algo_inst.inputNames()): dot_code += ''
    else:
        for name in algo_inst.inputNames():
            typestr = edt_dot_code[ algo_inst.getInputType(name) ]
            dot_code += indent+'        <tr><td port="'+name+'_i">'+name+'<br/>['+typestr+']</td></tr>\n'

    dot_code +=         indent+'    </table></td>\n\n'+\
                        indent+'    <td><table border="0">\n'

    # add algo name:
    dot_code +=         indent+'        <tr><td valign="top" colspan="2"><font color="white" point-size="18">'+algo_inst.name()+'</font></td></tr>\n'

    # add parameters:
    if not algo_inst.parameterNames(): dot_code += ''
    else:
        for name in algo_inst.parameterNames():
            value = algo_inst.paramValue(name)
            dot_code += indent+'        <tr>\n'+\
                        indent+'            <td border="0" valign="top" align="right"><font color="white">'+name+'</font></td>\n'+\
                        indent+'            <td border="0" valign="top" align="left"><font color="white">'+str(value)+'</font></td>\n'+\
                        indent+'        </tr>\n'

    dot_code +=         indent+'    </table></td>\n\n'+\
                        indent+'    <td><table border="0" bgcolor="white">\n'

    # add outputs:
    if not len(algo_inst.outputNames()): dot_code += ''
    else:
        for name in algo_inst.outputNames():
            typestr = edt_dot_code[ algo_inst.getOutputType(name) ]
            dot_code += indent+'        <tr><td port="'+name+'_o">'+name+'<br/>['+typestr+']</td></tr>\n'

    dot_code +=         indent+'    </table></td>\n'+\
                        indent+'</tr></table>'


    return dot_code


def generate_dot_network(configure_log, composite_algo_inst):
    # make connections
    dot_code ='\n// connecting the network\n'
    for algo_name, properties in configure_log.iteritems():
        for left_connector, right_connectors in properties['instance'].connections.iteritems():
            for right_connector in right_connectors:
                if isinstance(right_connector, streaming._StreamConnector):
                    dot_code += '        _'+inner_algo_name(left_connector.output_algo, configure_log).lower()+':'+left_connector.name+'_o:e'+' -> '+\
                                '_'+inner_algo_name(right_connector.input_algo, configure_log).lower()+':'+right_connector.name + '_i:w;\n'
                if isinstance(right_connector, types.NoneType):
                    inneralgoname = inner_algo_name(left_connector.output_algo, configure_log).lower()
                    dot_code += '        nowhere_'+inneralgoname+' [shape="box", style="rounded,filled", fillcolor="grey50", color="transparent" label="Nowhere" fontcolor="white" fontsize="18"];\n'+\
                                '        _'+inneralgoname+':'+left_connector.name+'_o:e'+' -> nowhere_'+inneralgoname+';\n'

    # make connections from floating inputs
    for name, connector in composite_algo_inst.inputs.iteritems():
        innerinputname = connector.name
        inneralgoname = inner_algo_name(connector.input_algo, configure_log).lower()
        dot_code += '        '+name+':e -> _'+inneralgoname+':'+innerinputname+'_i:w;\n'

    # make connections from floating outputs
    for name, connector in composite_algo_inst.outputs.iteritems():
        inneroutputname = connector.name
        inneralgoname = inner_algo_name(connector.output_algo, configure_log).lower()
        dot_code += '        _'+inneralgoname+':'+inneroutputname+'_o:e -> '+name+':w;\n'

    return dot_code


def generate_dot_cluster(configure_log, clustername, composite_algo_inst):
    ''' creates a cluster in dot language surrounded by a dashed, lightgrey line'''

    dot_code  = '    subgraph cluster_0 {\n'\
                '        color=lightgrey;\n'\
                '        style=dashed;\n'\
                '        label='+clustername+';\n\n'

    # for each algo in the cluster, declare it in dot:
    for algo_name, properties in configure_log.iteritems():
        dot_code += generate_dot_algo(algo_name, properties['instance'])

    # create the connections
    dot_code += generate_dot_network(configure_log, composite_algo_inst)

    # close the cluster code
    dot_code += '    }\n'

    return dot_code


def translate(composite_algo, output_filename, dot_graph=False):
    '''Takes in a class that is derived from essentia.streaming.CompositeBase and an output-filename
       and writes output-filename.h and output-filename.cpp versions of the given class.'''

    if not inspect.isclass(composite_algo):
        raise TypeError('"composite_algo" argument must be a class')

    if not streaming.CompositeBase in inspect.getmro(composite_algo):
        raise TypeError('"composite_algo" argument must inherit from essentia.streaming.CompositeBase')

    param_names, _, _, default_values = inspect.getargspec(composite_algo.__init__)
    param_names.remove('self')

    # these marker objects are used to track where config params travel in the network
    marker_objs = {}
    if not default_values and param_names: # python vars have no type so we cannot know what type they are!!
        raise TypeError('"composite_algo" arguments must have default values')
    if param_names:
        for param_name, value in zip(param_names, default_values):
            marker_objs[param_name] = MarkerObject(value)

    ### Before we call their function we need to neuter all of the configure methods of each
    ### streaming algorithm so that our markers won't cause the configure method to vomit

    configure_log = {}

    def dummy_configure(self, **kwargs):
        lbl = 0
        algo_name = self.name()+'_'+str(lbl)

        # increment lbl to generate a unique name for inner algo
        lowered_algo_names = [name.lower() for name in configure_log.keys()]
        while algo_name.lower() in lowered_algo_names:
            algo_name = algo_name[:algo_name.index('_')+1] + str(lbl)
            lbl +=1

        # algo_name is now unique

        configure_log[algo_name] = {}
        configure_log[algo_name]['instance'] = self
        configure_log[algo_name]['parameters'] = kwargs

        # We need to actually call the internal configure method because algorithms like silencerate
        # need to be configured so we can use its outputs. However we can't use our marker objects,
        # so we remove the marker objects that don't have a default value associated with them, and
        # for those that do have a default value, we use that value instead of the MarkerObject
        # itself
        kwargs_no_markers = dict(kwargs)

        for key, value in kwargs.iteritems():
            if value in marker_objs.values():
                if value.default_value == None:
                    del kwargs_no_markers[key]
                else:
                    kwargs_no_markers[key] = value.default_value

        self.real_configure(**kwargs_no_markers)

    # iterate over all streaming_algos
    streaming_algos = inspect.getmembers( streaming,
                                          lambda obj: inspect.isclass(obj) and \
                                                      _essentia.StreamingAlgorithm in inspect.getmro(obj) )
    streaming_algos = [member[1] for member in streaming_algos]
    for algo in streaming_algos:
        algo.real_configure = algo.configure
        algo.configure = dummy_configure

    ### Now generate an instance of their composite algorithm ###

    algo_inst = composite_algo(**marker_objs)

    # overwrite the dummy configure with the real configure method, so
    # translate can be called several times in the same file for a different
    # compositebase without entering in an infinite loop
    for algo in streaming_algos:
        algo.configure = algo.real_configure

    ### Do some checking on their network ###
    for algo in [ logitem['instance'] for logitem in configure_log.values() ]:
        if isinstance(algo, streaming.VectorInput):
            raise TypeError('essentia.streaming.VectorInput algorithms are not allowed for translatable composite algorithms')

        if isinstance(algo, streaming.AudioLoader) or \
           isinstance(algo, streaming.EasyLoader) or \
           isinstance(algo, streaming.MonoLoader) or \
           isinstance(algo, streaming.EqloudLoader):
            raise TypeError('No type of AudioLoader is allowed for translatable composite algorithms')

        if isinstance(algo, streaming.AudioWriter) or \
           isinstance(algo, streaming.MonoWriter):
            raise TypeError('No type of AudioWriter is allowed for translatable composite algorithms')

        if isinstance(algo, streaming.FileOutput):
            raise TypeError('essentia.streaming.FileOutput algorithms are not allowed for translatable composite algorithms')


    def sort_by_key(configure_log):
        # sort algorithms and conf values:
        sitems = configure_log.items()
        sitems.sort()
        sorted_algos = []
        sorted_params= []
        for k,v in sitems:
            sorted_params.append(v)
            sorted_algos.append(k)
        return sorted_algos, sorted_params

    sorted_algos, sorted_params = sort_by_key(configure_log)

    ### generate .h code ###

    h_code = '''// Generated automatically by essentia::translate

#ifndef STREAMING_''' + composite_algo.__name__.upper() + '''
#define STREAMING_''' + composite_algo.__name__.upper()+ '''

#include "streamingalgorithmcomposite.h"

class '''+composite_algo.__name__+''' : public essentia::streaming::AlgorithmComposite {
 protected:
'''

    for algo_name in sorted_algos:
        h_code += '  essentia::streaming::Algorithm* _'+algo_name.lower()+';\n'

    h_code += '''
 public:
  '''+composite_algo.__name__+'''();

  ~'''+composite_algo.__name__+'''() {
'''
    for algo_name in sorted_algos:
        h_code += '    delete _'+algo_name.lower()+';\n'

    h_code += '''  }

  void declareParameters() {
'''
    if param_names:
        for param_name, default_value in zip(param_names, default_values):
            h_code += '    declareParameter("'+param_name+'", "", "", '

            if isinstance(default_value, basestring): h_code += '"'+default_value+'"'
            else:                                     h_code += str(default_value)

            h_code += ');\n'

    h_code += '''  }

  void configure();
  void createInnerNetwork();
  void reset();

  static const char* name;
  static const char* version;
  static const char* description;
};
#endif
'''

    ### Generate .cpp code ###

    cpp_code = '''// Generated automatically by essentia::translate

#include "'''+output_filename+'''.h"
#include "algorithmfactory.h"
#include "taskqueue.h"

using namespace std;
using namespace essentia;
using namespace essentia::streaming;

const char* '''+composite_algo.__name__+'''::name = "'''+composite_algo.__name__+'''";
const char* '''+composite_algo.__name__+'''::version = "1.0";
const char* '''+composite_algo.__name__+'''::description = DOC("");\n\n'''


################################
# CONSTRUCTOR
################################
    cpp_code += composite_algo.__name__+'''::'''+composite_algo.__name__+'''(): '''
    for algo_name in sorted_algos: cpp_code += '_' + algo_name.lower() + '(0), '
    cpp_code = cpp_code[:-2] + ''' {
  setName("''' + composite_algo.__name__ + '''");
  declareParameters();
  AlgorithmFactory& factory = AlgorithmFactory::instance();\n\n'''

    # create inner algorithms
    for algo_name in sorted_algos:
        cpp_code += '  _'+algo_name.lower()+' = factory.create("'+algo_name[:algo_name.rindex('_')]+'");\n'

    cpp_code+='}\n\n'

################################
# INNER NETWORK
################################
# declaration of inputs and output and connecting the network should not be
# done in the constructor, as there are algos like silencerate which
# inputs/outputs depend on the configuration parameters. Hence, it is safer to
# do it in the configure() function

    cpp_code += 'void ' + composite_algo.__name__ + '::createInnerNetwork() {\n'

    # declare inputs
    for input_alias, connector in algo_inst.inputs.iteritems():
        input_owner_name = None
        input_name = None

        for algo_name, properties in zip(sorted_algos, sorted_params): #configure_log.iteritems():
            if properties['instance'] == connector.input_algo:
                input_owner_name = algo_name
                input_name = connector.name
                break

        if not input_owner_name:
            raise RuntimeError('Could not determine owner of the \''+input_alias+'\' input')

        cpp_code += '  declareInput(_'+input_owner_name.lower()+'->input("'+input_name+'"), "'+input_alias+'", "");\n'

    cpp_code += '\n'

    # declare outputs
    aliases, connectors = sort_by_key(algo_inst.outputs)
    for output_alias, connector in zip(aliases, connectors):
        output_owner_name = None
        output_name = None

        for algo_name, properties in zip(sorted_algos, sorted_params): #configure_log.iteritems():
            if properties['instance'] == connector.output_algo:
                output_owner_name = algo_name
                output_name = connector.name
                break

        if not output_owner_name:
            raise RuntimeError('Could not determine owner of the \''+output_alias+'\' output')

        cpp_code += '  declareOutput(_'+output_owner_name.lower()+'->output("'+output_name+'"), "'+output_alias+'", "");\n'

    cpp_code += '\n'

    # make connections
    for algo_name, properties in zip(sorted_algos, sorted_params): #configure_log.iteritems():
        for left_connector, right_connectors in properties['instance'].connections.iteritems():
            for right_connector in right_connectors:
                if isinstance(right_connector, streaming._StreamConnector):
                    cpp_code += '  connect( _'+\
                                inner_algo_name(left_connector.output_algo, configure_log).lower() + \
                                '->output("'+left_connector.name+'"), _' + \
                                inner_algo_name(right_connector.input_algo, configure_log).lower() + \
                                '->input("'+right_connector.name+'") );\n'

                elif isinstance(right_connector, types.NoneType):
                    cpp_code += '  connect( _'+\
                                inner_algo_name(left_connector.output_algo, configure_log).lower() + \
                                '->output("'+left_connector.name+'"), NOWHERE );\n'

    cpp_code = cpp_code[:-1]
    cpp_code += '''
}\n\n'''


################################
# CONFIGURE
################################

    cpp_code += 'void '+composite_algo.__name__+'::configure() {\n'

    # configure method

    # create local variable for every composite parameter
    for composite_param_name in param_names:
        param_edt = find_edt(composite_param_name, marker_objs[composite_param_name], configure_log)
        cpp_code += '  '+edt_cpp_code[param_edt]+' '+composite_param_name + \
                    ' = parameter("'+composite_param_name+'").to' + \
                    edt_parameter_code[param_edt]+'();\n'

    cpp_code += '\n'

    # configure inner algorithms
    for algo_name, properties in zip(sorted_algos, sorted_params): #configure_log.iteritems():
        # skip if inner algorithm wasn't configured explicitly
        if not properties['parameters']: continue

        for param_name, value in properties['parameters'].iteritems():
            type = common.determineEdt(value)
            if 'LIST' in str(type) or 'VECTOR' in str(type):
                if type in [common.Edt.VECTOR_STRING]:
                    cpp_code += '  const char* ' + param_name + '[]  = {'
                    for s in value: cpp_code += '\"' + s + '\"' + ','
                elif type in[common.Edt.VECTOR_REAL, common.Edt.LIST_REAL]:
                    cpp_code += '  Real ' + param_name + '[] = {'
                    for f in value: cpp_code +=  str(f) + ','
                elif type in [common.Edt.VECTOR_INT, common.Edt.LIST_INT]:
                    cpp_code += '  int' + param_name + '[] = {'
                    for i in value: cpp_code +=  str(i) + ','
                cpp_code = cpp_code[:-1]+'};\n'

        cpp_code += '  _'+algo_name.lower()+'->configure('

        for param_name, value in properties['parameters'].iteritems():
            if isinstance(value, MarkerObject):
                # figure out which composite param it is
                composite_param_name = None
                for marker_name, marker_obj in marker_objs.iteritems():
                    if marker_obj == value:
                        composite_param_name = marker_name
                        break

                if not composite_param_name:
                    raise RuntimeError('Could not determine which composite parameter to use to configure inner algorithm \''+algo_name+'\'s parameter \''+param_name+'\'')

                cpp_code += '"'+param_name+'", '+composite_param_name+', '

            else:
                type = common.determineEdt(value)
                if 'LIST' in str(type) or 'VECTOR' in str(type):
                    if type in [common.Edt.VECTOR_STRING]:
                        cpp_code += '"'+param_name+'", '+'arrayToVector<string>(' + param_name + ')  '
                    elif type in[common.Edt.VECTOR_REAL, common.Edt.LIST_REAL]:
                        cpp_code += '"'+param_name+'", '+'arrayToVector<Real>(' + param_name + ')  '
                    elif type in [common.Edt.VECTOR_INT, common.Edt.LIST_INT]:
                        cpp_code += '"'+param_name+'", '+'arrayToVector<int>(' + param_name + ')  '
                elif isinstance(value, basestring):
                    cpp_code += '"'+param_name+'", "'+value+'", '
                elif isinstance(value, bool):
                    if value: cpp_code += '"'+param_name+'", true, '
                    else:     cpp_code += '"'+param_name+'", false, '
                else:
                    cpp_code += '"'+param_name+'", '+str(value)+', '

        cpp_code = cpp_code[:-2] + ');\n'
    cpp_code += '  createInnerNetwork();\n}\n\n'

################################
# RESET
################################

    cpp_code += 'void '+composite_algo.__name__+'::reset() {\n'
    for algo_name in sorted_algos:
        cpp_code += '  _' + algo_name.lower() + '->reset();\n'
    cpp_code += '}\n\n'

################################
# DESTRUCTOR
################################
# see h_code. Each algo from the composite is deleted separately instead of
# calling deleteNetwork
#    cpp_code += composite_algo.__name__+'''::~'''+composite_algo.__name__+'''() {
#  deleteNetwork(_''' + input_owner_name.lower() + ''');
#}'''
#    cpp_code +='\n'
#
################################
# end of cpp code
################################

    if dot_graph:
        ### generate .dot code ###
        dot_code  = 'digraph ' + output_filename +' {\n'
        dot_code += '    rankdir=LR\n' # if instead of top-down left-right is prefered
        # general formatting options:
        dot_code += '    node [color=black, fontname=Verdana, weight=1, fontsize=8, shape=Mrecord]\n'
        dot_code += '    edge [color=black, style=solid, weight=1, arrowhead="dotnormal", arrowtail="dot", arrowsize=1, fontsize=6]\n'

        # for each input generate nodes
        for name in algo_inst.inputs.keys():
            dot_code += '    '+name+' [label="'+name+'"];\n'

        dot_code += generate_dot_cluster(configure_log, composite_algo.__name__, algo_inst)

        # for each output generate nodes
        for name in algo_inst.outputs.keys():
            dot_code += '    '+name+' [label="'+name+'"];\n'

        dot_code += '}'

    ### Write files ###

    f = open(output_filename+'.h', 'w')
    f.write(h_code)
    f.close()

    f = open(output_filename+'.cpp', 'w')
    f.write(cpp_code)
    f.close()

    if dot_graph:
        f = open(output_filename+'.dot', 'w')
        f.write(dot_code)
        f.close()

