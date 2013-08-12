#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
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



import essentia.standard
import essentia.streaming
import os, re, subprocess

def replace_math_symbols(s):
    while True:
        s, replaced = re.subn(r'(\W)pi(\W)', r'\1π\2', s, 1)
        if not replaced: break

    while True:
        s, replaced = re.subn(r'(\W)inf(\W)', r'\1∞\2', s, 1)
        if not replaced: break

    return s

def TR(s):
    '''Escape necessary characters for ReST formatting'''
    if s == '': return '""'

    s = s.replace('*', '\\*')

    s = replace_math_symbols(s)

    return s

def TR_DESC(s):
    '''Escape necessary characters / adjust layout for ReST formatting for the description field'''
    # find all bullet lists which are not preceded by an empty line and adds one (also
    # add 1 at the end just to make sure)
    s = re.sub(r'([^\n])((?:\n[ \t]+-.*)+)', r'\1\n\2\n', s)

    # replace '|' when it is used for absolute value
    s = s.replace('|', '\\|')

    # replace math constants
    s = replace_math_symbols(s)

    # make sure there is always an empty line before the references
    s = s.replace('References:\n', '\nReferences:\n')

    # add an empty line in all cases at the end, makes ReST happier when we're in the
    # middle of a bullet list (such as the refs section)
    s = s + '\n\n'

    return s

def doc2rst(algo_doc):
    lines = [ algo_doc['name'],
              '=' * len(algo_doc['name']),
              ''
              ]

    if algo_doc['inputs']:
        lines += [ 'Inputs',
                   '------',
                   ''
                   ]

        for inp in algo_doc['inputs']:
            lines.append(' - **%s** (*%s*) - %s' % (inp['name'], inp['type'], TR(inp['description'])))

        lines.append('')

    if algo_doc['outputs']:
        lines += [ 'Outputs',
                   '-------',
                   ''
                   ]

        for out in algo_doc['outputs']:
            lines.append(' - **%s** (*%s*) - %s' % (out['name'], out['type'], TR(out['description'])))

        lines.append('')

    if algo_doc['parameters']:
        lines += [ 'Parameters',
                   '----------',
                   ''
                   ]

        for param in algo_doc['parameters']:
            range_str = param['type']
            if param['range'] is not None:
                range_str += ' ∈ ' + TR(param['range'])
            if param['default'] is not None:
                range_str += ', default = ' + TR(param['default'])

            lines.append(' - **%s** (*%s*) :' % (param['name'], range_str))
            lines.append('     ' + TR(param['description']))

        lines.append('')

    lines += [ 'Description',
               '-----------',
               '',
               TR_DESC(algo_doc['description'])
               ]

    return '\n'.join(lines)


def write_doc(filename, algo_doc):
    open(filename, 'w').write(doc2rst(algo_doc))



def rst2html(rst, layout_type):
    try:
        html, _ = subprocess.Popen([ 'rst2html.py', '-v', '--initial-header-level=2', '--input-encoding=UTF-8:strict' ],
                                   stdin=subprocess.PIPE, stdout = subprocess.PIPE).communicate(rst)
    except OSError:
        try:
            html, _ = subprocess.Popen([ 'rst2html', '-v', '--initial-header-level=2', '--input-encoding=UTF-8:strict' ],
                                       stdin=subprocess.PIPE, stdout = subprocess.PIPE).communicate(rst)

        except OSError:
            print '*'*100
            print 'ERROR: You need to install rst2html! (in the python docutils package)'
            print '*'*100
            raise

    html = re.search('<body>(.*)</body>', html, re.DOTALL).groups()[0]

    algo_name = re.search('<h1.*?>(.*)</h1>', html).groups()[0]

    html = '''
{% extends "algo_description_layout_''' + layout_type + '''.html" %}
{% set title = "Algorithm reference: ''' + algo_name + '''" %}

{% block algo_description %}

''' + html + '''

{% endblock %}
'''

    return html


def write_html_doc(filename, algo_doc, layout_type):
    open(filename, 'w').write(rst2html(doc2rst(algo_doc), layout_type))


def write_algorithms_reference():
    '''Write all files necessary to have a complete algorithms reference in the sphinx doc.
    That includes:
     - write the _templates/algorithms_reference.html template
     - write each separate algo doc as an html template in the _templates/reference folder
     - write a essentia_reference.py file that contains a list of those (to be included in conf.py)
     - write the templates for both std & streaming algos to have the full list of algos as a sidebar
    '''

    try: os.mkdir('_templates/reference')
    except: pass

    # write the doc for all the std algorithms
    std_algo_list = [ algo for algo in dir(essentia.standard) if algo[0].isupper() ]
    std_algo_html_list = []

    for algoname in std_algo_list:
        print 'generating doc for standard algorithm:', algoname, '...'

        std_algo_html_list += [ '<a class="reference internal" href="reference/std_' + algoname + '.html">' +
                                '<em>' + algoname + '</em></a> / ' ]

        write_html_doc('_templates/reference/std_' + algoname + '.html',
                       getattr(essentia.standard, algoname).__struct__,
                       layout_type = 'std')

    # write the template for the std algorithms
    html = '''
{% extends "layout.html" %}
{% block body %}

<div class="sphinxsidebar">
  <div class="sphinxsidebarwrapper">

    <p><strong>Standard algorithms</strong></p>

'''

    for algoname in std_algo_list:
        html += '<div><a href="std_%s.html">%s</a></div>\n' % (algoname, algoname)

    html += '''
  </div>
</div>

{% block algo_description %}
{% endblock %}

{% endblock %}
'''
    open('_templates/algo_description_layout_std.html', 'w').write(html)


    # write the doc for all the streaming algorithms
    streaming_algo_list = [ algo for algo in dir(essentia.streaming) if algo[0].isupper() and algo not in [ 'CompositeBase', 'VectorInput' ] ]
    streaming_algo_html_list = []

    for algoname in streaming_algo_list:
        print 'generating doc for streaming algorithm:', algoname, '...'

        streaming_algo_html_list += [ '<a class="reference internal" href="reference/streaming_' + algoname + '.html">' +
                                      '<em>' + algoname + '</em></a> / ' ]

        write_html_doc('_templates/reference/streaming_' + algoname + '.html',
                       getattr(essentia.streaming, algoname).__struct__,
                       layout_type = 'streaming')


    # write the template for the streaming algorithms
    html = '''
{% extends "layout.html" %}
{% block body %}

<div class="sphinxsidebar">
  <div class="sphinxsidebarwrapper">

    <p><strong>Streaming algorithms</strong></p>

'''

    for algoname in streaming_algo_list:
        html += '<div><a href="streaming_%s.html">%s</a></div>\n' % (algoname, algoname)

    html += '''
  </div>
</div>

{% block algo_description %}
{% endblock %}

{% endblock %}
'''
    open('_templates/algo_description_layout_streaming.html', 'w').write(html)


    # write the essentia_reference.py file (to be included in conf.py)
    with open('essentia_reference.py', 'w') as algo_ref:
        algo_ref.write('''
standard_algorithms = {
''')
        for algo in std_algo_list:
            algo_ref.write("  'reference/std_%s': 'reference/std_%s.html',\n" % (algo, algo))

        algo_ref.write('''  }

streaming_algorithms = {
''')

        for algo in streaming_algo_list:
            algo_ref.write("  'reference/streaming_%s': 'reference/streaming_%s.html',\n" % (algo, algo))

        algo_ref.write('''  }

essentia_algorithms = {}
essentia_algorithms.update(standard_algorithms)
essentia_algorithms.update(streaming_algorithms)

''')


    # write the algorithms_reference.html file (main ref file)
    html = '''
{% extends "layout.html" %}
{% set title = "Algorithms reference" %}
{% block body %}

  <div class="section" id="algorithms-reference">
<h1>Algorithms reference<a class="headerlink" href="#algorithms-reference" title="Permalink to this headline">¶</a></h1>
<p>Here is the complete list of algorithms which you can access from the Python interface.</p>
<p>The C++ interface allows access to the same algorithms, and also some more which are templated
and hence are not available in python.</p>

<div class="section" id="standard-algorithms">
<h2>Standard algorithms<a class="headerlink" href="#standard-algorithms" title="Permalink to this headline">¶</a></h2>
<p>
'''
    html += '\n'.join(std_algo_html_list)[:-3]

    html += '''
</p></div>

<div class="section" id="streaming-algorithms">
<h2>Streaming algorithms<a class="headerlink" href="#streaming-algorithms" title="Permalink to this headline">¶</a></h2>
<p>
'''
    html += '\n'.join(streaming_algo_html_list)[:-3]

    html += '''
</p>
</div>
</div>

{% endblock %}
'''

    open('_templates/algorithms_reference.html', 'w').write(html)



if __name__ == '__main__':
    write_algorithms_reference()
