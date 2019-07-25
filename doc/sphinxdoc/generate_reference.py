#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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
import sys


std_algo_list = [ algo for algo in dir(essentia.standard) if algo[0].isupper() ]
streaming_algo_list = [ algo for algo in dir(essentia.streaming) if algo[0].isupper() and algo not in [ 'CompositeBase'] ]


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

def algo_link(algoname, mode):
    mode_abbr = 'std' if mode == 'standard' else 'streaming'
    return "%s `(%s) <%s_%s.html>`__" % (algoname, mode, mode_abbr, algoname)


def related_algos(algo_doc):

    lines = []
    if algo_doc['mode'] == 'standard mode' and algo_doc['name'] in streaming_algo_list:
        lines.append(algo_link(algo_doc['name'], 'streaming'))
    elif algo_doc['mode'] == 'streaming mode' and algo_doc['name'] in std_algo_list:
        lines.append(algo_link(algo_doc['name'], 'standard'))

    lines += [algo_link(a, 'standard') for a in std_algo_list
                                            if a != algo_doc['name'] and
                                            algo_doc['description'].count(a)]

    lines += [algo_link(a, 'streaming') for a in streaming_algo_list
                                            if a != algo_doc['name'] and
                                            algo_doc['description'].count(a)]

    if lines:
        return ['See also',
                '--------',
                '',
                '\n'.join(sorted(lines))]
    return []


def doc2rst(algo_doc, sphinxsearch=False):
    if sphinxsearch:
        # dummy rst files used to append algorithms to the sphinx HTML search
        lines = [':orphan:',
                 ''
                 ]
        header = 'Algorithm reference - ' + algo_doc['name'] + ' (' + algo_doc['mode'] + ')'
        lines += [header, '=' * len(header), '']
    else:
        # actual rst files used to render HTMLs
        lines = [ algo_doc['name'], '=' * len(algo_doc['name']), '']
        lines += [algo_doc['mode'] + ' | ' + algo_doc['category'] + ' category', '']

    if algo_doc['inputs']:
        lines += [ 'Inputs',
                   '------',
                   ''
                   ]

        for inp in algo_doc['inputs']:
            lines.append(' - ``%s`` (*%s*) - %s' % (inp['name'], inp['type'], TR(inp['description'])))

        lines.append('')

    if algo_doc['outputs']:
        lines += [ 'Outputs',
                   '-------',
                   ''
                   ]

        for out in algo_doc['outputs']:
            lines.append(' - ``%s`` (*%s*) - %s' % (out['name'], out['type'], TR(out['description'])))

        lines.append('')

    if algo_doc['parameters']:
        lines += [ 'Parameters',
                   '----------',
                   ''
                   ]

        for param in algo_doc['parameters']:
            range_str = param['type']
            if param['range'] is not None:
                range_str += ' ∈ ' + TR(param['range'].replace(",", ", "))
            if param['default'] is not None:
                range_str += ', default = ' + TR(param['default'])

            lines.append(' - ``%s`` (*%s*) :' % (param['name'], range_str))
            lines.append('     ' + TR(param['description']))

        lines.append('')

    lines += [ 'Description',
               '-----------',
               '',
               TR_DESC(algo_doc['description'])
               ]

    lines += related_algos(algo_doc)

    return '\n'.join(lines)


def write_doc(filename, algo_doc):
    open(filename, 'w').write(doc2rst(algo_doc, sphinxsearch=True))


def rst2html(rst, layout_type):
    try:
        html, _ = subprocess.Popen([ 'rst2html.py', '-v', '--initial-header-level=2', '--input-encoding=UTF-8:strict' ],
                                   stdin=subprocess.PIPE,
                                   stdout = subprocess.PIPE,
                                   encoding='utf8').communicate(rst)
    except OSError:
        try:
            html, _ = subprocess.Popen([ 'rst2html', '-v', '--initial-header-level=2', '--input-encoding=UTF-8:strict' ],
                                       stdin=subprocess.PIPE,
                                       stdout = subprocess.PIPE,
                                       encoding='utf8').communicate(rst)

        except OSError:
            print('*'*100)
            print('ERROR: You need to install rst2html! (in the python docutils package)')
            print('*'*100)
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

    # folder for storing HTMLs
    try: os.mkdir('_templates/reference')
    except: pass

    # folder for storing RSTs for sphinx search index
    try: os.mkdir('reference')
    except: pass


    # write the algorithms reference organized by categories
    # generate html documentation for each algorithm and create an overall list of algorithms
    algos = {}
    for algoname in std_algo_list:
        algos.setdefault(algoname, {})
        algos[algoname]['standard'] = getattr(essentia.standard, algoname).__struct__
        # __struct__ does not contain mode (perhaps it should),
        # therefore, doing a workaround so that doc2rst can know the mode
        algos[algoname]['standard']['mode'] = 'standard mode'

        # Ugly patch for MusicExtractor and FreesoundExtractor to add links to
        # their detailed documentation.
        if algoname == 'MusicExtractor':
            algos[algoname]['standard']['description'] = \
                algos[algoname]['standard']['description'].replace("essentia_streaming_extractor_music",
                    "`essentia_streaming_extractor_music <../streaming_extractor_music.html>`__")

        elif algoname == 'FreesoundExtractor':
            algos[algoname]['standard']['description'] = \
                algos[algoname]['standard']['description'].replace("essentia_streaming_extractor_freesound",
                    "`essentia_streaming_extractor_freesound <../freesound_extractor.html>`__")

        print('generating doc for standard algorithm: %s ...' % algoname)
        write_doc('reference/std_' + algoname + '.rst', algos[algoname]['standard'])
        write_html_doc('_templates/reference/std_' + algoname + '.html',
                       algos[algoname]['standard'],
                       layout_type = 'std')

    for algoname in streaming_algo_list:
        algos.setdefault(algoname, {})
        algos[algoname]['streaming'] = getattr(essentia.streaming, algoname).__struct__
        algos[algoname]['streaming']['mode'] = 'streaming mode'

        print('generating doc for streaming algorithm: %s ...' % algoname)
        write_doc('reference/streaming_' + algoname + '.rst', algos[algoname]['streaming'])
        write_html_doc('_templates/reference/streaming_' + algoname + '.html',
                       algos[algoname]['streaming'],
                       layout_type = 'streaming')


    # write the template for the std algorithms
    html = '''
{% extends "layout.html" %}
{% block body %}

<div class="algo-description-container">
{% block algo_description %}
{% endblock %}
</div>
<div class="algo-list">
    <h4><strong>Standard algorithms</strong></h4>
'''

    links = ['<a href="std_%s.html">%s</a>' % (algoname, algoname) for algoname in std_algo_list]
    html += ' | '.join(links)

    html += '''
</div>

{% endblock %}
'''
    open('_templates/algo_description_layout_std.html', 'w').write(html)


    # write the template for the streaming algorithms
    html = '''
{% extends "layout.html" %}
{% block body %}

<div class="algo-description-container">
{% block algo_description %}
{% endblock %}
</div>

<div class="algo-list">
    <h4><strong>Streaming algorithms</strong></h4>

'''

    links = ['<a href="streaming_%s.html">%s</a>' % (algoname, algoname) for algoname in streaming_algo_list]
    html += ' | '.join(links)

    html += '''
</div>

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
    algo_categories_html = {}
    for algoname in algos:
        std_algo = None
        streaming_algo = None
        if 'standard' in algos[algoname]:
            std_algo = algos[algoname]['standard']
        if 'streaming' in algos[algoname]:
            streaming_algo = algos[algoname]['streaming']

        if std_algo and streaming_algo:
            # both standard and streaming mode exist
            if (std_algo['category'] == streaming_algo['category']):
                category = std_algo['category']
            else:
                print("WARNING: %s categories differ for standard (%s) and streaming (%s) modes" % (algoname, std_algo['category'], streaming_algo['category']))
                sys.exit()

            if (std_algo['description'] == streaming_algo['description']):
                description = std_algo['description'].split('.')[0]
            else:
                print("WARNING: %s description differ for standard (%s) and streaming (%s) modes" % (algoname, std_algo['description'], streaming_algo['description']))
                sys.exit()

        if std_algo:
            category = std_algo['category']
            description = std_algo['description'].split('.')[0]
        else:
            category = streaming_algo['category']
            description = streaming_algo['description'].split('.')[0]

        # Description for many algorithms starts with "This algorithm..." and we do not want to show that
        description = description.replace('This algorithm ', '')
        if len(description):
            description = description[0].capitalize() + description[1:]

        links = []
        if std_algo:
            links.append('<a class="reference internal" href="reference/std_' + algoname + '.html"><em>standard</em></a>')
        if streaming_algo:
            links.append('<a class="reference internal" href="reference/streaming_' + algoname + '.html"><em>streaming</em></a>')
        algo_html = '<div class="algo-info">' + '<header><h3>' + algoname + '</h3></header>' + '<span>(' + ', '.join(links) + ')</span>' + '<div>' + description + '</div></div>'
        algo_categories_html.setdefault(category, [])
        algo_categories_html[category].append(algo_html)


    html = '''
{% extends "layout.html" %}
{% set title = "Algorithms reference" %}
{% block body %}

  <div class="section" id="algorithms-reference">
<h1>Algorithms reference<a class="headerlink" href="#algorithms-reference" title="Permalink to this headline">¶</a></h1>
<p>Here is the complete list of algorithms which you can access from the Python interface.</p>
<p>The C++ interface allows access to the same algorithms, and also some more which are templated
and hence are not available in python.</p>

<div class="section" id="algorithms">
'''
    for category in algo_categories_html:
        category_id = re.sub('[^0-9a-zA-Z]+', '', category.lower())
        html += '<section><h2 id=' + category_id + '>' + category + '</h2>'
        html += '\n'.join(sorted(algo_categories_html[category]))
        html += '</section>'
    html += '''
</div>

</div>

{% endblock %}
'''

    open('_templates/algorithms_reference.html', 'w').write(html)


    # Copy convert FAQ.md to rst and copy to documenation folder
    subprocess.call(['pandoc', '../../FAQ.md', '-o', 'FAQ.rst'])

    # Convert research_papers.md to rst
    subprocess.call(['pandoc', 'research_papers.md', '-o', 'research_papers.rst'])


    # Convert python notebook tutorials to rst
    subprocess.call(['jupyter', 'nbconvert', '../../src/examples/tutorial/*.ipynb', '--to', 'rst', '--output-dir', '.'])

if __name__ == '__main__':
    print("Loading Essentia with python=%s" % sys.executable)
    write_algorithms_reference()
