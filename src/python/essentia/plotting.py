import os
import pylab
import math


def descriptorPlot(name, xData, yData, pool, options):

    pylab.figure()
    pylab.title(name)
    pylab.plot(xData, yData)
    duration = int(math.floor(pool.value('metadata.duration_processed')))
    maxValue = max(yData)
    minValue = min(yData)

    # plotting segments lines
    if options['segmentation']['doSegmentation']:
        segments = pool.value('segmentation.timestamps')
        for segment in segments:
            pylab.plot([segment[0], segment[0]], [minValue, maxValue], 'r-')
            pylab.plot([segment[1], segment[1]], [minValue, maxValue], 'r-')

    pylab.axis([-2, duration + 2, minValue, maxValue])
    if not os.path.exists('plots'):
        os.mkdir('plots')
    figureName = 'plots/' + name + '.png'
    print 'Plotting ' + name + '...'
    pylab.savefig(figureName)

    return figureName


def descriptorPlotHTML(namespace, name, audio, pool, options):

    try:
        # plot name
        descName = namespace + '.' + name
        # plot x data
        scopes = pool.value(namespace + '.' + name + '.' + 'scope']
        descScopes = []
        for scope in scopes:
            descScopes.append(scope[0])
        # plot y data
        descValues = pool.value(namespace + '.' + name)
        # plotting
        try:
            figureName = descriptorPlot(descName, descScopes, descValues, pool, options)
            htmlCode = '<img src ="' + figureName  + '"/>'
        except RuntimeError:
            # special case: descriptors with more than one dimension (mfcc, barkbands, etc...)
            htmlCode = ''
            for i in range(len(descValues[0])):
                descSubName = descName + '.' + str(i)
                descSubValues = []
                for value in descValues:
                    descSubValues.append(value[i])
                figureName = descriptorPlot(descSubName, descScopes, descSubValues, pool, options)
                htmlCode += '<img src ="' + figureName  + '"/>'

    except KeyError:
        htmlCode = ''
        print "WARNING: the descriptor", descName, "doesn't exist"

    return htmlCode


def namespace_comp(ns1, ns2):
    if ns1 == 'special': return -1
    if ns2 == 'special': return 1
    return cmp(ns1, ns2)


def compute(inputFilename, audio, pool, options):

    htmlCode = '<p align="center"><b><font size=6>' + inputFilename + '</font></b></p>'
    html = False
    descriptors = options['plotsList']
    namespaces = [ ns for ns in descriptors ]
    namespaces.sort(namespace_comp)
    htmlFile = open(inputFilename + '.html', 'w')
    htmlCode = '<p align="center"><b><font size=6>' + inputFilename + '</font></b></p>'
    html = False

    # plot signal
    descName = "signal"
    descValues = audio
    descScopes = []
    for time in range(len(audio)):
        descScopes.append(time / options['sampleRate'])
    figureName = descriptorPlot(descName, descScopes, descValues, pool, options)
    htmlCode += '<img src ="' + figureName  + '"/>'

    # plot descriptors, one by one
    for namespace in namespaces:
        names = [ n for n in descriptors[namespace] ]
        names.sort()
        for name in names:
            htmlCode += descriptorPlotHTML(namespace, name, audio, pool, options)

    # write HTML file
    htmlFile = open(inputFilename + '.html', 'w')
    htmlFile.write(htmlCode)
    htmlFile.close()

