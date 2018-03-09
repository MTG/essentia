

import os
import numpy as np
from scipy.signal import spectrogram
from collections import defaultdict

import essentia
from essentia.standard import *

from test_hpcp_parameters import *
from test_hpcp_plots import *


def testWindowWholeRange(signals):
    # 'test0': 'HPCP and window type in different scales'
    testKey = 'testWindowWholeRange'
    testTitle = testTitles[testKey]
    directory = testKey
    if not os.path.exists(directory):
        os.mkdir(directory)        
    os.chdir(directory)
    # Comparison of window types:
    windowtypes = ['hamming',
                 'hann',
                 'triangular',
                 'square',
                 'blackmanharris62',
                 'blackmanharris70',
                 'blackmanharris74',
                 'blackmanharris92']
    # Comparison of signals:
    signalkey = test2SignalsMapping[testKey]

    frameSize = frameSize_
    hopSize = frameSize / 2
    # hpcp and window instantiation
    hpcp = HPCP(size=hpcpSize, bandPreset=False, minFrequency=12, maxFrequency=12000)
    w = Windowing()
    # html string
    text = parametersText(hpcp, w, frameSize_, hopSize)
    html = ["<p>" + "Params: " + text + "</p>\n"]
    for skey in signalkey:
        signal = signals[skey]
        signalname = skey
        for windowtype in windowtypes:
            title = "signal=" + signalname + " windowtype=" + windowtype
            w = Windowing(type=windowtype, normalized=win_normalized)
            speaks = SpectralPeaks(maxPeaks=speaks_max)
            html.append(plotHPCP(signal, signalname, hpcp, w, speaks, frameSize, hopSize, title))
            close('all')
    html = "\n\n\n".join(html)
    Html_file= open(testTitle + '.html',"w")
    Html_file.write(html)
    Html_file.close()
    os.chdir('..')

    
def testFrameSize(signals):
    # 'test1': 'HPCP, window type and frame size in the lower scales', 
    testKey = 'testFrameSize'
    testTitle = testTitles[testKey]
    directory = testKey
    if not os.path.exists(directory):
        os.mkdir(directory)        
    os.chdir(directory)
    # Comparison of frames 
    frameSize = 2048*np.power(2,np.arange(3))
    hopSize = frameSize / 2
    # Comparison of signals:
    signalkey = test2SignalsMapping[testKey]
    # Comparison of window types:
    windowtypes = ['hamming','hann','blackmanharris74']
    # hpcp and window instantiation
    hpcp = HPCP(size=hpcpSize, bandPreset=False, minFrequency=12, maxFrequency=12000)
    w = Windowing()
    # html string
    text = parametersText(hpcp, w, frameSize, hopSize)
    html = ["<p>" + "Params: " + text + "</p>\n"]
    for skey in signalkey:
        signal = signals[skey]
        signalname = skey
        for wtype in windowtypes:
            w = Windowing(type=wtype, normalized=win_normalized)
            for frSize in frameSize:
                hopSize = frSize / 2
                title = "signal=" + signalname + " windowtype=" + wtype + " frSize=" + str(frSize)
                speaks = SpectralPeaks(maxPeaks=speaks_max)
                html.append(plotHPCP(signal, skey, hpcp, w, speaks, frSize, hopSize, title))
                close('all')
    html = "\n\n\n".join(html)
    Html_file= open(testTitle + ".html","w")
    Html_file.write(html)
    Html_file.close()
    os.chdir('..')
    
def testMaxFreq(signals):
    # 'test2': 'HPCP, window type and max frequency in the upper octaves',
    testKey = 'testMaxFreq'
    testTitle = testTitles[testKey]
    directory = testKey
    if not os.path.exists(directory):
        os.mkdir(directory)        
    os.chdir(directory)
    # Comparison of HPCP maximum frequency 
    maxFreqList = 1000.*np.array([2., 4., 7., 10., 13.])
    # Comparison of signals:
    signalkey = test2SignalsMapping[testKey]
    # Comparison of window types:
    windowtypes = ['hamming','hann','blackmanharris74']
    
    frameSize = frameSize_
    hopSize = frameSize / 2
    # hpcp and window instantiation
    hpcp = HPCP(size=hpcpSize, bandPreset=False, minFrequency=12, maxFrequency=12000)
    w = Windowing()
    # html string
    text = parametersText(hpcp, w, frameSize, hopSize)
    html = ["<p>" + "Params: " + text + "</p>\n"]
    for skey in signalkey:
        signal = signals[skey]
        signalname = skey
        for wtype in windowtypes:
            w = Windowing(type=wtype, normalized=win_normalized)
            for maxFreq in maxFreqList:
                hpcp.maxFrequency = maxFreq
                title = "signal=" + signalname + " windowtype=" + wtype + " maxFrequency=" + str(maxFreq)
                speaks = SpectralPeaks(maxPeaks=speaks_max)
                html.append(plotHPCP(signal, skey, hpcp, w, speaks, frameSize, hopSize, title))
                close('all')
    html = "\n\n\n".join(html)
    Html_file= open(testTitle + ".html","w")
    Html_file.write(html)
    Html_file.close()
    os.chdir('..')
    
def testNonLinearPostProcessing(signals):
    # 'test3': 'non-linear post-processing (nonLinear parameter)',
    testKey = 'testJordiNL'
    testTitle = testTitles[testKey]
    directory = testKey
    if not os.path.exists(directory):
        os.mkdir(directory)        
    os.chdir(directory)
    # Comparison of non-linear post-processing parameter
    NLList = [True, False]
    # Comparison of signals:
    signalkey = test2SignalsMapping[testKey]
    # Comparison of window types:
    windowtypes = ['hamming','hann','blackmanharris74']
    
    frameSize = frameSize_
    hopSize = frameSize / 2
    # hpcp and window instantiation
    hpcp = HPCP(size=hpcpSize, bandPreset=False, minFrequency=12, maxFrequency=12000)
    w = Windowing()
    # html string
    text = parametersText(hpcp, w, frameSize, hopSize)
    html = ["<p>" + "Params: " + text + "</p>\n"]
    for skey in signalkey:
        signal = signals[skey]
        signalname = skey
        for wtype in windowtypes:
            w = Windowing(type=wtype, normalized=win_normalized)
            for nonLinearItem in NLList:
                hpcp.nonLinear = nonLinearItem
                title = "signal=" + signalname + " windowtype=" + wtype + " nonLinear=" + str(nonLinearItem)
                speaks = SpectralPeaks(maxPeaks=speaks_max)
                html.append(plotHPCP(signal, skey, hpcp, w, speaks, frameSize, hopSize, title))
                close('all')
    html = "\n\n\n".join(html)
    Html_file= open(testTitle + ".html","w")
    Html_file.write(html)
    Html_file.close()
    os.chdir('..')

def testWeigth(signals):
    # 'test4': 'Weigthing function and window type',
    testKey = 'testWeigth'
    testTitle = testTitles[testKey]
    directory = testKey
    if not os.path.exists(directory):
        os.mkdir(directory)        
    os.chdir(directory)
    # Comparison of HPCP weigth function 
    weigthList = ['none','cosine','squaredCosine']
    # Comparison of signals:
    signalkey = test2SignalsMapping[testKey]
    # Comparison of window types:
    windowtypes = ['hamming','hann','blackmanharris74']
    
    frameSize = frameSize_
    hopSize = frameSize / 2
    # hpcp and window instantiation
    hpcp = HPCP(size=hpcpSize, bandPreset=False, minFrequency=12, maxFrequency=12000)
    w = Windowing()
    # html string
    text = parametersText(hpcp, w, frameSize, hopSize)
    html = ["<p>" + "Params: " + text + "</p>\n"]
    for skey in signalkey:
        signal = signals[skey]
        signalname = skey
        for wtype in windowtypes:
            w = Windowing(type=wtype, normalized=win_normalized)
            for weigth in weigthList:
                hpcp.weigthType = weigth
                title = "signal=" + signalname + " windowtype=" + wtype + " weigth=" + str(weigth)
                speaks = SpectralPeaks(maxPeaks=speaks_max)
                html.append(plotHPCP(signal, skey, hpcp, w, speaks, frameSize, hopSize, title))
                close('all')
    html = "\n\n\n".join(html)
    Html_file= open(testTitle + ".html","w")
    Html_file.write(html)
    Html_file.close()
    os.chdir('..')
    
def testLPFilterMeanSpectrum(signals):
    # 'test5': 'Mean spectrum of low pass filtered white noise',
    testKey = 'testLPFilterMeanSpectrum'
    title = testTitles[testKey]
    directory = testKey
    if not os.path.exists(directory):
        os.mkdir(directory)            
    os.chdir(directory)
    # Frame configuration
    frameSize = frameSize_ # not sure if it should be frameSize = 2048 (now frameSize_ = 4096)
    hopSize = frameSize / 2
    # Comparison of signals:
    signalkey = test2SignalsMapping[testKey]
    # Comparison of window types:
    windowtypes = ['hamming','hann','blackmanharris74']
    # hpcp and window instantiation
    hpcp = HPCP(size=hpcpSize, bandPreset=False, minFrequency=12, maxFrequency=12000)
    w = Windowing()
    speaks = SpectralPeaks(maxPeaks=speaks_max)
    # html string
    text = parametersText(hpcp, w, frameSize, hopSize)
    html = ["<p>" + "Params: " + text + "</p>\n"]
    hpcpmeanpool = defaultdict(dict)
    spectrummeanpool = defaultdict(dict)
    for skey in signalkey:
        signal = signals[skey]
        signalname = skey
        for wtype in windowtypes:
            w = Windowing(type=wtype, normalized=win_normalized)
            _, spectrogram, _, hpcpmean,_ = extractHPCP(signal, frameSize, hopSize, w, speaks, hpcp, str(skey))
            hpcpmeanpool[wtype][signalname] = hpcpmean
            spectrummeanpool[wtype][signalname] = np.mean(spectrogram, axis=1)
    for wtype in windowtypes:
        plotTitle = title + ", windowtype " + wtype
        html.append(plotmeanHPPC_Spectrum(hpcpmeanpool,spectrummeanpool,signalkey,frameSize,hopSize,wtype,title))
    html = "\n\n\n".join(html)
    Html_file= open(title + ".html","w")
    Html_file.write(html)
    Html_file.close()
    os.chdir('..')

def testBPFilterMeanSpectrum(signals):
    # 'test6': 'Mean spectrum of band pass filtered white noise',
    testKey = 'testBPFilterMeanSpectrum'
    title = testTitles[testKey]
    directory = testKey
    if not os.path.exists(directory):
        os.mkdir(directory)            
    os.chdir(directory)
    # Frame configuration
    frameSize = frameSize_ # not sure if it should be frameSize = 2048 (now frameSize = 4096)
    hopSize = frameSize / 2
    # Comparison of signals:
    signalkey = test2SignalsMapping[testKey]
    # Comparison of window types:
    windowtypes = ['hamming','hann','blackmanharris74']
    # hpcp and window instantiation
    hpcp = HPCP(size=hpcpSize, bandPreset=False, minFrequency=12, maxFrequency=12000)
    w = Windowing()
    speaks = SpectralPeaks(maxPeaks=speaks_max)
    # html string
    text = parametersText(hpcp, w, frameSize, hopSize)
    html = ["<p>" + "Params: " + text + "</p>\n"]
    hpcpmeanpool = defaultdict(dict)
    spectrummeanpool = defaultdict(dict)
    for skey in signalkey:
        signal = signals[skey]
        signalname = skey
        for wtype in windowtypes:
            w = Windowing(type=wtype, normalized=win_normalized)
            _, spectrogram, _, hpcpmean,_ = extractHPCP(signal, frameSize, hopSize, w, speaks, hpcp, str(skey))
            hpcpmeanpool[wtype][signalname] = hpcpmean
            spectrummeanpool[wtype][signalname] = np.mean(spectrogram, axis=1)
    for wtype in windowtypes:
        plotTitle = title + ", windowtype " + wtype
        html.append(plotmeanHPPC_Spectrum(hpcpmeanpool,spectrummeanpool,signalkey,frameSize,hopSize,wtype,title))
    html = "\n\n\n".join(html)
    Html_file= open(title + ".html","w")
    Html_file.write(html)
    Html_file.close()
    os.chdir('..')
        
def testLPFilterFrameSpeaks(signals):
    # 'test7': 'Frame spectral peaks of low pass filtered white noise',
    testKey = 'testLPFilterFrameSpeaks'
    testTitle = testTitles[testKey]
    directory = testKey
    if not os.path.exists(directory):
        os.mkdir(directory)        
    os.chdir(directory)
    # Comparison of signals:
    signalkey = test2SignalsMapping[testKey]
    # hpcp
    hpcp = HPCP(size=hpcpSize, bandPreset=False, minFrequency=12, maxFrequency=12000)
    # Spectrum and speaks
    spectrum = Spectrum()
    speaks = SpectralPeaks(maxPeaks=speaks_max)
    # html string
    html = ["<p>" + testKey + "</p>\n"]
    for skey in signalkey:
        signal = signals[skey]
        signalname = skey
        # title = 'Spectrum ' + signalname
        pfreq, pmagn = speaks(spectrum(signal))
        html.append(plotSpectrum(pfreq, pmagn, signalname))
    html.append(plotSpectrumComparison(signals, signalkey, speaks, spectrum))
    html = "\n\n\n".join(html)
    Html_file= open(testTitle + ".html","w")
    Html_file.write(html)
    Html_file.close()
    os.chdir('..')
    
def testBPFilterFrameSpeaks(signals):
    # 'test8': 'Frame spectral peaks of band pass filtered white noise',
    testKey = 'testBPFilterFrameSpeaks'
    testTitle = testTitles[testKey]
    directory = testKey
    if not os.path.exists(directory):
        os.mkdir(directory)        
    os.chdir(directory)
    # Comparison of signals:
    signalkey = test2SignalsMapping[testKey]
    # hpcp
    hpcp = HPCP(size=hpcpSize, bandPreset=False, minFrequency=12, maxFrequency=12000)
    # Spectrum and speaks
    spectrum = Spectrum()
    speaks = SpectralPeaks(maxPeaks=speaks_max)
    # html string
    html = ["<p>" + testKey + "</p>\n"]
    for skey in signalkey:
        signal = signals[skey]
        signalname = skey
        # title = 'Spectrum ' + signalname
        pfreq, pmagn = speaks(spectrum(signal))
        html.append(plotSpectrum(pfreq, pmagn, signalname))
    html.append(plotSpectrumComparison(signals, signalkey, speaks, spectrum))
    html = "\n\n\n".join(html)
    Html_file= open(testTitle + ".html","w")
    Html_file.write(html)
    Html_file.close()
    os.chdir('..')

def testNormalizeWindow(signals):
    # 'test9': 'Window normalization for different window types',
    testKey = 'testNormalizeWindow'
    testTitle = testTitles[testKey]
    directory = testKey
    if not os.path.exists(directory):
        os.mkdir(directory)        
    os.chdir(directory)
    # Comparison of HPCP weigth function 
    windowNormalizedList = [False, True]
    # Comparison of signals:
    signalkey = test2SignalsMapping[testKey]
    # Comparison of window types:
    windowtypes = ['hamming','hann','blackmanharris74']
    # hpcp and window instantiation
    hpcp = HPCP(size=hpcpSize, bandPreset=False, minFrequency=12, maxFrequency=12000)
    w = Windowing()
    frameSize = frameSize_
    hopSize = frameSize / 2
    # html string
    text = parametersText(hpcp, w, frameSize, hopSize)
    html = ["<p>" + "Params: " + text + "</p>\n"]
    for skey in signalkey:
        signal = signals[skey]
        signalname = skey
        for wtype in windowtypes:
            for windowNormalized in windowNormalizedList:
                w = Windowing(type=wtype, normalized=windowNormalized)
                title = "signal=" + signalname + " windowtype=" + wtype + " windowNormalized=" + str(windowNormalized)
                speaks = SpectralPeaks(maxPeaks=speaks_max)
                html.append(plotHPCP(signal, skey, hpcp, w, speaks, frameSize, hopSize, title))
                close('all')
    html = "\n\n\n".join(html)
    Html_file= open(testTitle + ".html","w")
    Html_file.write(html)
    Html_file.close()
    os.chdir('..')
    
def testNormalizeHPCP(signals):
    # 'test10': 'HPCP normalization'
    testKey = 'testNormalizeHPCP'
    testTitle = testTitles[testKey]
    directory = testKey
    if not os.path.exists(directory):
        os.mkdir(directory)        
    os.chdir(directory)
    # Comparison of HPCP normalization 
    HPCPNormalizedList = ['none', 'unitSum', 'unitMax']
    # Comparison of signals:
    signalkey = test2SignalsMapping[testKey]
    # Comparison of window types:
    windowtypes = ['hamming','hann','blackmanharris74']
    # hpcp and window instantiation
    hpcp = HPCP(size=hpcpSize, bandPreset=False, minFrequency=12, maxFrequency=12000)
    w = Windowing()
    frameSize = frameSize_
    hopSize = frameSize / 2
    # html string
    text = parametersText(hpcp, w, frameSize, hopSize)
    html = ["<p>" + "Params: " + text + "</p>\n"]
    for skey in signalkey:
        signal = signals[skey]
        signalname = skey
        for wtype in windowtypes:
            w = Windowing(type=wtype, normalized=win_normalized)
            for HPCPNormalized in HPCPNormalizedList:
                hpcp.normalized = HPCPNormalized
                title = "signal=" + signalname + " windowtype=" + wtype + " windowNormalized=" + str(HPCPNormalized)
                speaks = SpectralPeaks(maxPeaks=speaks_max)
                html.append(plotHPCP(signal, skey, hpcp, w, speaks, frameSize, hopSize, title))
    html = "\n\n\n".join(html)
    Html_file= open(testTitle + ".html","w")
    Html_file.write(html)
    Html_file.close()
    os.chdir('..')

