import numpy as np
import essentia
from essentia.standard import *

from matplotlib.pyplot import imshow, show, plot, title, ylim, legend, figure, grid, GridSpec, subplot, text, tight_layout, yticks, locator_params, close
import matplotlib.pyplot as plt

from test_hpcp_parameters import *
from test_hpcp_extract import *


def plotSpectrum(pfreq, pmagn, signalname):
    fig = figure(figsize=(8, 5), dpi=80, facecolor='w', edgecolor='k')         
    title = 'Spectrum ' + str(signalname)
    fig.suptitle(title)
    plot(pfreq*fs/2, pmagn)   
    figname = title + '.png'
    fig.savefig(figname) 
    html_ = "<p><img src=\"" + figname + "\"></p>\n"
    close('all')
    return html_
    
def plotSpectrumComparison(signals, signalkey, speaks, spectrum):
    fig = figure(figsize=(8, 5), dpi=80)         
    for skey in signalkey:
        signal = signals[skey]
        pfreq, pmagn = speaks(spectrum(signal))
        plot(pfreq*fs/2, pmagn)   
    legend(signalkey)
    title = 'Spectrum BPF of WN'
    fig.suptitle(title)
    figname = title + '.png'
    fig.savefig(figname)
    close('all')
    html_ = "<p><img src=\"" + figname + "\"></p>\n"
    return html_

def plotHPCP(signal, signalname, hpcp, w, speaks, frameSize, hopSize, figtitle, show=False):
    hpcp_, _, _, hpcpmean, hpcpmedian = extractHPCP(signal, frameSize, hopSize, w, speaks, hpcp, signal)
    fig = figure(num=None, figsize=(8, 5), dpi=80, facecolor='w', edgecolor='k')         
    gs = GridSpec(4, 4)
    ax1 = subplot(gs[:,:-1])
    ax2 = subplot(gs[:,-1])
    ax1.imshow(hpcp_[::-1,:], aspect = 'auto') 
    ax1.set_title(figtitle)
    ax2.plot(hpcpmean,xrange(36))
    yticks(np.arange(36), np.arange(35,-1,-1))
    locator_params(axis='x', nbins=4)
    tight_layout()
    if show:
        show()
    figname = figtitle + '.png'
    fig.savefig(figname, bbox_inches='tight')
    html = "<p><img src=\"" + figname + "\"></p>\n"  
    return html

def plotmeanHPPC_Spectrum(hpcpmeanpool,spectrummean,signalsKey,frameSize,hopSize,wtype,title):
    figtitle = 'Mean HPCP ' + title + ' window type ' + str(wtype[0])
    fig = figure(num=None, figsize=(8, 5), dpi=80, facecolor='w', edgecolor='k')
    fignamehpcp = figtitle
    plt.title(figtitle)
    for skey in signalsKey:
        plot(hpcpmeanpool[str(wtype)][str(skey)])
        tight_layout()
    legend(signalsKey)
    # show()
    fignamehpcp = fignamehpcp + '.png'
    fig.savefig(fignamehpcp,bbox_inches='tight')
    close('all')                                            
    figtitle = 'Spectrum ' + title + ' window type ' + str(wtype[0])
    fig = figure(num=None, figsize=(8, 5), dpi=80, facecolor='w', edgecolor='k')
    fignamescptr = figtitle
    plt.title(figtitle)
    for skey in signalsKey:
        plot(spectrummean[str(wtype)][str(skey)])
        tight_layout()
    legend(signalsKey)
    fignamescptr = fignamescptr + '.png'
    fig.savefig(fignamescptr,bbox_inches='tight')
    close('all')
    html = "<p>" + title + "</p>\n" 
    html += "<p><img src=\"" + fignamehpcp + "\"></p>\n\n"  
    html += "<p><img src=\"" + fignamescptr + "\"></p>\n\n"      
    return html

def parametersText(hpcp, w, frameSize, hopSize):
    text = '<ul>' + \
    '\n<li>hpcp size ' + str(hpcp.paramValue('size')) + '</li>' \
    '\n<li>referenceFrequency ' + str(hpcp.paramValue('referenceFrequency')) + '</li>' \
    '\n<li>harmonics ' + str(hpcp.paramValue('harmonics')) + '</li>' \
    '\n<li>bandPreset ' + str(hpcp.paramValue('bandPreset')) + '</li>' \
    '\n<li>minFrequency ' + str(hpcp.paramValue('minFrequency')) + '</li>' \
    '\n<li>maxFrequency ' + str(hpcp.paramValue('maxFrequency')) + '</li>' \
    '\n<li>bandSplitFrequency ' + str(hpcp.paramValue('bandSplitFrequency')) + '</li>' \
    '\n<li>weightType ' + str(hpcp.paramValue('weightType')) + '</li>' \
    '\n<li>nonLinear ' + str(hpcp.paramValue('nonLinear')) + '</li>' \
    '\n<li>windowSize ' + str(hpcp.paramValue('windowSize')) + '</li>' \
    '\n<li>sampleRate ' + str(hpcp.paramValue('sampleRate')) + '</li>' \
    '\n<li>maxShifted ' + str(hpcp.paramValue('maxShifted')) + '</li>' \
    '\n<li>normalized ' + str(hpcp.paramValue('normalized')) + '</li>' \
    '\n<li>frame/hop size ' + str(frameSize) + '/' + str(hopSize) + '</li>' \
    '\n<li>window type ' + str(w.paramValue('type')) + '</li>\n</ul>'
    
    return text
