import sys, os
import essentia.standard as ess
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from AudioDSP.HPSS import median_filtering as MF
from AudioDSP.HPSS import HPSS_essentia
from AudioDSP.visualization import visualization as V
from AudioDSP import utils as U

def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)

def updateSpectrogam(mX, mX_tmp, flag):
    if mX is None:
        return mX_tmp
    else:
        return np.vstack((mX, mX_tmp))

messageToPrint = "\nHello Essentia!\n\
\tDONE: Enhanced Harmonic Spectrogram obtained!\n\
\tDONE: Enhanced Percussive Spectrogram perfectly matches the AudioDSP implementation\n\
\tDONE:Improve Harmonic Spectrogram (HPSS.cpp, vector/matrix issue)\n\
\tNEXT: Create first pull request\n\
\tNEXT: Test when parameters are changed (helloworld.py)\n\
\tNEXT: Create essentiaHPSS conda environment and use it as target for installation (waf, waf.sh)\n\
\tNEXT: store intermediate using YamlOutput (HPSS.cpp, helloworld.py)\n\
\tNEXT: find solution for the harmonic enhanced spectrogram time shifting issue\n\
\t\tUPDATE: adding buffer didn't solve: we need to ignore the output of the first frames to avoid the need to use 'np.roll' (HPSS.cpp, helloworld.py)\n\
\t\tUPDATE: ISSUE DETECTED: the first 'harmonicKernel/2 + 1' iteration generate a mH which is not legal (we should wait to be in the middle of the median filter to get the first output correct)\n\
\t\tUPDATE: PROPOSED SOLUTION: delay all the outputs by 'harmonicKernel/2 + 1' (processing necessary to have the first legitimate mH for the output. A buffer is needed to save the already valid value of mX and mP\n\
\tNEXT: call helloworld.py in a JupyterNotebook\n\
\n\
Things to ask:\n\
\tHarmonic enhanced spectrogram time shifting issue: I should know the first 'half median' spectrums to return the first one?\n\
\tHow much safe is to use '_frameSize' instead of mX.size()? Is it checked with 'const_spectrum.size() != _frameSize' I guess (HPSS.cpp) \n\
\tINHERIT('kernelSize') but I would like to name the parameter in HPSS\n\ 'PercussivKernelSize' and pass it as 'kernelSize' to medianFilter (HPSS.cpp))\n\
\tHow to check which version is more efficient? (HPSS.cpp, rolling of harmonic matrix)\
"

M = 1024
N = 1024
H = 512
fs = 44100
win_harm = 17
win_perc = 17
margin_harm = 1
margin_perc = 1
power = 2.0

spectrum = ess.Spectrum(size=N)
window = ess.Windowing(size=M, type='hann')
HPSS = ess.HPSS(frameSize=N)  # harmonicKernel=win_harm, percussiveKernel=win_perc)
    # harmonicMargin=margin_harm, percussiveMargin=margin_perc, power=power)


x = ess.MonoLoader(filename = 'test/audio/recorded/dubstep.wav', sampleRate = fs)() #audio file input
x = x[:fs]

# todo: computation is really slow. Must be the vstacking in python!
f = 0; mP_ess=None; mH_ess=None; mX=None; # flag first frame and declaration variables to be filled
for frame in ess.FrameGenerator(x, frameSize=M, hopSize=H, startFromZero=True): #frameGenerator divide the audio signal in frames
    #each frame is processed individually inside the for body
    mX_tmp = spectrum(window(frame))       #compute spectrum for single frame

    mP_tmp_ess, mH_tmp_ess = HPSS(mX_tmp) # function to test (creating only enhanced spectrograms now)

    mP_ess = updateSpectrogam(mP_ess, mP_tmp_ess, f); mH_ess = updateSpectrogam(mH_ess, mH_tmp_ess, f); mX = updateSpectrogam(mX, mX_tmp, f) ; f = 1

    # Computing harmonic and percussive masks
    # mask_harm, mask_perc = MF.compute_masks(mH, mP, power=power, margin_harm=margin_harm, margin_perc=margin_perc)

# y_perc = HPSS_essentia.compute_ISTFT(Y_perc, frameSize=frameSize, hopSize=hopSize, zeroPadding=zeroPadding,
# windowType=windowType, len=x.shape[0])

    # Computing harmonic and percussive enhanced magnitude spectrograms
mH_DSP, mP_DSP = MF.compute_enhanced_spectrograms(mX, win_harm=win_harm, win_perc=win_perc, test_mode=False)

# TODO: compare masking to create harmonic enhanced spectrogram with relevant plot

# time-shift
shift = (int(np.floor(win_harm / 2)))  # middle of the buffer
mH_ess = np.roll(mH_ess, -shift, 0)
# mP_ess = np.roll(mP_ess, -shift, 0)

[tx, fx] = U.getSpectrogramAxis(np.ndarray(shape=(mX.shape[0], N)), fs, H) # problem with second dimension of mX, should be doubled!

vmin=-60; vmax=U.amp2db(np.max([mX, mH_ess, mH_DSP])); cmap='YlGnBu'
fig = V.createFigure(title="Checking Harmonic Spectrogram")
originalSpectrogram_subplot = fig.add_subplot(4, 2, 1)
originalSpectrogram_subplot.set_title("Original Spectrogram")
originalSpectrogram_subplot.pcolormesh(tx, fx, np.transpose(U.amp2db(mX)), vmin=vmin, vmax=vmax, cmap='Greys')
essentia_subplot = fig.add_subplot(4, 2, 3)
essentia_subplot.set_title("Essentia Harmonic Spectrogram")
essentia_subplot.pcolormesh(tx, fx, np.transpose(U.amp2db(mH_ess)), vmin=vmin, vmax=vmax, cmap=cmap)
AudioDSP_subplot = fig.add_subplot(4, 2, 5)
AudioDSP_subplot.set_title("AudioDSP Harmonic Spectrogram")
AudioDSP_subplot.pcolormesh(tx, fx, np.transpose(U.amp2db(mH_DSP)), vmin=vmin, vmax=vmax, cmap=cmap)
diff_subplot = fig.add_subplot(4, 2, 7)
diff_subplot.set_title("Difference between Harmonic Spectrograms")
diff_subplot.pcolormesh(tx, fx, np.transpose(U.amp2db(np.abs(mH_DSP-mH_ess))), vmin=vmin, vmax=vmax, cmap='Reds')

vmin=-60; vmax=U.amp2db(np.max([mX, mP_ess, mP_DSP]))
originalSpectrogram_subplot = fig.add_subplot(4, 2, 2)
originalSpectrogram_subplot.set_title("Original Spectrogram")
originalSpectrogram_subplot.pcolormesh(tx, fx, np.transpose(U.amp2db(mX)), vmin=vmin, vmax=vmax, cmap='Greys')
essentia_subplot = fig.add_subplot(4, 2, 4)
essentia_subplot.set_title("Essentia Percussive Spectrogram")
essentia_subplot.pcolormesh(tx, fx, np.transpose(U.amp2db(mP_ess)), vmin=vmin, vmax=vmax, cmap=cmap)
AudioDSP_subplot = fig.add_subplot(4, 2, 6)
AudioDSP_subplot.set_title("AudioDSP Percussive Spectrogram")
AudioDSP_subplot.pcolormesh(tx, fx, np.transpose(U.amp2db(mP_DSP)), vmin=vmin, vmax=vmax, cmap=cmap)
diff_subplot = fig.add_subplot(4, 2, 8)
diff_subplot.set_title("Difference between HarPercussivemonic Spectrograms")
diff_subplot.pcolormesh(tx, fx, np.transpose(U.amp2db(np.abs(mP_DSP-mP_ess))), vmin=vmin, vmax=vmax, cmap='Reds')

move_figure(fig, 500, 500)

plt.show()

print(messageToPrint)