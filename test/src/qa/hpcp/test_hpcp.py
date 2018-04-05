from test_hpcp_parameters import *
from test_hpcp_generatesignals import *
from test_hpcp_tests import *
import os


def main():
    tests = ['testWindowWholeRange',
             'testFrameSize', 
             'testMaxFreq',
             'testJordiNL',
             'testWeigth',
             'testLPFilterMeanSpectrum',
             'testBPFilterMeanSpectrum',
             'testLPFilterFrameSpeaks',
             'testBPFilterFrameSpeaks', 
             'testNormalizeWindow',
             'testNormalizeHPCP']
    # Generate the needed signals
    signals = getSignalsList(tests)
    outputDirectory = 'results'
    initDirectory = os.getcwd()
    if not os.path.exists(outputDirectory):
        os.mkdir(outputDirectory)
    os.chdir(outputDirectory)
    # Do the tests
    for test in tests:
        print ("Running test " + test + "...")
        eval(test + '(signals)')
    os.chdir(initDirectory)

if __name__ == "__main__":
    import sys
    main()


