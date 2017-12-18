import essentia
from essentia.standard import *

from test_hpcp_parameters import *
from test_hpcp_generatesignals import *
from test_hpcp_tests import *



def main():
    # List of the tests to take
    # It should be an input argument
    # Put all the tests you want to run
    # tests = ['testBPFilterFrameSpeaks']
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
    signals, signalsKey = getSignalsList(tests)

    # Do the tests
    for test in tests:
        print (test)
        eval(test + '(signals,signalsKey)')




if __name__ == "__main__":
    import sys
    main()


