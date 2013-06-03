#!/usr/bin/env python

# this script is used in the preinstall stage to find out whether python2.5 is
# installed

import sys

def isCorrectVersion() :
    version = sys.version.split()[0].split('.')
    return int(version[0]) >= 2 and int(version[1]) >= 5


if __name__ == '__main__':
    if isCorrectVersion():
        sys.exit(0);
    else: sys.exit(1);
