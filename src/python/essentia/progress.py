#!/usr/bin/python

import sys
import logging

class Progress:
    def __init__(self, total, format = '%d%% done...'):
        if total < 0:
            raise IndexError('total needs to be >= 0')
        self.total = total
        self.current = 0
        self.format = '\r' + format

        # small hack: if total = 0, then we're already done, so pretend we
        # actually did something and just print 100%
        if total == 0:
            self.total = 1
            self.current = 1

    def percent(self, n):
        return n*100 / self.total

    def verbose(self):
        return logging.getLogger().isEnabledFor(logging.INFO)

    def update(self, current):
        shouldUpdate = (self.percent(current) > self.percent(self.current))
        self.current = current
        if shouldUpdate:
            self.updateDisplay()

    def updateDisplay(self):
        if self.verbose():
            print self.format % self.percent(self.current),
            sys.stdout.flush()

    def finish(self):
        self.update(self.total)
        # also account for the final end of line
        if self.verbose():
            print
