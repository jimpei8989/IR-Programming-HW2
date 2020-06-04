import os, sys, time
import pickle

_LightGray = '\x1b[38;5;251m'
_Bold = '\x1b[1m'
_Underline = '\x1b[4m'
_Orange = '\x1b[38;5;215m'
_SkyBlue = '\x1b[38;5;38m'
_Reset = '\x1b[0m'

SEED = 0x06902029

class EventTimer():
    def __init__(self, name = '', verbose = True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            print(_LightGray + '------------------ Begin "' + _SkyBlue + _Bold + _Underline + self.name + _Reset + _LightGray + '" ------------------' + _Reset, file = sys.stderr)
        self.beginTimestamp = time.time()
        return self

    def __exit__(self, type, value, traceback):
        elapsedTime = time.time() - self.beginTimestamp
        if self.verbose:
            print(_LightGray + '------------------ End "' + _SkyBlue + _Bold + _Underline + self.name + _Reset + _LightGray + ' (Elapsed ' + _Orange + f'{elapsedTime:.4f}' + _Reset + 's)" ------------------' + _Reset + '\n', file = sys.stderr)

    def gettime(self):
        return time.time() - self.beginTimestamp

def pickleSave(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)

def pickleLoad(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
