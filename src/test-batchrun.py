import os
import sys
import tensorflow as tf

#from KW_SelectCutFrames import *
from KW_CopyPaste import *

vidpaths = open(sys.argv[1])
for vidpath in vidpaths:
    vidpath = vidpath.split('\n')[0]
    print('process video: ' + vidpath)
    print(main(vidpath))


vidpaths.close()
