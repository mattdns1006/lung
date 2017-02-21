import glob
import numpy as np
from crop import showCrop

files = glob.glob("preprocessedData/*/aug_y*")
print(len(files))


maxes = []
for f in files[:2000]:
    f = np.fromfile(f,dtype=np.int8)
    f.resize(64,64,64)
    maxes.append(f.max())

maxes = np.array(maxes)
print(float((maxes==1).sum())/maxes.size)
    #showCrop(f)
