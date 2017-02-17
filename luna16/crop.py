import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, pdb, glob,cv2,sys,argparse
import scipy.stats as stats
from params import *
from tqdm import tqdm
from cubify import Cubify

PATIENTS = glob.glob("preprocessedData/*/orig.nrrd")
PATIENTS.sort()

def showCrop(crop):
    cropMiddle = crop.shape[0]/2
    z = y = x = cropMiddle
    zAxis = crop[z]
    yAxis = crop[:,y]
    xAxis = crop[:,:,x]
    plt.figure(figsize=(20,20))
    plt.subplot(131)
    plt.imshow(zAxis,cmap=cm.gray);
    plt.subplot(132)
    plt.imshow(yAxis,cmap=cm.gray);
    plt.subplot(133)
    plt.imshow(zAxis,cmap=cm.gray);
    plt.show()


def show(img,coords):
    plt.figure(figsize=(20,20))
    z, y, x = coords 
    zAxis = img[z]
    yAxis = img[:,y]
    xAxis = img[:,:,x]
    #cv2.circle(zAxis,(x,y),10,color=(255,0,0),thickness=3)
    plt.subplot(131)
    ax = plt.gca()
    ax.imshow(zAxis,cmap=cm.gray);
    anno = plt.Circle((x,y),6,color="r",fill=False)
    ax.add_artist(anno)
    plt.subplot(132)
    ax = plt.gca()
    ax.imshow(yAxis,cmap=cm.gray);
    anno = plt.Circle((x,z),6,color="r",fill=False)
    ax.add_artist(anno)
    plt.subplot(133)
    ax = plt.gca()
    ax.imshow(xAxis,cmap=cm.gray);
    anno = plt.Circle((y,z),6,color="r",fill=False)
    ax.add_artist(anno)
    plt.show()

def aug(showImgs=0,removePrevious=0):
    if removePrevious == 1:
        pathsX = glob.glob("preprocessedData/*/aug_x*")
        for xPath in pathsX:
            yPath = xPath.replace("_x_","_y_")
            os.remove(xPath)
            os.remove(yPath)
    def showing():
        if showImgs == 1:
            show(lungs,noduleCoords)
            showCrop(lungsCrop)
            showCrop(maskCrop)
    cropSize = IN_SIZE[0]/2 
    maxTranslation = cropSize/3
    print("Max translation of nodule in x,y,z is +- {0}.".format(maxTranslation))
    saveSitk = 0
    getCoords = lambda row: np.array([row.z,row.y,row.x])
    totalCount = 0
    for patient in tqdm(PATIENTS[:]):
        patientDir = patient.replace("orig.nrrd","")
        csv = pd.read_csv(patient.replace("orig.nrrd","coord.csv"))
        nNodules = csv.shape[0]
        lungs = sitk.ReadImage(patient)
        lungs = sitk.GetArrayFromImage(lungs)
        mask = sitk.ReadImage(patient.replace("orig","mask"))
        mask = sitk.GetArrayFromImage(mask)
        dims = np.array(lungs.shape)
        count = 0

        # For every nodule make another file pair with no nodule in it (balanced)
        minRange = np.array([0,0,0]) + cropSize
        maxRange = dims - cropSize
        noNodule = False
        #showing()


        for n in xrange(5): # make lots of data...

            # Nodule data 
            for nodule in xrange(nNodules):
                noduleCoords = getCoords(csv.ix[nodule])
                noduleCoords = noduleCoords.round().astype(np.int16)
                noduleCoords += np.random.randint(-maxTranslation,maxTranslation,3) # Add some translation
                start = noduleCoords - cropSize
                end = noduleCoords + cropSize
                if np.any(start < 0):
                    where = np.where(start<0)
                    start[where] = 0 
                    end[where] = cropSize * 2
                elif np.any(dims-end < 0):
                    where = np.where(dims-end<0)
                    end[where] = dims[where]
                    start[where] = dims[where] - cropSize*2

                lungsCrop = lungs[start[0]:end[0],start[1]:end[1],start[2]:end[2]]
                maskCrop = mask[start[0]:end[0],start[1]:end[1],start[2]:end[2]]
                wpX = patientDir + "aug_x_{0}.bin".format(count)
                wpY = patientDir + "aug_y_{0}.bin".format(count)
                lungsCrop.tofile(wpX)
                maskCrop.tofile(wpY)
                count += 1
                if saveSitk == True:
                    wpX = patientDir + "aug_x_{0}.nrrd".format(count)
                    wpY = patientDir + "aug_y_{0}.nrrd".format(count)
                    sitk.WriteImage(sitk.GetImageFromArray(lungsCrop),wpX)
                    sitk.WriteImage(sitk.GetImageFromArray(maskCrop),wpY)
                #showCrop(lungsCrop)
                #showCrop(maskCrop)

            # Non (possibly) nodule data
            for randomCrop in xrange(nNodules):
                # Max and min ranges to sample from
                z, y, x = [np.random.randint(minRange[i],maxRange[i]) for i in [0,1,2]]
                noduleCoords = np.array([z,y,x])

                start = noduleCoords - cropSize
                end = noduleCoords + cropSize
                lungsCrop = lungs[start[0]:end[0],start[1]:end[1],start[2]:end[2]]
                maskCrop = mask[start[0]:end[0],start[1]:end[1],start[2]:end[2]]

                wpX = patientDir + "aug_x_{0}.bin".format(count)
                wpY = patientDir + "aug_y_{0}.bin".format(count)
                lungsCrop.tofile(wpX)
                maskCrop.tofile(wpY)
                count += 1
                if saveSitk == True:
                    wpX = patientDir + "aug_x_{0}.nrrd".format(count)
                    wpY = patientDir + "aug_y_{0}.nrrd".format(count)
                    sitk.WriteImage(sitk.GetImageFromArray(lungsCrop),wpX)
                    sitk.WriteImage(sitk.GetImageFromArray(maskCrop),wpY)


def makeCsvs():
    directory = "csvs/"
    if not os.path.exists(directory):
        os.mkdir(directory)
    clean()
    pathsX = glob.glob("preprocessedData/*/aug_x*")
    pathsY = [x.replace("_x","_y") for x in pathsX] 
    df = pd.DataFrame({"x":pathsX,"y":pathsY})
    os.chdir(directory)
    df.to_csv("train.csv",index=0)
    split = int(0.9*df.shape[0])
    train = df.ix[:split]
    test = df.ix[split:]
    train.to_csv("trainCV.csv",index=0)
    test.to_csv("testCV.csv",index=0)
    print("CSVs made with train/test shapes = {0}/{1}".format(train.shape,test.shape))

def clean():
    # Make sure all files are same size
    pathsX = glob.glob("preprocessedData/*/aug_x*")
    pathsY = [x.replace("_x","_y") for x in pathsX] 
    pathsToRemove = [x for x in pathsX if os.path.getsize(x) != 524288]
    for xPath in pathsToRemove:
        yPath = xPath.replace("_x_","_y_")
        os.remove(xPath)
        os.remove(yPath)
        print("Removed {0}.".format(xPath))

def slicer():
    '''
    Split NRRD CT Scan file into 64/64/64 disjoint cubes for inference
    '''

    for patient in PATIENTS[-1:]:
        sliceDir = "sliced/"
        patientDir = patient.replace("orig.nrrd","") + sliceDir
        if not os.path.exists(patientDir):
            os.mkdir(patientDir)

        scan = sitk.ReadImage(patient)
        scan = sitk.GetArrayFromImage(scan)
        shape = np.array(scan.shape)
        multiple = 4
        desired = np.multiply(IN_SIZE,multiple) 

        # crop the image to fit desired --- NOTE WE ARE LOSING INFO (EDGE) HERE BE CAREFUL
        excess = np.abs(desired - shape)
        excess /= 2
        scan = scan[excess[0]:-excess[0],excess[1]:-excess[1],excess[2]:-excess[2]]
        scan = cubify(scan,IN_SIZE)
        nCubes = scan.shape[0]
        pdb.set_trace()
        
        #while True:
        #    # Pad until divisible by 64
        #    if np.any(shape>desired) == True:
        #        multiple += 1
        #        desired = np.multiply(IN_SIZE,multiple) 
        #    else:
        #        difference = desired - shape
        #        padding = ((0,difference[0]),(0,difference[1]),(0,difference[2]))
        #        scan = np.pad(scan,padding,"constant")
        #        break

        #N = scan.shape[0]/IN_SIZE[0]
        #split = np.split(scan,N)
        #slices = []
        #slices = [np.split(x,N,1) for x in split]#
        #for j in xrange(N):
        #    for k in xrange(N):
        #        slices[j][k] = np.split(slices[j][k],N,2)

        #count = 0
        #for i in xrange(N):
        #    for j in xrange(N):
        #        for k in xrange(N):
        #            wp = patientDir + "sliced_{0}_{1}_{2}.bin".format(i,j,k)
        #            count += 1
        #            slices[i][j][k].tofile(wp)
        paths = glob.glob(patientDir + "*.bin")
        y = ["dummy.bin" for x in paths]
        csv = pd.DataFrame({"x":paths,"y":y})
        csv.to_csv(patientDir + "csv.csv", index=0)
        sitk.WriteImage(sitk.GetImageFromArray(scan),patientDir + "orig.nrrd")

def grouper(patientDir):
    N = 4
    scan = []
    for i in xrange(1):
	    for j in xrange(1):
		    for k in xrange(N):
			    suffix = "{0}_{1}_{2}".format(i,j,k)
			    path = patientDir + "sliced/sliced_{0}_fittedX.bin".format(suffix)
			    print(path)
			    img = np.fromfile(path,dtype=np.float32).reshape(IN_SIZE)
			    scan.append(img)
			    #showCrop(img)

    scan = np.array(scan)
    pdb.set_trace()
    showCrop(scan)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show",type=bool,help="show images")
    args = parser.parse_args()
    #aug(args.show,removePrevious=0)
    #clean()
    #makeCsvs()
    slicer()
    #grouper(PATIENTS[-1].replace("orig.nrrd",""))




            
