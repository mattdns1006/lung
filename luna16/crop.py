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


        for n in xrange(30): # make lots of data...

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

    split = int(0.9*df.shape[0])
    train = df.ix[:split]
    train = train.sample(frac=1).reset_index(drop=True)
    test = df.ix[split:]
    train.to_csv("trainCV.csv",index=0)
    test.to_csv("testCV.csv",index=0)
    print("CSVs made with train/test shapes = {0}/{1}".format(train.shape,test.shape))

    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv("train.csv",index=0)


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

def pad(arr,desiredShape):
    shape = arr.shape    
    difference = desiredShape - shape
    padding = ((0,difference[0]),(0,difference[1]),(0,difference[2]))
    arr = np.pad(arr,padding,"constant")
    return arr 

def slicer(patientDir):
    '''
    Split NRRD CT Scan file into 64/64/64 disjoint cubes for inference
    '''
    multiple = 4
    desired = np.multiply(IN_SIZE,multiple) 
    newshape = np.array(IN_SIZE)

    cuber = Cubify(oldshape=desired,newshape=newshape)
    

    sliceDir = patientDir + "sliced/"
    if not os.path.exists(sliceDir):
        os.mkdir(sliceDir)

    patient = patientDir + "/orig.nrrd"
    scan = sitk.ReadImage(patient)
    scan = sitk.GetArrayFromImage(scan)
    print(scan.dtype)
    shape = np.array(scan.shape)


    # crop the image to fit desired --- NOTE WE ARE LOSING INFO (EDGE) HERE BE CAREFUL
    excess = np.abs(desired - shape)
    excess1 = excess/2
    excess2 = excess - excess1
    scan = scan[excess1[0]:-excess2[0],excess1[1]:-excess2[1],excess1[2]:-excess2[2]]

    scan = pad(scan,desired)

    sitk.WriteImage(sitk.GetImageFromArray(scan),sliceDir + "orig.nrrd")

    scan = cuber.cubify(scan)
    nCubes = scan.shape[0]

    for arrNo in range(nCubes):
        wp = sliceDir + "sliced_{0}.bin".format(arrNo)
        scan[arrNo].tofile(wp)
    paths = glob.glob(sliceDir + "*.bin")
    y = ["dummy.bin" for x in paths]
    csv = pd.DataFrame({"x":paths,"y":y})
    csv.to_csv(sliceDir + "csv.csv", index=0)


def grouper(patientDir):
    multiple = 4
    desired = np.multiply(IN_SIZE,multiple) 
    newshape = np.array(IN_SIZE)
    cuber = Cubify(oldshape=desired,newshape=newshape)

    patientDir += "sliced/"
    nCubes = len(glob.glob(patientDir+"sliced_*_y.bin"))
    scan = np.empty((nCubes,IN_SIZE[0],IN_SIZE[1],IN_SIZE[2]))
    for i in xrange(nCubes):

        path = patientDir + "sliced_{0}_y.bin".format(i)
        img = np.fromfile(path,dtype=np.float32).reshape(IN_SIZE)
        scan[i] = img

    scan = np.array(scan)
    scan = cuber.uncubify(scan)
    sitk.WriteImage(sitk.GetImageFromArray(scan),patientDir + "predicted.nrrd")

    #showCrop(scan)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show",type=bool,help="show images")
    args = parser.parse_args()
    #aug(args.show,removePrevious=0)
    #clean()
    makeCsvs()
    patientDir = PATIENTS[1].replace("orig.nrrd","")
    print(patientDir)
    #slicer(patientDir)
    #grouper(patientDir)




            
