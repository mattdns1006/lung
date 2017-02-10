import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, pdb, glob,cv2,sys,argparse
from tqdm import tqdm

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

def main(showImgs=0,segmentation=0):
    def showing():
        if showImgs == 1:
            show(lungs,noduleCoords)
            showCrop(lungsCrop)
            showCrop(maskCrop)
    cropSize = 35 
    saveSitk = 0
    patients = glob.glob("preprocessedData/*/orig.nrrd")
    getCoords = lambda row: np.array([row.z,row.y,row.x])
    patients.sort()
    totalCount = 0
    for patient in tqdm(patients[:]):
        print(patient)
        patientDir = patient.replace("orig.nrrd","")
        csv = pd.read_csv(patient.replace("orig.nrrd","coord.csv"))
        nNodules = csv.shape[0]
        lungs = sitk.ReadImage(patient)
        lungs = sitk.GetArrayFromImage(lungs)
        mask = sitk.ReadImage(patient.replace("orig","mask"))
        mask = sitk.GetArrayFromImage(mask)
        dims = np.array(lungs.shape)
        count = 0
        for nodule in xrange(nNodules):
            noduleCoords = getCoords(csv.ix[nodule])
            noduleCoords = noduleCoords.round().astype(np.int16)
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


        # For every nodule make another file pair with no nodule in it (balanced)
        minRange = np.array([0,0,0]) + cropSize
        maxRange = dims - cropSize
        noNodule = False
        #showing()


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
        totalCount += count
        if totalCount % 100 == 0:
            print("Total count so far = {0}.".format(totalCount))


def makeCsvs():
    directory = "csvs/"
    if not os.path.exists(directory):
        os.mkdir(directory)

    pathsX = glob.glob("preprocessedData/*/aug_x*")
    pathsY = [x.replace("_x","_y") for x in pathsX] 
    df = pd.DataFrame({"x":pathsX,"y":pathsY})
    os.chdir(directory)
    df.to_csv("train.csv",index=0)
    split = int(0.85*df.shape[0])
    train = df.ix[:split]
    test = df.ix[split:]
    train.to_csv("trainCV.csv",index=0)
    test.to_csv("testCV.csv",index=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show",type=bool,help="show images")
    args = parser.parse_args()
    #main(args.show)
    makeCsvs()




            
