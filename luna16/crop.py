import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, pdb, glob,cv2
from tqdm import tqdm

def showCrop(crop):
    cropMiddle = lungsCrop.shape[0]/2
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

if __name__ == "__main__":
    showImgs = 0 
    cropSize = 32 
    patients = glob.glob("preprocessedData/*/orig.nrrd")
    getCoords = lambda row: np.array([row.z,row.y,row.x])
    patients.sort()
    pdb.set_trace()
    for patient in tqdm(patients):
        patientDir = patient.replace("orig.nrrd","")
        csv = pd.read_csv(patient.replace("orig.nrrd","coord.csv"))
        nNodules = csv.shape[0]
        lungs = sitk.ReadImage(patient)
        lungs = sitk.GetArrayFromImage(lungs)
        mask = sitk.ReadImage(patient.replace("orig","mask"))
        mask = sitk.GetArrayFromImage(mask)
        dims = lungs.shape
        count = 0
        pdb.set_trace()
        for nodule in xrange(nNodules):
            noduleCoords = getCoords(csv.ix[nodule])
            noduleCoords = noduleCoords.round().astype(np.int16)
            start = noduleCoords - cropSize
            end = noduleCoords + cropSize
            if np.any(start < 0):
                print(start,end)
                where = np.where(start<0)
                start[where] = 0 
                end[where] = cropSize * 2
            elif np.any(dims-end < 0):
                print(start,end)
                where = np.where(dims-end<0)
                end[where] = dims[where]
                start[where] = dims[where] - crop*2


            lungsCrop = lungs[start[0]:end[0],start[1]:end[1],start[2]:end[2]]
            maskCrop = mask[start[0]:end[0],start[1]:end[1],start[2]:end[2]]
            wpX = patientDir + "x_{0}.nrrd".format(count)
            wpY = patientDir + "y_{0}.nrrd".format(count)
            sitk.WriteImage(sitk.GetImageFromArray(lungsCrop),wpX)
            sitk.WriteImage(sitk.GetImageFromArray(maskCrop),wpY)
            count += 1

            if showImgs == 1:
                show(lungs,noduleCoords)
                showCrop(lungsCrop)
                showCrop(maskCrop)



            
