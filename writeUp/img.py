import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import matplotlib.cm as cm
import pandas as pd
import cv2
import pdb

def getPlanes(crop,shift=0,show=0):
    cropMiddle = crop.shape[0]/2 + shift
    z = y = x = cropMiddle
    zAxis = crop[z]
    yAxis = crop[:,y]
    xAxis = crop[:,:,x]
    if show == 1:
        plt.figure(figsize=(20,20))
        plt.subplot(131)
        plt.imshow(zAxis,cmap=cm.gray);
        plt.subplot(132)
        plt.imshow(yAxis,cmap=cm.gray);
        plt.subplot(133)
        plt.imshow(zAxis,cmap=cm.gray);
        plt.show()
    return [zAxis,yAxis,xAxis]

def main():
    cropSize = 32

    getCoords = lambda row: np.array([row.z,row.y,row.x])


    patient = "../luna16/preprocessedData/1.3.6.1.4.1.14519.5.2.1.6279.6001.216252660192313507027754194207/orig.nrrd"
    patientDir = patient.replace("orig.nrrd","")
    csv = pd.read_csv(patient.replace("orig.nrrd","coord.csv"))
    nNodules = csv.shape[0]
    nodule = 0
    noduleCoords = getCoords(csv.ix[nodule])
    lungs = sitk.ReadImage(patient)
    lungs = sitk.GetArrayFromImage(lungs)
    mask = sitk.ReadImage(patient.replace("orig","mask"))
    mask = sitk.GetArrayFromImage(mask)*255.0
    ext = "nrrd"
    count = 0

    start = noduleCoords - cropSize
    end = noduleCoords + cropSize
    start, end = [x.astype(np.uint16) for x in [start,end]]
    lungs = lungs.astype(np.float32)
    lungs = (lungs - lungs.min())/(lungs.max() - lungs.min())
    lungs *= 255.0

    lungsCrop = lungs[start[0]:end[0],start[1]:end[1],start[2]:end[2]].astype(np.uint8)
    maskCrop = mask[start[0]:end[0],start[1]:end[1],start[2]:end[2]]

    wpX = "imgs/x_{0}.{1}".format(count,ext)
    wpY = "imgs/y_{0}.{1}".format(count,ext)
    count += 1
    #if ext == "nrrd":
    #    sitk.WriteImage(sitk.GetImageFromArray(lungsCrop),wpX)
    #    sitk.WriteImage(sitk.GetImageFromArray(maskCrop),wpY)
    #else:
    #    lungsCrop.tofile(wpX)
    #    maskCrop.tofile(wpY)
    ext = "jpg"
    for plane in range(0,8,2):
        print(plane)
        show= 0 
        z,y,x = getPlanes(lungsCrop,shift = plane-4,show=0)
        z1,y1,x1 = getPlanes(maskCrop, shift= plane-4,show=0)
        wpz = "imgs/zl_{0}.{1}".format(plane,ext)
        wpz1 = "imgs/zm_{0}.{1}".format(plane,ext)
        wpy = "imgs/yl_{0}.{1}".format(plane,ext)
        wpy1 = "imgs/ym_{0}.{1}".format(plane,ext)
        wpx = "imgs/xl_{0}.{1}".format(plane,ext)
        wpx1 = "imgs/xm_{0}.{1}".format(plane,ext)
        cv2.imwrite(wpz,z)
        cv2.imwrite(wpy,y)
        cv2.imwrite(wpx,x)
        cv2.imwrite(wpz1,z1)
        cv2.imwrite(wpy1,y1)
        cv2.imwrite(wpx1,x1)



if __name__ == "__main__":
    main()
