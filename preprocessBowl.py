import sys,glob
sys.path.append("/home/msmith/kaggle/lung/luna16/")
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, pdb, dicom
import scipy.ndimage
import matplotlib.pyplot as plt
import SimpleITK as sitk
from params import *
from crop import slicer,grouper
from tqdm import tqdm

def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, new_spacing

def preprocess(path):
    first_patient = load_scan(path)
    first_patient_pixels = get_pixels_hu(first_patient)
    pix_resampled, spacing = resample(first_patient_pixels, first_patient, [RESOLUTION,RESOLUTION,RESOLUTION])
    print("Patient --> {0}".format(first_patient))
    print("Shape before resampling\t", first_patient_pixels.shape)
    print("Shape after resampling\t", pix_resampled.shape)
    patient = path.split("/")[-2]
    wp = "preprocessedData/" + patient + "/"
    if not os.path.exists(wp):
        os.mkdir(wp)
    sitk.WriteImage(sitk.GetImageFromArray(pix_resampled),wp + "orig.nrrd")
    slicer(wp) # Make CUBES

if __name__ == "__main__":
    BOWL17_DIRS = glob.glob("/home/msmith/kaggle/lung/stage1/*/")
    BOWL17_DIRS.sort()
    print("Ther are {0} patients in BOWL 17.".format(len(BOWL17_DIRS)))
    prep = 1
    postPrep = 0
    count = 0
    assert prep + postPrep == 1, "Only doing prep or postprep not both"
    for i in tqdm(xrange(len(BOWL17_DIRS))):
        path = BOWL17_DIRS[i]
        if prep == 1:
            preprocess(path)
        elif postPrep == 1:
            preprocessedPath = "preprocessedData/"+ path.split("/")[-2] + "/"
            if os.path.exists(preprocessedPath + "sliced/sliced_0_yPred.bin"):
                print(preprocessedPath)
                count += 1
                grouper(preprocessedPath)
                print(count)




