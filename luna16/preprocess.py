import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import dicom,dicom.UID
from dicom.dataset import Dataset, FileDataset
from tqdm import tqdm
import datetime, time
import scipy.misc
import glob
import SimpleITK as sitk

IMAGE_PATHS = glob.glob("/home/msmith/luna16/subset*/*.mhd")
CANDIDATES = pd.read_csv("candidates_V2.csv")
ANNOTATIONS = pd.read_csv("annotations.csv")


def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    p = p[:,:,::-1]
    
    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

def read_ct_scan(folder_name):
        # Read the slices from the dicom file
        slices = [dicom.read_file(folder_name + filename) for filename in os.listdir(folder_name)]
        
        # Sort the dicom slices in their respective order
        slices.sort(key=lambda x: int(x.InstanceNumber))
        
        # Get the pixel values for all the slices
        slices = np.stack([s.pixel_array for s in slices])
        slices[slices == -2000] = 0
        return slices

def plot_ct_scan(scan):
    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(25, 25))
    for i in range(0, scan.shape[0], 5):
        plots[int(i / 20), int((i % 20) / 5)].axis('off')
        plots[int(i / 20), int((i % 20) / 5)].imshow(scan[i], cmap=plt.cm.bone) 

def show(im):
    plt.imshow(im,cmap=cm.gray); plt.show()


def get_segmented_lungs(im, plot=False):
    
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < 604
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone) 
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone) 
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(im, cmap=plt.cm.bone) 
        
    return im

def segment_lung_from_ct_scan(ct_scan):
    return np.asarray([get_segmented_lungs(slice) for slice in ct_scan])

def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)
    
    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)
    
    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    
    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    
    return ct_scan, origin, spacing

'''
This function is used to convert the world coordinates to voxel coordinates using 
the origin and spacing of the ct_scan
'''
def world_2_voxel(world_coordinates, origin, spacing):
    stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
    voxel_coordinates = stretched_voxel_coordinates / spacing
    return voxel_coordinates

'''
This function is used to convert the voxel coordinates to world coordinates using 
the origin and spacing of the ct_scan.
'''
def voxel_2_world(voxel_coordinates, origin, spacing):
    stretched_voxel_coordinates = voxel_coordinates * spacing
    world_coordinates = stretched_voxel_coordinates + origin
    return world_coordinates

def seq(start, stop, step=1):
	n = int(round((stop - start)/float(step)))
	if n > 1:
		return([start + step*i for i in range(n+1)])
	else:
		return([])

'''
This function is used to create spherical regions in binary masks
at the given locations and radius.
'''
def draw_circles(imageShape,cands,origin,spacing,resize_factor):
	#make empty matrix, which will be filled with the mask


	image_mask = np.zeros(imageShape.round().astype(np.uint16))

	#run over all the nodules in the lungs
        nodule_coords = []
        RESIZE_SPACING = resize_factor

	for ca in cands.values:
		#get middel x-,y-, and z-worldcoordinate of the nodule
		radius = np.ceil(ca[4])/2
		coord_x = ca[1]
		coord_y = ca[2]
		coord_z = ca[3]
		image_coord = np.array((coord_z,coord_y,coord_x))

		#determine voxel coordinate given the worldcoordinate
		image_coord = world_2_voxel(image_coord,origin,spacing)
                nodule_coords.append(image_coord)

		#determine the range of the nodule
		#noduleRange = seq(-radius, radius, SPACING[0])
		noduleRange = np.linspace(-radius, radius, 20)

		#create the mask
		for x in noduleRange:
			for y in noduleRange:
				for z in noduleRange:

					coords = world_2_voxel(np.array((coord_z+z,coord_y+y,coord_x+x)),origin,spacing)
                                        distance = np.linalg.norm(image_coord-coords)
					if distance < radius:
                                                coords = np.round(coords).astype(np.uint16)
						image_mask[int(coords[0]),int(coords[1]),int(coords[2])] = int(1)

        image_mask = image_mask.astype(np.int8)	
        nodule_coords = pd.DataFrame(nodule_coords)
        nodule_coords.columns = ["z","y","x"]
	return image_mask, nodule_coords

'''
This function takes the path to a '.mhd' file as input and 
is used to create the nodule masks and segmented lungs after 
rescaling to 1mm size in all directions. It saved them in the .npz
format. It also takes the list of nodule locations in that CT Scan as 
input.
'''
def rescale(imagePath):
        fp = imagePath.split("/")[-1].replace(".mhd","")
        savePath = "preprocessedData/" + fp + "/"
        if not os.path.exists(savePath):    
            os.mkdir(savePath)

        ##################### LOAD ######################
        start = time.time()
	img, origin, spacing = load_itk(imagePath)
        end = time.time()

        resize_factor = spacing / [1.0, 1.0, 1.0]
        new_real_shape = img.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize = new_shape / img.shape
        print("size {0} == > {1}".format(img.shape,new_real_shape))
        new_spacing = spacing / real_resize
	#calculate resize factor

        ##################### RESIZE ######################
        def resize():
            #resize image
            start = time.time()
            lung_img = scipy.ndimage.interpolation.zoom(img, real_resize)

            end = time.time()
            sitk.WriteImage(sitk.GetImageFromArray(lung_img),savePath+"orig.nrrd")
            print("Resizing took {0} seconds.".format(end-start))
    
        # Segment the lung structure
        def segment():
            start = time.time()
            lung_img = lung_img + 1024
            lung_mask = segment_lung_from_ct_scan(lung_img)
            lung_img = lung_img - 1024
            end = time.time()
            print("Segmenting took {0} seconds.".format(end-start))
            sitk.WriteImage(sitk.GetImageFromArray(lung_mask),savePath+"segment.nrrd")

        ##################### DRAW MASKS ######################
        annotations = getAnnotations(imagePath)
        start = time.time()
	nodule_mask, coords = draw_circles(new_real_shape,annotations,origin,new_spacing,resize_factor)
        end = time.time()
        print("Drawing masks took {0} seconds.".format(end-start))

        sitk.WriteImage(sitk.GetImageFromArray(nodule_mask),savePath+"mask.nrrd")
        coords.to_csv(savePath+"coord.csv",index=0)


def getAnnotations(imagePath):
    fp = imagePath.split("/")[-1].replace(".mhd","")
    loc = ANNOTATIONS.seriesuid==fp
    return ANNOTATIONS[loc]

def mkDirs():
    dirs = ["preprocessedData/","aug/"]
    for path in dirs:
        if not os.path.exists(path):
            os.mkdir(path)
        
if __name__ == "__main__":
    import pdb
    count = 0
    mkDirs()
    IMAGE_PATHS.sort()
    for i in tqdm(xrange(len(IMAGE_PATHS))):
        path = IMAGE_PATHS[i]
        nNodules = getAnnotations(path).shape[0]
        if nNodules > 0:
            print(IMAGE_PATHS[i])
            rescale(path)
            print("\n")





