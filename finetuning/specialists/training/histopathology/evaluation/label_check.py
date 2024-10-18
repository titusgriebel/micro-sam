import numpy as np 
from glob import glob
import h5py 
import tifffile
import os
import PIL
from PIL import Image
import cv2
import skimage.io


def open_hdf5_file(file_path):
  try:
    with h5py.File(file_path, 'r') as f:
      # Do something with the HDF5 file
      print(f.keys())  # List dataset names
  except FileNotFoundError as e:
    print(f"Error opening HDF5 file: {e}")

file_path = '/scratch/users/u11644/data/pannuke/pannuke_fold_3.h5'
#open_hdf5_file(file_path)



def empty_labels(hdf5_file):
    with h5py.File(hdf5_file, 'r') as f:
        instance_labels = f['labels/instances']
        #print(instance_labels.shape)   
        empty_count=0          
        for i in range(instance_labels.shape[0]):
            img = instance_labels[i]
            img_as_npy = img.astype(np.uint8)
            unique_elements = np.unique(img_as_npy)
            if len(unique_elements) == 1:
               print(f'Image {i+1} does not contain labels!')
               empty_count+=1
    print(f'{empty_count} of the labels from hdf5 file do not contain labels')
#empty_labels('/scratch/users/u11644/data/pannuke/pannuke_fold_3.h5')            

image_path = '/scratch/users/u11644/data/monusac/monusac_test/by_organs/kidney/images'
label_path = '/scratch/users/u11644/data/monusac/monusac_test/complete_images'

def check_for_empty_tiff(path):
   empty_count = 0
   for filename in sorted(os.listdir(path)):
        image_path = os.path.join(path, filename)
        with tifffile.TiffFile(image_path) as tif:
            photo = Image.open(image_path)
            data = np.array(photo)
            unique_elements = np.unique(data)
            #print(np.unique(data))
            print(np.shape(data))
            if len(unique_elements) == 1:
               print(f'Image {(os.listdir(path).index(filename))+1} does not contain labels!')
               empty_count+=1
   print(f'{empty_count} labels were empty')
            

#check_for_empty_tiff(image_path)
   
def delete_alpha_channel(path):
   for filename in os.listdir(path):
      image_path = os.path.join(path, filename)
      #with tifffile.TiffFile(image_path) as tif:
      data = skimage.io.imread(image_path)
      print(data.shape)
         #data = np.array(Image.open(image_path))
      if data.shape[-1] == 4:
         cleansed_data = data[:,:,:3]
         #os.remove(filename)
         output_path = os.path.join(path, f'{filename}')
         tifffile.imwrite(output_path, cleansed_data)
         print(f'Image {(os.listdir(path).index(filename))+1} was successfully cleansed of its alpha channel')
delete_alpha_channel(image_path)

# data = skimage.io.imread('/scratch/users/u11644/data/monusac/monusac_test/complete_images/0001.tiff')
# shape = data.shape
# print(shape)