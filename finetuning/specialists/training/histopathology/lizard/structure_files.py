import os
import shutil
from glob import glob

def create_complete_dataset(path):
    datapath_list = glob(os.path.join(path,'split1','images', '*.tiff'))
    for split in ['split2', 'split3']:
        for item in split:
            
    print(len(datapath_list))
    new_list = []
    for item in 