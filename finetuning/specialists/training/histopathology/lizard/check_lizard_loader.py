import os
import shutil
from glob import glob
import argparse

import torch

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model

from lizard2 import get_dataloaders

def check_lizard_loaders():
    patch_shape = (1, 512, 512)
    testloader = get_dataloaders(patch_shape,'/mnt/lustre-grete/usr/u12649/scratch/data/lizard', 'split1')
    for x, y in testloader:
        print('Image  shape:', x.shape)
        print('Label  shape:', y.shape)
        break


check_lizard_loaders()

