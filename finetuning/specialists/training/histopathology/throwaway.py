from torch_em.data import datasets
import micro_sam.training as sam_training
import torch 

checkpoint = torch.load('/scratch-grete/projects/nim00007/sam/models/vanilla/sam_vit_b_01ec64.pth')

print(checkpoint.keys())