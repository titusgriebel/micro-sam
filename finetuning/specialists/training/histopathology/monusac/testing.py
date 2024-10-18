import os

def len_identical():
    identical = len(os.listdir('/scratch/users/u11644/models/evaluation/monusac_eval/instance_eval/instance_segmentation_with_decoder/inference')) == len(os.listdir('/scratch/users/u11644/data/monusac/monusac_test/complete_labels'))
    print(identical)
    print(len(os.listdir('/scratch/users/u11644/models/evaluation/monusac_eval/instance_eval/instance_segmentation_with_decoder/inference')))
    print(len(os.listdir('/scratch/users/u11644/data/monusac/monusac_test/complete_labels')))
   

#len_identical()
def make_experiment_folders(directory,organ_types=None):
    if organ_types is not None:
        for organ in organ_types:
            os.makedirs(os.path.join(directory,f'{organ}', 'amg_eval'))
            os.makedirs(os.path.join(directory,f'{organ}', 'instance_eval'))
            os.makedirs(os.path.join(directory,f'{organ}', 'it_prompt_box_eval'))
            os.makedirs(os.path.join(directory,f'{organ}', 'it_prompt_point_eval'))
    else:
        os.makedirs(os.path.join(directory,'complete_dataset', 'amg_eval'))
        os.makedirs(os.path.join(directory,'complete_dataset', 'instance_eval'))
        os.makedirs(os.path.join(directory,'complete_dataset', 'it_prompt_box_eval'))
        os.makedirs(os.path.join(directory,'complete_dataset', 'it_prompt_point_eval'))
organ_list = ['kidney', 'prostate', 'lung', 'breast']
make_experiment_folders('/scratch/users/u11644/models/evaluation/monusac_eval/',organ_list)