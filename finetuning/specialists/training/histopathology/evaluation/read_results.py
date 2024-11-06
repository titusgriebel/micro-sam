import pandas as pd
import os

def read_instance_csv(path):
    result_dict = {
        'dataset':[],
        'msa':[],
        'sa50':[],
        'sa75':[]
    }
    for dataset in ['pannuke', 'lynsec', 'cryonuseg', 'lizard', 'tnbc']:
        dataset_path = os.path.join(path, f'{dataset}_eval', 'instance/results/instance_segmentation_with_decoder.csv')
        df = pd.read_csv(dataset_path)
        result_dict['msa'].append(df.loc[0, 'msa'])
        result_dict['sa50'].append(df.loc[0, 'sa50'])
        result_dict['sa75'].append(df.loc[0, 'sa75'])
        result_dict['dataset'].append(dataset)
    for dataset in ['monusac', 'monuseg']:
        dataset_path = os.path.join(path, f'{dataset}_eval', 'complete_dataset', 'instance/results/instance_segmentation_with_decoder.csv')
        df = pd.read_csv(dataset_path)
        #print(df.head())
        result_dict['msa'].append(df.loc[0, 'mSA'])
        result_dict['sa50'].append(df.loc[0, 'SA50'])
        result_dict['sa75'].append(df.loc[0, 'SA75'])
        result_dict['dataset'].append(dataset)
    df = pd.DataFrame(result_dict)
    print('Results of instance segmentation evaluation:')
    print(df.head(7))
    df.to_csv('/mnt/lustre-grete/usr/u12649/scratch/all_results.csv', index=False)


#read_instance_csv('/mnt/lustre-grete/usr/u12649/scratch/models/evaluation/')

def read_amg_csv(path):
    result_dict = {
        'dataset':[],
        'msa':[],
        'sa50':[],
        'sa75':[]
    }
    for dataset in ['pannuke','lynsec', 'cryonuseg', 'lizard', 'tnbc']:
        dataset_path = os.path.join(path, f'{dataset}_eval', 'amg/results/amg.csv')
        df = pd.read_csv(dataset_path)
        result_dict['msa'].append(df.loc[0, 'msa'])
        result_dict['sa50'].append(df.loc[0, 'sa50'])
        result_dict['sa75'].append(df.loc[0, 'sa75'])
        result_dict['dataset'].append(dataset)
    for dataset in ['monusac', 'monuseg']:
        dataset_path = os.path.join(path, f'{dataset}_eval', 'complete_dataset', 'amg/results/amg.csv')
        df = pd.read_csv(dataset_path)
        # print(df.head())
        result_dict['msa'].append(df.loc[0, 'mSA'])
        result_dict['sa50'].append(df.loc[0, 'SA50'])
        result_dict['sa75'].append(df.loc[0, 'SA75'])
        result_dict['dataset'].append(dataset)
    df = pd.DataFrame(result_dict)
    print('Results of amg evaluation:')
    print(df.head(7))
    df.to_csv('/mnt/lustre-grete/usr/u12649/scratch/all_amg_results.csv', index=False)
#read_amg_csv('/mnt/lustre-grete/usr/u12649/scratch/models/evaluation/')


def read_it_boxes_csv(path):
    result_dict = {
        'dataset':[],
        'msa_1st':[],
        'msa_8th':[],
        'sa50_1st':[],
        'sa50_8th':[],
        'sa75_1st':[],
        'sa75_8th':[]
    }
    for dataset in ['pannuke','lynsec', 'cryonuseg', 'lizard', 'tnbc']:
        dataset_path = os.path.join(path, f'{dataset}_eval', 'boxes/results/iterative_prompts_start_box.csv')
        df = pd.read_csv(dataset_path)
        # print(df.head(8))
        result_dict['msa_1st'].append(df.loc[0, 'msa'])
        result_dict['sa50_1st'].append(df.loc[0, 'sa50'])
        result_dict['sa75_1st'].append(df.loc[0, 'sa75'])
        result_dict['msa_8th'].append(df.loc[7, 'msa'])
        result_dict['sa50_8th'].append(df.loc[7, 'sa50'])
        result_dict['sa75_8th'].append(df.loc[7, 'sa75'])
        result_dict['dataset'].append(dataset)
    for dataset in ['monusac', 'monuseg']:
        dataset_path = os.path.join(path, f'{dataset}_eval', 'complete_dataset', 'boxes/results/iterative_prompts_start_box.csv')
        df = pd.read_csv(dataset_path)
        # print(df.head())
        result_dict['msa_1st'].append(df.loc[0, 'mSA'])
        result_dict['sa50_1st'].append(df.loc[0, 'SA50'])
        result_dict['sa75_1st'].append(df.loc[0, 'SA75'])
        result_dict['msa_8th'].append(df.loc[7, 'mSA'])
        result_dict['sa50_8th'].append(df.loc[7, 'SA50'])
        result_dict['sa75_8th'].append(df.loc[7, 'SA75'])
        result_dict['dataset'].append(dataset)
    df = pd.DataFrame(result_dict)
    print('Results of iterative prompting with boxes evaluation:')
    print(df.head(7))
    df.to_csv('/mnt/lustre-grete/usr/u12649/scratch/all_amg_results.csv', index=False)
read_it_boxes_csv('/mnt/lustre-grete/usr/u12649/scratch/models/evaluation/')

def read_it_points_csv(path):
    result_dict = {
        'dataset':[],
        'msa_1st':[],
        'msa_8th':[],
        'sa50_1st':[],
        'sa50_8th':[],
        'sa75_1st':[],
        'sa75_8th':[]
    }
    for dataset in ['pannuke','lynsec', 'cryonuseg', 'lizard', 'tnbc']:
        dataset_path = os.path.join(path, f'{dataset}_eval', 'points/results/iterative_prompts_start_point.csv')
        df = pd.read_csv(dataset_path)
        # print(df.head(8))
        result_dict['msa_1st'].append(df.loc[0, 'msa'])
        result_dict['sa50_1st'].append(df.loc[0, 'sa50'])
        result_dict['sa75_1st'].append(df.loc[0, 'sa75'])
        result_dict['msa_8th'].append(df.loc[7, 'msa'])
        result_dict['sa50_8th'].append(df.loc[7, 'sa50'])
        result_dict['sa75_8th'].append(df.loc[7, 'sa75'])
        result_dict['dataset'].append(dataset)
    for dataset in ['monusac', 'monuseg']:
        dataset_path = os.path.join(path, f'{dataset}_eval', 'complete_dataset', 'points/results/iterative_prompts_start_point.csv')
        df = pd.read_csv(dataset_path)
        # print(df.head())
        result_dict['msa_1st'].append(df.loc[0, 'mSA'])
        result_dict['sa50_1st'].append(df.loc[0, 'SA50'])
        result_dict['sa75_1st'].append(df.loc[0, 'SA75'])
        result_dict['msa_8th'].append(df.loc[7, 'mSA'])
        result_dict['sa50_8th'].append(df.loc[7, 'SA50'])
        result_dict['sa75_8th'].append(df.loc[7, 'SA75'])
        result_dict['dataset'].append(dataset)
    df = pd.DataFrame(result_dict)
    print('Results of iterative prompting with points evaluation:')
    print(df.head(7))
    df.to_csv('/mnt/lustre-grete/usr/u12649/scratch/all_amg_results.csv', index=False)
read_it_points_csv('/mnt/lustre-grete/usr/u12649/scratch/models/evaluation/')