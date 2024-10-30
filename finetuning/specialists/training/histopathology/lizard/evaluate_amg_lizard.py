import os
from glob import glob
from natsort import natsorted
from micro_sam.evaluation.evaluation import run_evaluation
from micro_sam.evaluation.inference import run_amg
import random

#from evaluate_instance_segmentation_monusac import get_test_paths, get_val_paths

from util import get_default_arguments, get_pred_paths, VANILLA_MODELS


def get_all_paths(organ_type=None):
    if organ_type is not None:
        path = os.path.join('/mnt/lustre-grete/usr/u12649/scratch/data/lizard/loaded_dataset', organ_type)
    else:
        path = '/mnt/lustre-grete/usr/u12649/scratch/data/lizard/loaded_dataset/'
    complete_image_paths = []
    complete_label_paths = []

    for split in ['split1', 'split2', 'split3']:
        complete_image_paths = complete_image_paths+ glob(os.path.join(path, split, 'images/*'))
        complete_label_paths = complete_label_paths+ glob(os.path.join(path, split, 'labels/*'))
    complete_image_paths = natsorted(complete_image_paths)
    complete_label_paths = natsorted(complete_label_paths)
    val_count = round(len(complete_image_paths)*0.05)
    val_indices = random.sample(range(0, (len(complete_image_paths))), val_count)
    val_indices.sort()
    print(val_indices, 'test')
    val_image_paths = []
    val_label_paths = []
    val_indices = sorted(val_indices, reverse=True)
    for item in val_indices:
        val_image_path = complete_image_paths.pop(item)
        val_label_path = complete_label_paths.pop(item)
        val_image_paths.append(val_image_path)
        val_label_paths.append(val_label_path)
    
    assert len(val_image_paths) == len(val_label_paths)
    print(val_image_paths, val_label_paths)
    # breakpoint()    
    # for val in val_label_paths:
    #     index = complete_label_paths.index(val)
    #     assert index == complete_image_paths.index(val), 'different list indices'
    #     complete_image_paths.pop(index)
    #     complete_image_paths.pop(index)
    assert len(complete_image_paths) == len(complete_label_paths), 'Incongruity between test labels and images'
    assert len(val_image_paths) == len(val_label_paths), 'Incongruity between val labels and images'
    print(f'Complete images count: {len(complete_image_paths)}, val images count: {len(val_image_paths)}')
    #breakpoint()
    return complete_image_paths, complete_label_paths, val_image_paths, val_label_paths  

def get_val_paths():
    path = '/mnt/lustre-grete/usr/u12649/scratch/data/lizard/loaded_dataset/complete_dataset'
    val_image_paths = glob(os.path.join(path, 'val_images/*.tiff'))
    val_label_paths = glob(os.path.join(path, 'val_labels/*.tiff'))
    assert len(val_image_paths) == len(val_label_paths)
    return val_image_paths, val_label_paths

def get_test_paths():
    path = '/mnt/lustre-grete/usr/u12649/scratch/data/lizard/loaded_dataset/complete_dataset'
    test_image_paths = glob(os.path.join(path, 'images/*.tiff'))
    test_label_paths = glob(os.path.join(path, 'labels/*.tiff'))
    assert len(test_image_paths) == len(test_label_paths)
    return test_image_paths, test_label_paths

# def get_test_paths(organ_type):
#     assert organ_type is None
#     if organ_type is not None:
#             path = os.path.join('/mnt/lustre-grete/usr/u12649/scratch/data/lizard/loaded_dataset', organ_type)
#     else:
#         path = '/mnt/lustre-grete/usr/u12649/scratch/data/lizard/loaded_dataset/'
#     test_image_paths = natsorted(glob(os.path.join(path, 'images/*')))
#     test_label_paths = natsorted(glob(os.path.join(path, 'labels/*')))
#     print(len(test_image_paths), len(test_label_paths))
#     return test_image_paths, test_label_paths

def run_amg_inference(model_type, checkpoint, experiment_folder, organ_type=None):
    val_image_paths, val_gt_paths = get_val_paths()
    test_image_paths, _ = get_test_paths()
    prediction_folder = run_amg(
        checkpoint,
        model_type,
        experiment_folder,
        val_image_paths,
        val_gt_paths,
        test_image_paths
    )
    return prediction_folder

def eval_amg(prediction_folder, experiment_folder, organ_type=None):
    print("Evaluating", prediction_folder)
    _, gt_paths = get_test_paths() #test
    pred_paths = get_pred_paths(prediction_folder)
    save_path = os.path.join(experiment_folder, "results", "amg.csv")
    res = run_evaluation(gt_paths, pred_paths, save_path=save_path)
    print(res)


def main():
    args = get_default_arguments()
    if args.checkpoint is None:
        ckpt = VANILLA_MODELS[args.model]
    else:
        ckpt = args.checkpoint

    prediction_folder = run_amg_inference(args.model, ckpt, args.experiment_folder, args.organ_type)
    eval_amg(prediction_folder, args.experiment_folder, args.organ_type) #deleted args.dataset as an argument for eval_amg due to error occurence


if __name__ == "__main__":
    main()
