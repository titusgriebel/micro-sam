import os
from natsort import natsorted
from glob import glob
from micro_sam.evaluation.evaluation import run_evaluation
from micro_sam.evaluation.inference import run_instance_segmentation_with_decoder

#rom util import get_paths  # comment this and create a custom function with the same name to run ais on your data
from util import get_pred_paths, get_default_arguments

def get_test_paths():
    image_path = '/mnt/lustre-grete/usr/u12649/scratch/data/pannuke_tif/fold3/images'
    label_path = '/mnt/lustre-grete/usr/u12649/scratch/data/pannuke_tif/fold3/labels'
    image_paths = natsorted(glob(os.path.join(image_path, '*.tiff')))
    gt_paths = natsorted(glob(os.path.join(label_path, '*.tiff')))
    print(len(image_paths), len(gt_paths))
    return image_paths, gt_paths
    
    
def get_val_paths():
    image_path = '/mnt/lustre-grete/usr/u12649/scratch/data/pannuke_tif/fold2/val_images'
    label_path = '/mnt/lustre-grete/usr/u12649/scratch/data/pannuke_tif/fold2/val_labels'
    image_paths = natsorted(glob(os.path.join(image_path, '*.tiff')))
    gt_paths = natsorted(glob(os.path.join(label_path, '*.tiff')))
    return image_paths, gt_paths

def run_instance_segmentation_with_decoder_inference(model_type, checkpoint, experiment_folder): #removed dataset_name as argument
    val_image_paths, val_gt_paths = get_val_paths()
    test_image_paths, _ = get_test_paths()
    prediction_folder = run_instance_segmentation_with_decoder(
        checkpoint,
        model_type,
        experiment_folder,
        val_image_paths,
        val_gt_paths,
        test_image_paths
    )
    return prediction_folder


def eval_instance_segmentation_with_decoder(prediction_folder, experiment_folder): #removed dataset_name as argument
    print("Evaluating", prediction_folder)
    _, gt_paths = get_test_paths()
    pred_paths = get_pred_paths(prediction_folder)
    save_path = os.path.join(experiment_folder, "results", "instance_segmentation_with_decoder.csv")
    res = run_evaluation(gt_paths, pred_paths, save_path=save_path)
    print(res)


def main():
    args = get_default_arguments()

    prediction_folder = run_instance_segmentation_with_decoder_inference(
        args.model, args.checkpoint, args.experiment_folder
    )
    eval_instance_segmentation_with_decoder(prediction_folder, args.experiment_folder)


if __name__ == "__main__":
    main()
