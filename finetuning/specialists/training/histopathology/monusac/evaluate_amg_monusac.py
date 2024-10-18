import os

from micro_sam.evaluation.evaluation import run_evaluation
from micro_sam.evaluation.inference import run_amg
from finetuning.specialists.training.histopathology.monusac.evaluate_instance_segmentation_monusac import get_test_paths, get_val_paths

from finetuning.specialists.training.histopathology.util import get_pred_paths, get_default_arguments, VANILLA_MODELS


def run_amg_inference(model_type, checkpoint, experiment_folder, organ_type=None):
    val_image_paths, val_gt_paths = get_val_paths(organ_type)
    test_image_paths, _ = get_test_paths(organ_type)
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
    _, gt_paths = get_test_paths(organ_type)
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
