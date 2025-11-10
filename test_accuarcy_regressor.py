import json
from splae.datasets.transforms import get_transforms
from splae.estimation.regressor import AccuracyEstimatorRegressor
from splae.evaluation.evaluator import MetricsEvaluator
from splae.evaluation.metrics import DiceMetric, ASSDMetric
from splae.estimation.utils import create_experiment_result_df
from splae.runner import AdaptationRunner
from splae.adapt.iplc import IPLCGenerator, PLVisualizer
from splae.models import load_unet
from splae.datasets import load_dataset2d
from monai.transforms import Compose, AsDiscrete
import torch
import os
import traceback
from tqdm import tqdm
from torch import nn
from monai.transforms import ScaleIntensity
from torch.utils.data import DataLoader
from splae.datasets.datasets import MNMsDataset2D, MSMDataset2D
from splae.estimation.base import BaseAccuracyEstimator
from splae.estimation.utils import create_experiment_result_df, save_experiment_summary, merge_and_save_results_by_grouped_keys, create_final_summary_per_grouped_keys

from monai.transforms import (
    ScaleIntensityd,
    NormalizeIntensityd,
    ClipIntensityPercentilesd,
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    ResizeWithPadOrCropd,
    Resized,
    Activations,
    AsDiscrete
)

def test(adapted_model, score_model_dice, score_model_assd, post_transforms, test_loader, evaluator, device):
    results = []
    img_transform = Compose([ScaleIntensity( minv=0, maxv=1.0)])
    with torch.no_grad():
        score_model_dice.eval()
        score_model_assd.eval()
        for i, data in enumerate(tqdm(test_loader)):
            img = data['img'].to(device)
            gt_mask = data['gt_mask'].to(device)
            if adapted_model is not None:
                seg = adapted_model(img)
                seg = seg.softmax(dim=1)
                argmax_seg = post_transforms(seg)
            else:
                logits = data['seg_logits'].to(device).squeeze(1)
                seg = logits.softmax(dim=1)
                argmax_seg = post_transforms(seg)
            img = img_transform(img)
            pred_score_dice = score_model_dice(img, seg).squeeze(0)
            pred_score_assd = score_model_assd(img, seg).squeeze(0)
            pred_score_assd = torch.exp(pred_score_assd)
            pred_score = {'Dice': pred_score_dice, 'ASSD': pred_score_assd}
            real_score = evaluator.process(argmax_seg, gt_mask)
            real_score = evaluator.finalize()
            zip_results = BaseAccuracyEstimator._zip_results(real_score, pred_score, data['metainfo']['img_name'][0])
            results.append(zip_results)
        results_df =create_experiment_result_df(results, test_loader.dataset.CLASS_LABELS)
        return results_df


def open_json(json_path):
    """
    Open and parse a JSON file
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {json_path}: {e}")
        return None

def run_experiment(source, target, ds_name, adapt_type, device):
    ds_root = f'/home/dsi/nahum92/sfda/datasets/{ds_name.lower()}_2d'
    num_classes = 4 if ds_name == 'MNMs' else 2

    if adapt_type == 'tent':
        seg_logits_subfolder = f'seg_logits/tent/{source}'
    else:
        seg_logits_subfolder = None
    transforms = get_transforms(ds_name, to_onehot=False, load_seg_logits=seg_logits_subfolder is not None)
    test_dataset = load_dataset2d(ds_name, data_root=f'{ds_root}/{target}', transforms=transforms, split='test', seg_logits_subfolder=seg_logits_subfolder)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    target_model = None
    if adapt_type != 'tent':
        target_model_path = f'/home/dsi/nahum92/sfda/experiments/adapt_{source}_to_{target}_{adapt_type}_{ds_name}/checkpoints/best_model.pth'
        target_model = load_unet(target_model_path, num_classes, device=device)

    dice_model_path = f'/home/dsi/nahum92/sfda/experiments/score_model_Dice_{source}_to_{target}_{ds_name}_{adapt_type}/checkpoints/best_model.pth'
    dice_score_model = AccuracyEstimatorRegressor(in_channels_img=1, in_channels_seg=num_classes,out_dims_score=num_classes, score_type="Dice").to(device)
    dice_score_model.load_state_dict(torch.load(dice_model_path))
    assd_model_path = f'/home/dsi/nahum92/sfda/experiments/score_model_ASSD_{source}_to_{target}_{ds_name}_{adapt_type}/checkpoints/best_model.pth'
    assd_score_model = AccuracyEstimatorRegressor(in_channels_img=1, in_channels_seg=num_classes,out_dims_score=num_classes, score_type="ASSD").to(device)
    assd_score_model.load_state_dict(torch.load(assd_model_path))
    post_transforms = Compose([AsDiscrete(argmax=True, dim=1)])


    metrics = [DiceMetric(),ASSDMetric()]
    evaluator = MetricsEvaluator(metrics, num_classes, device=device)
    results_df = test(target_model, dice_score_model, assd_score_model, post_transforms, test_loader, evaluator, device)
    return results_df, test_loader.dataset.CLASS_LABELS

    


def run_all_experiments():
    """
    Load dataset pairs from JSON file and run experiments for each pair with all classifiers
    """

    # Load dataset pairs from JSON file
    json_file_path = '/home/dsi/nahum92/sfda/dataset_pairs.json'
    datasets_pairs = open_json(json_file_path)
    if datasets_pairs is None:
        print("No dataset pairs to process. Exiting.")
        return []

    # Configuration

    classifiers_types = ["score_model"]
    adapt_type = 'iplc'
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Available classifiers: {classifiers_types}")
    output_dir = f'/home/dsi/nahum92/sfda/accuarcy_regressor_results4/{adapt_type}'
    
    # Track results
    all_results = []
    total_experiments = 0
    
    # Calculate total experiments
    for dataset_pairs in datasets_pairs:
        pairs = dataset_pairs['source_target_pairs']
        total_experiments += len(pairs) * len(classifiers_types)
    
    print(f"Total experiments to run: {total_experiments}")
    
    experiment_count = 0
    
    # Run experiments for each dataset
    for dataset_pairs in datasets_pairs:
        dataset_name = dataset_pairs['dataset_name']
        source_target_pairs = dataset_pairs['source_target_pairs']
        
        print(f"\n{'='*80}")
        print(f"Processing dataset: {dataset_name}")
        print(f"Number of source-target pairs: {len(source_target_pairs)}")
        print(f"{'='*80}")
        
        # Run experiments for each source-target pair
        for source, target in source_target_pairs:
            print(f"\nProcessing pair: {source} -> {target}")
            
            # Run experiments with each classifier
            for classifier_type in classifiers_types:
                experiment_count += 1
                print(f"\n[{experiment_count}/{total_experiments}] Running experiment:")
                print(f"  Dataset: {dataset_name}")
                print(f"  Source: {source}")
                print(f"  Target: {target}")
                print(f"  Classifier: {classifier_type}")
                
                try:
                    # Run the experiment
                    result_df, class_labels = run_experiment(
                        source=source,
                        target=target,
                        ds_name=dataset_name,
                        adapt_type=adapt_type,
                        device=device
                    )
                    
                    # 1) Save individual experiment summary
                    save_experiment_summary(
                        result_df, dataset_name, source, target, classifier_type, class_labels,output_dir=output_dir
                    )
                    
                    # Store result with metadata
                    experiment_result = {
                        'dataset_name': dataset_name,
                        'source': source,
                        'target': target,
                        'classifier_type': classifier_type,
                        'result_df': result_df,
                        'class_labels': class_labels,
                        'status': 'success'
                    }
                    all_results.append(experiment_result)
                    
                    print(f"  ✓ Experiment completed successfully")
                    
                except Exception as e:
                    # print the traceback
                    print(traceback.format_exc())
                    print(f"  ✗ Experiment failed: {str(e)}")
                    experiment_result = {
                        'dataset_name': dataset_name,
                        'source': source,
                        'target': target,
                        'classifier_type': classifier_type,
                        'result': None,
                        'status': 'failed',
                        'error': str(e)
                    }
                    all_results.append(experiment_result)
    
    # Print summary
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Total experiments: {total_experiments}")
    
    successful_experiments = [r for r in all_results if r['status'] == 'success']
    failed_experiments = [r for r in all_results if r['status'] == 'failed']
    
    print(f"Successful experiments: {len(successful_experiments)}")
    print(f"Failed experiments: {len(failed_experiments)}")
    
    if failed_experiments:
        print(f"\nFailed experiments:")
        for exp in failed_experiments:
            print(f"  - {exp['dataset_name']}: {exp['source']}->{exp['target']}, {exp['classifier_type']}: {exp['error']}")
    
    # 2) Merge and save results by classifier and dataset
    print(f"\n{'='*80}")
    print("MERGING RESULTS BY CLASSIFIER AND DATASET")
    print(f"{'='*80}")
    merged_files = merge_and_save_results_by_grouped_keys(all_results, output_dir=output_dir, group_by_keys=['dataset_name', 'classifier_type'])
    
    # 3) Create final summary per dataset per classifier
    print(f"\n{'='*80}")
    print("CREATING FINAL SUMMARIES PER DATASET PER CLASSIFIER")
    print(f"{'='*80}")
    final_summaries = create_final_summary_per_grouped_keys(merged_files, output_dir=output_dir)
    
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Individual experiment summaries: {len(successful_experiments)} files")
    print(f"Merged result files: {len(merged_files)} files")
    print(f"Final summary files: {len(final_summaries)} files")
    
    return all_results

def main():
    """
    Main function to run all RCA experiments
    """
    print("Starting RCA experiments...")
    results = run_all_experiments()
    
    if results:
        print(f"\nAll experiments completed. Total results: {len(results)}")
    else:
        print("No experiments were run.")

if __name__ == "__main__":
    main()
