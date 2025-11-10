import os
from splae.estimation.splae import SPLAE
from splae.models.epl.medsam import MedSamEPL
from splae.models.epl.morph import MorphEPL
from splae.utils import open_json

import torch
from splae.models import load_unet
from splae.datasets import load_dataset2d
from splae.datasets.transforms import get_transforms
from splae.evaluation.metrics import DiceMetric, ASSDMetric
from splae.estimation.utils import create_experiment_result_df, save_experiment_summary, \
    merge_and_save_results_by_grouped_keys, create_final_summary_per_grouped_keys
from monai.transforms import Compose, AsDiscrete
from datetime import datetime
from splae.visualization.visualizer import init_visualizer


def run_experiment(source,
                   target,
                   dataset_type,
                   adapt_type,
                   epl_type,
                   metrics=None,
                   timestamp=None,
                   device='cpu'):
    # Load datasets
    n_classes = 4 if dataset_type == 'MNMs' else 2
    transforms = get_transforms(dataset_type, to_onehot=False)
    ds_root = f'/home/dsi/nahum92/sfda/datasets/{dataset_type.lower()}_2d'
    if adapt_type == 'tent':
        seg_logits_subfolder = f'seg_logits/tent/{source}'
    else:
        seg_logits_subfolder = None
    eval_dataset = load_dataset2d(dataset_type, data_root=f'{ds_root}/{target}', transforms=transforms, split='test', seg_logits_subfolder=seg_logits_subfolder)
    post_transform = Compose([AsDiscrete(argmax=True, dim=1)])

    # Load source model
    source_model_path = f'/home/dsi/nahum92/sfda/experiments/train_source_{source}_{dataset_type}/checkpoints/best_model.pth'
    source_model = load_unet(source_model_path, n_classes, device=device)
    
    # Load adapted model
    target_model = None
    if adapt_type != 'tent':
        target_model_path = f'/home/dsi/nahum92/sfda/experiments/adapt_{source}_to_{target}_{adapt_type}_{dataset_type}/checkpoints/best_model.pth'
        target_model = load_unet(target_model_path, n_classes, device=device)

    if metrics is None:
        metrics = [DiceMetric()]
    exp_name = f'splae_{source}_to_{target}_{epl_type}_{dataset_type}'
    if timestamp is not None:
        exp_name = os.path.join(timestamp, exp_name)
    visualizer_cfg = dict(classes=eval_dataset.CLASS_LABELS, palette=list(eval_dataset.PALLETE.values()), save_dir=os.path.join('visualizations', exp_name))
    visualizer = init_visualizer(visualizer_cfg)
    if epl_type == 'medsam':
        epl_model = MedSamEPL(
            sam_model_type="vit_b",
            sam_weights_path="/home/dsi/nahum92/sfda/sam-med2d_b.pth",
            image_size=256,
            device=device,
            sample_times=10,
            num_classes=n_classes,
            ignore_index=None
        )
    elif epl_type == 'morph':
        epl_model = MorphEPL(num_classes=n_classes, ignore_index=None)
    else :
        epl_model = None
        print("EPL not recognized. Using raw source models as pseudo-label generator.")

    splae = SPLAE(source_model =source_model,
                    target_model=target_model,
                    num_classes=n_classes,
                    metrics=metrics,
                    epl_model=epl_model,
                    post_transform=post_transform,
                    visualizer=visualizer,
                    device=device)
    results = splae.run_evaluation(eval_dataset)
    results_df = create_experiment_result_df(results, class_labels=eval_dataset.CLASS_LABELS)
    return results_df, eval_dataset.CLASS_LABELS





def run_all_experiments():
    """
    Load dataset pairs from JSON file and run experiments for each pair with all classifiers
    """

    # Load dataset pairs from JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file_path = '/home/dsi/nahum92/sfda/dataset_pairs.json'
    datasets_pairs = open_json(json_file_path)
    if datasets_pairs is None:
        print("No dataset pairs to process. Exiting.")
        return []

    # Configuration
    metrics = [DiceMetric(), ASSDMetric()]
    epl_types = ['medsam', None]
    adapt_type = 'dpl'
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Available epl types: {epl_types}")
    output_dir = f'/home/dsi/nahum92/sfda/splae_experiments_results/{adapt_type}/{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    # Track results
    all_results = []
    total_experiments = 0

    # Calculate total experiments
    for dataset_pairs in datasets_pairs:
        pairs = dataset_pairs['source_target_pairs']
        total_experiments += len(pairs) * len(epl_types)

    print(f"Total experiments to run: {total_experiments}")

    experiment_count = 0

    # Run experiments for each dataset
    for dataset_pairs in datasets_pairs:
        dataset_name = dataset_pairs['dataset_name']
        source_target_pairs = dataset_pairs['source_target_pairs']

        print(f"\n{'=' * 80}")
        print(f"Processing dataset: {dataset_name}")
        print(f"Number of source-target pairs: {len(source_target_pairs)}")
        print(f"{'=' * 80}")

        # Run experiments for each source-target pair
        for source, target in source_target_pairs:
            print(f"\nProcessing pair: {source} -> {target}")

            # Run experiments with each classifier
            for epl_type in epl_types:
                if epl_type is None:
                    epl_type_str = 'no_epl'
                else:
                    epl_type_str = epl_type
                experiment_count += 1
                print(f"\n[{experiment_count}/{total_experiments}] Running experiment:")
                print(f"  Dataset: {dataset_name}")
                print(f"  Source: {source}")
                print(f"  Target: {target}")
                print(f"  EPL: {epl_type_str}")

                try:
                    # Run the experiment
                    result_df, class_labels = run_experiment(
                        source=source,
                        target=target,
                        dataset_type=dataset_name,
                        adapt_type=adapt_type,
                        epl_type=epl_type,
                        metrics=metrics,
                        timestamp=timestamp,
                        device=device
                    )

                    # 1) Save individual experiment summary
                    save_experiment_summary(
                        result_df, dataset_name, source, target, epl_type_str, class_labels, output_dir=output_dir, timestamp=timestamp
                    )

                    # Store result with metadata
                    experiment_result = {
                        'dataset_name': dataset_name,
                        'source': source,
                        'target': target,
                        'epl_type': epl_type_str,
                        'result_df': result_df,
                        'class_labels': class_labels,
                        'status': 'success'
                    }
                    all_results.append(experiment_result)

                    print(f"  ✓ Experiment completed successfully")

                except Exception as e:
                    print(f"  ✗ Experiment failed: {str(e)}")
                    #print backtrace
                    import traceback
                    traceback.print_exc()
                    experiment_result = {
                        'dataset_name': dataset_name,
                        'source': source,
                        'target': target,
                        'epl_type': epl_type_str,
                        'result': None,
                        'status': 'failed',
                        'error': str(e)
                    }
                    all_results.append(experiment_result)

    # Print summary
    print(f"\n{'=' * 80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total experiments: {total_experiments}")

    successful_experiments = [r for r in all_results if r['status'] == 'success']
    failed_experiments = [r for r in all_results if r['status'] == 'failed']

    print(f"Successful experiments: {len(successful_experiments)}")
    print(f"Failed experiments: {len(failed_experiments)}")

    if failed_experiments:
        print(f"\nFailed experiments:")
        for exp in failed_experiments:
            print(
                f"  - {exp['dataset_name']}: {exp['source']}->{exp['target']}, {exp['epl_type']}: {exp['error']}")

    # 2) Merge and save results by classifier and dataset
    print(f"\n{'=' * 80}")
    print("MERGING RESULTS BY EPL AND DATASET")
    print(f"{'=' * 80}")
    merged_files = merge_and_save_results_by_grouped_keys(all_results, output_dir=output_dir,
                                                          group_by_keys=['dataset_name', 'epl_type'],timestamp=timestamp)

    # 3) Create final summary per dataset per classifier
    print(f"\n{'=' * 80}")
    print("CREATING FINAL SUMMARIES PER DATASET PER EPL")
    print(f"{'=' * 80}")
    final_summaries = create_final_summary_per_grouped_keys(merged_files, output_dir=output_dir)

    print(f"\n{'=' * 80}")
    print("PROCESSING COMPLETE")
    print(f"{'=' * 80}")
    print(f"Individual experiment summaries: {len(successful_experiments)} files")
    print(f"Merged result files: {len(merged_files)} files")
    print(f"Final summary files: {len(final_summaries)} files")

    return all_results


def main():
    """
    Main function to run all RCA experiments
    """
    print("Starting SPLAE experiments...")
    results = run_all_experiments()

    if results:
        print(f"\nAll experiments completed. Total results: {len(results)}")
    else:
        print("No experiments were run.")


if __name__ == "__main__":
    main()