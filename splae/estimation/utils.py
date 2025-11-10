from datetime import datetime
import os

import numpy as np
import pandas as pd

def create_experiment_result_df(results, class_labels):
    data = []
    for result in results:
        row = {'img_name': result['img_name']}
        certainty_score = result.get('certainty_score', None)
        if certainty_score is not None:
            row['certainty_score'] = certainty_score
        reliability_score = result.get('reliability_score', None)
        if reliability_score is not None:
            row['reliability_score'] = reliability_score
        forground_ratio = result.get('forground_ratio', None)
        if forground_ratio is not None:
            row['forground_ratio'] = forground_ratio
        results_by_metric = result['results_by_metric']
        for metric in results_by_metric.keys():
            real_scores = results_by_metric[metric]['real_score']
            pred_scores = results_by_metric[metric]['pred_score']
            gaps = results_by_metric[metric]['gaps']
            row[f'mean_score_{metric}'] = float(np.mean(real_scores))
            row[f'mean_pred_{metric}'] = float(np.mean(pred_scores))
            row[f'mean_gap_{metric}'] = float(np.mean(gaps))
            for class_id, (real_score, pred_score, gap) in enumerate(zip(real_scores, pred_scores, gaps)):
                class_name = class_labels[class_id]
                row[f'real_{metric}_{class_name}'] = real_score
                row[f'pred_{metric}_{class_name}'] = pred_score
                row[f'gap_{metric}_{class_name}'] = gap
        data.append(row)
    df = pd.DataFrame(data)
    return df


def create_experiment_summary_df(result_df, class_labels):
    """
    Create a summary DataFrame for a single experiment similar to the existing summary CSV format
    """
    summary_data = []

    # Get metric columns (assuming they start with 'mean_score_', 'mean_pred_', 'mean_gap_')
    metric_columns = [col for col in result_df.columns if col.startswith('mean_score_')]
    metrics = [col.replace('mean_score_', '') for col in metric_columns]

    for metric in metrics:
        # Get columns for this metric
        real_score_col = f'mean_score_{metric}'
        pred_score_col = f'mean_pred_{metric}'
        gap_col = f'mean_gap_{metric}'

        # Calculate overall statistics
        real_scores = result_df[real_score_col].dropna()
        pred_scores = result_df[pred_score_col].dropna()
        gaps = result_df[gap_col].dropna()

        valid_samples = len(gaps)
        total_samples = len(result_df)

        row = {
            'metric': metric,
            'real_score_mean': float(real_scores.mean()) if len(real_scores) > 0 else 0.0,
            'pred_score_mean': float(pred_scores.mean()) if len(pred_scores) > 0 else 0.0,
            'gap_mean': float(gaps.mean()) if len(gaps) > 0 else 0.0,
            'gap_std': float(gaps.std()) if len(gaps) > 0 else 0.0,
            'gap_min': float(gaps.min()) if len(gaps) > 0 else 0.0,
            'gap_max': float(gaps.max()) if len(gaps) > 0 else 0.0,
            'valid_samples': valid_samples,
            'total_samples': total_samples
        }

        # Add per-class statistics
        for class_name in class_labels:
            real_class_col = f'real_{metric}_{class_name}'
            pred_class_col = f'pred_{metric}_{class_name}'
            gap_class_col = f'gap_{metric}_{class_name}'

            if real_class_col in result_df.columns:
                real_class_scores = result_df[real_class_col].dropna()
                pred_class_scores = result_df[pred_class_col].dropna()
                class_gaps = result_df[gap_class_col].dropna()

                row.update({
                    f'{class_name}_real_score_mean': float(real_class_scores.mean()) if len(
                        real_class_scores) > 0 else 0.0,
                    f'{class_name}_pred_score_mean': float(pred_class_scores.mean()) if len(
                        pred_class_scores) > 0 else 0.0,
                    f'{class_name}_gap_mean': float(class_gaps.mean()) if len(class_gaps) > 0 else 0.0,
                    f'{class_name}_gap_std': float(class_gaps.std()) if len(class_gaps) > 0 else 0.0,
                    f'{class_name}_gap_min': float(class_gaps.min()) if len(class_gaps) > 0 else 0.0,
                    f'{class_name}_gap_max': float(class_gaps.max()) if len(class_gaps) > 0 else 0.0,
                    f'{class_name}_valid_samples': len(class_gaps)
                })
            else:
                row.update({
                    f'{class_name}_real_score_mean': 0.0,
                    f'{class_name}_pred_score_mean': 0.0,
                    f'{class_name}_gap_mean': 0.0,
                    f'{class_name}_gap_std': 0.0,
                    f'{class_name}_gap_min': 0.0,
                    f'{class_name}_gap_max': 0.0,
                    f'{class_name}_valid_samples': 0
                })

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    return summary_df


def save_experiment_summary(result_df, dataset_name, source, target, classifier_type, class_labels,
                            output_dir, timestamp=None):
    """
    Save individual experiment summary to CSV file
    """
    os.makedirs(output_dir, exist_ok=True)

    summary_df = create_experiment_summary_df(result_df, class_labels)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{source}_to_{target}_{dataset_name}_{classifier_type}_{timestamp}_summary.csv"
    filepath = os.path.join(output_dir, filename)

    summary_df.to_csv(filepath, index=False)
    print(f"  ✓ Summary saved to: {filepath}")

    return filepath, summary_df


def merge_and_save_results_by_grouped_keys(all_results, output_dir,group_by_keys, timestamp=None):
    """
    Merge all results from same classifier and same dataset into single DataFrames
    """
    os.makedirs(output_dir, exist_ok=True)

    # Group results by dataset and classifier
    grouped_results = {}
    print(all_results)
    for result in all_results:
        if result['status'] == 'success' and result['result_df'] is not None:
            group_key = "_".join([str(result[k]) for k in group_by_keys])
            if group_key not in grouped_results:
                grouped_results[group_key] = []

            # Add experiment metadata to the result_df
            result_df = result['result_df'].copy()
            result_df['experiment_source'] = result['source']
            result_df['experiment_target'] = result['target']
            for key in group_by_keys:
                result_df[f'experiment_{key}'] = result[key]

            grouped_results[group_key].append(result_df)

    # Save merged results for each group
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_files = {}
    for key, result_dfs in grouped_results.items():
        if result_dfs:
            merged_df = pd.concat(result_dfs, ignore_index=True)
            filename = f"results_{key}_{timestamp}.csv"
            filepath = os.path.join(output_dir, filename)
            merged_df.to_csv(filepath, index=False)
            merged_files[key] = filepath
            print(f"  ✓ Merged results saved to: {filepath} ({len(result_dfs)} experiments)")

    return merged_files


def create_final_summary_per_grouped_keys(merged_files, output_dir):
    """
    Create final summary per dataset per classifier aggregating all results
    """
    final_summaries = {}

    for key, filepath in merged_files.items():
        # Load the merged results
        merged_df = pd.read_csv(filepath)

        # Get unique metrics and classes from the data
        metric_columns = [col for col in merged_df.columns if col.startswith('mean_score_')]
        metrics = [col.replace('mean_score_', '') for col in metric_columns]

        # Get class names from the columns
        class_columns = [col for col in merged_df.columns if col.startswith('real_') and '_' in col]
        class_names = set()
        for col in class_columns:
            parts = col.split('_')
            if len(parts) >= 3:
                class_name = '_'.join(parts[2:])  # Everything after 'real_metric_'
                class_names.add(class_name)
        class_names = sorted(list(class_names))

        summary_data = []

        for metric in metrics:
            # Get columns for this metric
            real_score_col = f'mean_score_{metric}'
            pred_score_col = f'mean_pred_{metric}'
            gap_col = f'mean_gap_{metric}'

            # Calculate overall statistics across all experiments
            real_scores = merged_df[real_score_col].dropna()
            pred_scores = merged_df[pred_score_col].dropna()
            gaps = merged_df[gap_col].dropna()

            valid_samples = len(gaps)
            total_samples = len(merged_df)

            row = {
                'metric': metric,
                'real_score_mean': float(real_scores.mean()) if len(real_scores) > 0 else 0.0,
                'pred_score_mean': float(pred_scores.mean()) if len(pred_scores) > 0 else 0.0,
                'gap_mean': float(gaps.mean()) if len(gaps) > 0 else 0.0,
                'gap_std': float(gaps.std()) if len(gaps) > 0 else 0.0,
                'gap_min': float(gaps.min()) if len(gaps) > 0 else 0.0,
                'gap_max': float(gaps.max()) if len(gaps) > 0 else 0.0,
                'valid_samples': valid_samples,
                'total_samples': total_samples
            }

            # Add per-class statistics
            for class_name in class_names:
                real_class_col = f'real_{metric}_{class_name}'
                pred_class_col = f'pred_{metric}_{class_name}'
                gap_class_col = f'gap_{metric}_{class_name}'

                if real_class_col in merged_df.columns:
                    real_class_scores = merged_df[real_class_col].dropna()
                    pred_class_scores = merged_df[pred_class_col].dropna()
                    class_gaps = merged_df[gap_class_col].dropna()

                    row.update({
                        f'{class_name}_real_score_mean': float(real_class_scores.mean()) if len(
                            real_class_scores) > 0 else 0.0,
                        f'{class_name}_pred_score_mean': float(pred_class_scores.mean()) if len(
                            pred_class_scores) > 0 else 0.0,
                        f'{class_name}_gap_mean': float(class_gaps.mean()) if len(class_gaps) > 0 else 0.0,
                        f'{class_name}_gap_std': float(class_gaps.std()) if len(class_gaps) > 0 else 0.0,
                        f'{class_name}_gap_min': float(class_gaps.min()) if len(class_gaps) > 0 else 0.0,
                        f'{class_name}_gap_max': float(class_gaps.max()) if len(class_gaps) > 0 else 0.0,
                        f'{class_name}_valid_samples': len(class_gaps)
                    })
                else:
                    row.update({
                        f'{class_name}_real_score_mean': 0.0,
                        f'{class_name}_pred_score_mean': 0.0,
                        f'{class_name}_gap_mean': 0.0,
                        f'{class_name}_gap_std': 0.0,
                        f'{class_name}_gap_min': 0.0,
                        f'{class_name}_gap_max': 0.0,
                        f'{class_name}_valid_samples': 0
                    })

            summary_data.append(row)

        final_summary_df = pd.DataFrame(summary_data)

        # Save final summary
        filename = f"final_summary_{key}.csv"
        filepath = os.path.join(output_dir, filename)
        final_summary_df.to_csv(filepath, index=False)
        final_summaries[key] = filepath
        print(f"  ✓ Final summary saved to: {filepath}")

    return final_summaries
