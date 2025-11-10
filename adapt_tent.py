import torch
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm
from splae.datasets.datasets import load_dataset2d
from splae.datasets.transforms import get_transforms
from splae.evaluation import MetricsEvaluator
from splae.evaluation import DiceMetric
from splae.models.segmentor import load_unet
from monai.transforms import (
    Compose,
    AsDiscrete,
)

from splae.utils import open_json
from splae.adapt.tent import CrossEntropyLoss
import numpy as np
import nibabel as nib

def save_logits_as_nifti(logits, img_names, data_root, split, source, method='tent'):
    """
    Save model logits as NIfTI files in a hierarchical directory structure.
    
    Args:
        logits (torch.Tensor): Model output logits of shape (B, C, H, W)
        img_names (list): List of base names of the image files without extension
        data_root (str): Root directory of the dataset
        split (str): Split name (train/valid/test)
        source (str): Source model name
        method (str): Method name (e.g., 'tent', 'pl', etc.)
    """
    # Create hierarchical directory structure: seg_logits/method/source/
    seg_logits_dir = os.path.join(data_root, split, 'seg_logits', method, source)
    os.makedirs(seg_logits_dir, exist_ok=True)
    
    # Convert logits to numpy array and move to CPU if needed
    if isinstance(logits, torch.Tensor):
        logits_np = logits.detach().cpu().numpy()
    else:
        logits_np = logits
    
    # Ensure logits are float32 for NIfTI compatibility
    logits_np = logits_np.astype(np.float32)
    
    # Handle batch dimension - loop through each item in the batch
    batch_size = logits_np.shape[0]
    for i in range(batch_size):
        # Extract single item from batch (remove batch dimension)
        single_logits = logits_np[i]  # Shape: (C, H, W)
        
        # Get corresponding image name
        if isinstance(img_names, list) and i < len(img_names):
            img_name = img_names[i]
        else:
            print(f"No image name found for batch {i}")
            continue
        
        # Create NIfTI image with identity affine
        nifti_img = nib.Nifti1Image(single_logits, affine=np.eye(4))
        
        # Create filename with method suffix
        filename = f"{img_name}_seg_logits.nii.gz"
        filepath = os.path.join(seg_logits_dir, filename)
        
        # Save the NIfTI file
        nib.save(nifti_img, filepath)

def finalize_metrics(evaluator, classes):
    """
    Normalize, log, and return scores per class for metrics after validation.

    Args:
        evaluator: MetricsEvaluator instance.
        epoch (int): Current epoch number.
        mode (str): Mode string for logging.

    Returns:
        dict: A dictionary containing scores per class and mean for each metric.
    """
    scores_per_class = {}
    metrics = evaluator.finalize()
    for metric_name, metric in metrics.items():
        log_str = f"{metric_name}   "
        class_scores = {}
        for i, score in enumerate(metric):
            class_name = classes[i] if i < len(classes) else f"class_{i}"
            log_str += f"{class_name}: {score:.4f}  "
            class_scores[class_name] = score
        mean_score = metric.mean()
        log_str += f"m{metric_name}: {mean_score:.4f}"
        class_scores['mean'] = mean_score
        scores_per_class[metric_name] = class_scores
        print(log_str)

    return scores_per_class

def adapt_single_tent(source, target, dataset, device):
    source_model_path = f'./experiments/train_source_{source}_{dataset}/checkpoints/best_model.pth'
    save_path = f'./experiments/adapt_{source}_to_{target}_tent_{dataset}/checkpoints'
    os.makedirs(save_path, exist_ok=True)
    num_classes = 4 if dataset == 'MNMs' else 2
    print("Loading source model...")
    source_model = load_unet(source_model_path, num_classes=num_classes, device=device)
    tent_model = load_unet(source_model_path, num_classes=num_classes, device=device)
    print("Loading optimizer...")
    transforms = get_transforms(dataset, to_onehot=False)
    post_transforms = Compose([AsDiscrete(argmax=True, dim=1, dtype=torch.long)])
    data_root = f'./datasets/{dataset.lower()}_2d/{target}'
    tent_dataset_train = load_dataset2d(dataset, data_root=data_root, split='train', transforms=transforms)
    tent_dataset_valid = load_dataset2d(dataset, data_root=data_root, split='valid', transforms=transforms)
    tent_dataset_test = load_dataset2d(dataset, data_root=data_root, split='test', transforms=transforms)
    tent_dataloader_train = DataLoader(tent_dataset_train, batch_size=32, num_workers=0, shuffle=False)
    tent_dataloader_valid = DataLoader(tent_dataset_valid, batch_size=32, num_workers=0, shuffle=False)
    tent_dataloader_test = DataLoader(tent_dataset_test, batch_size=32, num_workers=0, shuffle=False)
    test_dataloader = DataLoader(tent_dataset_test, batch_size=1, num_workers=0, shuffle=False)
    print(f"Test dataset size: {len(tent_dataset_test)}")

    metrics = [DiceMetric()]
    evaluator_source_vs_gt= MetricsEvaluator(metrics, num_classes=num_classes, ignore_index=-1)
    evaluator_tent_vs_gt = MetricsEvaluator(metrics, num_classes=num_classes, ignore_index=-1)

    source_model.eval()
    tent_model.eval()
    param_list = []
    num_iters = 1
    criterion = CrossEntropyLoss(use_sigmoid=False, loss_weight=1.0, reduction='none')
    for name, param in tent_model.named_parameters():
        if param.requires_grad:
            if "norm" in name or "bn" in name:
                print(param.requires_grad)
            if param.requires_grad and ("norm" in name or "bn" in name):
                param_list.append(param)
            else:
                param.requires_grad=False
    optimizer = torch.optim.Adam(param_list, lr=0.0005, betas=(0.5, 0.999),weight_decay=0.0001)
    with torch.no_grad():
        for idx, data in enumerate(tqdm.tqdm(tent_dataloader_test)):
            img = data['img'].to(device)
            gt_mask = data['gt_mask'].to(device)
            result_source = source_model(img)
            result_source = F.softmax(result_source, dim=1)
            result_source = post_transforms(result_source)
            evaluator_source_vs_gt.process(result_source, gt_mask)
        source_vs_gt_dice = finalize_metrics(evaluator_source_vs_gt, classes=tent_dataset_test.CLASS_LABELS)
        print(f"Source vs GT Dice: {source_vs_gt_dice}")


    for dataloader, split in zip([tent_dataloader_train, tent_dataloader_valid, tent_dataloader_test], ['train', 'valid','test']):
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            for _ in range(num_iters):
                img = data['img'].to(device)
                gt_mask = data['gt_mask'].to(device)
                with torch.no_grad():
                    result = tent_model(img)
                    result = F.softmax(result, dim=1)
                    result = post_transforms(result)
                result_with_grad = tent_model(img)
                loss = torch.mean(criterion(result_with_grad, result.squeeze(1)))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if split == 'test':
                    result_with_grad_argmax = F.softmax(result_with_grad, dim=1)
                    result_with_grad_argmax = post_transforms(result_with_grad_argmax)
                    evaluator_tent_vs_gt.process(result_with_grad_argmax, gt_mask)
                    # if 'metainfo' in data and 'img_name' in data['metainfo']:
                    #     img_names = data['metainfo']['img_name']  # This should be a list for batch
                    #     save_logits_as_nifti(result_with_grad, img_names, data_root, split, source, method='tent')

    adapted_vs_gt_dice = finalize_metrics(evaluator_tent_vs_gt, classes=tent_dataset_test.CLASS_LABELS)
    print(f"Iteration {i} - Adapted vs GT Dice: {adapted_vs_gt_dice}")
    improvement = adapted_vs_gt_dice['Dice']['mean'] - source_vs_gt_dice['Dice']['mean']
    print(f"Improvement: {improvement}")

def main():
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    json_file_path = '/home/dsi/nahum92/sfda/dataset_pairs.json'
    datasets_pairs = open_json(json_file_path)
    
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
            adapt_single_tent(source, target, dataset_name, device)

if __name__ == "__main__":
    main()