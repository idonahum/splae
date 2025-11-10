import json
from splae.datasets.transforms import get_transforms
from splae.estimation.utils import create_experiment_result_df
from splae.runner import AdaptationRunner
from splae.adapt.iplc import IPLCGenerator, PLVisualizer
from splae.models import load_unet
from splae.datasets import load_dataset2d
from monai.transforms import Compose, AsDiscrete
import torch
import os
from tqdm import tqdm
from torch import nn
from monai.transforms import ScaleIntensity
from torch.utils.data import DataLoader
from splae.datasets.datasets import MNMsDataset2D, MSMDataset2D
from splae.estimation.base import BaseAccuracyEstimator
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
def get_batched_scores(pred, gt, evaluator, score_type):
    batch_scores = []
    for pred, gt in zip(pred, gt):
        evaluator.process(pred, gt)
        batch_scores.append(evaluator.finalize()[score_type])
    return torch.stack(batch_scores)

def test(adapted_model, score_model_dice, score_model_assd, post_transforms, test_loader, evaluator, score_type, device):
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
            pred_score_dice = score_model_dice(img, seg)
            pred_score_assd = score_model_assd(img, seg)
            pred_score = {'Dice': pred_score_dice, 'ASSD': pred_score_assd}
            real_score = evaluator.process(argmax_seg, gt_mask)
            real_score = evaluator.finalize()
            real_score['ASSD'] = torch.log(real_score['ASSD'] + 1e-3)
            results.append(BaseAccuracyEstimator._zip_results(real_score, pred_score, data['metainfo']['img_name']))
        results_df =create_experiment_result_df(results, test_loader.dataset.CLASS_LABELS)
        print(f"Results DF: {results_df.head()}")
        return results_df

def validate(source_model, score_model, post_transforms, val_loader, evaluator, score_type, best_score, device, epoch):
    img_transform = Compose([ScaleIntensity( minv=0, maxv=1.0)])
    gaps_score = []
    with torch.no_grad():
        score_model.eval()
        for i, data in enumerate(tqdm(val_loader)):
            img = data['img'].to(device)
            gt_mask = data['gt_mask'].to(device)
            with torch.no_grad():
                seg = source_model(img)
                seg = seg.softmax(dim=1)
                argmax_seg = post_transforms(seg)
            img = img_transform(img)
            pred_scores = score_model(img, seg)
            real_scores = get_batched_scores(argmax_seg, gt_mask, evaluator, score_type).to(device)
            if score_type == 'ASSD':
                real_scores = torch.log(real_scores + 1e-3)
            valid_mask = torch.isfinite(real_scores)
            if not valid_mask.any():
                continue 
            pred_scores_mean = pred_scores[valid_mask].mean(dim=0)
            real_scores_mean = real_scores[valid_mask].mean(dim=0)
            gaps_score.append(torch.abs(pred_scores_mean - real_scores_mean))
        gaps_tensor = torch.stack(gaps_score)
        mean_gap_score_per_class = gaps_tensor.mean(dim=0)
        print(f"Mean {score_type} Gap per class: {mean_gap_score_per_class}")
        mean_gap_score = mean_gap_score_per_class.mean()
        print(f"Mean {score_type} Gap: {mean_gap_score}")
        return mean_gap_score

def train(source_model, score_model, post_transforms, train_loader, val_loader, optimizer, loss_mse, evaluator, score_type, device, checkpoint_path):
    img_transform = Compose([ScaleIntensity( minv=0, maxv=1.0)])
    best_score = float('inf')

    for epoch in range(8):
        score_model.train()
        for i, data in enumerate(tqdm(train_loader)):
            img = data['img'].to(device)
            gt_mask = data['gt_mask'].to(device)
            with torch.no_grad():
                seg = source_model(img)
                seg = seg.softmax(dim=1)
                argmax_seg = post_transforms(seg)

            real_scores = get_batched_scores(argmax_seg, gt_mask, evaluator, score_type).to(device)
            if score_type == 'ASSD':
                real_scores = torch.log(real_scores + 1e-3)
            img = img_transform(img)
            pred_scores = score_model(img, seg)
            # size is [B, num_classes], first average over batch
            valid_mask = torch.isfinite(real_scores) & (real_scores > 0)
            if valid_mask.any():
                loss = loss_mse(pred_scores[valid_mask], real_scores[valid_mask])
            else:
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} complete, validating")
        score = validate(source_model, score_model, post_transforms, val_loader, evaluator, score_type, best_score, device, epoch)
        if score < best_score:
            best_score = score
            torch.save(score_model.state_dict(), f"{checkpoint_path}/best_model.pth")
            print(f"New best {score_type} score: {best_score}, saving model, epoch {epoch}")

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

def run_experiment(source, ds_name, score_type, device):
    ds_root = f'/home/dsi/nahum92/sfda/datasets/{ds_name.lower()}_2d'
    num_classes = 4 if ds_name == 'MNMs' else 2
    experiment_name = f"train_score_model_{score_type}_{source}_{ds_name}"
    

    experiment_path = f"/home/dsi/nahum92/sfda/experiments/{experiment_name}"
    os.makedirs(experiment_path, exist_ok=True)
    checkpoint_path = os.path.join(experiment_path, f"checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)
    transforms = get_transforms(ds_name, to_onehot=False)
    if len(source) > 1:
        domains = [domain for domain in source]
        data_root = f'{ds_root}'
    else:
        domains = None
        data_root = f'{ds_root}/{source}'
    train_dataset = load_dataset2d(ds_name, data_root=data_root, transforms=transforms, split='train', domains=domains)
    val_dataset = load_dataset2d(ds_name, data_root=data_root, transforms=transforms, split='valid', domains=domains)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    source_model_path = f'/home/dsi/nahum92/sfda/experiments/train_source_{source}_{ds_name}/checkpoints/best_model.pth'
    source_model = load_unet(source_model_path, num_classes, device=device)
    score_model = AccuracyEstimatorRegressor(in_channels_img=1, in_channels_seg=num_classes,out_dims_score=num_classes, score_type=score_type).to(device)
    post_transforms = Compose([AsDiscrete(argmax=True, dim=1)])


    metrics = [DiceMetric(),ASSDMetric()]
    evaluator = MetricsEvaluator(metrics, num_classes, device=device)
    loss_mse = nn.MSELoss()
    optimizer = torch.optim.Adam(score_model.parameters(), lr=1e-3)
    train(source_model, score_model, post_transforms, train_loader, val_loader, optimizer, loss_mse, evaluator, score_type, device, checkpoint_path)
    # best_model_path = os.path.join(checkpoint_path, f"best_model.pth")
    # score_model.load_state_dict(torch.load(best_model_path))
    # test(target_model, score_model, post_transforms, test_loader, evaluator, score_type, device)

    


if __name__ == '__main__':
    from splae.estimation import AccuracyEstimatorRegressor
    from splae.evaluation import DiceMetric, ASSDMetric
    from splae.evaluation import MetricsEvaluator
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    json_file_path = '/home/dsi/nahum92/sfda/dataset_pairs.json'
    finished_exp = {}
    datasets_pairs = open_json(json_file_path)
    for dataset_pairs in datasets_pairs:
        dataset_name = dataset_pairs['dataset_name']
        source_target_pairs = dataset_pairs['source_target_pairs']
        for source, target in source_target_pairs:
            for score_type in ['ASSD','Dice']:
                if f"{source}_{dataset_name}_{score_type}" in finished_exp:
                    continue
                print(f"Running experiment: {source} , {dataset_name}, {score_type}")
                run_experiment(source, dataset_name, score_type, device)
                finished_exp[f"{source}_{dataset_name}_{score_type}"] = True