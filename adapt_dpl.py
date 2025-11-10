from torch.optim.lr_scheduler import StepLR

from splae.adapt.dpl import DPLGenerator
from splae.runner import DPLRunner
import torch

if __name__ == '__main__':
    ## Use This script to run DPL adaptation ##
    from splae.models.segmentor import SegmentationModel
    from splae.models.backbones import UNet
    from splae.models.heads import FCNHead, ConvHead

    # Device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    ds_name = 'MSM'
    source = 'BCDEF'
    target = 'A'
    data_root = f'/home/dsi/nahum92/sfda/datasets/msm_2d/{target}'

    method = 'dpl'
    experiment_name = f"adapt_{source}_to_{target}_{method}_{ds_name}"
    load_from = f'/home/dsi/nahum92/sfda/experiments/train_source_{source}_{ds_name}/checkpoints/best_model.pth'
    # Model
    backbone = UNet(in_channels=1, base_channels=16)
    head = ConvHead(in_channels=16, num_classes=2)
    model = SegmentationModel(backbone, head)
    # Datasets and loaders
    from torch.utils.data import DataLoader
    from splae.datasets.datasets import MNMsDataset2D, MSMDataset2D
    from monai.transforms import (
        ScaleIntensityd,
        NormalizeIntensityd,
        ClipIntensityPercentilesd,
        EnsureChannelFirstd,
        Compose,
        LoadImaged,
        ResizeWithPadOrCropd,
        Activations,
        AsDiscrete, Resized
)

    transforms = Compose([
        LoadImaged(keys=['img', 'gt_mask'], allow_missing_keys=True),
        EnsureChannelFirstd(keys=['img', 'gt_mask'], allow_missing_keys=True),
        ClipIntensityPercentilesd(keys=['img'], lower=1, upper=99),
        NormalizeIntensityd(keys=['img'], subtrahend=None, divisor=None),  # Zero mean, unit variance
        Resized(keys=['img'], spatial_size=(256, 256)),
        Resized(keys=['gt_mask'], spatial_size=(256, 256), mode='nearest'),
    ])

    post_transforms = Compose([AsDiscrete(argmax=True)])


    train_dataset = MNMsDataset2D(data_root, split='train', transforms=transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=8, num_workers=4, shuffle=True)

    val_dataset = MNMsDataset2D(data_root, split='valid', transforms=transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=4, shuffle=False)

    test_dataset = MNMsDataset2D(data_root, split='test', transforms=transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False)

    # Optimizer
    from torch.optim import SGD, Adam

    optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))

    # LR Scheduler , use PolyLR
    from splae.schedulers import PolyLR, ConstantLR

    lr_scheduler = PolyLR(optimizer, begin=0, end=20, power=0.9, eta_min=1e-5)

    # lr_scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    # lr_scheduler = ConstantLR(optimizer=optimizer)

    # Losses
    from torch.nn import CrossEntropyLoss
    from splae.models.losses import CrossEntropyLoss, DiceLoss


    losses = [CrossEntropyLoss(loss_weight=1.0, ignore_index=-1, reduction='none'),DiceLoss(loss_weight=3.0, softmax=True,to_onehot_y=True, include_background=False,reduction='none')]

    # Metrics
    from splae.evaluation import DiceMetric

    metrics = [DiceMetric()]

    pallete = list(MSMDataset2D.PALLETE.values())
    # Pseduo Label Geneartor

    pl_generator = DPLGenerator(device=device, num_classes=2, conf_threshold=0.7)
    # Trainer
    runner = DPLRunner(
        model=model,
        pl_generator=pl_generator,
        optimizer=optimizer,
        losses=losses,
        lr_scheduler=lr_scheduler,
        metrics=metrics,
        device=device,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        test_loader=test_dataloader,
        epochs=20,
        val_interval=1,
        save_dir=f'./experiments',
        # random_seed=2787988297,
        load_from=load_from,
        post_transforms=post_transforms,
        experiment_name=experiment_name,
        visualization_cfg=dict(train_interval=40, val_interval=40, classes=MSMDataset2D.CLASS_LABELS,
                               palette=pallete),
        checkpoint_cfg=dict(interval=1, save_best_by='Dice'),
        logger_cfg=dict(log_interval=50)
    )
    runner.test()
    runner.train()
