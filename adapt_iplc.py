from splae.runner import AdaptationRunner
from splae.adapt.iplc import IPLCGenerator
import torch

if __name__ == '__main__':
    from splae.models.segmentor import SegmentationModel
    from splae.models.backbones import UNet
    from splae.models.heads import ConvHead

    # Device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    ds_name = 'MSM'
    source = 'BCDEF'
    target = 'A'    
    data_root = '/home/dsi/nahum92/sfda/datasets/msm_2d/A'

    method = 'iplc'
    experiment_name = f"adapt_{source}_to_{target}_{method}_{ds_name}"
    load_from = '/home/dsi/nahum92/sfda/experiments/train_source_BCDEF_MSM/checkpoints/best_model.pth'
    # Model
    backbone = UNet(in_channels=1, base_channels=16)
    head = ConvHead(in_channels=16, num_classes=2)
    model = SegmentationModel(backbone, head)
    # Datasets and loaders
    from torch.utils.data import DataLoader
    from splae.datasets.datasets import MNMsDataset2D, MSMDataset2D
    from monai.transforms import (
        NormalizeIntensityd,
        ClipIntensityPercentilesd,
        EnsureChannelFirstd,
        Compose,
        LoadImaged,
        ResizeWithPadOrCropd,
        Activations,
        AsDiscrete
    )

    transforms = Compose([
        LoadImaged(keys=['img', 'gt_mask'], allow_missing_keys=True),
        EnsureChannelFirstd(keys=['img', 'gt_mask'], allow_missing_keys=True),
        ClipIntensityPercentilesd(keys=['img'], lower=1, upper=99),
        NormalizeIntensityd(keys=['img'], subtrahend=None, divisor=None),  # Zero mean, unit variance
        ResizeWithPadOrCropd(keys=['img'], spatial_size=(256, 256)),
        ResizeWithPadOrCropd(keys=['gt_mask'], spatial_size=(256, 256), value=-1),
    ])

    post_transforms = Compose([AsDiscrete(argmax=True)])

    ## Dataset set to MSM, change to MNMsDataset2D for MNMs
    train_dataset = MSMDataset2D(data_root, split='train', transforms=transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=2, num_workers=4, shuffle=True)

    val_dataset = MSMDataset2D(data_root, split='valid', transforms=transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=4, shuffle=True)

    test_dataset = MSMDataset2D(data_root, split='test', transforms=transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=True)

    # Optimizer
    from torch.optim import SGD

    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)

    # LR Scheduler , use PolyLR
    from splae.schedulers import PolyLR, ConstantLR

    lr_scheduler = ConstantLR(optimizer=optimizer)

    # Losses
    from torch.nn import CrossEntropyLoss
    from splae.models.losses import DiceLoss, CrossEntropyLoss, WeightedDiceLoss, CurvatureLoss, DiceLossV2

    losses = [
        WeightedDiceLoss(include_background=True, to_onehot_y=True, softmax=True, loss_weight=1.0, squared_pred=True),
        CurvatureLoss(softmax=True, loss_weight=0.01, device=device)]

    # Metrics
    from splae.evaluation import DiceMetric

    metrics = [DiceMetric()]

    pallete = list(MSMDataset2D.PALLETE.values())

    pl_generator = IPLCGenerator(model_type="vit_b",
                                    weights_path="/home/dsi/nahum92/sfda/sam-med2d_b.pth",
                                    device=device,
                                    num_classes=2,
                                    samples_times=10,
                                    ignore_index=-1,
                                    encoder_adapter=True,
                                    visualizer=None,
                                    image_size=256,
                                    benchmark=False)
    # Trainer
    runner = AdaptationRunner(
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
        save_dir='./experiments',
        # random_seed=2787988297,
        load_from=load_from,
        post_transforms=post_transforms,
        experiment_name=experiment_name,
        visualization_cfg=dict(train_interval=40, val_interval=20, classes=['background', 'prostate'],
                               palette=pallete),
        checkpoint_cfg=dict(interval=1, save_best_by='Dice'),
        logger_cfg=dict(log_interval=20)
    )
    runner.test()
    runner.train()
