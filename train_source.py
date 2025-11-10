from splae.runner import Runner
import torch


if __name__ == '__main__':
    from splae.models.segmentor import SegmentationModel
    from splae.models.backbones import UNet
    from splae.models.heads import ConvHead

    # Use this script to train source model
    experiment_name = 'test_adapt_B_to_A_iplc_MNMs'
    load_from= '/home/dsi/nahum92/sfda/experiments/adapt_A_to_D_iplc_MNMs/checkpoints/best_model.pth'
    data_root = '/home/dsi/nahum92/sfda/datasets/mnms_2d/D'
    domains = ['B', 'C', 'D', 'E', 'F']
    # Model
    backbone = UNet(in_channels=1, base_channels=16)
    head = ConvHead(in_channels=16, num_classes=4)
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
        AsDiscrete,
        Resized
    )

    transforms = Compose([
        LoadImaged(keys=['img', 'gt_mask'], allow_missing_keys=True),
        EnsureChannelFirstd(keys=['img', 'gt_mask'], allow_missing_keys=True),
        ClipIntensityPercentilesd(keys=['img'], lower=1, upper=99),
        ScaleIntensityd(keys=['img'], minv=-1.0, maxv=1.0),
        
        Resized(keys=['img'], spatial_size=(256, 256)),
        Resized(keys=['gt_mask'], spatial_size=(256, 256), mode='nearest'),
    ])

    post_transforms = Compose([AsDiscrete(argmax=True)])



    train_dataset = MNMsDataset2D(data_root, split='train', transforms=transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=4, num_workers=4, shuffle=True)

    val_dataset = MNMsDataset2D(data_root, split='valid', transforms=transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=4, shuffle=True)

    test_dataset = MNMsDataset2D(data_root, split='test', transforms=transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=True)

    # # Optimizer
    # from torch.optim import SGD
    #
    # optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
    #
    # # LR Scheduler , use PolyLR
    # from SamSFDA.schedulers import PolyLR
    # lr_scheduler = PolyLR(optimizer, begin=0, end=400, power=0.9, eta_min=0.0001)
    # # lr_scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    from torch.optim import AdamW

    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    # LR Scheduler , use PolyLR
    from splae.schedulers import PolyLR
    lr_scheduler = PolyLR(optimizer, begin=0, end=100, power=0.9, eta_min=0.0001)
    # Losses
    from torch.nn import CrossEntropyLoss
    from splae.models.losses import DiceLoss, CrossEntropyLoss

    losses = [CrossEntropyLoss(loss_weight=1.0), DiceLoss(include_background=True, to_onehot_y=True, softmax=True, loss_weight=3.0, squared_pred=True)]

    # Metrics
    from splae.evaluation import DiceMetric

    metrics = [DiceMetric()]

    # Device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    pallete = list(MSMDataset2D.PALLETE.values())
    # Trainer
    runner = Runner(
        model=model,
        optimizer=optimizer,
        losses=losses,
        lr_scheduler=lr_scheduler,
        metrics=metrics,
        device=device,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        test_loader=test_dataloader,
        epochs=100,
        val_interval=4,
        save_dir='./experiments',
        post_transforms=post_transforms,
        experiment_name=experiment_name,
        visualization_cfg=dict(train_interval=None, val_interval=20, classes=['background', 'LV', 'MYO', 'RV'], palette=pallete),
        checkpoint_cfg=dict(interval=4, save_best_by='Dice'),
        logger_cfg=dict(log_interval=20),
        load_from=load_from,
    )
    runner.test()
    # runner.train()
