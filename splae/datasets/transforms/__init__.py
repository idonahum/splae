from monai.transforms import Compose, LoadImaged, ScaleIntensityd, EnsureChannelFirstd, ClipIntensityPercentilesd, \
    Resized, AsDiscreted, NormalizeIntensityd

def get_transforms(dataset_type, to_onehot=True, load_seg_logits=False):
    assert dataset_type in ['MNMs', 'MSM'], f"Unsupported dataset type: {dataset_type}. Supported types are 'MNMs' and 'MSM'."
    scale_transform = ScaleIntensityd(keys=['img'], minv=-1, maxv=1)
    keys = ['img', 'gt_mask']
    if load_seg_logits:
        keys.append('seg_logits')
    transforms = [
        LoadImaged(keys=keys, allow_missing_keys=True),
        EnsureChannelFirstd(keys=keys, allow_missing_keys=True),
        ClipIntensityPercentilesd(keys=['img'], lower=1, upper=99),
        scale_transform,
        Resized(keys=['img'], spatial_size=(256, 256)),
        Resized(keys=['gt_mask'], spatial_size=(256, 256), mode='nearest'),
    ]
    if to_onehot:
        transforms.append(AsDiscreted(keys=['gt_mask'], to_onehot=4 if dataset_type == 'MNMs' else 2))
    transforms = Compose(transforms)
    return transforms