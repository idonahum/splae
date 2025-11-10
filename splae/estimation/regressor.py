from torch import nn
from splae.models.backbones.unet import UNetEncoder
import torch.nn.functional as F
import torch
class AccuracyEstimatorRegressor(nn.Module):
    def __init__(self,
                 score_type='dice',
                 in_channels_img=1,        # e.g. grayscale MRI
                 in_channels_seg=4,        # one-hot seg maps (K classes)
                 base_channels=16,
                 num_stages=5,
                 out_dims_score=1):
        super().__init__()
        # reuse your encoder
        in_channels = in_channels_img + in_channels_seg
        self.encoder = UNetEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            num_stages=num_stages,
            strides=(1, 1, 1, 1, 1),
            num_convs=(2, 2, 2, 2, 2),
            downsamples=(True, True, True, True),
            dilations=(1, 1, 1, 1, 1),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.score_type = score_type
        feat_dim = base_channels * 2 ** (num_stages - 1)  # channels at bottleneck

        if self.score_type == 'Dice':
            self.score_head = nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, out_dims_score),
                nn.Sigmoid()  # Dice per class ∈ [0,1]
            )
        elif self.score_type == 'ASSD':
            self.score_head = nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, out_dims_score)
                # log(ASSD+eps), one per class
            )

    def forward(self, img, seg):
        """
        img: [B, Cin_img, H, W]
        seg: [B, H, W] int labels or [B, Cin_seg, H, W] one-hot
        """
        # concat along channel axis
        x = torch.cat([img, seg], dim=1)  # [B, C_total, H, W]

        enc_outs = self.encoder(x)
        bottleneck = enc_outs[-1]  # [B, C, H', W']
        g = self.gap(bottleneck).flatten(1)

        pred_score = self.score_head(g)

        return pred_score