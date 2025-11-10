import os

import torch.nn as nn
from splae.models.heads import ConvHead
from splae.models.backbones import UNet
import torch

class SegmentationModel(nn.Module):
    def __init__(self,
                 backbone,
                 head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, return_features=False):
        x = self.backbone(x)
        return self.head(x, return_features=return_features)

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, '_initialize_weights'):
                m._initialize_weights()

def load_unet(model_path, num_classes, in_channels=1, device='cpu'):
    """Load pre-trained source model."""
    if not os.path.exists(model_path):
        raise ValueError(f"Source model path not found: {model_path}")

    try:
        backbone = UNet(in_channels=in_channels, base_channels=16)
        head = ConvHead(in_channels=16, num_classes=num_classes)

        model = SegmentationModel(backbone, head)

        # Load checkpoint
        checkpoint = torch.load(model_path)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()

        print(f"✓ Successfully loaded source model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading source model: {e}")
        return None