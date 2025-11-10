from abc import abstractmethod
import torch.nn as nn

from splae.models.layers import ConvModule


class BaseDecodeHead(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_classes,
                 dropout_ratio=0.1):
        super(BaseDecodeHead, self).__init__()
        self._check_inputs(out_channels, num_classes)
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio

        self.conv_seg = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout2d(self.dropout_ratio)
        else:
            self.dropout = None

    @abstractmethod
    def forward(self, inputs):
        pass

    def cls_seg(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        output = self.conv_seg(x)
        return output

    @staticmethod
    def _check_inputs(out_channels, num_classes):
        assert num_classes == out_channels or out_channels == 1, (
            f"Mismatch between num_classes ({num_classes}) and out_channels ({out_channels}). "
            "Ensure num_classes equals out_channels or out_channels is 1 for binary segmentation."
        )

    def _initialize_weights(self):
        """
        Initialize weights of the UNet.
        - Conv2d: Kaiming Normal
        - BatchNorm2d, GroupNorm: Constant Initialization
        """
        if hasattr(self.conv_seg, 'weight') and self.conv_seg.weight is not None:
            nn.init.normal_(self.conv_seg.weight, 0, 0.01)
        if hasattr(self.conv_seg, 'bias') and self.conv_seg.bias is not None:
            nn.init.constant_(self.conv_seg.bias, 0)


class FCNHead(BaseDecodeHead):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_classes,
                 num_convs=1,
                 kernel_size=3,
                 dilation=1,
                 dropout_ratio=0.1,
                 use_index=-1,
                 norm_cfg=dict(type='BatchNorm2d'),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__(in_channels, hidden_channels, out_channels, num_classes, dropout_ratio)
        self.num_convs = num_convs
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.use_index = use_index

        if num_convs == 0:
            assert self.in_channels == self.hidden_channels

        conv_padding = (kernel_size // 2) * dilation
        if num_convs == 0:
            convs = nn.Identity()
        else:
            convs = [ConvModule(
                self.in_channels,
                self.hidden_channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)]
            for i in range(num_convs - 1):
                convs.append(
                    ConvModule(
                        self.hidden_channels,
                        self.hidden_channels,
                        kernel_size=kernel_size,
                        padding=conv_padding,
                        dilation=dilation,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            convs = nn.Sequential(*convs)
        self.convs = convs

    def forward(self, inputs):
        if isinstance(inputs, list):
            inputs = inputs[self.use_index]
        output = self.convs(inputs)
        output = self.cls_seg(output)
        return output

    def _initialize_weights(self):
        """
        Initialize weights of the UNet.
        - Conv2d: Kaiming Normal
        - BatchNorm2d, GroupNorm: Constant Initialization
        """
        for conv_module in self.convs:
            if hasattr(conv_module, 'weight') and conv_module.weight is not None:
                nn.init.kaiming_normal_(conv_module.weight, mode='fan_out', nonlinearity='relu')
            if hasattr(conv_module, 'bias') and conv_module.bias is not None:
                nn.init.constant_(conv_module.bias, 0)

        super()._initialize_weights()


class ConvHead(BaseDecodeHead):
    def __init__(self,
                 in_channels,
                 num_classes,
                 use_index=-1):
        super().__init__(in_channels, in_channels, num_classes, num_classes, dropout_ratio=0)
        self.use_index = use_index

    def forward(self, inputs, return_features=False):
        if isinstance(inputs, list):
            inputs = inputs[self.use_index]   # pick one decoder output
        feats = inputs
        logits = self.conv_seg(feats)
        return (logits, feats) if return_features else logits