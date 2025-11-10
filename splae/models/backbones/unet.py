import torch
from torch import nn
from splae.models.layers import build_upsample_layer, ConvModule
from monai.networks.nets import UNet

class UNetConvBlock(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 norm_cfg=dict(type='BatchNorm2d'),
                 dropout_cfg=None,
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        for i in range(num_convs):
            self.add_module(f'conv{i}', ConvModule(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=3,
                stride=stride if i == 0 else 1,
                padding=1 if i == 0 else dilation,
                dilation=dilation,
                norm_cfg=norm_cfg,
                dropout_cfg=dropout_cfg,
                act_cfg=act_cfg))


class UpConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 skip_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 norm_cfg=dict(type='BatchNorm2d'),
                 dropout_cfg=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 upsample_cfg=dict(type='InterpConv')):
        super().__init__()
        self.conv_block = UNetConvBlock(
            in_channels=2*skip_channels,
            out_channels=out_channels,
            num_convs=num_convs,
            stride=stride,
            dilation=dilation,
            norm_cfg=norm_cfg,
            dropout_cfg=dropout_cfg,
            act_cfg=act_cfg)

        if upsample_cfg is not None:
            self.upsample_block = build_upsample_layer(cfg = upsample_cfg,
                                                 in_channels=in_channels,
                                                 out_channels=skip_channels,
                                                 norm_cfg=norm_cfg,
                                                 act_cfg=act_cfg)

        else:
            self.upsample_block = ConvModule(
                in_channels=in_channels,
                out_channels=skip_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

    def forward(self, x, skip):
        x = self.upsample_block(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class UNetEncoder(nn.Module):
    def __init__(self,
                 in_channels=1,
                 base_channels=64,
                 num_stages=5,
                 strides=(1, 1, 1, 1, 1),
                 num_convs=(2, 2, 2, 2, 2),
                 downsamples=(True, True, True, True),
                 dilations=(1, 1, 1, 1, 1),
                 norm_cfg=dict(type='BatchNorm2d'),
                 dropout_cfg=None,
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        self.num_stages = num_stages
        self.strides = strides
        self.num_convs = num_convs
        self.downsamples = downsamples
        self.dilations = dilations

        for i in range(num_stages):
            block = nn.ModuleList()
            if i != 0:
                if strides[i] == 1 and downsamples[i-1]:
                    block.append(nn.MaxPool2d(kernel_size=2, stride=2))
            block.append(UNetConvBlock(
                in_channels=in_channels,
                out_channels=base_channels * 2**i,
                num_convs=num_convs[i],
                stride=strides[i],
                dilation=dilations[i],
                norm_cfg=norm_cfg,
                dropout_cfg=dropout_cfg,
                act_cfg=act_cfg))
            self.add_module(f'down_block{i}', block)
            in_channels = base_channels * 2**i

    def forward(self, x):
        enc_outs = []
        for i in range(self.num_stages):
            block = getattr(self, f'down_block{i}')
            for layer in block:
                x = layer(x)
            enc_outs.append(x)
        return enc_outs


class UNetDecoder(nn.Module):
    def __init__(self,
                 base_channels=64,
                 num_stages=5,
                 num_convs=(2, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 strides=(1, 1, 1, 1, 1),
                 downsamples=(True, True, True, True),
                 dropout_cfg=None,
                 norm_cfg=dict(type='BatchNorm2d'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 upsample_cfg=dict(type='InterpConv')):

        super().__init__()
        self.num_stages = num_stages
        self.num_convs = num_convs
        self.dilations = dilations

        for i in range(num_stages - 1):
            upsample = (strides[num_stages - 1 - i] != 1 or downsamples[num_stages - 2 - i])
            self.add_module(f'up_block{i}', UpConvBlock(
                in_channels=base_channels * 2**(num_stages - i - 1),
                skip_channels=base_channels * 2**(num_stages - i - 2),
                out_channels=base_channels * 2**(num_stages - i - 2),
                num_convs=num_convs[i],
                dilation=dilations[i],
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                dropout_cfg=dropout_cfg,
                upsample_cfg=upsample_cfg if upsample else None))

    def forward(self, x, skips):
        dec_outs = [x]
        for i in range(self.num_stages - 1):
            x = getattr(self, f'up_block{i}')(x, skips[self.num_stages - 2 - i])
            dec_outs.append(x)
        return dec_outs


class UNet(nn.Module):
    def __init__(self,
                 in_channels=1,
                 base_channels=16,
                 num_stages=5,
                 strides=(1, 1, 1, 1, 1),
                 enc_num_convs=(2, 2, 2, 2, 2),
                 dec_num_convs=(2, 2, 2, 2),
                 downsamples=(True, True, True, True),
                 enc_dilations=(1, 1, 1, 1, 1),
                 dec_dilations=(1, 1, 1, 1),
                 enc_dropout=None,
                 dec_dropout=None,
                 norm_cfg=dict(type='BatchNorm2d'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 upsample_cfg=dict(type='InterpConv')):
        super().__init__()
        self._check_input(num_stages, strides, enc_num_convs, dec_num_convs,
                          downsamples, enc_dilations, dec_dilations)
        self.num_stages = num_stages
        self.strides = strides
        self.enc_num_convs = enc_num_convs
        self.dec_num_convs = dec_num_convs
        self.downsamples = downsamples
        self.enc_dilations = enc_dilations
        self.dec_dilations = dec_dilations

        self.encoder = UNetEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            num_stages=num_stages,
            strides=strides,
            num_convs=enc_num_convs,
            downsamples=downsamples,
            dilations=enc_dilations,
            norm_cfg=norm_cfg,
            dropout_cfg=enc_dropout,
            act_cfg=act_cfg)

        self.decoder = UNetDecoder(
            base_channels=base_channels,
            num_stages=num_stages,
            num_convs=dec_num_convs,
            dilations=dec_dilations,
            strides=strides,
            downsamples=downsamples,
            dropout_cfg=dec_dropout,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            upsample_cfg=upsample_cfg)

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights of the UNet.
        - Conv2d: Kaiming Normal
        - BatchNorm2d, GroupNorm: Constant Initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        enc_outs = self.encoder(x)
        dec_outs = self.decoder(enc_outs[-1], enc_outs[:-1])
        return dec_outs

    @staticmethod
    def _check_input(num_stages, strides, enc_num_convs, dec_num_convs,
                     downsamples, enc_dilations, dec_dilations):
        assert num_stages == len(strides) == len(enc_num_convs) == (len(
            dec_num_convs) + 1) == (len(downsamples)+1) == len(enc_dilations) == (len(
                dec_dilations) + 1), (f"The length of strides and enc_num_convs should be equal to {num_stages} AND the length of dec_num_convs should be equal to num_stages - 1 AND the length of downsamples should be equal to num_stages - 1 AND the length of enc_dilations should be equal to num_stages AND the length of dec_dilations should be equal to f{num_stages - 1},"
                                      f"Received - strides: {len(strides)}, enc_num_convs: {len(enc_num_convs)}, dec_num_convs: {len(dec_num_convs)}, downsamples: {len(downsamples)}, enc_dilations: {len(enc_dilations)}, dec_dilations: {len(dec_dilations)}")


if __name__ == '__main__':
    unet = UNet()
    print(unet)
    dummpy_input = torch.randn(1, 1, 256, 256)
    output = unet(dummpy_input)