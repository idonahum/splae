import warnings
from torch import nn


class ConvModule(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 padding=0,
                 groups=1,
                 bias='auto',
                 norm_cfg=None,
                 dropout_cfg=None,
                 act_cfg=None,
                 order: tuple = ('conv', 'norm', 'dropout', 'act')):
        super().__init__()
        self.order = order

        if norm_cfg is not None and self.order.index('conv') < self.order.index('norm'):
            if bias == 'auto':
                bias = False
            elif bias:
                warnings.warn('Unnecessary setting for bias since BatchNorm2d has been used.')

        op_dict = {'conv': nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias), 'norm': None, 'dropout': None, 'act': None}

        if norm_cfg is not None:
            op_dict['norm'] = build_batch_norm_layer(norm_cfg, out_channels)
        if dropout_cfg is not None:
            pass
        if act_cfg is not None:
            op_dict['act'] = build_activation_layer(act_cfg)

        for op in self.order:
            if op_dict[op] is not None:
                self.add_module(op, op_dict[op])


class DeconvModule(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BatchNorm2d'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 kernel_size=4,
                 scale_factor=2):
        super().__init__()

        self._check_kernel_and_scale_factor(kernel_size, scale_factor)

        stride = scale_factor
        padding = (kernel_size - scale_factor) // 2
        self.add_module('deconv', nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding))

        self.add_module('norm', build_batch_norm_layer(norm_cfg, out_channels))
        self.add_module('act', build_activation_layer(act_cfg))

    @staticmethod
    def _check_kernel_and_scale_factor(kernel_size, scale_factor):
        assert (kernel_size - scale_factor >= 0) and \
               (kernel_size - scale_factor) % 2 == 0, \
            f'kernel_size should be greater than or equal to scale_factor ' \
            f'and (kernel_size - scale_factor) should be even numbers, ' \
            f'while the kernel size is {kernel_size} and scale_factor is ' \
            f'{scale_factor}.'


class InterpConv(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_first=False,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 norm_cfg=dict(type='BatchNorm2d'),
                 dropout_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(scale_factor=2, mode='bilinear', align_corners=False)):
        super().__init__()
        op_dict = {'conv': ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm_cfg=norm_cfg,
            dropout_cfg=dropout_cfg,
            act_cfg=act_cfg), 'upsample': nn.Upsample(**upsample_cfg)}
        if conv_first:
            self.add_module('conv', op_dict['conv'])
            self.add_module('upsample', op_dict['upsample'])
        else:
            self.add_module('upsample', op_dict['upsample'])
            self.add_module('conv', op_dict['conv'])


def build_batch_norm_layer(norm_cfg, num_features):
    """
    Build batch normalization layer dynamically using torch.nn.modules.batchnorm.

    Args:
        norm_cfg (dict): A dictionary containing the normalization configuration.
                         It must have a key 'type' specifying the normalization type
                         (e.g., 'BatchNorm2d', 'SyncBatchNorm') and any other arguments
                         required by the normalization class.
        num_features (int): The number of features for the normalization layer.

    Returns:
        nn.Module: An instance of the normalization layer.

    Raises:
        AttributeError: If the specified normalization type is not found in torch.nn.modules.batchnorm.
    """
    cfg_ = norm_cfg.copy()
    norm_type = cfg_.pop('type')

    # Attempt to get the normalization class from torch.nn.modules.batchnorm
    try:
        norm_class = getattr(nn.modules.batchnorm, norm_type)
        return norm_class(num_features, **cfg_)
    except AttributeError:
        raise NotImplementedError(f"Normalization {norm_type} not found in torch.nn.modules.batchnorm")


def build_activation_layer(act_cfg):
    """
    Build activation layer dynamically using torch.nn.modules.activation.

    Args:
        act_cfg (dict): A dictionary containing the activation configuration.
                        It must have a key 'type' specifying the activation type
                        (e.g., 'ReLU', 'LeakyReLU') and any other arguments required
                        by the activation class.

    Returns:
        nn.Module: An instance of the activation layer.

    Raises:
        AttributeError: If the specified activation type is not found in torch.nn.modules.activation.
    """
    cfg_ = act_cfg.copy()
    act_type = cfg_.pop('type')

    # Attempt to get the activation class from torch.nn.modules.activation
    try:
        act_class = getattr(nn.modules.activation, act_type)
        return act_class(**cfg_)
    except AttributeError:
        raise NotImplementedError(f"Activation {act_type} not found in torch.nn.modules.activation")


def build_upsample_layer(cfg, *args, **kwargs):
    cfg_ = cfg.copy()
    upsample_type = cfg_.pop('type')
    if upsample_type == 'InterpConv':
        return InterpConv(*args, **kwargs, **cfg_)
    elif upsample_type == 'DeconvModule':
        return DeconvModule(*args, **kwargs, **cfg_)
    else:
        raise NotImplementedError(f"Upsample {upsample_type} not implemented")
