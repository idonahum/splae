from monai.losses import DiceLoss as _DiceLoss
from monai.networks.utils import one_hot
from torch.nn import CrossEntropyLoss as _CrossEntropyLoss
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(_DiceLoss):
    def __init__(
            self,
            include_background= True,
            to_onehot_y=False,
            sigmoid=False,
            softmax=False,
            ignore_index=-1,
            loss_weight=1.0,
            other_act=None,
            squared_pred= False,
            jaccard: bool = False,
            reduction= "mean",
            eps = 0.001,
            batch: bool = False,
            class_weight= None):
        super().__init__(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=eps,
            smooth_dr=eps,
            batch=batch,
            weight=class_weight
        )
        self.eps = eps
        self.ignore_index = ignore_index
        assert loss_weight > 0, 'loss_weight should be greater than 0'
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        one_hot_target = target
        if one_hot_target.shape != pred.shape:
            one_hot_target = self._one_hot_dice(pred, target)

        if self.sigmoid:
            pred = torch.sigmoid(pred)

        elif self.softmax and pred.shape[1] != 1:
            pred = pred.softmax(dim=1)

        if self.ignore_index is not None:
            num_classes = pred.shape[1]
            pred = pred[:, torch.arange(num_classes) != self.ignore_index, :, :]
            one_hot_target = one_hot_target[:, torch.arange(num_classes) != self.ignore_index, :, :]
            assert pred.shape[1] != 0

        pred = pred.flatten(1)
        one_hot_target = one_hot_target.flatten(1).float()

        dice = self._dice_loss(pred, one_hot_target)
        loss = 1.0 - dice
        loss = self.reduct(loss)
        return loss * self.loss_weight

    def reduct(self, loss):
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()

    def _one_hot_dice(self, pred, target):
        num_classes = pred.shape[1]
        if pred.dim() == target.dim():
            target = torch.squeeze(target, dim=1)
        min = -1 if self.ignore_index is not None and self.ignore_index < 0 else 0
        max = num_classes-1 if self.ignore_index is not None and self.ignore_index < 0 else num_classes
        one_hot_target = torch.clamp(target, min, max)
        one_hot_target = one_hot_target + 1 if min == -1 else one_hot_target
        one_hot_target = torch.nn.functional.one_hot(one_hot_target.long(),
                                                     num_classes + 1)

        min_idx = 1 if self.ignore_index is not None and self.ignore_index < 0 else 0
        max_idx = num_classes+1 if self.ignore_index is not None and self.ignore_index < 0 else num_classes
        one_hot_target = one_hot_target[..., min_idx:max_idx].permute(0, 3, 1, 2)
        return one_hot_target

    def _dice_loss(self, pred, one_hot_target):
        intersection = torch.sum(pred * one_hot_target, dim=1)
        if self.squared_pred:
            pred_sum = torch.sum(pred ** 2, dim=1)
            target_sum = torch.sum(one_hot_target ** 2, dim=1)
            nominator = 2.0 * intersection
        else:
            pred_sum = torch.sum(pred, dim=1)
            target_sum = torch.sum(one_hot_target, dim=1)
            nominator = 2.0 * intersection + self.eps

        if self.jaccard:
            denominator = pred_sum + target_sum - intersection
        else:
            denominator = pred_sum + target_sum

        denominator = denominator + 2*self.eps if self.squared_pred else denominator+self.eps
        dice = nominator / denominator
        return dice

    @property
    def name(self):
        return "dice_loss"


class WeightedDiceLoss(DiceLoss):
    def forward(self,pred, target_data):
        entropy = None
        if isinstance(target_data, dict):
            target = target_data.get('target')
            entropy = target_data.get('depth_weight_entropy')
            assert entropy is not None, 'entropy should not be None'
            assert target is not None, 'target should not be None'
            entropy = entropy.unsqueeze(1)
        else:
            target = target_data
        one_hot_target = target

        if one_hot_target.shape != pred.shape:
            one_hot_target = self._one_hot_dice(pred, target)

        if self.sigmoid:
            pred = torch.sigmoid(pred)

        elif self.softmax and pred.shape[1] != 1:
            pred = pred.softmax(dim=1)

        if self.ignore_index is not None:
            num_classes = pred.shape[1]
            pred = pred[:, torch.arange(num_classes) != self.ignore_index, :, :]
            one_hot_target = one_hot_target[:, torch.arange(num_classes) != self.ignore_index, :, :]
            assert pred.shape[1] != 0

        pred = pred * entropy if entropy is not None else pred
        one_hot_target = one_hot_target * entropy if entropy is not None else one_hot_target
        pred = pred.flatten(1)
        one_hot_target = one_hot_target.flatten(1).float()

        dice = self._dice_loss(pred, one_hot_target)
        loss = 1.0 - dice
        loss = self.reduct(loss)
        return loss * self.loss_weight

    @property
    def name(self):
        return "weighted_dice_loss"

class CrossEntropyLoss(_CrossEntropyLoss):
    def __init__(self, loss_weight=1.0, weight=None, ignore_index=-1, reduction='mean'):
        super().__init__(weight=weight, ignore_index=ignore_index, reduction=reduction)
        assert loss_weight > 0, 'loss_weight should be greater than 0'
        self.loss_weight = loss_weight

    def forward(self, input, target):
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch != n_target_ch and n_target_ch == 1:
            target = torch.squeeze(target, dim=1)
            target = target.long()
        return super().forward(input, target) * self.loss_weight

    @property
    def name(self):
        return "ce_loss"



class CurvatureLoss(nn.Module):
    def __init__(self, loss_weight=1.0, sigmoid=False, softmax=False, device=None):
        super().__init__()
        assert loss_weight > 0, 'loss_weight should be greater than 0'
        self.loss_weight = loss_weight
        self.sigmoid = sigmoid
        self.softmax = softmax

        # Set device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define fixed filters as buffers on the correct device
        self.register_buffer('laplace_filter', torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32,
                                                            device=self.device).view(1, 1, 3, 3))
        self.register_buffer('dx', torch.tensor([[-1, 0, 1]], dtype=torch.float32, device=self.device).view(1, 1, 1, 3))
        self.register_buffer('dy',
                             torch.tensor([[-1], [0], [1]], dtype=torch.float32, device=self.device).view(1, 1, 3, 1))
        self.register_buffer('dxx',
                             torch.tensor([[-1, 2, -1]], dtype=torch.float32, device=self.device).view(1, 1, 1, 3))
        self.register_buffer('dyy',
                             torch.tensor([[-1], [2], [-1]], dtype=torch.float32, device=self.device).view(1, 1, 3, 1))

    def forward(self, pred, target):
        if self.sigmoid:
            pred = torch.sigmoid(pred)

        elif self.softmax and pred.shape[1] != 1:
            pred = pred.softmax(dim=1)
        
        # Dynamically handle different numbers of classes
        num_classes = pred.shape[1]
        
        if num_classes == 2:  # Binary segmentation (e.g., MSM dataset)
            prob_1 = pred[:, 1]  # Only prostate class
            loss = self._compute_curvature_loss(prob_1)
        elif num_classes == 4:  # Multi-class segmentation (e.g., MNMs dataset)
            prob_1 = pred[:, 1]
            prob_2 = pred[:, 1] + pred[:, 2]
            prob_3 = pred[:, 3]
            loss = (
                self._compute_curvature_loss(prob_1) +
                self._compute_curvature_loss(prob_2) +
                self._compute_curvature_loss(prob_3)
            )
        else:
            # Generic case: compute curvature loss for all non-background classes
            loss = 0
            for i in range(1, num_classes):
                loss += self._compute_curvature_loss(pred[:, i])

        return loss * self.loss_weight

    def _compute_curvature_loss(self, prob_maps):
        B, H, W = prob_maps.shape
        loss = 0.0

        for b in range(B):
            prob = prob_maps[b].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

            laplace = F.conv2d(prob, self.laplace_filter, padding=1)

            gx = F.conv2d(laplace, self.dx, padding=(0, 1))
            gy = F.conv2d(laplace, self.dy, padding=(1, 0))

            hxx = F.conv2d(gx, self.dxx, padding=(0, 1))
            hyy = F.conv2d(gy, self.dyy, padding=(1, 0))
            hxy = F.conv2d(gx, self.dx, padding=(0, 1))
            hyx = F.conv2d(gy, self.dy, padding=(1, 0))

            numerator = (
                hxx * (1 + gy) ** 2 -
                2 * hxy * gx * gy +
                hyy * (1 + gx) ** 2
            )
            denominator = 2 * (1 + gx ** 2 + gy ** 2) ** (1.5)
            curvature = numerator / denominator

            neg_curv = F.relu(-curvature)
            if torch.sum(neg_curv != 0) > 0:
                avg_neg_curv = torch.sum(neg_curv) / torch.sum(neg_curv != 0).float()
                loss += avg_neg_curv

        return loss

    @property
    def name(self):
        return "curvature_loss"

class DiceLossV2(DiceLoss):
    def forward(self,pred, target_data):
        entropy = None
        if isinstance(target_data, dict):
            target = target_data.get('target')
            entropy = target_data.get('depth_weight_entropy')
            assert entropy is not None, 'entropy should not be None'
            assert target is not None, 'target should not be None'
            entropy = entropy.unsqueeze(1)
        else:
            target = target_data
        one_hot_target = target
        if one_hot_target.shape != pred.shape:
            one_hot_target = self._one_hot_dice(pred, target)

        if self.sigmoid:
            pred = torch.sigmoid(pred)

        elif self.softmax and pred.shape[1] != 1:
            pred = pred.softmax(dim=1)

        if self.ignore_index is not None:
            num_classes = pred.shape[1]
            pred = pred[:, torch.arange(num_classes) != self.ignore_index, :, :]
            one_hot_target = one_hot_target[:, torch.arange(num_classes) != self.ignore_index, :, :]
            assert pred.shape[1] != 0

        if entropy is not None:
            weight_map = entropy.expand_as(pred)
            intersection = torch.sum(weight_map * pred * one_hot_target, dim=(0, 2, 3))
            inputs_sum = torch.sum(weight_map * pred, dim=(0, 2, 3))
            targets_sum = torch.sum(weight_map * one_hot_target, dim=(0, 2, 3))
        else:
            intersection = torch.sum(pred * one_hot_target, dim=(0, 2, 3))
            inputs_sum = torch.sum(pred * pred, dim=(0, 2, 3))
            targets_sum = torch.sum(one_hot_target, dim=(0, 2, 3))
        smooth = 1e-4
        dice = (2.0 * intersection + smooth) / (inputs_sum + targets_sum + smooth)
        dice_loss_per_class = 1.0 - dice
        return dice_loss_per_class.mean()

    @property
    def name(self):
        return "dice_loss_v2"