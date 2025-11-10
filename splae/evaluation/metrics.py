from abc import ABC, abstractmethod
import numpy as np
import torch
import medpy.metric.binary as metrics


class BaseMetric(ABC):
    """Base class for all segmentation metrics."""
    def compute(self, **kwargs):
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def is_overlap(self):
        pass


class DiceMetric(BaseMetric):
    def compute(self, areas):
        intersection = areas.get('intersection')
        pred_area = areas.get('pred_area')
        gt_area = areas.get('gt_area')
        assert intersection is not None, 'Intersection area is not provided'
        assert pred_area is not None, 'Predicted area is not provided'
        assert gt_area is not None, 'Ground truth area is not provided'
        return self._compute(intersection, pred_area, gt_area)

    def _compute(self, intersection, pred_area, gt_area):
        return (2 * intersection) / (pred_area + gt_area + 1e-6)

    @property
    def name(self):
        return 'Dice'

    @property
    def is_overlap(self):
        return True

class IoUMetric(BaseMetric):
    def compute(self, areas):
        intersection = areas.get('intersection')
        pred_area = areas.get('pred_area')
        gt_area = areas.get('gt_area')
        union = areas.get('union')
        assert intersection is not None, 'Intersection area is not provided'
        assert union is not None, 'Union area is not provided'
        return intersection / (union + 1e-6)

    @property
    def name(self):
        return 'IoU'

    @property
    def is_overlap(self):
        return True

class AccuracyMetric(BaseMetric):
    def compute(self, areas):
        intersection = areas.get('intersection')
        gt_area = areas.get('gt_area')
        assert intersection is not None, 'Intersection area is not provided'
        assert gt_area is not None, 'Ground truth area is not provided'
        return intersection / (gt_area + 1e-6)

    @property
    def name(self):
        return 'Accuracy'

    @property
    def is_overlap(self):
        return True

class HausdorffMetric(BaseMetric):
    """Hausdorff distance metric using medpy."""
    def compute(self, pred_mask, gt_mask):
        """
        Args:
            pred_mask (torch.Tensor): Binary prediction mask
            gt_mask (torch.Tensor): Binary ground truth mask
        """
        # Convert to numpy if needed
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.cpu().numpy()
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.cpu().numpy()
        
        # Ensure binary masks
        pred_mask = (pred_mask > 0).astype(int)
        gt_mask = (gt_mask > 0).astype(int)
        
        if np.count_nonzero(pred_mask) == 0 or np.count_nonzero(gt_mask) == 0:
            return np.inf
        return metrics.hd(pred_mask, gt_mask)

    @property
    def name(self):
        return 'Hausdorff'

    @property
    def is_overlap(self):
        return False

class ASSDMetric(BaseMetric):
    """Average Symmetric Surface Distance metric using medpy."""
    def compute(self, pred_mask, gt_mask):
        """
        Args:
            pred_mask (torch.Tensor): Binary prediction mask
            gt_mask (torch.Tensor): Binary ground truth mask
        """
        # Convert to numpy if needed
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.cpu().numpy()
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.cpu().numpy()
        
        # Ensure binary masks
        pred_mask = (pred_mask > 0).astype(int)
        gt_mask = (gt_mask > 0).astype(int)
        
        if np.count_nonzero(pred_mask) == 0 or np.count_nonzero(gt_mask) == 0:
            return np.inf
        return metrics.assd(pred_mask, gt_mask)

    @property
    def name(self):
        return 'ASSD'
    
    @property
    def is_overlap(self):
        return False


class DiceMedpyMetric(BaseMetric):
    """Dice coefficient metric using medpy."""
    def compute(self, pred_mask, gt_mask):
        """
        Args:
            pred_mask (torch.Tensor): Binary prediction mask
            gt_mask (torch.Tensor): Binary ground truth mask
        """
        # Convert to numpy if needed
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.cpu().numpy()
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.cpu().numpy()
        
        # Ensure binary masks
        pred_mask = (pred_mask > 0).astype(int)
        gt_mask = (gt_mask > 0).astype(int)
        
        return metrics.dc(pred_mask, gt_mask)

    @property
    def name(self):
        return 'Dice_Medpy'
    
    @property
    def is_overlap(self):
        return True