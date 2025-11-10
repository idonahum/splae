import torch
import numpy as np


class MetricsEvaluator:
    def __init__(self, metrics, num_classes, ignore_index=None,device='cpu'):
        """
        Args:
            metrics (list): A list of metric class instances (e.g., [DiceMetric(), IoUMetric()])
        """
        self.metrics = metrics
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.computed_areas = []
        self.binary_masks = []  # Store binary masks for surface distance metrics
        self.device = device

    def compute_areas(self, pred_label: torch.Tensor, gt_label: torch.Tensor):
        """
        Computes the areas of the predicted and ground truth segmentation masks.

        Args:
            pred_label (torch.Tensor): Binary segmentation prediction
            gt_label (torch.Tensor): Ground truth segmentation

        Returns:
            dict: A dictionary containing the computed areas.
        """
        if self.ignore_index is not None:
            mask = gt_label != self.ignore_index
            pred_label = pred_label[mask]
            gt_label = gt_label[mask]

        intersection = pred_label[pred_label == gt_label]
        area_intersect = torch.histc(
            intersection.float(), bins=self.num_classes, min=0,
            max=self.num_classes - 1).cpu()
        area_pred_label = torch.histc(
            pred_label.float(), bins=self.num_classes, min=0,
            max=self.num_classes - 1).cpu()
        area_label = torch.histc(
            gt_label.float(), bins=self.num_classes, min=0,
            max=self.num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect
        return {"intersection": area_intersect, "union": area_union, "pred_area": area_pred_label, "gt_area": area_label}

    def _process_binary_mask(self, pred_label: torch.Tensor, gt_label: torch.Tensor):
        """
        Process a binary mask.
        """
        if self.ignore_index is not None:
            mask = gt_label != self.ignore_index
            pred_label_masked = pred_label[mask]
            gt_label_masked = gt_label[mask]
        else:
            pred_label_masked = pred_label
            gt_label_masked = gt_label
        return {'pred': pred_label_masked, 'gt': gt_label_masked}

    def process(self, pred_labels: torch.Tensor, gt_labels: torch.Tensor):
        """
        Computes all registered metrics.

        Args:
            pred_labels (torch.Tensor): Binary segmentation prediction
            gt_labels (torch.Tensor): Ground truth segmentation
        """
        # if np array, convert to torch tensor
        if isinstance(pred_labels, np.ndarray):
            pred_labels = torch.from_numpy(pred_labels).to(self.device)
        if isinstance(gt_labels, np.ndarray):
            gt_labels = torch.from_numpy(gt_labels).to(self.device)
        for pred_label, gt_label in zip(pred_labels, gt_labels):
            pred_label = pred_label.squeeze()
            gt_label = gt_label.squeeze()
            # Compute areas for area-based metrics
            self.computed_areas.append(self.compute_areas(pred_label, gt_label))
            
            # Store binary masks for surface distance metrics
            self.binary_masks.append(self._process_binary_mask(pred_label, gt_label))
            
            
    def _finalize_binary_masks(self, metric, get_best_scores=False):
        """
        Finalize the binary masks for distance metrics (per-class computation).
        """
        # For distance metrics, we need to compute per-class values
        num_classes = self.num_classes
        class_values = [[] for _ in range(num_classes)]
        
        for mask_pair in self.binary_masks:
            pred_mask = mask_pair['pred']
            gt_mask = mask_pair['gt']
            
            # Convert to numpy if needed
            if isinstance(pred_mask, torch.Tensor):
                pred_mask = pred_mask.cpu().numpy()
            if isinstance(gt_mask, torch.Tensor):
                gt_mask = gt_mask.cpu().numpy()
            
            # Compute metric for each class
            for class_idx in range(num_classes):  # Include all classes including background
                # Create binary masks for this class
                pred_binary = (pred_mask == class_idx).astype(int)
                gt_binary = (gt_mask == class_idx).astype(int)
                
                try:
                    value = metric.compute(pred_binary, gt_binary)
                    if not np.isinf(value):
                        class_values[class_idx].append(value)
                except Exception as e:
                    print(f"Warning: Could not compute {metric.name} for class {class_idx}: {e}")
                    continue
        
        # Compute final values per class
        final_values = []
        for class_idx in range(num_classes):
            if class_values[class_idx]:
                if get_best_scores:
                    final_values.append(min(class_values[class_idx]))
                else:
                    final_values.append(np.mean(class_values[class_idx]))
            else:
                final_values.append(np.inf)
        
        return torch.tensor(final_values, dtype=torch.float32)
    
    def _finalize_area_based_metrics(self, metric, get_best_scores=False):
        """
        Finalize the area-based metrics.
        """
        try:
            if get_best_scores:
                sample_scores = []
                for area in self.computed_areas:
                    sample_score = metric.compute(area)
                    sample_scores.append(sample_score)
                
                if sample_scores:
                    scores_tensor = torch.stack(sample_scores)
                    return torch.max(scores_tensor, dim=0)[0]
                else:
                    return torch.zeros(self.num_classes)
            else:
                # Compute area-based metrics with total areas
                total_intersection = sum([area["intersection"] for area in self.computed_areas])
                total_union = sum([area["union"] for area in self.computed_areas])
                total_pred = sum([area["pred_area"] for area in self.computed_areas])
                total_gt = sum([area["gt_area"] for area in self.computed_areas])
                total_areas = {"intersection": total_intersection, "union": total_union, "pred_area": total_pred, "gt_area": total_gt}
                return metric.compute(total_areas)
        except Exception as e:
            print(f"Warning: Could not compute {metric.name}: {e}")
            if get_best_scores:
                return torch.zeros(self.num_classes)
            else:
                return 0.0
    
    def finalize(self, get_best_scores=False):
        """
        Finalizes the computation of the metrics and returns the results.

        Args:
            get_best_scores (bool): If True, returns the best scores per class instead of averaged scores.
                                   For overlap metrics, returns max scores. For distance metrics, returns min scores.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        metrics = {}
        if not self.computed_areas and not self.binary_masks:
            return None
            
        for metric in self.metrics:
            metric_name = metric.name
            
            # Check if metric requires binary masks (surface distance metrics)
            # Use the is_overlap property to determine metric type
            if not metric.is_overlap or metric_name == 'Dice_Medpy':
                metrics[metric_name] = self._finalize_binary_masks(metric, get_best_scores)
            else:
                metrics[metric_name] = self._finalize_area_based_metrics(metric, get_best_scores)
        
        # Reset for next batch
        self.computed_areas = []
        self.binary_masks = []
        return metrics

