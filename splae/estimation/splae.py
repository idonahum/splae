import cv2
import numpy as np
from tqdm import tqdm
import torch
from scipy.ndimage import binary_fill_holes
from splae.estimation.base import BaseAccuracyEstimator
from splae.evaluation.evaluator import MetricsEvaluator
from splae.utils import reliability_score

class SPLAE(BaseAccuracyEstimator):
    def __init__(self,
                 source_model,
                 target_model,
                 num_classes,
                 metrics,
                 epl_model,
                 post_transform,
                 visualizer=None,
                 device='cpu'):
        self.evaluator = MetricsEvaluator(metrics, num_classes)
        self.source_model = source_model.to(device).eval()
        self.target_model = target_model.to(device).eval() if target_model is not None else None
        self.epl_model = epl_model
        self.visualizer = visualizer
        self.device = device
        self.post_transform = post_transform
        self.n_classes = num_classes

    @torch.no_grad()
    def run_evaluation(self, d_eval):
        results = []
        for sample_idx, sample in enumerate(tqdm(d_eval)):
            img = sample['img']
            logits_source = self.source_model(img.unsqueeze(0).to(self.device))
            if self.target_model is not None:
                logits_target = self.target_model(img.unsqueeze(0).to(self.device))
            else:
                logits_target = sample['seg_logits'].to(self.device)
            seg_source = self.post_transform(logits_source)
            seg_target = self.post_transform(logits_target)
            certainty_score = self._calculate_certainty_score(logits_target)
            rel_score = reliability_score(logits_target)
            forground_ratio = self._calculate_forground_ratio(logits_target)
            if self.epl_model is not None:
                enhanced_seg = self.epl_model.enhance(logits_source, seg_source, img)
            else:
                enhanced_seg = seg_source
            gt_mask = sample['gt_mask'].unsqueeze(0).to(self.device)
            if self.visualizer is not None:
                if sample_idx % 20 == 0:
                    self.visualizer.draw(
                        images=img.unsqueeze(0).to(self.device),
                        img_names=[sample['metainfo']['img_name']],
                        pred_masks=seg_source,
                        gt_masks=gt_mask,
                        pl_masks=enhanced_seg,
                        mode="test",
                    )
            real_score = self._calculate_score(seg_target, gt_mask)
            pred_score = self._calculate_score(seg_target, enhanced_seg)
            results.append(self._zip_results(real_score, pred_score, sample['metainfo']['img_name'], certainty_score, rel_score, forground_ratio))
        return results

    def _calculate_score(self, seg, label):
        self.evaluator.process(seg, label)
        return self.evaluator.finalize()

    def _calculate_certainty_score(self, seg_logits):
        seg_probs = seg_logits.softmax(dim=1).squeeze(0)  # (C, H, W)
        roi_mask = self._threshold_foreground(seg_probs)
        largest_component = self._find_largest_component(roi_mask)
        if largest_component is None:
            return 0.0
        compactness = self._compute_compactness(largest_component)
        if np.isnan(compactness):
            return 0.0
        return compactness

    def _calculate_forground_ratio(self, seg_logits):
        seg_probs = seg_logits.softmax(dim=1).squeeze(0)  # (C, H, W)
        roi_mask = self._threshold_foreground(seg_probs)
        return np.sum(roi_mask) / roi_mask.size

    @staticmethod
    def _threshold_foreground(seg_probs, threshold=0.5):
        """Threshold the sum of foreground probabilities."""
        roi_prob = torch.sum(seg_probs[1:], dim=0)  # (H, W)
        roi = (roi_prob > threshold).to(torch.uint8)
        return roi.cpu().detach().numpy()  # (H, W)

    def _find_largest_component(self, binary_mask):
        """Find the largest connected component in the binary mask."""
        num_labels, labels_im = cv2.connectedComponents(binary_mask)

        max_area = 0
        max_label = -1
        for label in range(1, num_labels):
            area = np.sum(labels_im == label)
            if area > max_area:
                max_area = area
                max_label = label

        if max_label == -1:
            return None  # No valid component

        largest_component_mask = (labels_im == max_label).astype(np.uint8)
        return largest_component_mask

    def _compute_compactness(self, binary_mask):
        """Compute shape compactness."""
        filled_mask = binary_fill_holes(binary_mask).astype(np.uint8)

        area = np.sum(filled_mask > 0)
        contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = np.sum([cv2.arcLength(c, closed=True) for c in contours])

        if perimeter == 0:
            return np.nan
        return (4 * np.pi * area) / (perimeter ** 2 + 1e-8)


