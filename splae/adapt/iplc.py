import pandas as pd
from splae.utils import norm_0_255, get_largest_component, gray2rgb, random_sample
from splae.adapt.base import SamPLGeneratorBase
from segment_anything.predictor_sammed import SammedPredictor
import torch
import numpy as np
import cv2

class IPLCGenerator(SamPLGeneratorBase):
    def __init__(self,
                 model_type,
                 weights_path,
                 image_size,
                 num_classes,
                 device,
                 encoder_adapter=True,
                 ignore_index=None,
                 visualizer=None,
                 samples_times=3,
                 benchmark=False):
        super(IPLCGenerator, self).__init__(model_type, weights_path, image_size, device, encoder_adapter, visualizer)
        self.num_classes = num_classes
        self._predictor = SammedPredictor(self.model)
        self.sample_times = samples_times
        self.ignore_index = ignore_index
        self.benchmark = benchmark

    @property
    def predictor(self):
        return self._predictor

    def _one_hot_dice(self, target, num_classes):
        if target.ndim == 4:
            target = target.squeeze(1)
        min = -1 if self.ignore_index is not None and self.ignore_index < 0 else 0
        max = num_classes - 1 if self.ignore_index is not None and self.ignore_index < 0 else num_classes
        one_hot_target = torch.clamp(target, min, max)
        one_hot_target = one_hot_target + 1 if min == -1 else one_hot_target
        one_hot_target = torch.nn.functional.one_hot(one_hot_target.long(),
                                                     num_classes + 1)

        min_idx = 1 if self.ignore_index is not None and self.ignore_index < 0 else 0
        max_idx = num_classes + 1 if self.ignore_index is not None and self.ignore_index < 0 else num_classes
        one_hot_target = one_hot_target[..., min_idx:max_idx].permute(0, 3, 1, 2)
        return one_hot_target

    def generate(self, images, seg_logits, ignore_index_mask=None, gt_masks=None):
        seg_probs = seg_logits.softmax(dim=1)
        seg_masks = seg_probs.argmax(dim=1).unsqueeze(1)
        if ignore_index_mask is not None:
            assert self.ignore_index is not None, "Ignore index mask is provided but ignore index is not set"
            seg_masks[ignore_index_mask] = self.ignore_index
        one_hot_masks = self._one_hot_dice(seg_masks, self.num_classes).cpu().detach().numpy()
        mask_input = seg_probs.cpu().detach().numpy()  # (B, C, H, W)
        pseudo_labels_one_hot, weight_entropy_maps = self.get_pl_label(
            images, one_hot_masks, mask_input, gt_masks=gt_masks
        )
        pseudo_labels = np.argmax(pseudo_labels_one_hot, axis=1)
        pseudo_labels = torch.tensor(pseudo_labels).float().unsqueeze(1).to(self.device)
        weight_entropy_maps = torch.tensor(weight_entropy_maps).float().to(self.device)
        valid_indices = np.arange(images.shape[0])
        return {'target': pseudo_labels, 'depth_weight_entropy': weight_entropy_maps, 'valid_indices': valid_indices}

    def get_pl_label(self, images, one_hot_masks, mask_input, gt_masks=None):
        # Step 1: Normalize images to 0-255
        normed_images = norm_0_255(images)
        batch_size, num_classes, height, width = one_hot_masks.shape

        weight_entropy_maps = np.zeros((batch_size, height, width), dtype=np.float64)
        sam_pseudo_labels = np.zeros((batch_size, num_classes, height, width), dtype=np.float32)
        for img_idx in range(batch_size):
            sam_masks = np.zeros((self.sample_times, num_classes, height, width), dtype=np.float32)
            rgb_img = gray2rgb(normed_images[img_idx])
            self.predictor.set_image(rgb_img)

            gt_mask = self._one_hot_dice(gt_masks[img_idx], self.num_classes).squeeze(0).cpu().detach().numpy() if gt_masks is not None else None
            largest_components, resized_mask_input = self.prepare_base_prompts(one_hot_masks[img_idx],
                                                                               mask_input[img_idx])


            for sample_idx in range(self.sample_times):
                sam_masks[sample_idx] = self.run_iter(gt_mask, resized_mask_input) if self.benchmark else self.run_iter(
                    largest_components, resized_mask_input
                )

            # Step 2: Get average probability maps
            if gt_masks is not None:
                probability_map = self.get_closest_mask_to_gt(sam_masks, gt_masks[img_idx])
            else:
                probability_map = self.get_average_probability_map(sam_masks, self.sample_times)
            # Step 3: Compute weight entropy map
            weight_entropy_maps[img_idx] = self.get_weight_entropy_map(probability_map)
            # Step 4: Finalize segmentation mask
            sam_pseudo_labels[img_idx] = self.finalize(probability_map)

            if self.visualizer is not None:
                self.visualizer.visualize_sample(rgb_img, gt_mask)

        return sam_pseudo_labels, weight_entropy_maps

    def prepare_base_prompts(self, one_hot_mask, mask_input):
        num_classes = one_hot_mask.shape[0]
        largest_components = np.zeros_like(one_hot_mask)
        resized_mask_input = np.zeros((num_classes, 64, 64), dtype=np.float64)
        for cls_idx in range(num_classes):
            # Get the largest component per class
            largest_component = get_largest_component(one_hot_mask[cls_idx])
            largest_components[cls_idx] = largest_component

            # Resize the mask input to 64x64
            resized_mask_input[cls_idx] = cv2.resize(mask_input[cls_idx], (64, 64), interpolation=cv2.INTER_LINEAR)
        return largest_components, resized_mask_input

    def run_iter(self, largest_components, resized_mask_input):
        num_classes = largest_components.shape[0]
        iter_sam_logits = np.zeros_like(largest_components, dtype=np.float64)
        for cls_idx in range(1, num_classes):
            points = self.sample_points(largest_components[cls_idx])
            if points is not None:
                labels = np.ones(points.shape[0], dtype=np.int32)
                _, scores, logits = self.predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    mask_input=resized_mask_input[cls_idx][None, ...],
                    multimask_output=True,
                )
            else:
                _, scores, logits = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    mask_input=resized_mask_input[cls_idx][None, ...],
                    multimask_output=True)
            iter_sam_logits[cls_idx] = logits.copy()

        total_forground = np.sum(iter_sam_logits, axis=0)  # H, W
        background = 1 - total_forground
        background[background < 0] = 0
        iter_sam_logits[0] = background
        return iter_sam_logits

    def sample_points(self, largest_component):
        return random_sample(largest_component)

    @staticmethod
    def get_average_probability_map(sam_logits, sample_times):
        num_classes = sam_logits.shape[1]
        probability_maps = np.zeros((num_classes, sam_logits.shape[2], sam_logits.shape[3]), dtype=np.float64)
        for cls_idx in range(num_classes):
            probability_maps[cls_idx] = np.sum(sam_logits[:, cls_idx], axis=0) / sample_times
        return probability_maps

    def get_weight_entropy_map(self, probability_maps):
        """
        Compute entropy of the segmentation probabilities.
        Args:
            probability_maps (np.ndarray): Segmentation probabilities of shape (C, H, W).
        Returns:
            np.ndarray: Entropy map of shape (H, W).
        """
        # Compute entropy
        entropy = -np.sum(probability_maps * np.log2(probability_maps + 1e-10), axis=0)
        weight_entropy_map = (2 - entropy) / 2
        return weight_entropy_map

    def finalize(self, probability_map):
        """
        Finalize the segmentation mask by applying a threshold.
        Args:
            probability_map (np.ndarray): Segmentation probabilities of shape (H, W).
        Returns:
            np.ndarray: Finalized segmentation mask of shape (H, W).
        """
        num_classes = probability_map.shape[0]
        one_hot_map = np.zeros((num_classes, probability_map.shape[1], probability_map.shape[2]), dtype=np.float32)
        argmax_map = np.argmax(probability_map, axis=0)
        for cls_idx in range(num_classes):
            class_map = (argmax_map == cls_idx).astype(np.float32)
            one_hot_map[cls_idx] = get_largest_component(class_map)
        return one_hot_map

    def get_closest_mask_to_gt(self, sam_masks: np.ndarray, gt_mask: np.ndarray,
                               ignore_index: int = None) -> np.ndarray:
        """
        Get the mask that is closest to the ground truth in terms of mean Dice score.
        Args:
            sam_masks (np.ndarray): Segmentation masks of shape (sample_times, C, H, W).
            gt_mask (np.ndarray): Ground truth mask of shape (1, H, W), values in [0, NUM_CLASSES-1] or ignore_index.
            ignore_index (int, optional): Label value to ignore in evaluation.
        Returns:
            np.ndarray: Closest mask to the ground truth, shape (C, H, W).
        """

        def dice_score(pred, target, num_classes):
            dice_scores = []
            for cls in range(num_classes):
                pred_cls = (pred == cls)
                target_cls = (target == cls)

                intersection = np.sum(pred_cls & target_cls)
                union = np.sum(pred_cls) + np.sum(target_cls)
                if union == 0:
                    continue  # Skip this class if it doesn't appear in either
                dice_scores.append(2 * intersection / union)

            return np.mean(dice_scores) if dice_scores else 0.0

        sample_times, num_classes, H, W = sam_masks.shape
        gt_mask = gt_mask.squeeze(0)  # Shape: (H, W)
        gt_mask = gt_mask.cpu().detach().numpy()

        best_score = -1
        best_mask = None

        for i in range(sample_times):
            pred_probs = sam_masks[i]  # shape (C, H, W)
            pred_mask = np.argmax(pred_probs, axis=0)  # shape (H, W)

            # Apply ignore mask to prediction
            if self.ignore_index is not None:
                pred_mask = pred_mask.copy()
                pred_mask[gt_mask == self.ignore_index] = self.ignore_index

            # Evaluate Dice
            score = dice_score(pred_mask, gt_mask, num_classes)
            if score > best_score:
                best_score = score
                best_mask = pred_probs  # Keep the original prob mask

        return best_mask