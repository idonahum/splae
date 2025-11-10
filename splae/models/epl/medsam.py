import cv2
import numpy as np
import torch

from splae.models.epl.base import EPLBase
from splae.utils import norm_0_255, get_largest_component, random_sample, gray2rgb

try:
    from segment_anything import sam_model_registry
    from segment_anything.predictor_sammed import SammedPredictor
    SAM_AVAILABLE = True
except ImportError as e:
    SAM_AVAILABLE = False
    print("Warning: SAM not available. SAM-based EPL will not work. Reason:", e)


class MedSamEPL(EPLBase):
    """
    SAM-based Enhanced Pseudo Labeling that uses Segment Anything Model to enhance pseudo labels.

    This implementation mimics the IPLCGenerator approach, using SAM to refine pseudo labels
    by leveraging the model's ability to segment objects with high precision.
    """

    def __init__(self,
                 sam_model_type: str,
                 sam_weights_path: str,
                 image_size: int,
                 num_classes: int,
                 device: str,
                 sample_times: int = 1,
                 ignore_index: int = None,
                 **kwargs):
        """
        Initialize the SAM-based EPL.

        Args:
            confidence_threshold: Minimum confidence threshold for pseudo labels
            sam_model_type: SAM model type ("vit_b", "vit_l", "vit_h")
            sam_weights_path: Path to SAM model weights
            image_size: Image size for SAM processing
            sample_times: Number of sampling iterations for SAM
            device: Device to run SAM on
            **kwargs: Additional parameters
        """
        if not SAM_AVAILABLE:
            raise ImportError("SAM is not available. Please install segment-anything.")

        self.sam_model_type = sam_model_type
        self.sam_weights_path = sam_weights_path
        self.image_size = image_size
        self.sample_times = sample_times
        self.num_classes = num_classes
        self.device = device
        self.ignore_index = ignore_index
        # Initialize SAM model and predictor
        self._initialize_sam()

    def _initialize_sam(self):
        """Initialize SAM model and predictor."""
        self.sam_model = sam_model_registry.get(self.sam_model_type)(
            self.image_size, self.sam_weights_path, encoder_adapter=True
        ).to(self.device)
        self.predictor = SammedPredictor(self.sam_model)
        print(f"SAM model initialized: {self.sam_model_type}")

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

    def enhance(self,
              logits: torch.Tensor,
              original_pseudo_labels: torch.Tensor,
              images: torch.Tensor,
              **kwargs) -> torch.Tensor:
        """
        Enhance pseudo labels using SAM.

        Args:
            logits: Model output logits of shape (B, C, H, W)
            original_pseudo_labels: Original pseudo labels of shape (B, H, W)
            **kwargs: Additional parameters

        Returns:
            enhanced_pseudo_labels: Enhanced pseudo labels of shape (B, H, W)
        """

        seg_probs = logits.softmax(dim=1)
        # seg_masks = original_pseudo_labels.unsqueeze(1)
        one_hot_masks = self._one_hot_dice(original_pseudo_labels, self.num_classes).cpu().detach().numpy()
        mask_input = seg_probs.cpu().detach().numpy()
        pseudo_labels_one_hot = self.get_pl_label(
            images, one_hot_masks, mask_input)
        pseudo_labels = np.argmax(pseudo_labels_one_hot, axis=1)
        pseudo_labels = torch.tensor(pseudo_labels).float().unsqueeze(1).to(self.device)
        return pseudo_labels

    def get_pl_label(self, images, one_hot_masks, mask_input):
        # Step 1: Normalize images to 0-255
        normed_images = norm_0_255(images)
        batch_size,_, height, width = one_hot_masks.shape

        sam_pseudo_labels = np.zeros((batch_size, self.num_classes, height, width), dtype=np.float32)
        for img_idx in range(batch_size):
            sam_masks = np.zeros((self.sample_times, self.num_classes, height, width), dtype=np.float32)
            rgb_img = gray2rgb(normed_images[img_idx])
            self.predictor.set_image(rgb_img)

            largest_components, resized_mask_input = self.prepare_base_prompts(one_hot_masks[img_idx],
                                                                               mask_input[img_idx])

            for sample_idx in range(self.sample_times):
                sam_masks[sample_idx] = self.run_iter(
                    largest_components, resized_mask_input
                )

            # Step 2: Get average probability maps
            probability_map = self.get_average_probability_map(sam_masks, self.sample_times)
            # Step 3: Finalize segmentation mask
            sam_pseudo_labels[img_idx] = self.finalize(probability_map)

        return sam_pseudo_labels

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
                    # mask_input=resized_mask_input[cls_idx][None, ...],
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

    @staticmethod
    def sample_points(largest_component):
        return random_sample(largest_component)

    @staticmethod
    def get_average_probability_map(sam_logits, sample_times):
        num_classes = sam_logits.shape[1]
        probability_maps = np.zeros((num_classes, sam_logits.shape[2], sam_logits.shape[3]), dtype=np.float64)
        for cls_idx in range(num_classes):
            probability_maps[cls_idx] = np.sum(sam_logits[:, cls_idx], axis=0) / sample_times
        return probability_maps

    @staticmethod
    def finalize(probability_map):
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


class IMedSamEPL(MedSamEPL):
    """
    Iterative MedSAM EPL with entropy-based convergence.
    Keeps refining each class until entropy stabilizes or max_iter is reached.
    """

    def __init__(self, *args, max_iter: int = 10, entropy_tol: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_iter = max_iter
        self.entropy_tol = entropy_tol

    @staticmethod
    def compute_entropy(prob_map):
        """Compute pixel-wise entropy."""
        eps = 1e-8
        return -np.sum(prob_map * np.log(prob_map + eps), axis=0)

def run_iterative_refinement(self, rgb_img, largest_components, mask_input):
    """
    Iteratively refine pseudo-labels using MedSAM until entropy stabilizes.
    """
    num_classes = largest_components.shape[0]
    prev_entropy = None
    aggregated_logits = None  # allocate lazily after first SAM call

    for it in range(self.max_iter):
        iter_logits = None
        self.predictor.set_image(rgb_img)

        for cls_idx in range(1, num_classes):
            # --- adaptive prompt sampling ---
            if it == 0:
                points = self.sample_points(largest_components[cls_idx])
            else:
                # focus new samples on uncertain (high-entropy) areas
                if 'entropy_map' in locals():
                    uncertain_mask = (entropy_map > np.percentile(entropy_map, 75)).astype(np.uint8)
                    points = random_sample(uncertain_mask)
                else:
                    points = self.sample_points(largest_components[cls_idx])

            if points is not None:
                labels = np.ones(points.shape[0], dtype=np.int32)
                _, _, logits = self.predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    multimask_output=True,
                )
            else:
                _, _, logits = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    mask_input=mask_input[cls_idx][None, ...],
                    multimask_output=True,
                )

            # --- dynamic shape allocation ---
            if aggregated_logits is None:
                h, w = logits.shape[-2:]
                aggregated_logits = np.zeros((num_classes, h, w), dtype=np.float64)
                iter_logits = np.zeros_like(aggregated_logits)
            elif iter_logits is None:
                iter_logits = np.zeros_like(aggregated_logits)

            iter_logits[cls_idx] = logits.copy()

        # --- background as complement of all foreground classes ---
        total_foreground = np.sum(iter_logits[1:], axis=0)
        iter_logits[0] = np.clip(1 - total_foreground, 0, 1)

        # --- running mean aggregation ---
        aggregated_logits = (aggregated_logits * it + iter_logits) / (it + 1)

        # --- compute entropy for convergence check ---
        prob_map = aggregated_logits / np.maximum(aggregated_logits.sum(0, keepdims=True), 1e-6)
        entropy_map = self.compute_entropy(prob_map)
        mean_entropy = float(entropy_map.mean())

        if prev_entropy is not None:
            delta = abs(mean_entropy - prev_entropy)
            if delta < self.entropy_tol:
                print(f"Entropy stabilized after {it+1} iterations (Δ={delta:.4f})")
                break
        prev_entropy = mean_entropy

    return aggregated_logits


    def get_pl_label(self, images, one_hot_masks, mask_input):
        normed_images = norm_0_255(images)
        batch_size, _, height, width = one_hot_masks.shape
        sam_pseudo_labels = np.zeros((batch_size, self.num_classes, height, width), dtype=np.float32)

        for img_idx in range(batch_size):
            rgb_img = gray2rgb(normed_images[img_idx])
            largest_components, resized_mask_input = self.prepare_base_prompts(
                one_hot_masks[img_idx], mask_input[img_idx]
            )
            aggregated_logits = self.run_iterative_refinement(rgb_img, largest_components, resized_mask_input)
            sam_pseudo_labels[img_idx] = self.finalize(aggregated_logits)

        return sam_pseudo_labels
