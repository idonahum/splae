import os
import torch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


class SegmentationVisualizer:
    def __init__(self, save_dir, classes, palette=None, background_index=0):
        """
        Initializes the SegmentationVisualizer.

        Args:
            save_dir (str): Directory to save visualizations.
            classes (tuple): Tuple of class names.
            palette (list, optional): List of colors for each class. If not provided, a palette is generated.
            background_index (int, optional): Index of the background class. Default is 0.
        """
        self.save_dir = save_dir
        self.classes = classes
        self.background_index = background_index

        # Generate a palette, ensuring the background is black
        if palette is None:
            self.palette = self._generate_palette(len(classes))
        else:
            self.palette = palette

        os.makedirs(save_dir, exist_ok=True)

    def _generate_palette(self, num_classes):
        """
        Generates a random color palette for the given number of classes, ignoring the background.

        Args:
            num_classes (int): Number of classes.

        Returns:
            list: List of RGB colors.
        """
        palette = [tuple(np.random.randint(0, 256, 3)) for _ in range(num_classes)]
        palette[self.background_index] = (0, 0, 0)  # Ensure background is black
        return palette

    def _apply_palette(self, mask):
        """Apply a color palette to a segmentation mask."""
        if mask is None:
            return None
        mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for label, color in enumerate(self.palette):
            mask_rgb[mask == label] = color
        return mask_rgb

    def _overlay_mask(self, image, mask_rgb):
        """Overlay an RGB mask on an image."""
        if mask_rgb is None:
            return image
        overlay = image.copy()
        alpha = 0.5  # Transparency
        overlay = (alpha * mask_rgb + (1 - alpha) * overlay).astype(np.uint8)
        return overlay

    def draw(self, images, img_names, pred_masks, gt_masks, pl_masks=None, mode='val', num_samples=1, iteration=None,
             epoch=None, ignore_index=-1, extra_visuals=None):
        """
        Visualizes and saves segmentation results.

        Args:
            images (torch.Tensor): Batch of input images of shape (B, C, H, W).
            img_names (List[str]): List of image names.
            pred_masks (torch.Tensor): Batch of predicted masks of shape (B, C, H, W).
            gt_masks (torch.Tensor): Batch of ground truth masks of shape (B, 1, H, W).
            pl_masks (torch.Tensor, optional): Batch of pseudo-label masks of shape (B, 1, H, W).
            mode (str, optional): Mode of operation, e.g., 'val' or 'train'.
            num_samples (int, optional): Number of samples to visualize. Defaults to 1.
            iteration (int, optional): Current iteration.
            epoch (int, optional): Current epoch.
            ignore_index (int, optional): Index of regions to ignore in visualization. Defaults to -1.
        """
        # Convert PyTorch tensors to NumPy arrays
        images = images.cpu().detach().numpy()  # (B, C, H, W)
        pred_masks = pred_masks.squeeze(1).cpu().detach().numpy()  # (B, H, W)
        gt_masks = gt_masks.squeeze(1).cpu().detach().numpy()  # (B, H, W)
        if pl_masks is not None:
            pl_masks = pl_masks.squeeze(1).cpu().detach().numpy()  # (B, H, W)

        # Randomly sample indices
        num_samples = min(num_samples, len(images))
        sampled_indices = np.random.choice(len(images), num_samples, replace=False)

        for idx in sampled_indices:
            img = np.transpose(images[idx], (1, 2, 0))  # Convert to (H, W, C)
            img = ((img + 1) / 2 * 255).astype(np.uint8)
            pred = pred_masks[idx]  # Shape (H, W)
            gt = gt_masks[idx]  # Shape (H, W)
            pl = pl_masks[idx] if pl_masks is not None else None  # Shape (H, W) or None

            # Mask ignored regions
            pred[gt == ignore_index] = 0  # Set ignored regions to background for visualization
            gt[gt == ignore_index] = 0  # Set ignored regions to background
            if pl is not None:
                pl[gt == ignore_index] = 0  # Set ignored regions to background

            # Convert masks to RGB using the palette
            pred_rgb = self._apply_palette(pred)  # Shape (H, W, 3)
            gt_rgb = self._apply_palette(gt)  # Shape (H, W, 3)
            pl_rgb = self._apply_palette(pl) if pl is not None else None  # Shape (H, W, 3) or None

            # Overlay masks on the image
            pred_overlay = self._overlay_mask(img, pred_rgb)  # Shape (H, W, 3)
            gt_overlay = self._overlay_mask(img, gt_rgb)  # Shape (H, W, 3)
            pl_overlay = self._overlay_mask(img, pl_rgb) if pl_rgb is not None else None  # Shape (H, W, 3) or None

            # Create a figure
            num_cols = 3 if pl is not None else 2
            fig, axes = plt.subplots(1, num_cols, figsize=(15, 8))
            axes[0].imshow(pred_overlay)
            axes[0].set_title("Prediction")
            axes[0].axis("off")

            axes[1].imshow(gt_overlay)
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            if pl is not None:
                axes[2].imshow(pl_overlay)
                axes[2].set_title("Pseudo Label")
                axes[2].axis("off")

            # Add a legend (exclude background)
            legend_patches = [
                Patch(color=np.array(color) / 255.0, label=cls_name)
                for cls_name, color in zip(self.classes, self.palette)
                if self.palette.index(color) != self.background_index
            ]
            if legend_patches:
                fig.legend(handles=legend_patches, loc="upper center", ncol=len(legend_patches))

            # Save the figure
            base_name = img_names[idx].split('.')[0]
            file_name = f"{base_name}_{mode}"
            if epoch is not None:
                file_name += f"_epoch{epoch}"
            if iteration is not None:
                file_name += f"_iter{iteration}"
            file_name += ".png"

            save_path = os.path.join(self.save_dir, file_name)
            plt.savefig(save_path)
            plt.close(fig)

def init_visualizer(visualization_cfg):
    """Initialize the visualizer."""
    visualization_dir = visualization_cfg.get("save_dir", "./visualizations")
    os.makedirs(visualization_dir, exist_ok=True)
    visualizer = SegmentationVisualizer(
        save_dir=visualization_dir,
        classes=visualization_cfg.get("classes"),
        palette=visualization_cfg.get("palette"),
        background_index=visualization_cfg.get("background_index", 0)
    )
    return visualizer
if __name__ == "__main__":
    import torch
    # Example test script
    visualizer = SegmentationVisualizer(
        save_dir="./visualizations",
        classes=["background", "class_1", "class_2", "class_3"],
        palette=[(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)],
        background_index=0
    )

    # Dummy data for testing
    images = torch.rand(4, 3, 256, 256)  # Batch of 4 RGB images
    img_names = ["image_1.nii.gz", "image_2.nii.gz", "image_3.nii.gz", "image_4.nii.gz"]
    pred_masks = torch.randint(0, 4, (4, 256, 256))  # Predicted masks
    gt_masks = torch.randint(0, 4, (4, 256, 256))  # Ground truth masks
    pl_masks = torch.randint(0, 4, (4, 256, 256))  # Pseudo-label masks

    visualizer.draw(
        images=images,
        img_names=img_names,
        pred_masks=pred_masks,
        gt_masks=gt_masks,
        pl_masks=pl_masks,
        mode="val",
        num_samples=2,
        iteration=100,
        epoch=5
    )

