import numpy as np
from splae.models.rca.elastix import register_label
from splae.evaluation.evaluator import MetricsEvaluator
from monai.transforms import Compose
from concurrent.futures import ThreadPoolExecutor, as_completed

class SAC:
    """Single-atlas classifier for evaluating segmentations given input image and label."""

    def __init__(self, metrics, num_classes, transform=Compose([])):
        self.evaluator = MetricsEvaluator(metrics, num_classes)
        self.num_classes = num_classes
        self.transform = transform
        self.image = None
        self.label = None

    def set_sample(self, image, label):
        """Set the reference atlas image and label."""
        self.image = image
        self.label = label.argmax(0).cpu().numpy().astype(np.uint8)

    def _process_single_atlas(self, sample, elx_params):
        """Process a single atlas sample - designed for parallel execution."""
        img, true_label = sample['img'], sample['gt_mask'].argmax(0).cpu().numpy().astype(np.uint8)
        img = self.transform(img)
        if self.num_classes == 1:
            true_label = true_label // true_label.max()

        pred_label = register_label(img, self.image, self.label, elx_params)
        return pred_label, true_label
    def predict(self, atlas, elx_params):
        """Predict the accuracy of input segmentation by transferring labels."""
        assert self.image is not None and self.label is not None, "Reference image and label must be set before prediction."
        # Determine optimal worker count
        max_workers = max(16, len(atlas))
        
        pred_labels = []
        true_labels = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all registration tasks
            future_to_idx = {
                executor.submit(self._process_single_atlas, sample, elx_params): idx 
                for idx, sample in enumerate(atlas)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    pred_label, true_label = future.result()
                    if pred_label is None or true_label is None:
                        continue
                    pred_labels.append(pred_label)
                    true_labels.append(true_label)
                except Exception as exc:
                    continue
        
        for pred_label, true_label in zip(pred_labels, true_labels):
            self.evaluator.process(pred_label, true_label)
        result_score = self.evaluator.finalize(get_best_scores=True)
    
        
        return result_score