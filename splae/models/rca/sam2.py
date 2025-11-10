import numpy as np
import torch

from splae.evaluation import MetricsEvaluator
from segmentanything2.sam2.build_sam import build_sam2_video_predictor
from monai.transforms import Compose
class SAM2:
    """Universal segmentation class with shared evaluation logic."""

    def __init__(self, config,
                    checkpoint,
                 metrics,
                 num_classes,
                 transform=Compose([]),
                 device='cpu'):
        self.evaluator = MetricsEvaluator(metrics, num_classes)
        self.device = torch.device(device)
        self.predictor = build_sam2_video_predictor(config, checkpoint, device=self.device)
        self.num_classes = num_classes
        self.transform = transform
        self.image = None
        self.label = None

    def set_sample(self, image, label):
        """Set the reference atlas image and label."""
        self.image = image
        self.label = label.squeeze(0).cpu().numpy().astype(np.uint8)

    @torch.no_grad()
    def predict(self, d_reference):
        """Predict the accuracy of input segmentation on reference dataset."""
        assert self.image is not None and self.label is not None, "Reference image and label must be set before prediction."

        for sample in d_reference:
            img, true_label = sample['img'], sample['gt_mask'].squeeze(0).cpu().numpy().astype(np.uint8)
            img = self.transform(img)
            all_images = np.array([self.image, img])  # create array with [sup_img, query_img]
            inference_state = self.predictor.init_state_by_np_data(all_images)
            for i in range(1, self.num_classes):
                # Add mask for each class to first frame (sup_img)
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_mask(inference_state=inference_state,
                                                                              frame_idx=0, obj_id=i,
                                                                              mask=self.label[i])

            out_frame_idx, out_obj_ids, out_mask_logits = next(self.predictor.propagate_in_video(inference_state,
                                                                                           start_frame_idx=1))  # Predict masks for second frame (query_img)

            class_preds = [(out_mask_logits[i] > 0.0).squeeze().cpu().numpy() for i in range(self.num_classes - 1)]
            background_pred = np.logical_not(np.logical_or.reduce(class_preds))  # Compute background predictions
            class_preds.insert(0, background_pred)
            class_preds = np.stack(class_preds, axis=0)
            pred_label = np.argmax(class_preds, axis=0)
            true_label = np.argmax(true_label, axis=0)

            self.predictor.reset_state(inference_state)

            # Evaluate metrics
            self.evaluator.process(pred_label, true_label)
        result_score = self.evaluator.finalize(get_best_scores=True)
        return result_score