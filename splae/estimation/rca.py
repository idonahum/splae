import itk
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
import torch
import numpy as np
import faiss
from torch.utils.data import Subset

from splae.estimation.base import BaseAccuracyEstimator
from splae.models.rca import UniverSeg, SAC, SAM2
from splae.utils import get_embedding
from splae.evaluation.evaluator import MetricsEvaluator
from monai.transforms import Compose, ScaleIntensity, ToTensor, ToNumpy, SqueezeDim

class BaseRCA(BaseAccuracyEstimator):
    def __init__(self,
                 target_model,
                 num_classes,
                 metrics,
                 embedding_model_name=None,
                 post_transform=None,
                 n_test=24,
                 device='cpu'):
        self.evaluator = MetricsEvaluator(metrics, num_classes)
        self.target_model = target_model.to(device).eval() if target_model is not None else None
        self.device = device
        self.k = n_test
        self.post_transform = post_transform
        self.processor, self.embedding_model = self.get_embedding_model_and_processer(embedding_model_name, device)
        self.n_classes = num_classes
        self.classifier_transform = Compose([])
        self._index_initialized = None
        self.index = None

    @staticmethod
    def get_embedding_model_and_processer(embedding_model_name, device='cpu'):
        if embedding_model_name is None:
            return None, None
        processor = AutoImageProcessor.from_pretrained(embedding_model_name)
        embedding_model = AutoModel.from_pretrained(embedding_model_name).to(device)
        return processor, embedding_model

    def _initialize_index(self, d_reference):
        """
        Initialize the FAISS index using the reference dataset.
        """
        self.index = faiss.IndexFlatIP(768)

        print('Populating faiss index...')
        for idx, data in enumerate(tqdm(d_reference)):
            img = self.classifier_transform(data['img'])
            embedding = get_embedding(img, self.embedding_model, self.processor,
                                      normalize=True, emb_method='output', device=self.device)
            self.index.add(embedding)

        self._index_initialized = True

    def select_k_closest(self, dataset, image, k):
        """
        Select the k closest images in the dataset to the input image in an embedding space.
        """
        if self.index is None:
            raise ValueError("FAISS index is not initialized. Call '_initialize_index' first.")
        img_embedding = get_embedding(image, self.embedding_model, self.processor, device=self.device)
        _, indices = self.index.search(img_embedding, k)
        subset = Subset(dataset, indices.reshape(-1))

        return subset

    @staticmethod
    def select_k_random(dataset, k):
        idxs = np.random.permutation(len(dataset))[:k]
        subset = Subset(dataset, idxs)
        return subset

    @torch.no_grad()
    def run_evaluation(self, d_reference, d_eval):
        """
        Run the evaluation using the selected model (UniverSeg or Atlas).
        """
        # Initialize the FAISS index if not already initialized
        if not self._index_initialized and self.embedding_model is not None:
            self._initialize_index(d_reference)

        print('\nRunning predictions...')
        return self._run_evaluation(d_reference, d_eval)

    @torch.no_grad()
    def _run_evaluation(self, d_reference, d_eval):
        results = []
        for sample_idx, sample in enumerate(tqdm(d_eval)):
            img = sample['img']
            if self.target_model is not None:
                seg = self.target_model(img.unsqueeze(0).to(self.device))
            else:
                seg = sample['seg_logits'].to(self.device)
            if self.post_transform is not None:
                seg = self.post_transform(seg)
            seg = seg.squeeze(0)
            gt_mask = sample['gt_mask'].to(self.device)
            if gt_mask.shape[0] == 1 and self.n_classes > 1:
                raise (ValueError("Ground truth mask should be one-hot encoded for multi-class segmentation"))

            real_score = self._calculate_real_score(seg, gt_mask)
            img = self.classifier_transform(img)
            if self.embedding_model is not None:
                selected_subset = self.select_k_closest(d_reference, img, self.k)
            else:
                selected_subset = self.select_k_random(d_reference, self.k)
            pred_score = self._calculate_pred_score(img, seg, selected_subset)
            results.append(self._zip_results(real_score, pred_score, sample['metainfo']['img_name']))
        return results

    def _calculate_real_score(self, seg, label):
        if self.n_classes > 1:
            seg, label = torch.argmax(seg, dim=0).unsqueeze(0), torch.argmax(label, dim=0).unsqueeze(0)
        self.evaluator.process(seg, label)
        return self.evaluator.finalize()

    def _calculate_pred_score(self, img, seg, selected_subset):
        raise NotImplementedError("Subclasses should implement this method.")


class AtlasRCA(BaseRCA):
    def __init__(self,
                 target_model,
                 num_classes,
                 metrics,
                 embedding_model_name=None,
                 post_transform=None,
                 n_test=24,
                 device='cpu'):
        super().__init__(target_model=target_model,
                            num_classes=num_classes,
                            metrics=metrics,
                            embedding_model_name=embedding_model_name,
                            post_transform=post_transform,
                            n_test=n_test,
                            device=device)

        self.elx_params = itk.ParameterObject.New()
        affine_params = self.elx_params.GetDefaultParameterMap("affine", 2)
        affine_params['MaximumNumberOfIterations'] = ['5']        # still low, but enough to converge
        affine_params['NumberOfResolutions'] = ['2']              # coarse + fine pyramid
        affine_params['Metric'] = ['AdvancedMeanSquares']         # faster than MI for 0–255 images
        self.elx_params.AddParameterMap(affine_params)

        # --- Lightweight B-spline stage (local refinement) ---
        bspline_params = self.elx_params.GetDefaultParameterMap("bspline", 2)
        # Remove the physical spacing parameter if it exists
        if 'FinalGridSpacingInPhysicalUnits' in bspline_params:
            del bspline_params['FinalGridSpacingInPhysicalUnits']
        bspline_params['MaximumNumberOfIterations'] = ['20']     # reasonable refinement budget
        bspline_params['NumberOfResolutions'] = ['2']             # multi-resolution
        bspline_params['Metric'] = ['AdvancedMeanSquares']        # faster metric
        bspline_params['Metric1Weight'] = ['1']                   # keep default balance
        self.elx_params.AddParameterMap(bspline_params)
        self.classifier_transform = Compose([ScaleIntensity(minv=0, maxv=255, dtype=np.uint8),SqueezeDim(), ToNumpy()])

        # Configure RCA classifier using SAC
        self.classifier = SAC(metrics, num_classes, self.classifier_transform)

    def _calculate_pred_score(self, img, seg, selected_subset):
        self.classifier.set_sample(img, seg)
        pred_score = self.classifier.predict(selected_subset, self.elx_params)
        return pred_score


class SAM2RCA(BaseRCA):
    def __init__(self,
                 target_model,
                 num_classes,
                 sam_config,
                 sam_checkpoint,
                 metrics,
                 embedding_model_name=None,
                 post_transform=None,
                 n_test=24,
                 device='cpu'):
        super().__init__(target_model=target_model,
                            num_classes=num_classes,
                            metrics=metrics,
                            embedding_model_name=embedding_model_name,
                            post_transform=post_transform,
                            n_test=n_test,
                            device=device)
        self.classifier_transform = Compose([ScaleIntensity(minv=0, maxv=255, dtype=np.uint8),SqueezeDim(), ToNumpy()])
        self.classifier = SAM2(config=sam_config, checkpoint=sam_checkpoint, metrics=metrics, num_classes=num_classes, transform=self.classifier_transform, device=device)
    
    def _calculate_pred_score(self, img, seg, selected_subset):
        self.classifier.set_sample(img, seg)
        pred_score = self.classifier.predict(selected_subset)
        return pred_score


def get_rca_model(rca_type, **kwargs):
    rca_type = rca_type.lower()
    if rca_type == 'atlas':
        return AtlasRCA(**kwargs)
    elif rca_type == 'sam2':
        return SAM2RCA(**kwargs)
    else:
        raise ValueError(f"Unknown RCA type: {rca_type}. Supported types are 'atlas', 'universseg', and 'sam2'.")
