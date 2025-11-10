import os

import torch
from torch.utils.data import Dataset
from monai.transforms import Compose, LoadImaged


class SegmentationDataset2D(Dataset):
    DEFAULT_IMG_SUFFICES = ['.png', '.jpg', '.jpeg']
    DEFAULT_MASK_SUFFICES = ['_gt.png', '_gt.jpg', '_gt.jpeg']
    DEFAULT_SEG_LOGITS_SUFFICES = ['_seg_logits.png', '_seg_logits.jpg', '_seg_logits.jpeg']

    def __init__(self,
                 data_root,
                 img_subfolder='img',
                 mask_subfolder='gt_masks',
                 seg_logits_subfolder=None,
                 split=None,
                 img_suffix=None,
                 mask_suffix=None,
                 seg_logits_suffix=None,
                 metainfo=None,
                 transforms=None):
        self.data_root = data_root
        self.img_subfolder = img_subfolder
        self.mask_subfolder = mask_subfolder
        self.split = split
        self.metainfo = metainfo

        self.img_suffices = [img_suffix] if img_suffix is not None else self.DEFAULT_IMG_SUFFICES
        self.mask_suffices = [mask_suffix] if mask_suffix is not None else self.DEFAULT_MASK_SUFFICES
        self.seg_logits_suffices = [seg_logits_suffix] if seg_logits_suffix is not None else self.DEFAULT_SEG_LOGITS_SUFFICES
        
        if split is not None:
            data_root = os.path.join(data_root, split)
        self.img_folder = os.path.join(data_root, img_subfolder)
        self.mask_folder = os.path.join(data_root, mask_subfolder)
        self.seg_logits_folder = None
        if seg_logits_subfolder is not None:
            self.seg_logits_folder = os.path.join(data_root, seg_logits_subfolder)
        self.samples_paths = self.load_paths()
        self.transforms = self._process_transforms(transforms)

    @staticmethod
    def _process_transforms(transforms):
        """
        Processes the input transforms, converting them into a MONAI Compose object.

        Args:
            transforms (Compose, list, or None): A MONAI Compose object, a list of dictionaries defining each transform,
                                                 or None.

        Returns:
            Compose: A MONAI Compose object containing the specified transforms.
        """
        # If transforms is None, create a default Compose with LoadImaged
        if transforms is None:
            return Compose([LoadImaged(keys=['img', 'gt_mask'], allow_missing_keys=True)])

        # If transforms is a list of dictionaries, build it into a pipeline
        if isinstance(transforms, list):
            from splae.datasets.transforms import build_transforms_pipeline
            transforms = build_transforms_pipeline(transforms)

        # Ensure transforms is a Compose object
        if not isinstance(transforms, Compose):
            raise ValueError("Transforms must be a MONAI Compose object, a list of dictionaries, or None.")

        # Check if the first transform is LoadImaged; prepend if missing
        if len(transforms.transforms) == 0 or not isinstance(transforms.transforms[0], LoadImaged):
            transforms = Compose(
                [LoadImaged(keys=['img', 'gt_mask'], allow_missing_keys=True)] + list(transforms.transforms))

        return transforms

    def __len__(self):
        return len(self.samples_paths)

    def load_paths(self):
        """
        Loads image and corresponding ground truth mask paths.

        Returns:
            List[Dict[str, Optional[str]]]: A list of dictionaries where each dictionary
            contains:
                - 'img': Path to the image.
                - 'gt_mask': Path to the corresponding mask or None if not found.
                - 'img_name': Base name of the image file without suffix.
        """
        image_mask_pairs = []

        # Traverse the image folder and find matching files
        for root, _, files in os.walk(self.img_folder):
            for file in files:
                # Get the suffix if it exists in img_suffices
                img_suffix = next((suffix for suffix in self.img_suffices if file.endswith(suffix)), None)
                if img_suffix:
                    img_path = os.path.join(root, file)
                    # Extract the base name without the suffix
                    img_name = file[: -len(img_suffix)]  # Remove the suffix from the file name

                    # Determine the corresponding mask path
                    mask_path = None
                    for mask_suffix in self.mask_suffices:
                        candidate_mask = img_name + mask_suffix
                        candidate_mask_path = os.path.join(self.mask_folder, candidate_mask)
                        if os.path.exists(candidate_mask_path):
                            mask_path = candidate_mask_path
                            break
                    seg_logits_path = None
                    if self.seg_logits_folder is not None:
                        for seg_logits_suffix in self.seg_logits_suffices:
                            candidate_seg_logits = img_name + seg_logits_suffix
                            candidate_seg_logits_path = os.path.join(self.seg_logits_folder, candidate_seg_logits)
                            if os.path.exists(candidate_seg_logits_path):
                                seg_logits_path = candidate_seg_logits_path
                                break
                    # Append the image, mask (or None), and image name as a dictionary
                    if mask_path:
                        image_mask_pairs.append({
                            'img': img_path,
                            'gt_mask': mask_path,
                            'seg_logits': seg_logits_path,
                            'img_name': img_name
                        })

        if not image_mask_pairs:
            raise FileNotFoundError(
                f"No valid images found in {self.img_folder} with suffixes {self.img_suffices}."
            )

        return image_mask_pairs

    def __getitem__(self, idx):
        """
        Fetches the image and mask at the specified index, applies transformations, and returns them.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            Dict[str, Any]: A dictionary with the following keys:
                - 'img': Transformed image.
                - 'gt_mask': Transformed ground truth mask (or None if no mask exists).
        """
        sample = self.samples_paths[idx]  # Get the image and mask paths
        img_path = sample.get('img')
        gt_mask_path = sample.get('gt_mask', None)
        seg_logits_path = sample.get('seg_logits', None)
        if sample.get('gt_mask') is None:
            sample.pop('gt_mask')
        sample = self.transforms(sample)  # Apply the transforms
        img, mask = sample['img'], sample.get('gt_mask', None)
        seg_logits = sample.get('seg_logits', None)
        sample_metainfo = {}
        sample_metainfo['img_name'] = sample['img_name']
        sample_metainfo['img_path'] = img_path
        sample_metainfo['gt_mask_path'] = gt_mask_path
        if seg_logits_path is not None:
            sample_metainfo['seg_logits_path'] = seg_logits_path
            return {'img': img, 'gt_mask': mask, 'metainfo': sample_metainfo, 'seg_logits': seg_logits}
        return {'img': img, 'gt_mask': mask, 'metainfo': sample_metainfo}


class MNMsDataset2D(SegmentationDataset2D):
    DEFAULT_IMG_SUFFICES = ['.nii.gz', '.nii']
    DEFAULT_MASK_SUFFICES = ['_gt.nii.gz', '_gt.nii']
    DEFAULT_SEG_LOGITS_SUFFICES = ['_seg_logits.nii.gz', '_seg_logits.nii']

    CLASS_LABELS = ['background', 'LV', 'MYO', 'RV']
    PALLETE = {
        'background': [0, 0, 0],
        'LV': [255, 0, 0],
        'MYO': [0, 255, 0],
        'RV': [0, 0, 255]
    }

    def __init__(self, data_root,
                 img_subfolder='images',
                 mask_subfolder='gt_masks',
                 seg_logits_subfolder=None,
                 split=None,
                 img_suffix=None,
                 mask_suffix=None,
                 seg_logits_suffix=None,
                 metainfo=None,
                 transforms=None):
        super().__init__(
            data_root=data_root,
            img_subfolder=img_subfolder,
            mask_subfolder=mask_subfolder,
            seg_logits_subfolder=seg_logits_subfolder,
            split=split,
            img_suffix=img_suffix,
            mask_suffix=mask_suffix,
            seg_logits_suffix=seg_logits_suffix,
            metainfo=metainfo,
            transforms=transforms
        )

class MSMDataset2D(SegmentationDataset2D):
    DEFAULT_IMG_SUFFICES = ['.nii.gz', '.nii']
    DEFAULT_MASK_SUFFICES = ['_gt.nii.gz', '_gt.nii']
    DEFAULT_SEG_LOGITS_SUFFICES = ['_seg_logits.nii.gz', '_seg_logits.nii']

    CLASS_LABELS = ['background', 'prostate']
    PALLETE = {
        'background': [0, 0, 0],
        'prostate': [255, 0, 0]
    }

    def __init__(self, data_root,
                 domains=None,
                 img_subfolder='images',
                 mask_subfolder='gt_masks',
                 seg_logits_subfolder=None,
                 split=None,
                 img_suffix=None,
                 mask_suffix=None,
                 seg_logits_suffix=None,
                 metainfo=None,
                 transforms=None):
        self.domains = domains
        super().__init__(
            data_root=data_root,
            img_subfolder=img_subfolder,
            mask_subfolder=mask_subfolder,
            seg_logits_subfolder=seg_logits_subfolder,
            split=split,
            img_suffix=img_suffix,
            mask_suffix=mask_suffix,
            seg_logits_suffix=seg_logits_suffix,
            metainfo=metainfo,
            transforms=transforms
        )
    
    def load_paths(self):
        """
        Loads image and corresponding ground truth mask paths from specified domains.
        If domains is None, assumes data_root is a domain folder and uses parent class method.

        Returns:
            List[Dict[str, Optional[str]]]: A list of dictionaries where each dictionary
            contains:
                - 'img': Path to the image.
                - 'gt_mask': Path to the corresponding mask or None if not found.
                - 'img_name': Base name of the image file without suffix.
                - 'domain': Domain name (e.g., 'A', 'B', 'C', etc.).
        """
        # If no domains specified, assume data_root is a domain folder
        if self.domains is None:
            # Use parent class method and add domain info
            image_mask_pairs = super().load_paths()
            return image_mask_pairs
        
        # Ensure domains is a list
        if isinstance(self.domains, str):
            self.domains = [self.domains]
        
        image_mask_pairs = []
        
        # Load paths from each specified domain
        for domain in self.domains:
            domain_data_root = os.path.join(self.data_root, domain)
            
            # Build domain-specific paths
            if self.split is not None:
                domain_data_root = os.path.join(domain_data_root, self.split)
            
            domain_img_folder = os.path.join(domain_data_root, self.img_subfolder)
            domain_mask_folder = os.path.join(domain_data_root, self.mask_subfolder)
            
            # Check if domain directories exist
            if not os.path.exists(domain_img_folder):
                print(f"Warning: Domain {domain} image folder not found: {domain_img_folder}")
                continue
            if not os.path.exists(domain_mask_folder):
                print(f"Warning: Domain {domain} mask folder not found: {domain_mask_folder}")
                continue
            
            # Traverse the domain image folder and find matching files
            for root, _, files in os.walk(domain_img_folder):
                for file in files:
                    # Get the suffix if it exists in img_suffices
                    img_suffix = next((suffix for suffix in self.img_suffices if file.endswith(suffix)), None)
                    if img_suffix:
                        img_path = os.path.join(root, file)
                        # Extract the base name without the suffix
                        img_name = file[: -len(img_suffix)]  # Remove the suffix from the file name

                        # Determine the corresponding mask path
                        mask_path = None
                        for mask_suffix in self.mask_suffices:
                            candidate_mask = img_name + mask_suffix
                            candidate_mask_path = os.path.join(domain_mask_folder, candidate_mask)
                            if os.path.exists(candidate_mask_path):
                                mask_path = candidate_mask_path
                                break
                        seg_logits_path = None
                        if self.seg_logits_folder is not None:
                            for seg_logits_suffix in self.seg_logits_suffices:
                                candidate_seg_logits = img_name + seg_logits_suffix
                                candidate_seg_logits_path = os.path.join(self.seg_logits_folder, candidate_seg_logits)
                                if os.path.exists(candidate_seg_logits_path):
                                    seg_logits_path = candidate_seg_logits_path
                                    break
                        # Append the image, mask (or None), image name, and domain as a dictionary
                        if mask_path:
                            image_mask_pairs.append({
                                'img': img_path,
                                'gt_mask': mask_path,
                                'seg_logits': seg_logits_path,
                                'img_name': img_name
                            })

        if not image_mask_pairs:
            available_domains_str = ', '.join(self.domains) if self.domains else 'all available'
            raise FileNotFoundError(
                f"No valid images found in domains {available_domains_str} with suffixes {self.img_suffices}."
            )

        return image_mask_pairs


class DPLDataset(Dataset):
    """
    Wraps original dataset with DPL pseudo labels.
    Each item: {
        'img': tensor,
        'gt_mask': tensor,
        'pseudo_label': tensor,
        'proto_pseudo': tensor,
        'uncertain_map': tensor (optional),
        'metainfo': dict
    }
    """
    def __init__(self, base_dataset, pseudo_label_dic, proto_pseudo_dic,
                 uncertain_dic=None, ignore_index=-1, transforms=None):
        self.base_dataset = base_dataset
        self.pseudo_label_dic = pseudo_label_dic
        self.proto_pseudo_dic = proto_pseudo_dic
        self.uncertain_dic = uncertain_dic if uncertain_dic is not None else {}
        self.ignore_index = ignore_index
        self.transforms = transforms

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # base dataset returns: { 'img', 'gt_mask', 'metainfo': {'img_name': ...}, ... }
        sample = self.base_dataset[idx]
        img_name = sample["metainfo"]["img_name"]

        # get pseudo and proto labels
        pseudo_label = torch.from_numpy(self.pseudo_label_dic[img_name]).long()
        proto_pseudo = torch.from_numpy(self.proto_pseudo_dic[img_name]).long()

        # optional uncertainty map
        if img_name in self.uncertain_dic:
            uncertain_map = torch.from_numpy(self.uncertain_dic[img_name]).float()
        else:
            uncertain_map = None

        # apply ignore_index safety (if needed)
        pseudo_label[pseudo_label == 255] = self.ignore_index  # generator used 255 as ignore index
        proto_pseudo[proto_pseudo == 255] = self.ignore_index

        # add to sample
        sample["pseudo_label"] = pseudo_label
        sample["proto_pseudo"] = proto_pseudo
        if uncertain_map is not None:
            sample["uncertain_map"] = uncertain_map

        # optional transforms (applied on dict)
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


def load_dataset2d(dataset_name, **kwargs):
    dataset_name = dataset_name.lower()
    if dataset_name == 'mnms':
        return MNMsDataset2D(**kwargs)
    elif dataset_name == 'msm':
        return MSMDataset2D(**kwargs)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets are: 'mnms', 'msm'.")

