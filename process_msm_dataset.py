#!/usr/bin/env python3
"""
MSM Dataset Processing Script

This script processes the MSM dataset by:
1. Converting 3D volumes to 2D slices
2. Applying domain-specific label mapping (domains A,B: index 2 -> 1, index 1 -> 0)
3. Skipping slices with empty ground truth
4. Tracking unique labels per domain
5. Splitting data into train/validation/test sets
6. Organizing output in the expected directory structure

Usage:
    python process_msm_dataset.py --input_dir /path/to/msm/dataset --output_dir /path/to/output
"""

import os
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import shutil
import logging
from datetime import datetime

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    print("scikit-learn not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.model_selection import train_test_split

# Set up logging
logger = logging.getLogger(__name__)


class MSMDatasetProcessor:
    """Process MSM dataset from 3D to 2D format with train/val/test splits."""
    
    def __init__(self, input_dir, output_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
        """
        Initialize the MSM dataset processor.
        
        Args:
            input_dir (str): Path to the MSM dataset directory
            output_dir (str): Path to the output directory
            train_ratio (float): Ratio of data for training
            val_ratio (float): Ratio of data for validation
            test_ratio (float): Ratio of data for testing
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        # Create output directory structure
        self._create_output_structure()
        
        # Set up logging to file
        self._setup_logging()
        
        # Statistics tracking
        self.stats = {
            'total_slices': 0,
            'skipped_empty_gt': 0,
            'processed_slices': 0,
            'domains_processed': 0,
            'cases_processed': 0,
            'unique_labels_per_domain': {}
        }
    
    def _create_output_structure(self):
        """Create the expected directory structure for 2D dataset."""
        domains = ['A', 'B', 'C', 'D', 'E', 'F']
        splits = ['train', 'valid', 'test']
        subdirs = ['images', 'gt_masks']
        
        for domain in domains:
            for split in splits:
                for subdir in subdirs:
                    (self.output_dir / domain / split / subdir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created output directory structure in {self.output_dir}")
    
    def _setup_logging(self):
        """Set up logging to both console and file."""
        # Create log file path in the output directory
        log_file = self.output_dir / f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # Console output
            ]
        )
        
        logger.info(f"Logging to file: {log_file}")
        self.log_file = log_file
    
    def _get_domain_directories(self):
        """Get all domain directories (A, B, C, D, E, F)."""
        domains = []
        for item in self.input_dir.iterdir():
            if item.is_dir() and item.name in ['A', 'B', 'C', 'D', 'E', 'F']:
                domains.append(item)
        return sorted(domains)
    
    def _get_case_pairs(self, domain_dir):
        """
        Get image and segmentation pairs for a domain.
        
        Args:
            domain_dir (Path): Path to domain directory
            
        Returns:
            List[tuple]: List of (image_path, segmentation_path) tuples
        """
        case_pairs = []
        
        # Get all .nii.gz files
        nii_files = list(domain_dir.glob('*.nii.gz'))
        
        # Group by case number
        cases = {}
        for nii_file in nii_files:
            # Extract case number and type
            name = nii_file.stem.replace('.nii', '')  # Remove .nii from .nii.gz
            
            if '_segmentation' in name.lower() or '_segmentation' in name:
                case_num = name.replace('_segmentation', '').replace('_Segmentation', '')
                case_type = 'segmentation'
            else:
                case_num = name
                case_type = 'image'
            
            if case_num not in cases:
                cases[case_num] = {}
            cases[case_num][case_type] = nii_file
        
        # Create pairs
        for case_num, files in cases.items():
            if 'image' in files and 'segmentation' in files:
                case_pairs.append((files['image'], files['segmentation']))
            else:
                logger.warning(f"Missing image or segmentation for case {case_num} in domain {domain_dir.name}")
        
        return case_pairs
    
    def _is_empty_gt(self, gt_array):
        """
        Check if ground truth is empty (all zeros - no segmentation).
        
        Args:
            gt_array (np.ndarray): Ground truth array (original labels)
            
        Returns:
            bool: True if ground truth is empty (no segmentation)
        """
        return np.all(gt_array == 0)
    
    def _process_3d_volume(self, img_path, gt_path, domain_name, case_name):
        """
        Process a 3D volume by extracting 2D slices.
        
        Args:
            img_path (Path): Path to 3D image
            gt_path (Path): Path to 3D ground truth
            domain_name (str): Name of the domain
            case_name (str): Name of the case
            
        Returns:
            List[dict]: List of slice information dictionaries
        """
        slices_info = []
        
        try:
            # Load 3D volumes
            img_nii = nib.load(img_path)
            gt_nii = nib.load(gt_path)
            
            img_array = img_nii.get_fdata()
            gt_array = gt_nii.get_fdata()
            
            # Ensure arrays have the same shape
            if img_array.shape != gt_array.shape:
                logger.warning(f"Shape mismatch for {case_name}: img {img_array.shape} vs gt {gt_array.shape}")
                return slices_info
            
            # Process each slice along the last dimension (assuming it's the slice dimension)
            num_slices = img_array.shape[-1]
            
            for slice_idx in range(num_slices):
                # Extract 2D slice
                img_slice = img_array[:, :, slice_idx]
                gt_slice = gt_array[:, :, slice_idx]
                
                # Apply domain-specific label mapping
                if domain_name in ['A', 'B']:
                    gt_slice = np.where(gt_slice == 1, 0, gt_slice)  
                    gt_slice = np.where(gt_slice == 2, 1, gt_slice)  
                
                # Skip if ground truth is empty (all zeros)
                if self._is_empty_gt(gt_slice):
                    self.stats['skipped_empty_gt'] += 1
                    continue
                
                # Track unique labels in this domain (after mapping)
                unique_labels = np.unique(gt_slice)
                if domain_name not in self.stats['unique_labels_per_domain']:
                    self.stats['unique_labels_per_domain'][domain_name] = set()
                self.stats['unique_labels_per_domain'][domain_name].update(unique_labels)
                
                # Create slice info
                slice_name = f"{case_name}_{slice_idx:03d}"
                slice_info = {
                    'img_slice': img_slice,
                    'gt_slice': gt_slice,
                    'slice_name': slice_name,
                    'domain': domain_name,
                    'case': case_name,
                    'slice_idx': slice_idx
                }
                
                slices_info.append(slice_info)
                self.stats['processed_slices'] += 1
        
        except Exception as e:
            logger.error(f"Error processing {case_name}: {str(e)}")
        
        return slices_info
    
    def _save_slice(self, slice_info, split):
        """
        Save a 2D slice to the appropriate directory.
        
        Args:
            slice_info (dict): Slice information dictionary
            split (str): Split name (train/valid/test)
        """
        slice_name = slice_info['slice_name']
        domain = slice_info['domain']
        
        # Create filenames
        img_filename = f"{domain}_{slice_name}.nii.gz"
        gt_filename = f"{domain}_{slice_name}_gt.nii.gz"
        
        # Save paths
        img_save_path = self.output_dir / domain / split / 'images' / img_filename
        gt_save_path = self.output_dir / domain / split / 'gt_masks' / gt_filename
        
        try:
            # Ensure proper data types for NIfTI compatibility
            img_slice = slice_info['img_slice'].astype(np.float32)
            gt_slice = slice_info['gt_slice'].astype(np.int16)  # Use int16 to preserve original labels
            
            # Create 2D NIfTI images
            img_2d = nib.Nifti1Image(img_slice, affine=np.eye(4))
            gt_2d = nib.Nifti1Image(gt_slice, affine=np.eye(4))
            
            # Save images
            nib.save(img_2d, img_save_path)
            nib.save(gt_2d, gt_save_path)
            
        except Exception as e:
            logger.error(f"Error saving slice {slice_name}: {str(e)}")
    
    def _split_data(self, all_slices):
        """
        Split data into train/validation/test sets.
        
        Args:
            all_slices (List[dict]): List of all slice information
            
        Returns:
            tuple: (train_slices, val_slices, test_slices)
        """
        # First split: separate test set
        train_val_slices, test_slices = train_test_split(
            all_slices, 
            test_size=self.test_ratio, 
            random_state=42,
            stratify=[s['domain'] for s in all_slices]  # Stratify by domain
        )
        
        # Second split: separate train and validation
        val_size = self.val_ratio / (self.train_ratio + self.val_ratio)
        train_slices, val_slices = train_test_split(
            train_val_slices,
            test_size=val_size,
            random_state=42,
            stratify=[s['domain'] for s in train_val_slices]
        )
        
        return train_slices, val_slices, test_slices
    
    def process_domain(self, domain_dir):
        """
        Process all cases in a domain.
        
        Args:
            domain_dir (Path): Path to domain directory
            
        Returns:
            List[dict]: List of all slice information from this domain
        """
        domain_name = domain_dir.name
        logger.info(f"Processing domain {domain_name}")
        
        # Get case pairs
        case_pairs = self._get_case_pairs(domain_dir)
        logger.info(f"Found {len(case_pairs)} cases in domain {domain_name}")
        
        all_slices = []
        
        for img_path, gt_path in tqdm(case_pairs, desc=f"Processing {domain_name}"):
            case_name = img_path.stem.replace('.nii', '')
            
            # Process 3D volume
            slices_info = self._process_3d_volume(img_path, gt_path, domain_name, case_name)
            all_slices.extend(slices_info)
            
            self.stats['cases_processed'] += 1
        
        self.stats['domains_processed'] += 1
        logger.info(f"Domain {domain_name}: {len(all_slices)} slices processed")
        
        return all_slices
    
    def process_dataset(self):
        """Process the entire MSM dataset."""
        logger.info("Starting MSM dataset processing")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Get all domain directories
        domain_dirs = self._get_domain_directories()
        logger.info(f"Found domains: {[d.name for d in domain_dirs]}")
        
        # Process all domains
        all_slices = []
        for domain_dir in domain_dirs:
            domain_slices = self.process_domain(domain_dir)
            all_slices.extend(domain_slices)
        
        self.stats['total_slices'] = len(all_slices)
        logger.info(f"Total slices collected: {len(all_slices)}")
        
        # Split data
        logger.info("Splitting data into train/validation/test sets")
        train_slices, val_slices, test_slices = self._split_data(all_slices)
        
        logger.info(f"Split sizes - Train: {len(train_slices)}, Val: {len(val_slices)}, Test: {len(test_slices)}")
        
        # Save slices
        logger.info("Saving slices to disk")
        
        for split, slices in [('train', train_slices), ('valid', val_slices), ('test', test_slices)]:
            logger.info(f"Saving {len(slices)} slices to {split} split")
            for slice_info in tqdm(slices, desc=f"Saving {split}"):
                self._save_slice(slice_info, split)
        
        # Print final statistics
        self._print_statistics()
    
    def _print_statistics(self):
        """Print processing statistics."""
        logger.info("=" * 50)
        logger.info("PROCESSING STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Domains processed: {self.stats['domains_processed']}")
        logger.info(f"Cases processed: {self.stats['cases_processed']}")
        logger.info(f"Total slices found: {self.stats['total_slices']}")
        logger.info(f"Slices with empty GT (skipped): {self.stats['skipped_empty_gt']}")
        logger.info(f"Slices processed: {self.stats['processed_slices']}")
        logger.info(f"Skip rate: {self.stats['skipped_empty_gt'] / (self.stats['total_slices'] + self.stats['skipped_empty_gt']) * 100:.1f}%")
        
        # Print unique labels per domain
        logger.info("\nUnique labels per domain:")
        for domain, labels in self.stats['unique_labels_per_domain'].items():
            sorted_labels = sorted(labels)
            logger.info(f"  Domain {domain}: {sorted_labels} (total: {len(sorted_labels)} unique labels)")
        
        logger.info("=" * 50)
        
        # Save detailed statistics to a separate file
        self._save_detailed_statistics()
    
    def _save_detailed_statistics(self):
        """Save detailed statistics to a file in the output directory."""
        stats_file = self.output_dir / f"processing_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(stats_file, 'w') as f:
            f.write("MSM Dataset Processing Statistics\n")
            f.write("=" * 50 + "\n")
            f.write(f"Processing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input directory: {self.input_dir}\n")
            f.write(f"Output directory: {self.output_dir}\n\n")
            
            f.write("PROCESSING STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Domains processed: {self.stats['domains_processed']}\n")
            f.write(f"Cases processed: {self.stats['cases_processed']}\n")
            f.write(f"Total slices found: {self.stats['total_slices']}\n")
            f.write(f"Slices with empty GT (skipped): {self.stats['skipped_empty_gt']}\n")
            f.write(f"Slices processed: {self.stats['processed_slices']}\n")
            
            if self.stats['total_slices'] + self.stats['skipped_empty_gt'] > 0:
                skip_rate = self.stats['skipped_empty_gt'] / (self.stats['total_slices'] + self.stats['skipped_empty_gt']) * 100
                f.write(f"Skip rate: {skip_rate:.1f}%\n")
            
            f.write("\nUNIQUE LABELS PER DOMAIN\n")
            f.write("-" * 30 + "\n")
            for domain, labels in self.stats['unique_labels_per_domain'].items():
                sorted_labels = sorted(labels)
                f.write(f"Domain {domain}: {sorted_labels} (total: {len(sorted_labels)} unique labels)\n")
            
            f.write("\n" + "=" * 50 + "\n")
        
        logger.info(f"Detailed statistics saved to: {stats_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Process MSM dataset from 3D to 2D format')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Path to MSM dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to output directory')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Ratio of data for training (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Ratio of data for validation (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                       help='Ratio of data for testing (default: 0.2)')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return
    
    # Create processor and run
    processor = MSMDatasetProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    processor.process_dataset()


if __name__ == '__main__':
    main()
