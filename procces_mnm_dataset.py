import random

import pandas as pd
import os
import SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def load_nifti_file(filepath):
    """Load a NIfTI file using SimpleITK and return the image data as a NumPy array."""
    try:
        image = sitk.ReadImage(filepath)
        return sitk.GetArrayFromImage(image)
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None


def create_folders_structure(dataset_root, splits, vendors):
    """Create the folder structure for the dataset."""
    for split in splits:
        split_dir = os.path.join(dataset_root, split)
        os.makedirs(os.path.join(split_dir, 'img'), exist_ok=True)
        os.makedirs(os.path.join(split_dir, 'lab'), exist_ok=True)
        for vendor in vendors:
            os.makedirs(os.path.join(split_dir, 'img', vendor), exist_ok=True)
            os.makedirs(os.path.join(split_dir, 'lab', vendor), exist_ok=True)


def save_sample_as_2d_slices(source_file_path, target_path, slice_idxs):
    """Save each slice of a NIfTI file as a 2D image."""
    image = load_nifti_file(source_file_path)
    if image is None:
        return  # Skip this file if loading failed

    suffix = '_gt.nii.gz' if source_file_path.endswith('_gt.nii.gz') else '.nii.gz'
    basename = os.path.basename(source_file_path).replace('_sa' + suffix, '')

    for i, slice_idx in enumerate(slice_idxs):
        try:
            slice_image = image[slice_idx, :, :]
            slice_itk = sitk.GetImageFromArray(slice_image)
            slice_path = os.path.join(target_path, f'{basename}_{i}{suffix}')
            sitk.WriteImage(slice_itk, slice_path)
        except Exception as e:
            print(f"Error processing slice {i} of file {source_file_path}: {e}")


def process_sample(source_file_path, target_path, slice_idxs):
    """Wrapper function for processing a sample, used in parallel execution."""
    try:
        save_sample_as_2d_slices(source_file_path, target_path, slice_idxs)
    except Exception as e:
        print(f"Error processing file {source_file_path}: {e}")


def split_data(
        img_paths,
        train_ratio: float = 0.7,
        valid_ratio: float = 0.1,
        test_ratio: float = 0.2,
        seed: int = 3407
):
    """
    Split a list of image paths into train, validation, and test sets.

    Parameters:
    - img_paths: List of image file paths.
    - train_ratio: Proportion of the data to be used for training.
    - valid_ratio: Proportion of the data to be used for validation.
    - test_ratio: Proportion of the data to be used for testing.
    - seed: Random seed for reproducibility.

    Returns:
    - train_paths: List of training image paths.
    - valid_paths: List of validation image paths.
    - test_paths: List of testing image paths.
    """
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    # Shuffle the list for randomness
    random.seed(seed)
    random.shuffle(img_paths)

    # Calculate the split indices
    total = len(img_paths)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    # Split the data
    train_paths = img_paths[:train_end]
    valid_paths = img_paths[train_end:valid_end]
    test_paths = img_paths[valid_end:]

    return train_paths, valid_paths, test_paths


def get_target_path_name(file, sample_vendor, target_split, dataset_root):
    if file.endswith('_gt.nii.gz'):
        target_path = os.path.join(dataset_root, target_split, 'lab', sample_vendor)
    else:
        target_path = os.path.join(dataset_root, target_split, 'img', sample_vendor)
    return target_path


def map_samples_to_vendors(source_to_target_splits, dataset_root, dataset_meta, vendors):
    samples_dirs_per_vendor = {vendor: [] for vendor in vendors}
    for source_split in source_to_target_splits:
        source_split_dir = os.path.join(dataset_root, source_split)
        for sample_dir in os.listdir(source_split_dir):
            sample_dir_path = os.path.join(source_split_dir, sample_dir)

            # Skip if the directory does not exist or is not a directory
            if not os.path.isdir(sample_dir_path):
                print(f"Skipping {sample_dir_path}, not a directory.")
                continue

            try:
                sample_vendor = dataset_meta[dataset_meta['External code'] == sample_dir]['Vendor'].values[0]
                # get values of ED and ES columns to one list
                slice_idxs = dataset_meta[dataset_meta['External code'] == sample_dir][['ED', 'ES']].values[0]
                samples_dirs_per_vendor[sample_vendor].append((sample_dir_path, slice_idxs))
            except IndexError:
                print(f"Vendor not found for sample {sample_dir}. Skipping.")
                continue

    return samples_dirs_per_vendor


def main():
    dataset_root = '/Users/ido.nahum/Downloads/MNMs'
    dataset_meta = pd.read_csv(os.path.join(dataset_root, '211230_M&Ms_Dataset_information_diagnosis_opendataset.csv'))

    source_to_target_splits = {
        'Training/Labeled': 'train',
        'Training/Unlabeled': 'train',
        'Validation': 'valid',
        'Testing': 'test'
    }

    target_splits = source_to_target_splits.values()
    vendors = dataset_meta['Vendor'].unique()
    create_folders_structure(dataset_root, target_splits, vendors)

    tasks = []
    with ProcessPoolExecutor() as executor:
        samples_dirs_per_vendor = map_samples_to_vendors(source_to_target_splits, dataset_root, dataset_meta, vendors)
        # split each vendor into train valid and test 70% , 10%, 20%
        for vendor in samples_dirs_per_vendor:
            train_paths, valid_paths, test_paths = split_data(samples_dirs_per_vendor[vendor])
            for target_split, source_split_paths in zip(target_splits, [train_paths, valid_paths, test_paths]):
                for source_sample_path, slice_idxs in source_split_paths:
                    for file in os.listdir(source_sample_path):
                        source_file_path = os.path.join(source_sample_path, file)
                        target_path = get_target_path_name(file, vendor, target_split, dataset_root)
                        tasks.append(executor.submit(process_sample, source_file_path, target_path, slice_idxs))

        # Use tqdm to display a progress bar
        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Processing Samples"):
            future.result()


if __name__ == '__main__':
    main()
