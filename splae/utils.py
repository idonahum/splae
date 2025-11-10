import json
import torch
import numpy as np
from scipy import ndimage
import faiss
from torchvision import transforms

def norm_0_255(images: torch.Tensor) -> np.array:
    """
    Normalize tensor images to 0-255 range and return as numpy array.
    Args:
        images: Tensor of shape (B, C, H, W) with pixel values in the range [-1, 1].

    Returns:
        Numpy array of shape (B, C, H, W) with pixel values in the range [0, 255].

    """
    x = ((images + 1) / 2) * 255
    x = torch.clamp(x, 0, 255).to(torch.uint8).cpu().detach().numpy()
    return x


def get_largest_component(mask):
    """
    Given a mask size (H, W), return the largest connected component.
    Args:
        mask: numpy array of shape (H, W).

    Returns:
        largest_component: numpy array of shape (H, W) with the largest connected component.

    """
    dim = mask.ndim
    assert dim == 2, f"Expected 2D mask, got {dim}D"
    if (mask.sum() == 0):
        return mask
    binary_structure = ndimage.generate_binary_structure(dim, 1)
    labeled_array, numpatches = ndimage.label(mask, structure=binary_structure)
    sizes = ndimage.sum(mask, labeled_array, range(1, numpatches + 1))
    max_label = np.where(sizes == sizes.max())[0] + 1
    largest_component = np.asarray(labeled_array == max_label[0], np.uint8)
    return largest_component


def gray2rgb(image):
    """
    Convert a grayscale image to RGB format.
    Args:
        image: numpy array of shape (H, W) or (H, W, 1) or (1, H, W).

    Returns:
        rgb_image: numpy array of shape (H, W, 3).

    """
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 1:
        return np.concatenate([image] * 3, axis=-1)
    elif image.shape[0] == 1:
        image = np.transpose(image, (1, 2, 0))
        return np.concatenate([image] * 3, axis=-1)
    else:
        raise ValueError("Input image must be grayscale or have a single channel.")


def random_sample(mask, num_samples=5):
    """
    Given a numpy mask size (H, W), randomly sample num_samples from the mask.
    Args:
        mask:
        num_samples:

    Returns:
        sampled_points: numpy array of shape (num_samples, 2) with sampled points.
    """
    true_indices = np.argwhere(mask)

    if len(true_indices) < num_samples:
        return None

    random_indices = np.random.choice(len(true_indices), num_samples, replace=False)
    sampled_points = true_indices[random_indices]
    sampled_points = np.flip(sampled_points, axis=1)
    return sampled_points


def get_embedding(image, model, processor, normalize=True, emb_method='patch_mean', device='cpu'):
    image = transforms.ToPILImage()(image)  # Convert tensor to PIL image

    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)

    # Extract embedding and convert to numpy
    if emb_method == 'patch_mean':
        embedding = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
    elif emb_method == 'output':
        embedding = outputs.pooler_output.detach().cpu().numpy()
    elif emb_method == 'flat_patch':
        embedding = outputs.last_hidden_state[:, 1:].view(1, -1).detach().cpu().numpy()  # first token is CLS
    else:
        raise ValueError(
            f"Invalid embedding method: {emb_method}. Choose from 'patch_mean', 'output', or 'flat_patch'.")

    if normalize: faiss.normalize_L2(embedding)

    return embedding


def get_embedding_size(model, processor):
    image_size = processor.crop_size["height"]
    patch_size = model.config.patch_size
    embeddings_size = image_size // patch_size

    return embeddings_size ** 2 * 768

def to_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def open_json(json_path):
    """
    Open and parse a JSON file
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {json_path}: {e}")
        return None

def one_hot(target, num_classes, ignore_index=None):
    if target.ndim == 4:
        target = target.squeeze(1)
    min = -1 if ignore_index is not None and ignore_index < 0 else 0
    max = num_classes - 1 if ignore_index is not None and ignore_index < 0 else num_classes
    one_hot_target = torch.clamp(target, min, max)
    one_hot_target = one_hot_target + 1 if min == -1 else one_hot_target
    one_hot_target = torch.nn.functional.one_hot(one_hot_target.long(),
                                                 num_classes + 1)

    min_idx = 1 if ignore_index is not None and ignore_index < 0 else 0
    max_idx = num_classes + 1 if ignore_index is not None and ignore_index < 0 else num_classes
    one_hot_target = one_hot_target[..., min_idx:max_idx].permute(0, 3, 1, 2)
    return one_hot_target

def predictive_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    probs: [B, C, H, W] tensor with softmax probabilities
    returns: [B] average entropy per image
    """
    eps = 1e-8
    ent = -(probs * (probs.clamp(min=eps).log())).sum(dim=1)  # [B, H, W]
    return ent.mean(dim=(1, 2))  # [B]

def prediction_margin(probs: torch.Tensor) -> torch.Tensor:
    """
    probs: [B, C, H, W]
    returns: [B] average margin per image
    """
    top2 = probs.topk(2, dim=1).values  # [B, 2, H, W]
    margin = (top2[:,0] - top2[:,1])    # [B, H, W]
    return margin.mean(dim=(1, 2))      # [B]

def reliability_score(probs: torch.Tensor) -> torch.Tensor:
    ent = predictive_entropy(probs)      # low is good
    marg = prediction_margin(probs)      # high is good
    
    # normalize (z-score or min-max across dataset)
    ent_n = (ent - ent.min()) / (ent.max() - ent.min() + 1e-8)
    marg_n = (marg - marg.min()) / (marg.max() - marg.min() + 1e-8)
    
    # combine: low entropy + high margin
    score = (1 - ent_n) + marg_n
    return score / 2.0  # average in [0,1]