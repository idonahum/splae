from typing import Optional, Tuple, Dict
import math
import numpy as np
import torch
import torch.nn.functional as F
from splae.models.epl import EPLBase
from tqdm import tqdm
def l2_normalize(x: torch.Tensor, dim: int) -> torch.Tensor:
    return F.normalize(x, p=2, dim=dim, eps=1e-6)

def upsample_to(img_hw: Tuple[int,int], feat: torch.Tensor) -> torch.Tensor:
    # feat: [B, D, Hf, Wf] -> [B, D, H, W]
    return F.interpolate(feat, size=img_hw, mode="bilinear", align_corners=False)


def _infer_grid(N: int, H: int, W: int, patch_size: int | None):
    # If patch size is known & fits, prefer it
    if patch_size is not None and H % patch_size == 0 and W % patch_size == 0:
        Hf, Wf = H // patch_size, W // patch_size
        if Hf * Wf == N:
            return Hf, Wf
    # Otherwise factor N close to the image AR
    ar = W / max(H, 1)
    best = None
    for d in range(1, int(math.sqrt(N)) + 1):
        if N % d:
            continue
        for Hf, Wf in ((d, N // d), (N // d, d)):
            score = abs((Wf / max(Hf, 1)) - ar)
            if best is None or score < best[0]:
                best = (score, Hf, Wf)
    return best[1], best[2]

class EPLMedDinoV3(EPLBase):
    def __init__(self, dinov3_model, source_model, device, resize_size=224, n_classes=4):
        super().__init__()
        self._sum_wf = torch.zeros(n_classes, dinov3_model.embed_dim, device=device)
        self._sum_w = torch.zeros(n_classes, device=device)
        self.source_model = source_model
        self.dinov3_model = dinov3_model
        self.device = device
        self.resize_size = resize_size
        self.n_classes = n_classes

    @property
    def prototypes(self):
        # Normalize to unit vectors
        normed = F.normalize(self._sum_wf / (self._sum_w[:,None] + 1e-8), dim=1)
        return normed  # [K, D]

    @torch.no_grad()
    def populate_prototypes(self, dataloader, top_p: float = 0.10, fg_gamma: float = 0.30, T: float = 0.5,
                            min_pixels: int = 64):
        self._sum_wf.zero_();
        self._sum_w.zero_()
        self.dinov3_model.eval();
        self.source_model.eval()

        for data in tqdm(dataloader):
            images = data["img"].to(self.device)  # [B,1,H,W]

            logits = self.source_model(images)  # [B,K,H,W]
            probs = torch.softmax(logits / T, dim=1)  # sharpen a bit

            feats = self._get_dense_features(images)  # [B,D,Hout,Wout]
            B, D, Hout, Wout = feats.shape

            probs = F.interpolate(probs, size=(Hout, Wout), mode="bilinear", align_corners=False)  # [B,K,Hout,Wout]
            feats = feats.permute(0, 2, 3, 1).reshape(-1, D)  # [N,D]
            feats = F.normalize(feats, dim=1)

            # foreground confidence (exclude background=0)
            fg_conf = probs[:, 1:].amax(dim=1, keepdim=True)  # [B,1,Hout,Wout]
            # background seed: low confident FG pixels
            bg_seed = (fg_conf < fg_gamma).float()  # [B,1,Hout,Wout]

            # hard PL (only for erosion), resized to grid
            hard_pl = probs.argmax(dim=1, keepdim=True).float()  # [B,1,Hout,Wout]

            # light erosion to drop boundary pixels (3x3)
            erode = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            # keep only interior of each class k: one-vs-rest mask then erode
            interior_masks = []
            for k in range(self.n_classes):
                mk = (hard_pl == k).float()
                mk_eroded = (erode(1 - erode(1 - mk)) > 0.5).float()
                interior_masks.append(mk_eroded)
            interior_masks = torch.cat(interior_masks, dim=1)  # [B,K,Hout,Wout]

            # flatten everything
            probs_f = probs.permute(0, 2, 3, 1).reshape(-1, self.n_classes)  # [N,K]
            interior_f = interior_masks.permute(0, 2, 3, 1).reshape(-1, self.n_classes)  # [N,K]
            bg_seed_f = bg_seed.view(-1)  # [N]

            # ---- background (k=0) from low-FG regions
            k = 0
            bg_conf = (1 - fg_conf).view(-1)  # [N]
            idx_bg = torch.nonzero(bg_seed_f, as_tuple=False).squeeze(1)
            if idx_bg.numel() >= min_pixels:
                topk_bg = max(1, int(idx_bg.numel() * top_p))
                vals_bg, ord_bg = torch.topk(bg_conf[idx_bg], topk_bg)
                f_bg = feats[idx_bg[ord_bg]]
                self._sum_wf[k] += (vals_bg.unsqueeze(1) * f_bg).sum(0)
                self._sum_w[k] += vals_bg.sum()

            # ---- foreground classes
            for k in range(1, self.n_classes):
                mask_k = (interior_f[:, k] > 0.5)  # only interior pixels
                if mask_k.sum() < min_pixels:
                    continue
                conf_k = probs_f[mask_k, k]
                Nk = conf_k.shape[0]
                topk = max(1, int(Nk * top_p))
                vals, idx = torch.topk(conf_k, topk)
                f = feats[mask_k][idx]
                self._sum_wf[k] += (vals.unsqueeze(1) * f).sum(0)
                self._sum_w[k] += vals.sum()

        print("Prototypes built:", self.prototypes.shape)

    @torch.no_grad()
    def _get_dense_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, 1, H, W] single-channel CT slices
        returns: [B, D, H_out, W_out] feature map aligned with resized input
        """
        # convert to 3-channel
        images = images.repeat(1, 3, 1, 1)

        # resize to square
        images_resized = F.interpolate(
            images, size=(self.resize_size, self.resize_size),
            mode="bilinear", align_corners=False
        )

        # forward through MedDINOv3
        feats = self.dinov3_model(images_resized.to(self.device), is_training=True)

        # we care about patch tokens
        patch_tokens = feats["x_norm_patchtokens"]  # [B, N, D]

        B, N, D = patch_tokens.shape
        H_out = W_out = int(N ** 0.5)

        # reshape to [B, D, H_out, W_out]
        patch_map = patch_tokens.view(B, H_out, W_out, D).permute(0, 3, 1, 2)
        return patch_map  # [B, D, H_out, W_out]

    @torch.no_grad()
    def enhance(self, logits, original_pseudo_labels, images,
                alpha: float = 0.20, beta: float = 0.10,  # beta = temperature for cosine
                unknown_thresh: float = 0.55, margin_thresh: float = 0.05, **kwargs):

        self.dinov3_model.eval();
        self.source_model.eval()

        B, _, H, W = logits.shape
        probs = torch.softmax(logits, dim=1)  # [B,K,H,W]

        feats = self._get_dense_features(images)  # [B,D,Hout,Wout]
        B, D, Hout, Wout = feats.shape
        feats = F.normalize(feats.permute(0, 2, 3, 1).reshape(-1, D), dim=1)  # [N,D]

        # cosine → proto_probs via softmax(cos/β)
        cos = feats @ self.prototypes.T  # [N,K]
        proto_probs = torch.softmax(cos / beta, dim=1)  # [N,K]

        probs_resized = F.interpolate(probs, size=(Hout, Wout), mode="bilinear", align_corners=False)
        probs_resized = probs_resized.permute(0, 2, 3, 1).reshape(-1, self.n_classes)  # [N,K]

        score = (1 - alpha) * proto_probs + alpha * probs_resized

        top2 = score.topk(2, dim=1)
        pred = top2.indices[:, 0]
        conf = top2.values[:, 0]
        margin = top2.values[:, 0] - top2.values[:, 1]
        pred[(conf < unknown_thresh) | (margin < margin_thresh)] = -1

        pred = pred.view(B, Hout, Wout)
        pred = F.interpolate(pred.unsqueeze(1).float(), size=(H, W), mode="nearest").squeeze(1).long()
        return pred
