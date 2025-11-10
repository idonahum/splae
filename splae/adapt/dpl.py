import torch
import torch.nn.functional as F
from splae.adapt.base import PLGeneratorBase
from splae.datasets import DPLDataset
from tqdm import tqdm


class DPLGenerator(PLGeneratorBase):
    def __init__(self, device, num_classes, ignore_index=-1, conf_threshold=0.7, mc_passes=10):
        """
        Args:
            device: torch device
            num_classes: number of segmentation classes
            ignore_index: value for ignored pixels (default: -1)
            conf_threshold: min confidence to accept pseudo labels
            mc_passes: number of stochastic forward passes for uncertainty (1 = deterministic)
        """
        super(DPLGenerator, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.conf_threshold = conf_threshold
        self.mc_passes = mc_passes

    def generate(self, source_model, train_dataloader):
        return self._generate_dpl_dataset(source_model, train_dataloader)

    def _generate_dpl_dataset(self, source_model, train_dataloader):
        source_model.to(self.device)
        source_model.eval()

        pseudo_label_dic = {}
        proto_pseudo_dic = {}
        uncertain_dic = {}

        with torch.no_grad():
            for batch in tqdm(train_dataloader, desc="Generating DPL dataset"):
                imgs = batch["img"].to(self.device)  # [B,1,H,W]
                img_names = batch["metainfo"]["img_name"]

                # ---- Monte Carlo forward passes (for uncertainty) ----
                logits_list, features_list = [], []
                for _ in range(self.mc_passes):
                    logits, features = source_model(imgs, return_features=True)
                    logits_list.append(logits.unsqueeze(0))
                    features_list.append(features.unsqueeze(0))

                logits_stack = torch.cat(logits_list, dim=0)  # [T,B,C,H,W]
                features = features_list[0].squeeze(0)        # use last features for proto
                probs = F.softmax(logits_stack, dim=2)        # [T,B,C,H,W]

                mean_probs = probs.mean(dim=0)                # [B,C,H,W]
                conf, pseudo_label = mean_probs.max(dim=1)    # [B,H,W]

                # pixel-wise uncertainty (mean std across classes)
                if self.mc_passes > 1:
                    std_map = probs.std(dim=0).mean(dim=1)    # [B,H,W]
                else:
                    std_map = torch.zeros_like(conf)

                # ---- Confidence filtering ----
                pseudo_label[conf < self.conf_threshold] = 255 # 255 is the ignore index now

                B, C, H, W = mean_probs.shape
                _, Fdim, h, w = features.shape

                # ---- Compute centroids for prototype pseudo ----
                up_pseudo = F.interpolate(
                    F.one_hot(pseudo_label.clamp(0, self.num_classes - 1), self.num_classes)
                    .permute(0, 3, 1, 2).float(),
                    size=(h, w), mode="nearest",
                )

                centroids = []
                for k in range(self.num_classes):
                    mask_k = up_pseudo[:, k:k+1, ...]
                    if mask_k.sum() == 0:
                        centroids.append(torch.zeros(Fdim, device=self.device))
                    else:
                        centroids.append((features * mask_k).sum(dim=[0, 2, 3]) / mask_k.sum())

                dist_maps = torch.cat(
                    [((features - c.view(1, -1, 1, 1)) ** 2).sum(dim=1, keepdim=True) for c in centroids],
                    dim=1,
                )

                proto_pseudo = dist_maps.argmin(dim=1)  # [B,h,w]
                proto_pseudo = F.interpolate(
                    proto_pseudo.unsqueeze(1).float(), size=(H, W), mode="nearest"
                ).squeeze(1).long()

                # ---- Save per-sample results ----
                for i, img_name in enumerate(img_names):
                    pseudo_label_dic[img_name] = pseudo_label[i].cpu().numpy()
                    proto_pseudo_dic[img_name] = proto_pseudo[i].cpu().numpy()
                    uncertain_dic[img_name] = std_map[i].cpu().numpy()

        return DPLDataset(
            train_dataloader.dataset,
            pseudo_label_dic,
            proto_pseudo_dic,
            uncertain_dic=uncertain_dic,  # add uncertainty if your dataset supports it
        )
