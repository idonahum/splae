import time

from torch.utils.data import DataLoader

import codename
import os
import sys
from collections import defaultdict
from loguru import logger
from splae.visualization import SegmentationVisualizer

import torch
import random
import numpy as np


class Runner:
    def __init__(self,
                 model,
                 save_dir,
                 optimizer=None,
                 losses=None,
                 lr_scheduler=None,
                 metrics=None,
                 device=None,
                 train_loader=None,
                 val_loader=None,
                 test_loader=None,
                 epochs=None,
                 val_interval=None,
                 load_from=None,
                 post_transforms=None,
                 ignore_index=-1,
                 experiment_name=None,
                 visualization_cfg=None,  # Visualization config with intervals and sample size
                 checkpoint_cfg=None,  # Checkpoint config with intervals and retention
                 logger_cfg=dict(log_interval=100),  # Logger config with intervals
                 random_seed=None):  # New argument for random seed
        """
        Initialize the Trainer class.
        """
        self.model = model
        self.optimizer = optimizer
        self.losses = losses
        self.lr_scheduler = lr_scheduler
        self.metrics = metrics
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.val_interval = val_interval
        self.save_dir = save_dir
        self.load_from = load_from
        self.visualization_cfg = visualization_cfg
        self.checkpoint_cfg = checkpoint_cfg
        self.logger_cfg = logger_cfg
        self.post_transforms = post_transforms
        self.ignore_index = ignore_index
        self.experiment_name = experiment_name or codename.codename(separator='_')
        self._best_metric = None
        self._best_model_path = None

        # Initialize directories
        self.exp_dir = os.path.join(save_dir, self.experiment_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        self.checkpoint_dir = self._init_directory("checkpoints") if checkpoint_cfg else None

        # Initialize logger and visualizer
        self.logger,  self.log_file = self.init_logger()
        self.visualizer = self.init_visualizer()
        self.evaluator = self.init_evaluator(self.metrics)
        # Set or generate random seed
        self.random_seed = self._set_random_seed(random_seed)

    def init_logger(self):
        """
        Initialize the logger using loguru.

        Returns:
            logger: Configured loguru logger instance.
        """
        # Create the logging directory if it doesn't exist
        os.makedirs(self.exp_dir, exist_ok=True)

        # Define log file path
        datetime_str = time.strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(self.exp_dir,f"{datetime_str}.log")

        # Remove the default loguru handler to avoid duplicate logs
        logger.remove()

        # Add a file handler for logs
        logger.add(
            log_file,
            level="INFO",
            format="{time:MM/DD HH:mm:ss} - {level} - {message}",
            rotation="10 MB",  # Rotate logs when they reach 10 MB
            retention="7 days",  # Keep logs for 7 days
            compression="zip",  # Compress old logs
        )

        # Add a console handler for logs
        logger.add(
            sink=sys.stdout,
            level="INFO",
            format="{time:MM/DD HH:mm:ss} - {level} - {message}",
            colorize=True,  # Enable colored output
        )

        return logger, log_file

    @staticmethod
    def _set_random_seed(seed):
        """
        Set or generate a random seed, ensuring reproducibility.

        Args:
            seed (int, optional): The seed to set. If None, a random seed is generated.

        Returns:
            int: The seed used.
        """
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return seed

    def _init_directory(self, subfolder_name):
        """
        Initialize a subdirectory in the experiment directory.

        Args:
            subfolder_name (str): Name of the subdirectory.

        Returns:
            str: Path to the subdirectory.
        """
        path = os.path.join(self.exp_dir, subfolder_name)
        os.makedirs(path, exist_ok=True)
        return path

    def init_visualizer(self):
        """Initialize the visualizer."""
        if not self.visualization_cfg:
            self.logger.info("Visualization is disabled.")
            return None
        visualization_dir = self._init_directory("visualizations")
        visualizer = SegmentationVisualizer(
            save_dir=visualization_dir,
            classes=self.visualization_cfg.get("classes"),
            palette=self.visualization_cfg.get("palette"),
            background_index=self.visualization_cfg.get("background_index", 0)
        )
        return visualizer

    def init_evaluator(self, metrics):
        from splae.evaluation import MetricsEvaluator
        return MetricsEvaluator(metrics, num_classes=4, ignore_index=self.ignore_index)

    def _log_experiment_setup(self):
        """Log experiment setup information."""
        self.logger.info(f"Experiment Name: {self.experiment_name}")
        self.logger.info(f"Saving directory: {self.exp_dir}")
        self.logger.info(f"Logs will be saved to {self.log_file}")
        self.logger.info(f"Seed: {self.random_seed}")
        if self.checkpoint_dir:
            self.logger.info(f"Checkpoints will be saved to {self.checkpoint_dir}")
        if self.visualizer:
            self.logger.info(f"Visualizations will be saved to {self.visualizer.save_dir}")
        if self.epochs:
            self.logger.info(f"Total epochs: {self.epochs}")
        if self.val_interval:
            self.logger.info(f"Validation interval: {self.val_interval}")

    def _before_test(self):
        assert self.test_loader is not None, "For testing, test_loader must be provided."
        assert self.metrics is not None, "For testing, metrics must be provided."
        assert self.load_from is not None, "For testing, load_from must be provided."
        self._load_checkpoint(self.load_from)
        self.model.to(self.device)
        self._log_experiment_setup()

    def test(self):
        self._before_test()
        self.validate(epoch=1, mode='test', data_loader=self.test_loader)

    def _before_train(self):
        """Perform actions before training the model."""
        # assert if any of the following is None
        assert self.train_loader is not None, "For training, train_loader must be provided."
        assert self.losses is not None, "For training, losses must be provided."
        assert self.optimizer is not None, "For training, optimizer must be provided."
        assert self.epochs is not None, "For training, epochs must be provided."
        assert self.lr_scheduler is not None, "For training, lr_scheduler must be provided."
        if self.val_interval:
            assert self.val_loader is not None, "Configured val_interval, but val_loader is None."
            assert self.metrics is not None, "Configured val_interval, but metrics is None."

        if self.load_from:
            self._load_checkpoint(self.load_from)
        else:
            self.model.init_weights()
        self.model.to(self.device)
        self._log_experiment_setup()

    def train(self):
        """Train the model for all epochs."""
        self._before_train()
        for epoch in range(1, self.epochs + 1):
            self.logger.info(f"Experiment: {self.experiment_name}")
            self.train_epoch(epoch)
            self._after_train_epoch(epoch)
        self._after_train(epoch)

    def _after_train(self,epoch):
        if self.test_loader:
            self.logger.info("Finished training. Starting testing.")
            if self._best_model_path:
                self._load_checkpoint(self._best_model_path)
            self.validate(epoch=epoch, mode='test', data_loader=self.test_loader)

    def train_epoch(self, epoch):
        """Train the model for one epoch."""
        self.model.train()
        epoch_loss = 0
        iter_start_time = time.time()
        for i, data in enumerate(self.train_loader):
            data, losses = self.run_iter(data, mode='train')
            epoch_loss += sum(losses.values()) / len(losses)
            iter_start_time = self._after_iter(data, losses, i+1, epoch, iter_start_time, mode='train')
        return epoch_loss / len(self.train_loader)

    def run_iter(self, data, mode='train'):
        """Train the model for one iteration."""
        inputs, targets = data["img"].to(self.device), data["gt_mask"].to(self.device)

        # Forward pass
        outputs = self.model(inputs)
        data.update({'pred_logits': outputs})
        self.post_process(data)

        # Calculate losses
        losses = self.get_losses(outputs, targets, reduction='none')
        total_loss = sum(losses.values())
        losses.update({'total_loss': total_loss})

        # Backward pass
        if mode == 'train':
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        return data, losses

    def post_process(self, data):
        """Apply post transforms to the outputs."""
        if self.post_transforms:
            pred_logits = data["pred_logits"]
            pred_mask = torch.stack([self.post_transforms(pred_logits[i]) for i in range(len(pred_logits))])
            data.update({'pred_mask': pred_mask})

    def get_losses(self, outputs, targets, reduction=None):
        """Calculate the losses."""
        losses = {}
        for loss_fn in self.losses:
            loss = loss_fn(outputs, targets) if targets is not None else 0
            if loss.ndim > 0 and reduction != 'none':
                if reduction == 'mean':
                    loss = loss.mean()
                elif reduction == 'sum':
                    loss = loss.sum()
                else:
                    # fallback to mean if unknown reduction
                    loss = loss.mean()
            # if loss_fn as name property, use it as key
            loss_name = loss_fn.name if hasattr(loss_fn, 'name') else loss_fn.__class__.__name__
            losses[loss_name] = loss
        return losses

    def validate(self, epoch, mode='val', data_loader=None):
        """
        Validate the model and calculate metrics.

        Args:
            mode:
            data_loader:
            epoch (int): Current epoch number.
        """
        self.model.eval()  # Set model to evaluation mode
        iter_start_time = time.time()
        data_loader = data_loader or self.val_loader
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                # Run the validation iteration
                data, losses = self.run_iter(data, mode=mode)
                iter_start_time = self._after_iter(
                    data, losses, i + 1, epoch, iter_start_time, mode=mode
                )

                # Get model outputs
                outputs = self._get_outputs(data)

                # Prepare ground truth and mask ignored regions
                targets = data["gt_mask"].to(self.device)

                # Compute metrics for the current batch
                self.evaluator.process(outputs, targets)

        # Normalize metrics across all batches
        return self._finalize_metrics(epoch, mode)

    def _get_outputs(self, data):
        """
        Retrieve the model outputs for metric calculation.

        Args:
            data (dict): Batch data from the dataloader.

        Returns:
            torch.Tensor: Model outputs (one-hot encoded or directly predicted masks).
        """
        if "pred_mask" in data and data["pred_mask"] is not None:
            return data["pred_mask"]  # Use precomputed masks if available
        else:
            return data["pred_logits"].softmax(1).argmax(1).to(self.device)


    def _finalize_metrics(self,epoch=1, mode='val'):
        """
        Normalize, log, and return mean scores for metrics after validation.

        Args:
            all_metrics (defaultdict): Accumulated metrics.
            num_batches (int): Number of batches in the validation loader.
            epoch (int): Current epoch number.

        Returns:
            dict: A dictionary containing mean scores for each metric.
        """
        classes = self.visualization_cfg.get("classes", ["class_0", "class_1", "class_2", "class_3"])
        mean_scores = {}
        metrics = self.evaluator.finalize()
        for metric_name, metric in metrics.items():
            log_str = f"Epoch({mode})  [{epoch}]  {metric_name}   "
            for i, score in enumerate(metric):
                class_name = classes[i] if i < len(classes) else f"class_{i}"
                log_str += f"{class_name}: {score:.4f}  "
            mean_score = metric.mean()
            log_str += f"m{metric_name}: {mean_score:.4f}"
            mean_scores[metric_name] = mean_score
            self.logger.info(log_str)

        return mean_scores

    def _after_train_epoch(self, epoch):
        """
        Perform actions after completing a training epoch, such as validation, checkpointing, and learning rate adjustment.

        Args:
            epoch (int): The current epoch number.
        """
        # Validate the model at specified intervals
        if epoch % self.val_interval == 0:
            metrics = self.validate(epoch)

            # Save the best model if 'save_best_by' is specified in checkpoint_cfg
            if self.checkpoint_cfg and "save_best_by" in self.checkpoint_cfg:
                self._save_best_model(epoch, metrics)

        # Save a regular checkpoint at specified intervals
        if self.checkpoint_cfg and epoch % self.checkpoint_cfg.get("interval", 1) == 0:
            self._save_checkpoint(epoch)

        # Step the learning rate scheduler
        if self.lr_scheduler:
            self.lr_scheduler.step()

    def _after_iter(self, data, losses, i, epoch, iter_start_time, mode='train'):
        if i % self.logger_cfg["log_interval"] == 0:
            self._log_interval(losses, i, epoch, iter_start_time, mode=mode)
            iter_start_time = time.time()
        if mode == 'train':
            if self.visualization_cfg.get('train_interval'):
                if i % self.visualization_cfg['train_interval'] == 0:
                    self._save_visualization(data, i, epoch, mode='train')
        elif mode == 'val' or mode == 'test':
            if self.visualization_cfg.get('val_interval'):
                if i % self.visualization_cfg['val_interval'] == 0:
                    self._save_visualization(data, i, epoch, mode=mode)
        return iter_start_time

    def _save_checkpoint(self, epoch):
        """
        Save the model checkpoint at the current epoch.

        Args:
            epoch (int): The current epoch number.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pth")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
        }, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _save_best_model(self, epoch, metrics):
        """
        Save the best model based on a specific metric defined in checkpoint_cfg.

        Args:
            epoch (int): The current epoch number.
            metrics (dict): A dictionary of validation metrics from the current epoch.
        """
        save_best_by = self.checkpoint_cfg["save_best_by"]
        current_metric = metrics.get(save_best_by, None)

        # Ensure the metric exists
        if current_metric is None:
            return

        # Initialize or update the best metric
        if not self._best_metric or current_metric > self._best_metric:
            self._best_metric = current_metric
            self._best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
                save_best_by: current_metric,
            }, self._best_model_path)
            self.logger.info(f"Best model saved: {self._best_model_path} with {save_best_by}: {current_metric:.4f}")

    def _log_interval(self, batch_losses, i, epoch, iter_start_time, mode='train'):
        time_per_iter = time.time() - iter_start_time
        losses_text = "  ".join([f"{loss_name}: {loss:.4f}" for loss_name, loss in batch_losses.items()])
        if mode == 'val' or mode == 'test':
            loader_len = len(self.val_loader) if mode == 'val' else len(self.test_loader)
            lr_text = ""
        elif mode == 'train':
            loader_len = len(self.train_loader)
            lr_text = f"lr: {self.lr_scheduler.get_last_lr()[0]:.6f}"
        self.logger.info(
            f"Epoch({mode})   [{epoch}][{i}/{loader_len}]  "
            f"time: {time_per_iter:.4f}  {lr_text}  {losses_text}"
        )

    def _save_visualization(self, data, i, epoch, mode='val'):
        """Save visualizations."""
        if data is None:
            return
        images = data['img']
        img_names = data['metainfo']['img_name']
        gt_masks = data['gt_mask']
        pred_masks = self._get_outputs(data)
        pl_masks = data.get('pl_mask', None)
        num_samples = self.visualization_cfg.get('num_samples', 1)
        self.visualizer.draw(images, img_names, pred_masks, gt_masks, pl_masks, mode,num_samples, i, epoch)

    def _load_checkpoint(self, checkpoint_path):
        """
        Load a model checkpoint from a file.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")


class AdaptationRunner(Runner):
    def __init__(self,
                 model,
                 save_dir,
                 load_from,
                 pl_generator=None,
                 optimizer=None,
                 losses=None,
                 lr_scheduler=None,
                 metrics=None,
                 device=None,
                 train_loader=None,
                 val_loader=None,
                 test_loader=None,
                 epochs=None,
                 val_interval=None,
                 post_transforms=None,
                 ignore_index=-1,
                 experiment_name=None,
                 visualization_cfg=None,  # Visualization config with intervals and sample size
                 checkpoint_cfg=None,  # Checkpoint config with intervals and retention
                 logger_cfg=dict(log_interval=100),  # Logger config with intervals
                 random_seed=None):
        super(AdaptationRunner, self).__init__(
            model=model,
            save_dir=save_dir,
            optimizer=optimizer,
            losses=losses,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=epochs,
            val_interval=val_interval,
            load_from=load_from,
            post_transforms=post_transforms,
            ignore_index=ignore_index,
            experiment_name=experiment_name,
            visualization_cfg=visualization_cfg,
            checkpoint_cfg=checkpoint_cfg,
            logger_cfg=logger_cfg,
            random_seed=random_seed
        )

        self.pl_generator = pl_generator

    def _before_train(self):
        assert self.pl_generator is not None, "For training, pl_generator must be provided."
        super(AdaptationRunner, self)._before_train()

    def run_iter(self, data, mode='train'):
        """Train the model for one iteration."""
        inputs, targets = data["img"].to(self.device), data["gt_mask"].to(self.device)

        # Forward pass
        outputs = self.model(inputs)
        data.update({'pred_logits': outputs})
        self.post_process(data)

        if mode == 'train':
            ignore_index_mask = None
            if self.ignore_index is not None:
                ignore_index_mask = (targets == self.ignore_index)
            pl_targets = self.pl_generator.generate(images=inputs, seg_logits=outputs, ignore_index_mask=ignore_index_mask, gt_masks=None)
            if pl_targets is None:
                losses = self.get_losses(outputs, pl_targets)
                return None, losses
            pl_mask = pl_targets.get('target')
            valid_indices = pl_targets.get('valid_indices')
            # change values of pl_mask to ignore_index
            if ignore_index_mask is not None:
                ignore_index_mask = ignore_index_mask[valid_indices]
                pl_mask[ignore_index_mask] = self.ignore_index
            extra_visuals = pl_targets.get('extra_visuals', None)
            if extra_visuals:
                data.update({'extra_visuals': extra_visuals})
            data.update({'pl_mask': pl_mask})
            data.update({'pred_logits': outputs[valid_indices]})
            data.update({'gt_mask': targets[valid_indices]})
            data.update({'img': inputs[valid_indices]})
            losses = self.get_losses(outputs[valid_indices], pl_targets)
            self.optimizer.zero_grad()
            total_loss = sum(losses.values())
            total_loss.backward()
            self.optimizer.step()
        else:
            losses = self.get_losses(outputs, targets)
            total_loss = sum(losses.values())

        losses.update({'total_loss': total_loss})
        return data, losses

    def _save_visualization(self, data, i, epoch, mode='val'):
        """Save visualizations."""
        if data is None:
            return
        images = data['img']
        img_names = data['metainfo']['img_name']
        gt_masks = data['gt_mask']
        pred_masks = self._get_outputs(data)
        pl_masks = data.get('pl_mask', None)
        extra_visuals = data.get('extra_visuals', None)
        num_samples = self.visualization_cfg.get('num_samples', 1)
        self.visualizer.draw(images, img_names, pred_masks, gt_masks, pl_masks, mode,num_samples, i, epoch, extra_visuals)

    def adapt_warmup(self):
        self._before_train()
        for i, data in enumerate(self.train_loader):
            inputs, targets = data["img"].to(self.device), data["gt_mask"].to(self.device)
            outputs = self.model(inputs)
            ignore_index_mask = None
            if self.ignore_index is not None:
                ignore_index_mask = (targets == self.ignore_index)
            self.pl_generator.warmup(seg_logits=outputs,
                                                    ignore_index_mask=ignore_index_mask)
        self.pl_generator.finalize_structure_priors()

class DPLRunner(AdaptationRunner):
    def __init__(self, 
                 model,
                 save_dir,
                 load_from,
                 pl_generator=None,
                 optimizer=None,
                 losses=None,
                 lr_scheduler=None,
                 metrics=None,
                 device=None,
                 train_loader=None,
                 val_loader=None,
                 test_loader=None,
                 epochs=None,
                 val_interval=None,
                 post_transforms=None,
                 ignore_index=-1,
                 experiment_name=None,
                 visualization_cfg=None,
                 checkpoint_cfg=None,
                 logger_cfg=dict(log_interval=100),
                 random_seed=None):
        super(DPLRunner, self).__init__(
            model=model,
            save_dir=save_dir,
            load_from=load_from,
            pl_generator=pl_generator,
            optimizer=optimizer,
            losses=losses,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=epochs,
            val_interval=val_interval,
            post_transforms=post_transforms,
            ignore_index=ignore_index,
            experiment_name=experiment_name,
            visualization_cfg=visualization_cfg,
            checkpoint_cfg=checkpoint_cfg,
            logger_cfg=logger_cfg,
            random_seed=random_seed
        )

    def _before_train(self):
        super(DPLRunner, self)._before_train()
        # Generate pseudo/proto dataset before training
        self.logger.info("Generating pseudo/proto dataset before training")
        train_dataset = self.pl_generator.generate(self.model, self.train_loader)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.train_loader.batch_size,
            num_workers=self.train_loader.num_workers,
            shuffle=True
        )
        self.logger.info("Pseudo/proto dataset generated")

    def run_iter(self, data, mode="train"):
        """Run one iteration with DPL logic."""
        inputs, targets = data["img"].to(self.device), data["gt_mask"].to(self.device)
        outputs = self.model(inputs)

        data.update({"pred_logits": outputs})
        self.post_process(data)

        if mode == "train":
            # ---- DPL logic ----
            pseudo = data["pseudo_label"].to(self.device)    # [B,H,W]
            proto  = data["proto_pseudo"].to(self.device)    # [B,H,W]

            # valid pixels (not ignore) and agreement between pseudo & proto
            valid_mask = (pseudo != self.ignore_index)
            agree_mask = (pseudo == proto)

            # optional uncertainty mask
            if "uncertain_map" in data:
                uncertain_map = data["uncertain_map"].to(self.device)  # [B,H,W]
                uncert_mask = (uncertain_map < 0.05)                   # threshold can be tuned
                keep_mask = valid_mask & agree_mask & uncert_mask
            else:
                keep_mask = valid_mask & agree_mask

            # loss per pixel (no reduction!)
            per_pix_losses = self.get_losses(outputs, pseudo, reduction='none')  # dict of [B,H,W]

            masked_losses = {}
            used = keep_mask.sum().item()
            for name, loss_map in per_pix_losses.items():
                # loss_map expected [B,H,W], keep_mask [B,H,W]
                if loss_map.ndim == 1:
                    masked_losses[name] = loss_map.mean()
                    continue
                if used > 0:
                    masked_losses[name] = (loss_map * keep_mask).sum() / keep_mask.sum()
                else:
                    masked_losses[name] = torch.tensor(0.0, device=self.device)

            total_loss = sum(masked_losses.values())
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            data.update({'pl_mask': pseudo})
            losses = masked_losses
            losses.update({"total_loss": total_loss})

        else:
            # standard supervised loss on GT
            losses = self.get_losses(outputs, targets)
            total_loss = sum(losses.values())
            losses.update({"total_loss": total_loss})

        return data, losses

