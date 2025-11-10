from torch.optim.lr_scheduler import _LRScheduler
import math

class PolyLR(_LRScheduler):
    def __init__(self,
                 optimizer,
                 eta_min=0.0,
                 power=1.0,
                 begin=0,
                 end=math.inf,
                 last_step=-1):
        """
        Implements the polynomial learning rate schedule with a minimum learning rate bound.

        Args:
            optimizer (Optimizer or OptimWrapper): Wrapped optimizer.
            eta_min (float): Minimum learning rate at the end of scheduling. Defaults to 0.
            power (float): The power of the polynomial. Defaults to 1.0.
            begin (int): Step at which to start updating the parameters. Defaults to 0.
            end (int): Step at which to stop updating the parameters. Defaults to INF.
            last_step (int): The index of last step. Used for resume without state dict. Defaults to -1.
        """
        self.eta_min = eta_min
        self.power = power
        self.begin = begin
        self.end = end
        super().__init__(optimizer, last_step)

    def get_lr(self):
        """
        Compute the new learning rates based on the current step, bounded by `eta_min`.

        Returns:
            List[float]: Updated learning rates for each parameter group.
        """
        if self.last_epoch < self.begin:  # Before scheduling starts
            return [group['lr'] for group in self.optimizer.param_groups]
        elif self.last_epoch >= self.end:  # After scheduling ends
            return [self.eta_min for _ in self.optimizer.param_groups]
        else:
            # Polynomial decay factor
            progress = (self.last_epoch - self.begin) / (self.end - self.begin)
            factor = (1 - progress) ** self.power
            return [
                max(base_lr * factor, self.eta_min)
                for base_lr in self.base_lrs
            ]

class ConstantLR(_LRScheduler):
    def __init__(self, optimizer, last_step=-1):
        """
        Implements a constant learning rate schedule.

        Args:
            optimizer (Optimizer or OptimWrapper): Wrapped optimizer.
            last_step (int): The index of last step. Used for resume without state dict. Defaults to -1.
        """
        super().__init__(optimizer, last_step)

    def get_lr(self):
        """
        Returns the current learning rates for all parameter groups.

        Returns:
            List[float]: Current learning rates for each parameter group.
        """
        return [group['lr'] for group in self.optimizer.param_groups]