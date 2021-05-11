from torch.optim.lr_scheduler import _LRScheduler
import math
import warnings


class CosineAnnealingWithWarmUp(_LRScheduler):
    def __init__(self, optimizer, T_max, W_steps, eta_min=0, last_epoch=-1, verbose=False):
        self.T_max = T_max
        self.W_steps = W_steps
        self.eta_min = eta_min
        super(CosineAnnealingWithWarmUp, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        if self.last_epoch == 0:
            return self.base_lrs
        elif self.last_epoch <= self.W_steps:
            return [base_lr * (self.last_epoch / self.W_steps) for base_lr in self.base_lrs]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * (self.last_epoch - self.W_steps) / (self.T_max - self.W_steps))) /
                (1 + math.cos(math.pi * (self.last_epoch - self.W_steps - 1) / (self.T_max - self.W_steps))) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]


class WarmUp(_LRScheduler):
    def __init__(self, optimizer, W_steps, eta_min=0, last_epoch=-1, verbose=False):
        self.W_steps = W_steps
        self.eta_min = eta_min
        super(WarmUp, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        if self.last_epoch == 0:
            return self.base_lrs
        elif self.last_epoch <= self.W_steps:
            return [base_lr * (self.last_epoch / self.W_steps) for base_lr in self.base_lrs]
        else:
            return self.base_lrs
