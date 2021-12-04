import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from monai.losses import DiceLoss
from monai.networks import one_hot


def plot_training(train_losses, val_losses):
    """Plots the training and the validation loss curve.

    Args:
        train_losses (float): The list of training loss values
        val_losses (float): The list of validation loss values
    """
    # Plot the training vs validation graph
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(train_losses)), train_losses, 'r', label="Training loss")
    plt.plot(range(len(val_losses)), val_losses, 'g', label="Validation loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Training iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.ylim((0, 10))
    plt.show()


def save_model(model, config, epoch=None):
    """Saves the trained model to the provided dir.

    Args:
        model (nn.Module): The trained model 
        config (dict): The parameter dictionary with model path
        epoch (int): The nth epoch the model was trained
    """
    # Get path to the model
    path = config['model_info']['model_path']

    if epoch is not None:
        # If epoch is provided, change the name
        path = path[:-4] + f"_ep{epoch}.pth"

    if not os.path.exists("models"):
        # Create `models` dir if it not exists
        os.makedirs("models")

    # Save the model
    torch.save(model.state_dict(), path)


def save_losses(train_losses, val_losses, config):
    """Saves the monitored losses to `models` dir.

    Args:
        train_losses (list(float)): The training losses
        val_losses (list(float)): The validation losses
        config (dict): The parameter dictionary with model path
    """
    # Get model path and optimizer name
    model_path = config["model_info"]["model_path"][:-4]
    op_name = config["optimizer_info"]["name"]

    with open(f"{model_path}_{op_name}_loss.npz","wb") as f:
        # Save the loss history to model path
        d = {'file': f, 'train_losses': train_losses, 'val_losses': val_losses}
        np.savez(**d)


def get_dice_score(output, y):
    """Calculates the dice score.

    Args:
        output (torch.tensor): The raw network outputs of shape (N, C, H, W)
        y (torch.tensor): The true mask of shape (N, H, W)
    
    Returns:
        (torch.tensor): The (1 - loss) values of shape (C - 1,)
    """
    # Get the raw predictions
    _, y_hat = torch.max(output, 1)
    y_hat = one_hot(y_hat.unsqueeze_(1), num_classes=output.shape[1])

    # Define the dice loss (ignore first channel)
    dice_loss_fn = DiceLoss(to_onehot_y=True, include_background=False)

    return 1 - dice_loss_fn(y_hat, y)


class AdaHessian(torch.optim.Optimizer):
    """
    Implements the AdaHessian algorithm from "ADAHESSIAN: An Adaptive Second OrderOptimizer for Machine Learning"
    Arguments:
        params (iterable) -- iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional) -- learning rate (default: 0.1)
        betas ((float, float), optional) -- coefficients used for computing running averages of gradient and the squared hessian trace (default: (0.9, 0.999))
        eps (float, optional) -- term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional) -- weight decay (L2 penalty) (default: 0.0)
        hessian_power (float, optional) -- exponent of the hessian trace (default: 1.0)
        update_each (int, optional) -- compute the hessian trace approximation only after *this* number of steps (to save time) (default: 1)
        n_samples (int, optional) -- how many times to sample `z` for the approximation of the hessian trace (default: 1)
    """

    def __init__(self, params, lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, 
                 hessian_power=1.0, update_each=1, n_samples=1, average_conv_kernel=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError(f"Invalid Hessian power value: {hessian_power}")

        self.n_samples = n_samples
        self.update_each = update_each
        self.average_conv_kernel = average_conv_kernel

        # use a separate generator that deterministically generates the same `z`s across all GPUs in case of distributed training
        self.generator = torch.Generator().manual_seed(2147483647)

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, hessian_power=hessian_power)
        super(AdaHessian, self).__init__(params, defaults)

        for p in self.get_params():
            p.hess = 0.0
            self.state[p]["hessian step"] = 0

    def get_params(self):
        """
        Gets all parameters in all param_groups with gradients
        """

        return (p for group in self.param_groups for p in group['params'] if p.requires_grad)

    def zero_hessian(self):
        """
        Zeros out the accumalated hessian traces.
        """

        for p in self.get_params():
            if not isinstance(p.hess, float) and self.state[p]["hessian step"] % self.update_each == 0:
                p.hess.zero_()

    @torch.no_grad()
    def set_hessian(self):
        """
        Computes the Hutchinson approximation of the hessian trace and accumulates it for each trainable parameter.
        """

        params = []
        for p in filter(lambda p: p.grad is not None, self.get_params()):
            if self.state[p]["hessian step"] % self.update_each == 0:  # compute the trace only each `update_each` step
                params.append(p)
            self.state[p]["hessian step"] += 1

        if len(params) == 0:
            return

        if self.generator.device != params[0].device:  # hackish way of casting the generator to the right device
            self.generator = torch.Generator(params[0].device).manual_seed(2147483647)

        grads = [p.grad for p in params]

        for i in range(self.n_samples):
            zs = [torch.randint(0, 2, p.size(), generator=self.generator, device=p.device) * 2.0 - 1.0 for p in params]  # Rademacher distribution {-1.0, 1.0}
            h_zs = torch.autograd.grad(grads, params, grad_outputs=zs, only_inputs=True, retain_graph=i < self.n_samples - 1)
            for h_z, z, p in zip(h_zs, zs, params):
                p.hess += h_z * z / self.n_samples  # approximate the expected values of z*(H@z)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional) -- a closure that reevaluates the model and returns the loss (default: None)
        """

        loss = None
        if closure is not None:
            loss = closure()

        self.zero_hessian()
        self.set_hessian()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.hess is None:
                    continue

                if self.average_conv_kernel and p.dim() == 4:
                    p.hess = torch.abs(p.hess).mean(dim=[2, 3], keepdim=True).expand_as(p.hess).clone()

                # Perform correct stepweight decay as in AdamW
                p.mul_(1 - group['lr'] * group['weight_decay'])

                state = self.state[p]

                # State initialization
                if len(state) == 1:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)  # Exponential moving average of gradient values
                    state['exp_hessian_diag_sq'] = torch.zeros_like(p.data)  # Exponential moving average of Hessian diagonal square values

                exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_hessian_diag_sq.mul_(beta2).addcmul_(p.hess, p.hess, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                k = group['hessian_power']
                denom = (exp_hessian_diag_sq / bias_correction2).pow_(k / 2).add_(group['eps'])

                # make update
                step_size = group['lr'] / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss