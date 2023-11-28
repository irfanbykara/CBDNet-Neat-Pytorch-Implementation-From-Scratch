
import torch
from typing import Tuple
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss


class TVLoss(_Loss):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        grad_h = inp[:, :, :, :-1] - inp[:, :, :, 1:]

        # Compute vertical gradient
        grad_v = inp[:, :, :-1, :] - inp[:, :, 1:, :]

        # Calculate L2 norm of gradients
        norm_h = torch.norm(grad_h, p=2, dim=3)
        norm_v = torch.norm(grad_v, p=2, dim=2)

        # Sum of L2 norms in both directions
        tv_loss = torch.sum(norm_h) + torch.sum(norm_v)

        return tv_loss


class AsymmetricLoss(_Loss):
    def __init__(self):
        super(AsymmetricLoss, self).__init__()

    def forward(self, inp: torch.Tensor, target: torch.Tensor,estimated_mos:torch.Tensor, true_mos: torch.Tensor, alpha:float=0.5) -> torch.Tensor:

        # Calculate the indicator function I
        indicator_I = (estimated_mos - true_mos) < 0
        indicator_I = indicator_I.float()  # Convert boolean to float (1.0 for True, 0.0 for False)

        # Calculate the loss terms
        loss_term1 = torch.abs(alpha - indicator_I)
        loss_term2 = (estimated_mos - true_mos) ** 2

        # Calculate the overall loss for each pixel
        pixelwise_loss = loss_term1 + loss_term2

        # Sum the losses over all pixels
        asymmetric_loss = torch.sum(pixelwise_loss)

        return asymmetric_loss


class TotalLoss(_Loss):
    def __init__(self, lambda_tv=0.5, lambda_asymm=0.5):
        super(TotalLoss, self).__init__()
        self.lambda_tv = lambda_tv
        self.lambda_asymm = lambda_asymm
        self.tv_loss = TVLoss()
        self.asymmetric_loss = AsymmetricLoss()

    def forward(self, inp: torch.Tensor, target: torch.Tensor,estimated_noise:torch.Tensor, true_mos: torch.Tensor) -> torch.Tensor:
        reconstruction_error = torch.sum((inp - target) ** 2)
        tv_loss = self.tv_loss(inp, target)
        asymmetric_loss = self.asymmetric_loss(inp,target,estimated_noise,true_mos)
        return reconstruction_error + self.lambda_tv * tv_loss + self.lambda_asymm * asymmetric_loss

