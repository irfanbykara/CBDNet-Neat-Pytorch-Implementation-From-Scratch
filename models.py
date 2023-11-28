
import torch
from torch.nn import Module


class CBDNetwork(Module):
    def __init__(self, noise_predictor: Module,
                 reconstruction_network: Module):
        super(CBDNetwork, self).__init__()
        self.noise_predictor = noise_predictor
        self.reconstruction_network = reconstruction_network

    def forward(self, x: torch.Tensor):
        """Forward pass for the network.

        :return: A torch.Tensor.
        """

        noise_level = self.noise_predictor(x)
        print(f"Noise level shape is : {noise_level.shape}")
        concat_img = torch.cat([x, noise_level], dim=1)
        out = self.reconstruction_network(concat_img) + x
        print(noise_level.shape)
        return noise_level, out

