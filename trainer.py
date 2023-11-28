
from typing import Callable, List
import torch
import torch.utils.data as data


class BaselineTrainer:
    def __init__(self, model: torch.nn.Module,
                 loss: Callable,
                 optimizer: torch.optim.Optimizer,
                 use_cuda=True):
        self.loss = loss
        self.use_cuda = use_cuda
        self.optimizer = optimizer
        self.model = model

        if use_cuda:
            self.model = model.to(device="cuda:0")

    def fit(self, train_data_loader: data.DataLoader,
            epoch: int):
        avg_loss = 0.
        self.model.training = True
        for e in range(epoch):
            print(f"Start epoch {e+1}/{epoch}")

            n_batch = 0
            for i, ((ref_img, dist_img),y) in enumerate(train_data_loader):
                # Reset previous gradients
                self.optimizer.zero_grad()

                # Move data to cuda is necessary:
                if self.use_cuda:
                    ref_img = ref_img.cuda()
                    dist_img = dist_img.cuda()

                # Make forward
                noise_level, out = self.model(dist_img)

                # estimated_mos = torch.mean(noise_level, dim=(1, 2, 3))
                y = y.view(y.shape[0], 1, 1, 1)

                true_mos = torch.ones_like(noise_level)
                true_mos *= y
                loss = self.loss(out, ref_img,noise_level,true_mos)
                loss.backward()

                # Adjust learning weights
                self.optimizer.step()
                avg_loss += loss.item()
                n_batch += 1

                print(f"\r{i+1}{len(train_data_loader)}: loss = {loss / n_batch}", end='')
            print()

        return avg_loss
