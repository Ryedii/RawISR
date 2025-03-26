from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as fu
from torch.utils.data import Dataset, DataLoader

from utils import psnr, ssim


class Trainer:
    def __init__(
            self,
            model,
            device,
            train_dataset,
            batch_size,
            max_epoch,
            lr,
            validate_freq=10000,
            validate_dataset=None,
            pretrained=None,
            loss=nn.L1Loss,
            optimizer=optim.Adam,
            disable_tqdm=True
    ):
        self.model = model
        self.device = device

        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.validate_freq = validate_freq
        self.validate_dataset = validate_dataset
        self.validate_loader = DataLoader(
            self.validate_dataset, batch_size=self.batch_size
        ) if validate_dataset is not None else None
        self.psnr = psnr.to(device)
        self.ssim = ssim.to(device)

        self.max_epoch = max_epoch
        if pretrained is not None:
            pass
        self.optimizer = optimizer(self.model, lr=lr)
        self.loss = loss

        self.disable_tqdm = disable_tqdm

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = .0
        for batch in tqdm(self.train_loader, desc=f'Epoch {epoch}', disable=self.disable_tqdm):
            low_resolution_raw = batch['lr_raw'].to(self.device)
            high_resolution_raw = batch['hr_raw'].to(self.device)
            super_resolution_raw = self.model(low_resolution_raw)
            loss = self.loss(super_resolution_raw, high_resolution_raw)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        with torch.no_grad():
            for batch in tqdm(self.validate_loader, desc=f'Validating epoch {epoch}', disable=self.disable_tqdm):
                low_resolution_raw = batch['lr_raw'].to(self.device)
                high_resolution_raw = batch['hr_raw'].to(self.device)
                upsampled_raw = self.model(low_resolution_raw)
                total_psnr += self.psnr(upsampled_raw, high_resolution_raw).cpu()
                total_ssim += self.ssim(upsampled_raw, high_resolution_raw).cpu()
        return total_psnr / len(self.validate_loader), total_ssim / len(self.validate_loader)

    def train(self):
        for epoch in range(1, self.max_epoch + 1):
            loss = self.train_epoch(epoch)
            # loss = 0.0

            if epoch % self.validate_freq == 0:
                psnr, ssim = self.validate(epoch)
                print(f'epoch {epoch}: train_loss = {loss:.6f}, val_PSNR = {psnr:.6f}, val_SSIM = {ssim:.6f}')

            if epoch % self.save_freq == 0 or epoch == self.max_epoch:
                self.save_model(epoch)
