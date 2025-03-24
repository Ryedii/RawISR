import os
import argparse
import json
import sys
from datetime import datetime

from tqdm import tqdm

import numpy as np

from model import Model as Model

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torchmetrics

TIME = datetime.now().strftime("%m%d-%H%M%S")


class RawData(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.lr_raw_list = [os.path.join(data_dir, 'lr', rawf) for rawf in
                            sorted(os.listdir(os.path.join(data_dir, 'lr')))]
        self.hr_raw_list = [os.path.join(data_dir, 'hr', rawf) for rawf in
                            sorted(os.listdir(os.path.join(data_dir, 'hr')))]

    def __len__(self):
        return len(self.lr_raw_list)

    def __getitem__(self, idx):
        lr_raw = np.load(self.lr_raw_list[idx]) if len(self.lr_raw_list) > idx else np.empty((0, 0, 4))
        hr_raw = np.load(self.hr_raw_list[idx]) if len(self.hr_raw_list) > idx else np.empty((0, 0, 4))
        return {
            'lr_raw': torch.from_numpy(lr_raw).float().permute(2, 0, 1),
            'hr_raw': torch.from_numpy(hr_raw).float().permute(2, 0, 1)
        }


class Trainer:
    def __init__(self, config, rank):
        self.log_dir = os.path.join(config['log_dir'], 'train' + TIME)
        if rank == 0:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(os.path.join(self.log_dir, 'config.json'), 'w') as f:
                json.dump(config, f, indent=4)
        self.disable_tqdm = config['disable_tqdm'] if rank == 0 else True
        self.rank = rank
        self.world_size = config['world_size']
        self.device = torch.device(f'cuda:{self.rank}')
        self.batch_size = config['batch_size']

        self.model = Model(config).to(self.device)
        self.model = DDP(self.model, device_ids=[self.rank])

        self.pretrained = config['pretrained']
        self.pretrained_model = config['pretrained_model']

        self.train_dataset_dir = config['train_dataset_dir']
        self.train_dataset = RawData(self.train_dataset_dir)
        self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=self.world_size, rank=self.rank)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, sampler=self.train_sampler, num_workers=2
        )

        self.if_validate = config['if_validate']
        self.validate_dataset_dir = config['validate_dataset_dir']
        self.validate_dataset = RawData(self.validate_dataset_dir)
        self.validate_sampler = DistributedSampler(self.validate_dataset, num_replicas=self.world_size, rank=self.rank)
        self.validate_loader = DataLoader(
            self.validate_dataset, batch_size=self.batch_size, sampler=self.validate_sampler, shuffle=False
        )

        self.max_epoch = config['max_epoch']

        self.validate_freq = config['validate_freq']
        self.save_freq = config['save_freq']

        self.learning_rate = config['learning_rate']
        self.decay_coef = config['decay_coef']
        self.decay_freq = config['decay_freq']
        self.max_decay_epoch = config['max_decay_epoch']
        self.lowest_learning_rate = config['lowest_learning_rate']
        self.T_max = config['T_max']
        self.eta_min = config['eta_min']
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: max(
                self.decay_coef ** (epoch // self.decay_freq),
                self.lowest_learning_rate / self.learning_rate,
            ) if epoch < self.max_decay_epoch else self.lowest_learning_rate / self.learning_rate
        )
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.T_max, eta_min=self.eta_min)
        # self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1)
        self.loss = nn.L1Loss()
        self.psnr = torchmetrics.image.PeakSignalNoiseRatio().to(self.device)
        self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure().to(self.device)

        if self.pretrained:
            self.load_model(self.pretrained_model)

    def load_model(self, path):
        state = torch.load(path)
        self.model.module.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        print(f'load model and optimizer from {path}')

    def save_model(self, epoch):
        if self.rank == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': self.model.module.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                }, os.path.join(self.log_dir, f'model_epoch_{epoch}.pth')
            )

    def train_epoch(self, epoch):
        self.model.train()
        self.train_sampler.set_epoch(epoch)
        total_loss = .0
        for batch in tqdm(self.train_loader, desc=f'Epoch {epoch}', disable=self.disable_tqdm):
            low_resolution_raw = batch['lr_raw'].to(self.device)
            high_resolution_raw = batch['hr_raw'].to(self.device)
            upsampled_raw = self.model(low_resolution_raw)
            loss = self.loss(upsampled_raw, high_resolution_raw)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        self.scheduler.step()
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

            if epoch % self.validate_freq == 0 or epoch == self.max_epoch:
                psnr, ssim = self.validate(epoch)
                if self.rank == 0:
                    print(f'epoch {epoch}: train_loss = {loss:.6f}, val_PSNR = {psnr:.6f}, val_SSIM = {ssim:.6f}')

            if epoch % self.save_freq == 0 or epoch == self.max_epoch:
                self.save_model(epoch)


def parse_args():
    parser = argparse.ArgumentParser(description="Please use a config file.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file."
    )
    return parser.parse_args()


def train_worker(rank, world_size, config):
    print(f"rank {rank} started.")
    try:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(33355)
        os.environ['TORCH_DIST_TIMEOUT'] = '60'
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        trainer = Trainer(config, rank)
        trainer.train()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    args = parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
        if config['task'] == 'train':
            world_size = config['world_size']
            mp.spawn(train_worker, args=(world_size, config), nprocs=world_size, join=True)
        if config['task'] == 'predict':
            model = torch.load(config['pretrained_model']).to(config['device'])
            model.eval()
