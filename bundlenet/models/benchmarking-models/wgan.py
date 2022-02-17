"""
    Adapted from code from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_div/wgan_div.py
    See LICENSE in this directory for the original license text (MIT).
"""
import os
import numpy as np
import json
import time

from bundlenet.util import BasicDataset

import torch.nn as nn
import torch.autograd as autograd
import torch


class WGAN_div(nn.Module):
    def __init__(self, cover_data, base_data, device='cpu', latent_dim=None):
        super().__init__()

        # save data
        self.cover_data = cover_data
        self.base_data = base_data

        # parse dimension data
        self.total_dim = cover_data.shape[-1]
        self.latent_dim = latent_dim
        if latent_dim is None:
            self.latent_dim = self.total_dim

        # create components and send to device
        self.device = device
        self.G = self.Generator(self.latent_dim, self.total_dim)
        self.G.to(self.device)
        self.D = self.Discriminator(self.total_dim)
        self.D.to(self.device)
    
    def train_net(self,
            lr=1e-4,
            num_epochs=100,
            batch_size=8,
            save_weights=False,
            n_critic=5,
            wt_dir='run_'+time.ctime().replace(' ', '_'),
            k=2,
            p=6
        ):
        # Save everything!
        os.makedirs(wt_dir, exist_ok=True)

        # print out the parameters used for setup and training (possibly for easier loading later on)
        init_params = {
            'device': self.device,
            'latent_dim': self.latent_dim
        }
        train_params = {
            'lr': lr,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'n_critic': n_critic,
            'save_weights': save_weights,
            'k': k,
            'p': p
        }
        with open(wt_dir + '/train_parameters.json', 'w') as f:
            f.write(json.dumps(train_params))
        with open(wt_dir + '/init_parameters.json', 'w') as f:
            f.write(json.dumps(init_params))
        
        optimizer_G = torch.optim.Adam(self.G.parameters(), lr=lr)
        optimizer_D = torch.optim.Adam(self.D.parameters(), lr=lr)

        num_batches = num_epochs*(num_epochs//batch_size)
        scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=num_batches//18, gamma=0.5)
        scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=num_batches//18, gamma=0.5)

        loader = self._make_dataloader(self.cover_data, self.base_data, batch_size)

        for epoch in range(num_epochs):
            # Track losses across each epoch for displaying during training
            epoch_G_loss, epoch_D_loss = 0, 0
            for batch, (bundle, base) in enumerate(loader):
                self.G.train()
                self.D.train()
                optimizer_D.zero_grad()

                real_data = torch.stack(bundle).to(self.device).requires_grad_(True)
                
                noise = torch.randn([len(real_data), self.latent_dim], device=self.device)
                generated_data = self.G(noise)

                real_validity = self.D(real_data)
                fake_validity = self.D(generated_data)

                # Compute W-div gradient penalty
                real_grad_out = torch.ones([len(real_data)], dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(1)
                real_grad = autograd.grad(
                    real_validity, real_data, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True
                )[0]
                real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

                fake_grad_out = torch.ones([len(real_data)], dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(1)
                fake_grad = autograd.grad(
                    fake_validity, generated_data, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True
                )[0]
                fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

                div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2

                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + div_gp
                epoch_D_loss += d_loss.item()

                d_loss.backward()
                optimizer_D.step()
                scheduler_D.step()

                optimizer_G.zero_grad()

                if batch % n_critic == 0:
                    # Generate a batch of images
                    fake_data = self.G(noise)

                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = self.D(fake_data)
                    g_loss = -torch.mean(fake_validity)

                    g_loss.backward()
                    epoch_G_loss += g_loss.item()
                    optimizer_G.step()
                    scheduler_G.step()
                
                # print our results
                print(f'Epoch {epoch:2d} || G Loss: {n_critic*epoch_G_loss/(batch + 1):8.5f} || D Loss: {epoch_D_loss/(batch+1)}', end='\r')

                if epoch > 0 and epoch % 20 == 0 and save_weights:
                    torch.save(self.G.state_dict(), wt_dir + f'/epoch-{epoch}-G-weights.pt')
                    torch.save(self.D.state_dict(), wt_dir + f'/epoch-{epoch}-D-weights.pt')
                
            print()
        
        if save_weights:
            torch.save(self.G.state_dict(), wt_dir + f'/final-G-weights.pt')
            torch.save(self.D.state_dict(), wt_dir + f'/final-D-weights.pt')

    def _make_dataloader(self, data, target, batch_size):
        dataset = BasicDataset(data, target)
        dataloader = torch.utils.data.DataLoader(dataset, collate_fn=lambda x:(samples for samples in zip(*x)), batch_size=batch_size, shuffle=True)
        return dataloader
    
    def sample_from_fiber(self, basept=None, n=1):
        if n == 1:
            # hack to fix the problem of only putting through a single point
            noise = torch.randn([2, self.latent_dim], device=self.device)
            return self.G(noise).detach().cpu()[0]
        else:
            noise = torch.randn([n, self.latent_dim], device=self.device)
            return self.G(noise).detach().cpu()

    class Generator(nn.Module):
        def __init__(self, latent_dim, data_dim):
            super().__init__()

            def block(in_feat, out_feat, normalize=True):
                layers = [nn.Linear(in_feat, out_feat)]
                if normalize:
                    layers.append(nn.BatchNorm1d(out_feat, 0.8))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return layers

            self.model = nn.Sequential(
                *block(latent_dim, 128, normalize=False),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, data_dim)
            )

        def forward(self, z):
            return self.model(z)


    class Discriminator(nn.Module):
        def __init__(self, data_dim):
            super().__init__()

            self.model = nn.Sequential(
                nn.Linear(data_dim, 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
            )

        def forward(self, img):
            return self.model(img)
