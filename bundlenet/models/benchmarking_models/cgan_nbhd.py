"""
    Adapted from code from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_div/wgan_div.py
    See LICENSE in this directory for the original license text (MIT).
"""
import os
import numpy as np
import json
import time

from bundlenet.util import BasicDataset
from random import sample
from sklearn.cluster import KMeans

import torch.nn as nn
import torch


class CGAN_nbhd(nn.Module):
    def __init__(self, cover_data, base_data, device='cpu', latent_dim=None, num_nbhds=25):
        super().__init__()

        # save data
        self.cover_data = cover_data
        self.base_data = base_data

        # parse dimension data
        self.total_dim = cover_data.shape[-1]
        self.latent_dim = latent_dim
        self.cond_dim = base_data.shape[-1]
        if latent_dim is None:
            self.latent_dim = self.total_dim
        
        self.num_nbhds = num_nbhds
        self.centers = self._create_nbhds()

        # create components and send to device
        self.device = device
        self.G = self.Generator(self.latent_dim, self.total_dim, 2*self.cond_dim)
        self.G.to(self.device)
        self.D = self.Discriminator(self.total_dim, 2*self.cond_dim)
        self.D.to(self.device)
    
    def train_net(self,
            lr=1e-4,
            num_epochs=100,
            batch_size=8,
            save_weights=False,
            wt_dir='run_'+time.ctime().replace(' ', '_')
        ):
        # Save everything!
        os.makedirs(wt_dir, exist_ok=True)
        torch.save(self.centers, wt_dir + '/centers.pt')

        # print out the parameters used for setup and training (possibly for easier loading later on)
        init_params = {
            'device': self.device,
            'latent_dim': self.latent_dim
        }
        train_params = {
            'lr': lr,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'save_weights': save_weights
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

        adversarial_loss = nn.MSELoss()

        for epoch in range(num_epochs):
            # Track losses across each epoch for displaying during training
            epoch_G_loss, epoch_D_loss = 0, 0
            for batch, (bundle, base) in enumerate(loader):
                self.G.train()
                self.D.train()
                optimizer_D.zero_grad()

                # Adversarial ground truths
                valid = torch.ones([len(bundle), 1], dtype=torch.float, requires_grad=False, device=self.device)
                fake = torch.zeros([len(bundle), 1], dtype=torch.float, requires_grad=False, device=self.device)

                # Configure input
                real_imgs = torch.stack(bundle).to(self.device)

                conditions = []
                for pt in base:
                    idx = (self.centers - pt.unsqueeze(0)).norm(dim=1).argmin()
                    conditions.append(torch.cat([self.centers[idx],pt]))
                conditions = torch.stack(conditions).to(self.device)

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Sample noise and labels as generator input
                z = torch.randn((len(bundle), self.latent_dim), device=self.device)
                # NOTE I had to change this here. We don't want to make any assumptions about the base space (that we're conditioning on).
                # Instead of randomly choosing integers, we will randomly sample the base space
                base_pts = sample(list(self.base_data), k=len(bundle))
                sample_conditions = []
                for pt in base_pts:
                    idx = (self.centers - pt.unsqueeze(0)).norm(dim=1).argmin()
                    sample_conditions.append(torch.cat([self.centers[idx],pt]))
                sample_conditions = torch.stack(sample_conditions).to(self.device)

                # Generate a batch of images
                gen_imgs = self.G(z, sample_conditions)

                # Loss measures generator's ability to fool the discriminator
                validity = self.D(gen_imgs, sample_conditions)
                g_loss = adversarial_loss(validity, valid)

                g_loss.backward()
                epoch_G_loss += g_loss.item()
                optimizer_G.step()
                scheduler_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Loss for real images
                validity_real = self.D(real_imgs, conditions)
                d_real_loss = adversarial_loss(validity_real, valid)

                # Loss for fake images
                validity_fake = self.D(gen_imgs.detach(), conditions)
                d_fake_loss = adversarial_loss(validity_fake, fake)

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2

                d_loss.backward()
                epoch_D_loss += d_loss.item()
                optimizer_D.step()
                scheduler_D.step()
                
                # print our results
                print(f'Epoch {epoch:2d} || G Loss: {epoch_G_loss/(batch + 1):8.5f} || D Loss: {epoch_D_loss/(batch+1)}', end='\r')

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
    
    def sample_from_fiber(self, basept, n=1):
        idx = (self.centers - basept.unsqueeze(0)).norm(dim=1).argmin()
        condition = torch.cat([self.centers[idx],basept])
        if n == 1:
            # hack to fix the problem of only putting through a single point
            noise = torch.randn([2, self.latent_dim], device=self.device)
            condition = condition.unsqueeze(0).repeat(2,1).to(self.device)
            ret = self.G(noise, condition).detach().cpu()[0]
            return ret
        else:
            z = torch.randn((n, self.latent_dim), device=self.device)
            condition = condition.unsqueeze(0).repeat(n,1).to(self.device)
            ret = self.G(z, condition).detach().cpu()
            return ret

    class Generator(nn.Module):
        def __init__(self, latent_dim, data_dim, cond_dim):
            super().__init__()

            self.label_emb = nn.Linear(cond_dim, cond_dim)

            def block(in_feat, out_feat, normalize=True):
                layers = [nn.Linear(in_feat, out_feat)]
                if normalize:
                    layers.append(nn.BatchNorm1d(out_feat, 0.8))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return layers

            self.model = nn.Sequential(
                *block(latent_dim + cond_dim, 128, normalize=False),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, data_dim)
            )

        def forward(self, noise, cond):
            # Concatenate label embedding and image to produce input
            gen_input = torch.cat((self.label_emb(cond), noise), -1)
            return self.model(gen_input)


    class Discriminator(nn.Module):
        def __init__(self, data_dim, cond_dim):
            super().__init__()

            self.label_embedding = nn.Linear(cond_dim, cond_dim)

            self.model = nn.Sequential(
                nn.Linear(cond_dim + data_dim, 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 512),
                nn.Dropout(0.4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 512),
                nn.Dropout(0.4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 1),
            )

        def forward(self, data, cond):
            # Concatenate label embedding and image to produce input
            d_in = torch.cat((data, self.label_embedding(cond)), -1)
            validity = self.model(d_in)
            return validity
    
    def _create_nbhds(self):
        """An idea here: use k-means to find an approximate evenly-spaced cover. Then find the minimum distance to each center for each data point and set that as
        the minimum radius (so that each point is in a neighborhood). We may want to increase the radius from there so there is some overlap."""
        # Use sklearn's implementation of k-means
        kmeans = KMeans(self.num_nbhds).fit(self.base_data)
        return torch.tensor(kmeans.cluster_centers_).float()