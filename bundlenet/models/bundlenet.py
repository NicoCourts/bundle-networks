import torch
import os
import json
import time

import torch.nn as nn
import numpy as np
import FrEIA.framework as Ff
import FrEIA.modules as Fm

from sklearn.cluster import KMeans
from random import sample, choices

from bundlenet.util import iConditionalAffineTransform, BasicDataset, InvLocalAffines


class BundleNet(nn.Module):
    """A net that tries to assign local trivializations to our data."""
    def __init__(self, base_data, cover_data,
            fixed_nbhds=True,
            num_nbhds=None,
            width=512,
            num_inv_blocks=5,
            nn_depth=5,
            incompressible=False,
            device='cpu',
            condition=True,
            total_dim=None,
            num_circles=None,
            convolutional=False
            ):
        super().__init__()

        # Save our data for future use
        self.base_dim = base_data.reshape(base_data.shape[0],-1).shape[-1]
        self.cover_dim = cover_data.reshape(cover_data.shape[0],-1).shape[-1]
        
        if total_dim is None:
            self.total_dim = max(self.base_dim, self.cover_dim)
        else:
            self.total_dim = total_dim

        self.base_data = torch.zeros(base_data.shape[0], total_dim)
        self.base_data[:,:self.base_dim] = base_data.reshape(cover_data.shape[0], -1)
        self.cover_data = torch.zeros(cover_data.shape[0], total_dim)
        self.cover_data[:,:self.cover_dim] = cover_data.reshape(cover_data.shape[0], -1)

        # Split off the parameters in latent space between circles and intervals (1-D Gaussians)
        if num_circles is None:
            num_circles = (self.total_dim - self.base_dim) // 2
        if 2*num_circles > self.total_dim - self.base_dim:
            raise ValueError('The number of circles must not exceed the number of free parameters divided by two.')
        self.num_circles = num_circles
        self.num_intervals = self.total_dim - self.base_dim - 2*self.num_circles
        
        # Save parameters for creating the model
        self.width = width
        self.num_inv_blocks = num_inv_blocks
        self.nn_depth = nn_depth
        self.incompressible = incompressible

        # Compute fixed neighborhoods, if desired.
        self.fixed_nbhds = fixed_nbhds
        self.num_nbhds = num_nbhds
        if self.fixed_nbhds:
            self.centers, self.base_nbhds, self.cover_nbhds = self._create_nbhds()


        # Create model and move things to the chosen device
        self.device = device
        self.condition = condition
        self.convolutional = convolutional
        if self.convolutional:
            raise NotImplementedError('Convolutional INNs have not yet been added.')
        else:
            self.model = self._construct_net_()
        self.model.to(self.device)
    
    def train_net(self,
            lr=1e-4,
            num_epochs=100,
            batch_size=8,
            weights=[10., 1., 1., 1.],
            save_weights=False,
            wt_dir='run_'+time.ctime().replace(' ', '_'),
            load_wt=None,
            sample_size=None
        ):
        if load_wt is not None:
            self.model.load_state_dict(torch.load(wt_dir + '/' + load_wt))
            self.model.to(self.device)
        
        # Save everything!
        os.makedirs(wt_dir, exist_ok=True)
        torch.save(self.centers, wt_dir + '/centers.pt')
        torch.save(self.base_nbhds, wt_dir + '/base_nbhds.pt')
        torch.save(self.cover_nbhds, wt_dir + '/cover_nbhds.pt')

        # print out the parameters used for setup and training (possibly for easier loading later on)
        init_params = {
            'fixed_nbhds': self.fixed_nbhds,
            'num_nbhds': self.num_nbhds,
            'width': self.width,
            'num_inv_blocks': self.num_inv_blocks,
            'nn_depth': self.nn_depth,
            'incompressible': self.incompressible,
            'device': self.device,
            'condition': self.condition,
            'total_dim': self.total_dim,
            'num_circles': self.num_circles,
            'convolutional': self.convolutional
        }
        train_params = {
            'lr': lr,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'weights': weights,
            'save_weights': save_weights,
            'sample_size': sample_size
        }
        with open(wt_dir + '/train_parameters.json', 'w') as f:
            f.write(json.dumps(train_params))
        with open(wt_dir + '/init_parameters.json', 'w') as f:
            f.write(json.dumps(init_params))

        # Set up two optimizers, one for the model itself and one for the "f" net
        model_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=0
        )
        num_batches = num_epochs*(num_epochs//batch_size)
        scheduler = torch.optim.lr_scheduler.StepLR(model_optimizer, step_size=num_batches//18, gamma=0.5)

        if self.fixed_nbhds:
            loader = self._make_dataloader(
                list(enumerate(self.cover_nbhds)),
                self.base_nbhds, batch_size
                )
        else:
            raise NotImplementedError

        for epoch in range(num_epochs):
            # Track losses across each epoch for displaying during training
            epoch_MSE_loss, epoch_KL_Fwd_loss, epoch_KL_Bwd_loss, epoch_Dist_loss = 0,0,0,0
            for batch, (bundle, base) in enumerate(loader):
                self.model.train()
                model_optimizer.zero_grad()

                batchMSE = 0.
                batchKLFwd, batchKLBwd = 0., 0.
                batchDist = 0.
                
                for (nbhd_idx, cover), basept in zip(bundle, base):
                    basept = basept.to(self.device)
                    cover = cover.to(self.device)
                    idxs = list(range(basept.shape[0]))

                    if sample_size is not None:
                        l = min(sample_size, len(idxs))
                        idxs = sample(idxs, k=l)
                    if self.condition:
                        yhat, _ = self(
                            cover[idxs], 
                            c=self.centers[nbhd_idx].to(self.device).float().unsqueeze(0) #condition on center of neighborhood (local trivialization)
                        )
                    else:
                        yhat, _ = self(
                            cover[idxs],
                            c=[nbhd_idx]
                        )
                    if yhat.isnan().any():
                        print('Got nan out of the net')
                        print('batch',batch)
                        print('cover[idxs] shape', cover[idxs].shape)
                        print('cover[idxs] is nan', cover[idxs].isnan().any())
                        print('nbhd idx', nbhd_idx)
                        raise RuntimeError
                    
                    # MSE Loss -- Learn that we're over base points
                    MSELoss = (yhat[:,:self.base_dim] - basept[idxs,:self.base_dim]).norm(dim=1).mean()

                    # fiber loss 
                    sub_idxs = idxs
                    # we can add further downsampling here!)
                    #sub_idxs = choices(idxs, k=sample_size//5)
                    samples = basept[sub_idxs].unsqueeze(1).repeat(1, 5, 1).reshape(-1, self.total_dim)

                    stds = yhat[:,self.base_dim:].std(0, unbiased=True)
                    means = yhat[:,self.base_dim:].mean(0)

                    # Generate our parameters for latent space and attach them to our base points
                    params = []
                    for _ in range(len(samples)):
                        sample_circle_norms = yhat[:,self.base_dim:self.base_dim + 2*self.num_circles].reshape(yhat.shape[0], self.num_circles, 2).norm(dim=-1)
                        norm_means = sample_circle_norms.mean(0)
                        norm_stds = sample_circle_norms.std(0, unbiased=True)
                        circle_radii = norm_stds*torch.randn(self.num_circles, device=self.device) + norm_means
                        circle_radii = circle_radii.unsqueeze(1).repeat(1, 2)

                        param = stds*torch.randn([self.total_dim - self.base_dim], device=self.device) + means
                        angles = 2*np.pi*torch.rand([self.num_circles], device=self.device)
                        circle_params = (circle_radii*torch.stack([angles.cos(), angles.sin()], dim=1)).reshape(-1)
                        param[:len(circle_params)] = circle_params
                        params.append(param)

                    samples[:,self.base_dim:] = torch.stack(params)

                    if self.condition:
                        z_hat, _ = self(
                        samples,
                        c=self.centers[nbhd_idx].to(self.device).float().unsqueeze(0).repeat(samples.shape[0], 1),
                        rev=True
                    )
                    else:
                        z_hat, _ = self(
                        samples,
                        c=[nbhd_idx],
                        rev=True
                    )

                    # Different fiber losses
                    DistLoss = 0
                    for pt in z_hat:
                        # Just minimize distance to closest point in the fiber
                        DistLoss += (pt.unsqueeze(0).repeat(cover[idxs].shape[0], 1) - cover[idxs]).norm(dim=1).min()
                    DistLoss /= len(z_hat)
                    
                    KLLossBwd = self.KL_divergence(z_hat, cover[idxs], k=1)
                    KLLossFwd = self.KL_divergence(cover[idxs], z_hat, k=1)

                    batchMSE += MSELoss
                    batchKLBwd += KLLossBwd
                    batchKLFwd += KLLossFwd
                    batchDist += DistLoss

                    if (batchDist + batchKLBwd + batchKLFwd + batchMSE).isnan().any():
                        raise RuntimeError('Got a NaN during training. Relevant variables:', batchDist, batchKLBwd, batchKLFwd, batchMSE)

                # Total loss
                loss = weights[0]*batchMSE/batch_size + weights[1]*batchKLBwd/batch_size + weights[2]*batchKLFwd/batch_size + weights[3]*batchDist/batch_size
                try:
                    loss.backward(retain_graph=True)
                except RuntimeError:
                    print('\n Got a bad gradient \n')
                model_optimizer.step()
                scheduler.step()
                
                epoch_MSE_loss += batchMSE.item()/batch_size
                epoch_KL_Fwd_loss += batchKLFwd.item()/batch_size
                epoch_KL_Bwd_loss += batchKLBwd.item()/batch_size
                epoch_Dist_loss += batchDist.item()/batch_size
                
                print(f'Epoch {epoch:5} || MSE: {epoch_MSE_loss/(batch + 1):8.5f} || KL-Fwd: {epoch_KL_Fwd_loss/(batch + 1):8.5f} || KL-Bwd: {epoch_KL_Bwd_loss/(batch + 1):8.5f} || Dist: {epoch_Dist_loss/(batch + 1):8.5f}', end='\r')

                if epoch > 0 and epoch % 20 == 0 and save_weights:
                    torch.save(self.model.state_dict(), wt_dir + f'/epoch-{epoch}-weights.pt')
                
            print()
        
        torch.save(self.model.state_dict(), wt_dir + f'/final-weights.pt')
    
    def forward(self, X, c=None, rev=False):
        if c is None:
            if self.condition:
                raise ValueError('If using conditions, a condition (c) must be passed.')
            else:
                raise ValueError('If not using conditions, you must pass an index (idx) for the neighborhood we\'re computing.')
        return self.model(X, c=c, rev=rev)

    # Code adapted from the implementation at
    # https://github.com/nhartland/KL-divergence-estimators/blob/master/src/knn_divergence.py
    def KL_divergence(self, s1, s2, k=1):
        n, m = len(s1), len(s2)
        D = torch.tensor(m / (n - 1), dtype=torch.float, device=self.device).log()
        d = torch.tensor(s1.shape[1]).float().to(self.device)

        
        for pt in s1:
            # Estimate densities using the kth nearest neighbor. Idea from:
            # Qing Wang, Sanjeev R. Kulkarni, and Sergio Verdú. "Divergence estimation for multidimensional densities via k-nearest-neighbor distances." Information Theory, IEEE Transactions on 55.5 (2009): 2392-2405.
            norms = (s2-pt).norm(dim=1).reshape(-1)
            nu = norms[~(norms == 0)].kthvalue(k=k)[0]

            norms = (s1-pt).norm(dim=1).reshape(-1)
            rho = norms[~(norms == 0)].kthvalue(k=k)[0]

            D += (d/n)*(nu/rho).log()

            # If we're getting infs or nans something is wrong.
            if D.isinf():
                raise RuntimeError('KL-Divergence got a inf. Relevant variables:', nu, rho, d, n)
            if D.isnan():
                raise RuntimeError('KL-Divergence got a nan. Relevant variables:', nu, rho, d, n)
        return D
    
    def sample_from_fiber(self, basept, n=1, estimate_size=200):
        samples = torch.zeros([n, self.total_dim], device=self.device, dtype=torch.float) 
        samples[:, :self.base_dim] = basept.unsqueeze(0).repeat(n, 1)

        idx = (self.centers.to(self.device) - samples[0]).norm(dim=1).argmin()
        center = self.centers[idx].to(self.device)
        
        # sample parameter space so we know the right parameters to use for generating data
        size = min(estimate_size, len(self.cover_nbhds[idx]))
        pts = torch.stack(sample(list(self.cover_nbhds[idx]), k=size)).to(self.device)
        if self.condition:
            outs = self(pts, c=center.unsqueeze(0).repeat(pts.shape[0],1))[0]
        else:
            outs = self(pts, c=[idx])[0]

        stds = outs[:,self.base_dim:].std(0, unbiased=True)
        means = outs[:,self.base_dim:].mean(0)

        # Generate our parameters for latent space and attach them to our base points
        sample_circle_norms = outs[:,self.base_dim:self.base_dim + 2*self.num_circles].reshape(size, self.num_circles, 2).norm(dim=-1)
        norm_means = sample_circle_norms.mean(0)
        norm_stds = sample_circle_norms.std(0, unbiased=True)

        params = []
        for _ in range(n):
            circle_radii = norm_stds*torch.randn(self.num_circles, device=self.device) + norm_means
            circle_radii = circle_radii.unsqueeze(1).repeat(1, 2)

            param = stds*torch.randn([self.total_dim - self.base_dim], device=self.device) + means
            angles = 2*np.pi*torch.rand([self.num_circles], device=self.device)
            circle_params = (circle_radii*torch.stack([angles.cos(), angles.sin()], dim=1)).reshape(-1)
            param[:len(circle_params)] = circle_params
            params.append(param)

        samples[:,self.base_dim:] = torch.stack(params)

        if self.condition:
            outs = self(samples, c=center.unsqueeze(0).repeat(samples.shape[0],1), rev=True)[0].detach().cpu()
        else:
            outs = self(samples, c=[idx], rev=True)[0].detach().cpu()

        return outs[:,:self.cover_dim]
    
    def _make_dataloader(self, data, target, batch_size):
        dataset = BasicDataset(data, target)
        dataloader = torch.utils.data.DataLoader(dataset, collate_fn=lambda x:(samples for samples in zip(*x)), batch_size=batch_size, shuffle=True)
        return dataloader
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

    def _create_nbhds(self):
        """An idea here: use k-means to find an approximate evenly-spaced cover. Then find the minimum distance to each center for each data point and set that as
        the minimum radius (so that each point is in a neighborhood). We may want to increase the radius from there so there is some overlap."""
        # Use sklearn's implementation of k-means
        kmeans = KMeans(self.num_nbhds).fit(self.base_data)

        for s in range(10):
            # Just in case there is a problem with empty neighborhoods (there shouldn't be)
            try:
                # For now just let the neighborhoods be disjoint according to the clustering
                base_nbhds = [[] for _ in range(self.num_nbhds)]
                cover_nbhds = [[] for _ in range(self.num_nbhds)]
                for i, pt in enumerate(self.base_data):
                    base_nbhds[kmeans.labels_[i]].append(pt)
                    cover_nbhds[kmeans.labels_[i]].append(self.cover_data[i])
                
                # Stack neighborhoods together
                for i in range(self.num_nbhds):
                    base_nbhds[i] = torch.stack(base_nbhds[i]).float()
                    cover_nbhds[i] = torch.stack(cover_nbhds[i]).float()
                
                return torch.tensor(kmeans.cluster_centers_).float(), base_nbhds, cover_nbhds
            except Exception as inst:
                print(f"Issue generating neighborhoods. Trying again... ({s+1}/10)")
                print(inst)
                continue
        raise RuntimeError('Could not generate neighborhoods. Check that the number of neighborhoods is smaller than the number of points.')

    def _fc_constr(self, c_in, c_out):
        """Plug-n-play fully connected net for INNs"""
        layers = [nn.Linear(c_in, self.width), nn.ReLU()]
        for _ in range(self.nn_depth):
            layers.append(nn.Linear(self.width,  self.width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.width,  c_out))
        
        return nn.Sequential(*layers)
    
    def _conv_constr(self, c_in, c_out):
        """Plug-n-play convolutional net for INNs"""
        layers = [nn.Conv2d(c_in, self.width, 3, padding=1), nn.ReLU()]
        for _ in range(self.nn_depth):
            layers.append(nn.Conv2d(self.width, self.width, 3, padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(self.width, c_out, 3, padding=1))
        
        return nn.Sequential(*layers)
    
    def _construct_net_(self):
        """Constructs INN with conditional affine giving us 'local trivializations'."""
        in1 = Ff.InputNode(self.total_dim, name='Input')
        if self.condition:
            condition = Ff.ConditionNode(self.total_dim, name='Condition')
            local_triv = Ff.Node(
                [in1.out0],
                iConditionalAffineTransform if self.incompressible else Fm.ConditionalAffineTransform,
                {'subnet_constructor':self._fc_constr},
                conditions=condition,
                name='Conditional affine'
            )
            layers = [in1, condition, local_triv]
        else:
            condition = Ff.ConditionNode(1, name='Condition')
            affines = Ff.Node(
                [in1.out0],
                InvLocalAffines,
                {'num_nbhds':self.num_nbhds, 'total_dim':self.total_dim, 'device':self.device},
                conditions=condition,
                name='affines')
            layers = [in1, condition, affines]

        for i in range(self.num_inv_blocks):
            layers.append(Ff.Node(
                [layers[-1].out0],
                Fm.GINCouplingBlock if self.incompressible else Fm.RNVPCouplingBlock,
                {'subnet_constructor':self._fc_constr},
                name=f'RNVP {i}')
            )
            layers.append(Ff.Node([layers[-1].out0], Fm.PermuteRandom, {}))

        layers.append(Ff.OutputNode([layers[-1].out0], name='Output'))

        return Ff.ReversibleGraphNet(layers, verbose=False)
