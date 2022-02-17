import torch
import argparse

from bundlenet import CGAN_triv

parser = argparse.ArgumentParser(description='Train a CGAN-triv.')
parser.add_argument('--epochs', default=2000, type=int)
parser.add_argument('--batch_size', default=5, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--latent_dim', default=10, type=int)
parser.add_argument('--device', default='cpu')
parser.add_argument('--wt_dir', default='CGAN_triv_sliced_torus')
args = parser.parse_args()

base_points = torch.load('../../datasets/sliced_torus_base.pt')
bundle_points = torch.load('../../datasets/sliced_torus_bundle.pt')
    
model = CGAN_triv(bundle_points, base_points,
            device=args.device,
            latent_dim=args.latent_dim,
            num_nbhds=25
        )

model.train_net(
        lr=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        save_weights=True,
        wt_dir='../../saved_weights/' + args.wt_dir
    )
