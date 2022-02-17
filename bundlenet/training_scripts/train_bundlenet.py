import torch
import argparse
import json
import time

from bundlenet import BundleNet

parser = argparse.ArgumentParser(description='Train a BundleNet.')
parser.add_argument('--epochs', default=2000, type=int)
parser.add_argument('--batch_size', default=5, type=int)
parser.add_argument('--weights', default=[10,1,1,1], type=json.loads)
parser.add_argument('--num_circles', default=3, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--total_dim', default=20, type=int)
parser.add_argument('--device', default='cpu')
parser.add_argument('--num_nbhds', default=25, type=int)
parser.add_argument('--wt_dir', default=f'weights_{int(time.time())}')
parser.add_argument('--incompressible', action='store_true')
parser.add_argument('--no_condition', action='store_false')
parser.add_argument('--load_wt', default=None)
args = parser.parse_args()

# Change dataset here
base_points = torch.load('../../datasets/airfoil_base.pt')
bundle_points = torch.load('../../datasets/airfoil_bundle.pt')
    
model = BundleNet(base_points, bundle_points,
            fixed_nbhds=True,
            num_nbhds=args.num_nbhds,
            width=512,
            num_inv_blocks=5,
            nn_depth=5,
            incompressible=args.incompressible,
            device=args.device,
            condition=args.no_condition,
            total_dim=args.total_dim,
            num_circles=args.num_circles,
            convolutional=False
        )

model.train_net(
        lr=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        weights=args.weights,
        save_weights=False,
        wt_dir='../../saved_weights/' + args.wt_dir + f'-{args.num_circles}-circles',
        load_wt=args.load_wt,
        sample_size=400
    )
