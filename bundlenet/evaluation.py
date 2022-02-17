import torch
import numpy as np
from random import choice, choices
from bundlenet import BundleNet, CGAN, CGAN_nbhd, CGAN_triv
from scipy.stats import bootstrap
from .util import MMD, MSMD, KL_divergence, Wasserstein1, Wasserstein2

def evaluate_torus(model, r=2, R=8, global_sample_size=4000, fiber_sample_size=200, num_fibers=15, iterations=10):
    do_fiber = type(model) is BundleNet or type(model) is CGAN or type(model) is CGAN_nbhd or type(model) is CGAN_triv
    print(f'do_fiber={do_fiber}')
    def sample_point(phi=None):
        """ Rejection sampling to uniformly sample the torus """
        if phi is None:
            a,b,c = 0,0,0
            while a == 0 or c > (R+r*(2*np.pi*a).cos())/(R + r):
                a,b,c = torch.rand([3], dtype=torch.float)
            
            phi = 2*np.pi*a
        else:
            b = torch.rand([1], dtype=torch.float)
        theta = 2*np.pi*b

        return torch.tensor([
            (R + r*theta.cos())*phi.cos(),
            (R + r*theta.cos())*phi.sin(), 
            r*theta.sin()])
    
    # Global losses first
    global_losses = {'MMD':[], 'MSMD':[], 'KL-fwd':[], 'KL-bwd':[], 'Wass1':[], 'Wass2':[]}
    print('Global Losses: ', end='')
    for i in range(iterations):
        print(i, end=' ')
        global_sample= []
        for _ in range(global_sample_size):
            global_sample.append(sample_point())
        global_sample = torch.stack(global_sample)[:,:3].cpu()

        global_generated = []
        for _ in range(global_sample_size):
            angle = 2*np.pi*torch.rand([1])
            basept = 4.5*torch.tensor([angle.cos(), angle.sin()], dtype=torch.float)
            global_generated.append(model.sample_from_fiber(basept=basept))
        global_generated = torch.stack(global_generated).squeeze()[:,:3].cpu()

        global_losses['MMD'].append(MMD(global_sample, global_generated))
        global_losses['MSMD'].append(MSMD(global_sample, global_generated))
        global_losses['KL-fwd'].append(KL_divergence(global_sample, global_generated))
        global_losses['KL-bwd'].append(KL_divergence(global_generated, global_sample))
        global_losses['Wass1'].append(Wasserstein1(global_sample, global_generated))
        global_losses['Wass2'].append(Wasserstein2(global_sample, global_generated))
    
    if do_fiber:
        # Fiber losses
        # num_iterations times: choose num_fibers base points, compute metrics, then average over chosen fibers
        fiber_losses = {'MMD':[], 'MSMD':[], 'KL-fwd':[], 'KL-bwd':[], 'Wass1':[], 'Wass2':[]}
        print('\nFiber Losses ', end='')
        for i in range(iterations):
            print(i, end=' ')
            iteration_losses = {'MMD':[], 'MSMD':[], 'KL-fwd':[], 'KL-bwd':[], 'Wass1':[], 'Wass2':[]}
            for _ in range(num_fibers):
                angle = 2*np.pi*torch.rand([1])
                basept = 4.5*torch.tensor([np.cos(angle), np.sin(angle)])

                fiber_generated = model.sample_from_fiber(basept=basept, n=fiber_sample_size)
                
                fiber_sample = []
                for _ in range(fiber_sample_size):
                    fiber_sample.append(sample_point(phi=angle))
                fiber_sample = torch.stack(fiber_sample)

                iteration_losses['MMD'].append(MMD(fiber_sample, fiber_generated))
                iteration_losses['MSMD'].append(MSMD(fiber_sample, fiber_generated))
                iteration_losses['KL-fwd'].append(KL_divergence(fiber_sample, fiber_generated))
                iteration_losses['KL-bwd'].append(KL_divergence(fiber_generated, fiber_sample))
                iteration_losses['Wass1'].append(Wasserstein1(fiber_sample, fiber_generated))
                iteration_losses['Wass2'].append(Wasserstein2(fiber_sample, fiber_generated))

            for key in iteration_losses.keys():
                fiber_losses[key].append(sum(iteration_losses[key])/len(iteration_losses[key]))
    
    global_outs, fiber_outs = {}, {}
    for key in global_losses.keys():
        global_outs[key] = (
            np.mean(global_losses[key]),
            bootstrap(torch.stack(global_losses[key]).reshape(1,-1), np.mean, confidence_level=0.95)
        )
        if do_fiber:
            fiber_outs[key] = (
                np.mean(fiber_losses[key]),
                bootstrap(torch.stack(fiber_losses[key]).reshape(1,-1), np.mean, confidence_level=0.95)
            )
    
    return {'global': global_outs, 'fiber': fiber_outs}

def evaluate_sliced_torus(model, r=2, R=8, global_sample_size=4000, fiber_sample_size=200, num_fibers=15, iterations=10):
    do_fiber = type(model) is BundleNet or type(model) is CGAN or type(model) is CGAN_nbhd or type(model) is CGAN_triv
    def sample_point(phi=None):
        a,b = torch.rand([2], dtype=torch.float)
        if phi is not None:
            a = phi/(2*np.pi)
        while (b-0.5).abs() - 0.01 > (a-0.5).abs():
            if phi is not None:
                b = torch.rand([1], dtype=torch.float)
            else:
                a,b = torch.rand([2], dtype=torch.float)
        phi = 2*np.pi*a
        theta = 2*np.pi*b

        return torch.tensor([
            (R + r*theta.cos())*phi.cos(),
            (R + r*theta.cos())*phi.sin(), 
            r*theta.sin()])
    
    # Global losses first
    global_losses = {'MMD':[], 'MSMD':[], 'KL-fwd':[], 'KL-bwd':[], 'Wass1':[], 'Wass2':[]}
    print('Global Losses: ', end='')
    for i in range(iterations):
        print(i, end=' ')
        global_sample= []
        for _ in range(global_sample_size):
            global_sample.append(sample_point())
        global_sample = torch.stack(global_sample)[:,:3].cpu()

        global_generated = []
        for _ in range(global_sample_size):
            angle = 2*np.pi*torch.rand([1])
            basept = 4.5*torch.tensor([angle.cos(), angle.sin()], dtype=torch.float)
            global_generated.append(model.sample_from_fiber(basept=basept))
        global_generated = torch.stack(global_generated).squeeze()[:,:3].cpu()

        global_losses['MMD'].append(MMD(global_sample, global_generated))
        global_losses['MSMD'].append(MSMD(global_sample, global_generated))
        global_losses['KL-fwd'].append(KL_divergence(global_sample, global_generated))
        global_losses['KL-bwd'].append(KL_divergence(global_generated, global_sample))
        global_losses['Wass1'].append(Wasserstein1(global_sample, global_generated))
        global_losses['Wass2'].append(Wasserstein2(global_sample, global_generated))
    
    if do_fiber:
        # Fiber losses
        # num_iterations times: choose num_fibers base points, compute metrics, then average over chosen fibers
        fiber_losses = {'MMD':[], 'MSMD':[], 'KL-fwd':[], 'KL-bwd':[], 'Wass1':[], 'Wass2':[]}
        print('\nFiber Losses ', end='')
        for i in range(iterations):
            print(i, end=' ')
            iteration_losses = {'MMD':[], 'MSMD':[], 'KL-fwd':[], 'KL-bwd':[], 'Wass1':[], 'Wass2':[]}
            for _ in range(num_fibers):
                angle = 2*np.pi*torch.rand([1])
                basept = 4.5*torch.tensor([np.cos(angle), np.sin(angle)])

                fiber_generated = model.sample_from_fiber(basept=basept, n=fiber_sample_size)
                
                fiber_sample = []
                for _ in range(fiber_sample_size):
                    fiber_sample.append(sample_point(phi=angle))
                fiber_sample = torch.stack(fiber_sample)

                iteration_losses['MMD'].append(MMD(fiber_sample, fiber_generated))
                iteration_losses['MSMD'].append(MSMD(fiber_sample, fiber_generated))
                iteration_losses['KL-fwd'].append(KL_divergence(fiber_sample, fiber_generated))
                iteration_losses['KL-bwd'].append(KL_divergence(fiber_generated, fiber_sample))
                iteration_losses['Wass1'].append(Wasserstein1(fiber_sample, fiber_generated))
                iteration_losses['Wass2'].append(Wasserstein2(fiber_sample, fiber_generated))

            for key in iteration_losses.keys():
                fiber_losses[key].append(sum(iteration_losses[key])/len(iteration_losses[key]))
    
    global_outs, fiber_outs = {}, {}
    for key in global_losses.keys():
        global_outs[key] = (
            np.mean(global_losses[key]),
            bootstrap(torch.stack(global_losses[key]).reshape(1,-1), np.mean, confidence_level=0.95)
        )
        if do_fiber:
            fiber_outs[key] = (
                np.mean(fiber_losses[key]),
                bootstrap(torch.stack(fiber_losses[key]).reshape(1,-1), np.mean, confidence_level=0.95)
            )
    
    return {'global': global_outs, 'fiber': fiber_outs}

def evaluate_moebius(model, fiber_width=2, R=8, global_sample_size=4000, fiber_sample_size=200, num_fibers=15, iterations=10):
    do_fiber = type(model) is BundleNet or type(model) is CGAN or type(model) is CGAN_nbhd or type(model) is CGAN_triv

    def sample_point(phi=None):
        s,t = torch.rand(2)
        s *= 2*np.pi
        if phi is not None:
            s = phi
        t = fiber_width*(2*t - 1)
        
        # Angle should be zero when s=0 and pi when s=2pi
        angle = s/2
        
        # Time for some calc 3! First find the tangent vector at the point
        #  This defines the plane that the fiber will lie on.
        tangent = torch.tensor([-np.sin(s), np.cos(s),0])
        # Remember the normal always points towards the center! Or just take the derivative again.
        normal = torch.tensor([-np.cos(s), -np.sin(s), 0])
        # Hack: we already know the third direction. Or cross product and normalize.
        binormal = torch.tensor([0,0,1.])
        # Change of basis and rotation matrices
        A = torch.stack([normal, binormal, tangent]).T
        rot = torch.tensor(
            [[np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,              0,             1]
            ])
        B = A @ rot @ A.inverse()
        
        # We did it! Just add the proper vector
        fiber_pt = torch.tensor([R*np.cos(s), R*np.sin(s), 0.]) + t*(B @ normal.T)

        return fiber_pt
    
    # Global losses first
    global_losses = {'MMD':[], 'MSMD':[], 'KL-fwd':[], 'KL-bwd':[], 'Wass1':[], 'Wass2':[]}
    print('Global Losses: ', end='')
    for i in range(iterations):
        print(i, end=' ')
        global_sample= []
        for _ in range(global_sample_size):
            global_sample.append(sample_point())
        global_sample = torch.stack(global_sample)[:,:3].cpu()

        global_generated = []
        for _ in range(global_sample_size):
            angle = 2*np.pi*torch.rand([1])
            basept = 4.5*torch.tensor([angle.cos(), angle.sin()], dtype=torch.float)
            global_generated.append(model.sample_from_fiber(basept=basept))
        global_generated = torch.stack(global_generated).squeeze()[:,:3].cpu()

        global_losses['MMD'].append(MMD(global_sample, global_generated))
        global_losses['MSMD'].append(MSMD(global_sample, global_generated))
        global_losses['KL-fwd'].append(KL_divergence(global_sample, global_generated))
        global_losses['KL-bwd'].append(KL_divergence(global_generated, global_sample))
        global_losses['Wass1'].append(Wasserstein1(global_sample, global_generated))
        global_losses['Wass2'].append(Wasserstein2(global_sample, global_generated))
    
    if do_fiber:
        # Fiber losses
        # num_iterations times: choose num_fibers base points, compute metrics, then average over chosen fibers
        fiber_losses = {'MMD':[], 'MSMD':[], 'KL-fwd':[], 'KL-bwd':[], 'Wass1':[], 'Wass2':[]}
        print('\nFiber Losses ', end='')
        for i in range(iterations):
            print(i, end=' ')
            iteration_losses = {'MMD':[], 'MSMD':[], 'KL-fwd':[], 'KL-bwd':[], 'Wass1':[], 'Wass2':[]}
            for _ in range(num_fibers):
                angle = 2*np.pi*torch.rand([1])
                basept = 4.5*torch.tensor([np.cos(angle), np.sin(angle)])

                fiber_generated = model.sample_from_fiber(basept=basept, n=fiber_sample_size)[:,:3]
                
                fiber_sample = []
                for _ in range(fiber_sample_size):
                    fiber_sample.append(sample_point(phi=angle))
                fiber_sample = torch.stack(fiber_sample)[:,:3]

                iteration_losses['MMD'].append(MMD(fiber_sample, fiber_generated))
                iteration_losses['MSMD'].append(MSMD(fiber_sample, fiber_generated))
                iteration_losses['KL-fwd'].append(KL_divergence(fiber_sample, fiber_generated))
                iteration_losses['KL-bwd'].append(KL_divergence(fiber_generated, fiber_sample))
                iteration_losses['Wass1'].append(Wasserstein1(fiber_sample, fiber_generated))
                iteration_losses['Wass2'].append(Wasserstein2(fiber_sample, fiber_generated))

            for key in iteration_losses.keys():
                fiber_losses[key].append(sum(iteration_losses[key])/len(iteration_losses[key]))
    
    global_outs, fiber_outs = {}, {}
    for key in global_losses.keys():
        global_outs[key] = (
            np.mean(global_losses[key]),
            bootstrap(torch.stack(global_losses[key]).reshape(1,-1), np.mean, confidence_level=0.95)
        )
        if do_fiber:
            fiber_outs[key] = (
                np.mean(fiber_losses[key]),
                bootstrap(torch.stack(fiber_losses[key]).reshape(1,-1), np.mean, confidence_level=0.95)
            )
    
    return {'global': global_outs, 'fiber': fiber_outs}

def evaluate_wine(model, global_sample_size=4000, fiber_sample_size=200, num_fibers=4, iterations=10):
    do_fiber = type(model) is BundleNet or type(model) is CGAN or type(model) is CGAN_nbhd or type(model) is CGAN_triv

    train_pts = torch.load('../datasets/wine_bundle.pt')
    test_pts = torch.load('../datasets/wine_bundle_test.pt')
    cover_pts = torch.cat([train_pts, test_pts], dim=0)

    train_pts = torch.load('../datasets/wine_base.pt')
    test_pts = torch.load('../datasets/wine_base_test.pt')
    base_pts = torch.cat([train_pts, test_pts], dim=0)
    
    global_sample_size = min(int(len(cover_pts)*0.75), global_sample_size)

    good_basepts = base_pts.unique(dim=0)

    pts = [[] for _ in good_basepts]
    for i, coverpt in enumerate(cover_pts):
        idx = (base_pts[i].unsqueeze(0) - good_basepts).norm(dim=1).argmin()
        pts[idx].append(coverpt)
    
    good_basepts = [pt for nbhd, pt in zip(pts, good_basepts) if len(nbhd)>2]

    def sample_point(idx=None):
        if idx is None:
            return choice(list(cover_pts))
        return choice(pts[idx])
    
    # Global losses first
    global_losses = {'MMD':[], 'MSMD':[], 'KL-fwd':[], 'KL-bwd':[], 'Wass1':[], 'Wass2':[]}
    print('Global Losses: ', end='')
    for i in range(iterations):
        print(i, end=' ')
        global_sample= []
        for _ in range(global_sample_size):
            global_sample.append(sample_point())
        global_sample = torch.stack(global_sample).cpu()

        global_generated = []
        for _ in range(global_sample_size):
            basept= choice(base_pts.unique(dim=0))
            global_generated.append(model.sample_from_fiber(basept=basept))
        global_generated = torch.stack(global_generated).squeeze().cpu()

        global_losses['MMD'].append(MMD(global_sample, global_generated))
        global_losses['MSMD'].append(MSMD(global_sample, global_generated))
        global_losses['KL-fwd'].append(KL_divergence(global_sample, global_generated))
        global_losses['KL-bwd'].append(KL_divergence(global_generated, global_sample))
        global_losses['Wass1'].append(Wasserstein1(global_sample, global_generated))
        global_losses['Wass2'].append(Wasserstein2(global_sample, global_generated))
    
    if do_fiber:
        # Fiber losses
        # num_iterations times: choose num_fibers base points, compute metrics, then average over chosen fibers
        fiber_losses = {'MMD':[], 'MSMD':[], 'KL-fwd':[], 'KL-bwd':[], 'Wass1':[], 'Wass2':[]}
        print('\nFiber Losses ', end='')
        for i in range(iterations):
            print(i, end=' ')
            iteration_losses = {'MMD':[], 'MSMD':[], 'KL-fwd':[], 'KL-bwd':[], 'Wass1':[], 'Wass2':[]}
            for _ in range(num_fibers):
                idx = choice(list(range(len(good_basepts))))
                basept = good_basepts[idx]

                round_sample_size = min(len(pts[idx]), fiber_sample_size)

                fiber_generated = model.sample_from_fiber(basept=basept, n=round_sample_size)
                
                fiber_sample = []
                for _ in range(round_sample_size):
                    fiber_sample.append(sample_point(idx=idx))
                fiber_sample = torch.stack(fiber_sample)

                iteration_losses['MMD'].append(MMD(fiber_sample, fiber_generated))
                iteration_losses['MSMD'].append(MSMD(fiber_sample, fiber_generated))
                iteration_losses['KL-fwd'].append(KL_divergence(fiber_sample, fiber_generated))
                iteration_losses['KL-bwd'].append(KL_divergence(fiber_generated, fiber_sample))
                iteration_losses['Wass1'].append(Wasserstein1(fiber_sample, fiber_generated))
                iteration_losses['Wass2'].append(Wasserstein2(fiber_sample, fiber_generated))

            for key in iteration_losses.keys():
                fiber_losses[key].append(sum(iteration_losses[key])/len(iteration_losses[key]))
    
    global_outs, fiber_outs = {}, {}
    for key in global_losses.keys():
        global_outs[key] = (
            np.mean(global_losses[key]),
            bootstrap(torch.stack(global_losses[key]).reshape(1,-1), np.mean, confidence_level=0.95)
        )
        if do_fiber:
            fiber_outs[key] = (
                np.mean(fiber_losses[key]),
                bootstrap(torch.stack(fiber_losses[key]).reshape(1,-1), np.mean, confidence_level=0.95)
            )
    
    return {'global': global_outs, 'fiber': fiber_outs}

def evaluate_airfoil(model, global_sample_size=4000, fiber_sample_size=200, num_fibers=15, iterations=10):
    do_fiber = type(model) is BundleNet or type(model) is CGAN or type(model) is CGAN_nbhd or type(model) is CGAN_triv

    train_pts = torch.load('../datasets/airfoil_bundle.pt')
    test_pts = torch.load('../datasets/airfoil_bundle_test.pt')
    cover_pts = torch.cat([train_pts, test_pts], dim=0)

    train_pts = torch.load('../datasets/airfoil_base.pt')
    test_pts = torch.load('../datasets/airfoil_base_test.pt')
    base_pts = torch.cat([train_pts, test_pts], dim=0)

    def sample_point(basept=None, K=15):
        if basept is None:
            return choice(list(cover_pts))
        
        # sample randomly from closest K points
        # Extra shuffle to the points in case there is a tie in distances. I want the tie to be broken randomly instead of sequentially
        shuffle_idxs = choices(list(range(len(cover_pts))), k=len(cover_pts))
        
        nbhd = cover_pts[shuffle_idxs][(base_pts[shuffle_idxs] - basept.unsqueeze(0)).norm(dim=1).topk(k=K, largest=False).indices]
        return choice(list(nbhd))
    
    # Global losses first
    global_losses = {'MMD':[], 'MSMD':[], 'KL-fwd':[], 'KL-bwd':[], 'Wass1':[], 'Wass2':[]}
    print('Global Losses: ', end='')
    for i in range(iterations):
        print(i, end=' ')
        global_sample = torch.stack(choices(list(cover_pts), k=global_sample_size))

        global_generated = []
        for _ in range(global_sample_size):
            basept = choice(list(base_pts))
            global_generated.append(model.sample_from_fiber(basept=basept))
        global_generated = torch.stack(global_generated).squeeze().cpu()[:,:global_sample.shape[-1]]

        global_losses['MMD'].append(MMD(global_sample, global_generated))
        global_losses['MSMD'].append(MSMD(global_sample, global_generated))
        global_losses['KL-fwd'].append(KL_divergence(global_sample, global_generated))
        global_losses['KL-bwd'].append(KL_divergence(global_generated, global_sample))
        global_losses['Wass1'].append(Wasserstein1(global_sample, global_generated))
        global_losses['Wass2'].append(Wasserstein2(global_sample, global_generated))
    
    if do_fiber:
        # Fiber losses
        # num_iterations times: choose num_fibers base points, compute metrics, then average over chosen fibers
        fiber_losses = {'MMD':[], 'MSMD':[], 'KL-fwd':[], 'KL-bwd':[], 'Wass1':[], 'Wass2':[]}
        print('\nFiber Losses ', end='')
        for i in range(iterations):
            print(i, end=' ')
            iteration_losses = {'MMD':[], 'MSMD':[], 'KL-fwd':[], 'KL-bwd':[], 'Wass1':[], 'Wass2':[]}
            for _ in range(num_fibers):
                basept = choice(list(base_pts))

                fiber_generated = model.sample_from_fiber(basept=basept, n=fiber_sample_size)
                
                fiber_sample = []
                for _ in range(fiber_sample_size):
                    fiber_sample.append(sample_point(basept=basept))
                fiber_sample = torch.stack(fiber_sample)

                iteration_losses['MMD'].append(MMD(fiber_sample, fiber_generated))
                iteration_losses['MSMD'].append(MSMD(fiber_sample, fiber_generated))
                iteration_losses['KL-fwd'].append(KL_divergence(fiber_sample, fiber_generated))
                iteration_losses['KL-bwd'].append(KL_divergence(fiber_generated, fiber_sample))
                iteration_losses['Wass1'].append(Wasserstein1(fiber_sample, fiber_generated))
                iteration_losses['Wass2'].append(Wasserstein2(fiber_sample, fiber_generated))

            for key in iteration_losses.keys():
                fiber_losses[key].append(sum(iteration_losses[key])/len(iteration_losses[key]))
    
    global_outs, fiber_outs = {}, {}
    for key in global_losses.keys():
        global_outs[key] = (
            np.mean(global_losses[key]),
            bootstrap(torch.stack(global_losses[key]).reshape(1,-1), np.mean, confidence_level=0.95)
        )
        if do_fiber:
            fiber_outs[key] = (
                np.mean(fiber_losses[key]),
                bootstrap(torch.stack(fiber_losses[key]).reshape(1,-1), np.mean, confidence_level=0.95)
            )
    
    return {'global': global_outs, 'fiber': fiber_outs}

def evaluate_airfoil_minimal(model, global_sample_size=4000, fiber_sample_size=200, num_fibers=15, iterations=10):
    train_pts = torch.load('../datasets/airfoil_bundle.pt')
    test_pts = torch.load('../datasets/airfoil_bundle_test.pt')
    cover_pts = torch.cat([train_pts, test_pts], dim=0)

    train_pts = torch.load('../datasets/airfoil_base.pt')
    test_pts = torch.load('../datasets/airfoil_base_test.pt')
    base_pts = torch.cat([train_pts, test_pts], dim=0)

    def sample_point(basept=None, K=15):
        if basept is None:
            return choice(list(cover_pts))
        
        # sample randomly from closest K points
        # Extra shuffle to the points in case there is a tie in distances. I want the tie to be broken randomly instead of sequentially
        shuffle_idxs = choices(list(range(len(cover_pts))), k=len(cover_pts))
        nbhd = cover_pts[shuffle_idxs][(base_pts[shuffle_idxs] - basept.unsqueeze(0)).norm(dim=1).topk(k=K, largest=False).indices]
        return choice(list(nbhd))
    
    # Global losses first
    global_losses = {'Wass1':[]}
    print('Global Losses: ', end='')
    for i in range(iterations):
        print(i, end=' ')
        global_sample = torch.stack(choices(list(cover_pts), k=global_sample_size))

        global_generated = []
        for _ in range(global_sample_size):
            basept = choice(list(base_pts))
            global_generated.append(model.sample_from_fiber(basept=basept))
        global_generated = torch.stack(global_generated).squeeze().cpu()[:,:global_sample.shape[-1]]

        global_losses['Wass1'].append(Wasserstein1(global_sample, global_generated))
    
    # Fiber losses
    # num_iterations times: choose num_fibers base points, compute metrics, then average over chosen fibers
    fiber_losses = {'Wass1':[]}
    print('\nFiber Losses ', end='')
    for i in range(iterations):
        print(i, end=' ')
        iteration_losses = {'Wass1':[]}
        for _ in range(num_fibers):
            basept = choice(list(base_pts))

            fiber_generated = model.sample_from_fiber(basept=basept, n=fiber_sample_size)
            
            fiber_sample = []
            for _ in range(fiber_sample_size):
                fiber_sample.append(sample_point(basept=basept))
            fiber_sample = torch.stack(fiber_sample)

            iteration_losses['Wass1'].append(Wasserstein1(fiber_sample, fiber_generated))

        for key in ['Wass1']:
            fiber_losses[key].append(sum(iteration_losses[key])/len(iteration_losses[key]))
    
    global_outs, fiber_outs = {}, {}
    for key in global_losses.keys():
        global_outs[key] = (
            np.mean(global_losses[key]),
            bootstrap(torch.stack(global_losses[key]).reshape(1,-1), np.mean, confidence_level=0.95)
        )
        fiber_outs[key] = (
            np.mean(fiber_losses[key]),
            bootstrap(torch.stack(fiber_losses[key]).reshape(1,-1), np.mean, confidence_level=0.95)
        )
    
    return {'global': global_outs, 'fiber': fiber_outs}