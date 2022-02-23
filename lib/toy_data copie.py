# <Source: https://github.com/rtqichen/ffjord/blob/master/lib/toy_data.py >

import numpy as np
import sklearn
import torch
import sklearn.datasets
from PIL import Image
import os

# Dataset iterator
def inf_train_gen(data, rng=None, batch_size=200):
    if rng is None:
        rng = np.random.RandomState()
        #print(rng)

    if data == "swissroll":
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data

    elif data == "circles":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3
        return data

    elif data == "moons":
        data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        data = data.astype("float32")
        return data

    elif data == "8gaussians":
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset

    elif data == "conditionnal8gaussians":
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        context = np.zeros((batch_size, 8))
        for i in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            context[i, idx] = 1
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset, context

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations)).astype("float32")

    elif data == "2spirals":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        return x.astype("float32")

    elif data == "checkerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        return np.concatenate([x1[:, None], x2[:, None]], 1).astype("float32") * 2

    elif data == "line":
        x = rng.rand(batch_size)
        #x = np.arange(0., 1., 1/batch_size)
        x = x * 5 - 2.5
        y = x #- x + rng.rand(batch_size)
        return np.stack((x, y), 1).astype("float32")
    elif data == "line-noisy":
        x = rng.rand(batch_size)
        x = x * 5 - 2.5
        y = x + rng.randn(batch_size)
        return np.stack((x, y), 1).astype("float32")
    elif data == "cos":
        x = rng.rand(batch_size) * 5 - 2.5
        y = np.sin(x) * 2.5
        return np.stack((x, y), 1).astype("float32")

    elif data == "sin_xunif":
        a = 0.5; k = 3;
        x = rng.rand(batch_size) * 5 - 2.5
        yy = rng.rand(batch_size)
        y = yy - (a/(2*np.pi*k))*np.cos(2*np.pi*k*yy) + (a/(2*np.pi*k))
        y = y * 5 - 2.5
        return np.stack((x, y), 1).astype("float32")
    elif data == "sin_yunif":
        a = 0.5; k = 3;
        y = rng.rand(batch_size) * 5 - 2.5
        xx = rng.rand(batch_size)
        x = xx - (a/(2*np.pi*k))*np.cos(2*np.pi*k*xx) + (a/(2*np.pi*k))
        x = x * 5 - 2.5
        return np.stack((x, y), 1).astype("float32")
    elif data == "banana_xnorm":
        a = 0.99; k = 3;
        y = rng.rand(batch_size)
        x = y - (a/(2*np.pi*k))*np.cos(2*np.pi*k*y) + 1
        return np.stack((x, y), 1).astype("float32")
    elif data == "banana_ynorm":
        a = 0.99; k = 3;
        y = rng.rand(batch_size)
        x = y - (a/(2*np.pi*k))*np.cos(2*np.pi*k*y) + 1
        return np.stack((x, y), 1).astype("float32")
    elif data == "joint_gaussian1":
        ###x2 = torch.distributions.Normal(0., 4.).sample((batch_size, 1))
        ###x1 = torch.distributions.Normal(0., 1.).sample((batch_size, 1)) + (x2**2)/4
        x2 = torch.distributions.Normal(0., 1.).sample((batch_size, 1))
        x1 = torch.distributions.Normal(0., 0.5).sample((batch_size, 1)) + (x2**2)/2
        return torch.cat((x1, x2), 1)
    elif data == "joint_gaussian2":
        ###x1 = torch.distributions.Normal(0., 4.).sample((batch_size, 1))
        ###x2 = torch.distributions.Normal(0., 1.).sample((batch_size, 1)) + (x1**2)/4
        x1 = torch.distributions.Normal(0., 1.).sample((batch_size, 1))
        x2 = torch.distributions.Normal(0., 0.5).sample((batch_size, 1)) + (x1**2)/2
        return torch.cat((x1, x2), 1)
    elif data == 'mixture_1':
        norm_means = np.array([[-1, -1],[1,1],[1, -1],[-1, 1],[0,0]])
        weights = np.array([0.2,0.2,0.2,0.2,0.2])
        n_components = np.shape(weights)[0]
        mixture_idx = np.random.choice(n_components, size=batch_size, replace=True, p=weights)
        x = np.array([np.random.multivariate_normal(norm_means[i],0.3*np.identity(2)) for i in mixture_idx])
        x = torch.from_numpy(x)
        x = x.float()
        return x
    elif data == 'mixture_11':
        norm_means = np.array([[-1, -1],[1,1],[1, -1],[-1, 1],[0,0]])
        weights = np.array([0.2,0.2,0.2,0.2,0.2])
        n_components = np.shape(weights)[0]
        mixture_idx = np.random.choice(n_components, size=batch_size, replace=True, p=weights)
        x = np.array([np.random.multivariate_normal(norm_means[i],0.3*np.identity(2)) for i in mixture_idx])
        x = torch.from_numpy(x)
        x = x.float()
        x = x[:,[1,0]]
        return x
    
    elif data == 'mixture_2':
        norm_means = np.array([[-1, -1],[1,1],[1, -1],[-1, 1]])
        norm_var = [0.3,0.5,0.7,0.9]
        weights = np.array([0.2,0.2,0.2,0.4])
        n_components = np.shape(weights)[0]
        mixture_idx = np.random.choice(n_components, size=batch_size, replace=True, p=weights)
        x = np.array([np.random.multivariate_normal(norm_means[i],norm_var[i]*np.identity(2)) for i in mixture_idx])
        x = torch.from_numpy(x)
        x = x.float()
        return x
    elif data == 'mixture_22':
        norm_means = np.array([[-1, -1],[1,1],[1, -1],[-1, 1]])
        norm_var = [0.3,0.5,0.7,0.9]
        weights = np.array([0.2,0.2,0.2,0.4])
        n_components = np.shape(weights)[0]
        mixture_idx = np.random.choice(n_components, size=batch_size, replace=True, p=weights)
        x = np.array([np.random.multivariate_normal(norm_means[i],norm_var[i]*np.identity(2)) for i in mixture_idx])
        x = torch.from_numpy(x)
        x = x.float()
        x = x[:,[1,0]]
        return x
    elif data == 'mixture_3':
        norm_means = np.array([[-1, -1],[1,1],[1, -1],[-1, 1]])
        norm_var = [0.5,0.5,0.5,0.5]
        weights = np.array([0.25,0.25,0.25,0.25])
        n_components = np.shape(weights)[0]
        mixture_idx = np.random.choice(n_components, size=batch_size, replace=True, p=weights)
        x = np.array([np.random.multivariate_normal(norm_means[i],norm_var[i]*np.identity(2)) for i in mixture_idx])
        x = torch.from_numpy(x)
        x = x.float()
        return x
    elif data == 'mixture_33':
        norm_means = np.array([[-1, -1],[1,1],[1, -1],[-1, 1]])
        norm_var = [0.5,0.5,0.5,0.5]
        weights = np.array([0.25,0.25,0.25,0.25])
        n_components = np.shape(weights)[0]
        mixture_idx = np.random.choice(n_components, size=batch_size, replace=True, p=weights)
        x = np.array([np.random.multivariate_normal(norm_means[i],norm_var[i]*np.identity(2)) for i in mixture_idx])
        x = torch.from_numpy(x)
        x = x.float()
        x = x[:,[1,0]]
        return x



    ###else:
    ###    return inf_train_gen("8gaussians", rng, batch_size)
