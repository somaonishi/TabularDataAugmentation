from typing import List

import numpy as np
import torch


class Transform:
    def __init__(self, transforms: list) -> None:
        self.transfroms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transfroms:
            x = transform(x)
        return x


def flip(x: torch.Tensor):
    return x.unsqueeze(0).fliplr().squeeze()


def mask_generator(x, p):
    """Generate mask vector.
    Args:
    - p_m: corruption probability
    - x: feature matrix
    Returns:
    - mask: binary mask matrix
    """
    mask = torch.bernoulli(torch.ones_like(x) * p)
    return mask.to(torch.float)


class Shuffle2:
    def __init__(self, x, p=0.3) -> None:
        self.p = p
        self.n, self.dim = x.shape
        self.x_bar = self.get_mask(x)

    def get_mask(self, x):
        x_bar = torch.zeros_like(x)
        for i in range(self.dim):
            idx = torch.randperm(self.n)
            x_bar[:, i] = x[idx, i]
        return x_bar

    def __call__(self, x, get_mask=False):
        """Generate corrupted samples.

        Args:
        m: mask matrix
        x: feature matrix

        Returns:
        m_new: final mask matrix after corruption
        x_tilde: corrupted feature matrix
        """
        m = mask_generator(x, self.p)

        n = len(x)
        np.random.randint(0, self.n, n)

        # Corrupt samples
        x_tilde = x * (1-m) + self.x_bar * m
        # Define new mask matrix
        m_new = 1 * (x != x_tilde)
        if get_mask:
            return x_tilde.to(torch.float), m_new.to(torch.float)
        else:
            return x_tilde.to(torch.float)


class Shuffle:
    def __init__(self, p=0.3) -> None:
        self.p = p

    def __call__(self, x, get_mask=False):
        """Generate corrupted samples.

        Args:
        m: mask matrix
        x: feature matrix

        Returns:
        m_new: final mask matrix after corruption
        x_tilde: corrupted feature matrix
        """
        m = mask_generator(x, self.p)
        # Parameters
        no, dim = x.shape
        # Randomly (and column-wise) shuffle data
        x_bar = torch.zeros_like(x)
        for i in range(dim):
            idx = torch.randperm(no)
            x_bar[:, i] = x[idx, i]

        # Corrupt samples
        x_tilde = x * (1-m) + x_bar * m
        # Define new mask matrix
        m_new = 1 * (x != x_tilde)
        if get_mask:
            return x_tilde.to(torch.float), m_new.to(torch.float)
        else:
            return x_tilde.to(torch.float)


class Noise:
    def __init__(self, eps=1.) -> None:
        self.eps = eps
        self.init = False

    def get_sampler(self, dim):
        self.init = True
        me = torch.zeros((dim))
        cov = torch.eye(dim) * self.eps
        self.sampler = torch.distributions.MultivariateNormal(me, cov)

    def __call__(self, x, get_noise=False):
        # Parameters
        no, dim = x.shape
        if not self.init:
            self.get_sampler(dim)

        # gen mask
        x_bar = self.sampler.sample((no,)).to(x.device)
        # Corrupt samples
        x_tilde = x + x_bar
        if get_noise:
            return x_tilde.to(torch.float), x_bar.to(torch.float)
        else:
            return x_tilde.to(torch.float)


class RandomFlip:
    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, x: torch.Tensor):
        n = len(x)
        flip_n = torch.bernoulli(torch.ones(n, 1) * self.p).to(x.device)
        noflip_n = torch.ones(n, 1).to(x.device) - flip_n
        x_flip = x.fliplr()
        return flip_n * x_flip + noflip_n * x


def add_dim(x: torch.Tensor):
    new_dim = torch.randn(len(x), device=x.device).view(-1, 1)
    new_x = torch.cat([x, new_dim], dim=1)
    return new_x


class RandomCollapse:
    def __init__(self, p=0.2) -> None:
        self.p = p

    def __call__(self, x: torch.Tensor):
        m = mask_generator(x, self.p)
        m = 1 - m
        return x * m


class Erasing:
    def __init__(self, erasing_rate=0.3) -> None:
        self.erasing_rate = erasing_rate
        self.init = False

    def get_all_mask(self, dim):
        self.init = True
        self.erasing_size = int(dim * self.erasing_rate)
        self.n_mask = dim - self.erasing_size + 1
        self.masks = torch.ones(self.n_mask, dim)
        for i in range(dim - self.erasing_size + 1):
            self.masks[i, i:i+self.erasing_size] = 0

    def __call__(self, x: torch.Tensor):
        n, dim = x.shape
        if not self.init:
            self.get_all_mask(dim)

        erasing_idx = torch.randint(0, self.n_mask, (n,))
        mask = self.masks[erasing_idx].to(x.device)
        return x * mask


class RandomErasing(Erasing):
    def __init__(self, p=0.5, erasing_rate=0.3) -> None:
        super().__init__(erasing_rate)
        self.p = p

    def get_mask(self, n):
        erasing_sample = torch.bernoulli(torch.ones(n, 1) * self.p)
        no_erasing_sample = torch.ones(n, 1) - erasing_sample
        erasing_idx = torch.randint(0, self.n_mask, (n,))
        mask = self.masks[erasing_idx]
        mask = mask * erasing_sample + no_erasing_sample
        return mask

    def __call__(self, x: torch.Tensor):
        n, dim = x.shape
        if not self.init:
            self.get_all_mask(dim)
        mask = self.get_mask(n).to(x.device)
        return x * mask


class RandomResize:
    def __init__(self, p=0.2) -> None:
        self.p = p

    def __call__(self, x: torch.Tensor):
        n = len(x)
        resize_n = torch.bernoulli(torch.ones(n, 1) * self.p)
        noresize_n = torch.ones(n, 1) - resize_n
        resize = torch.rand(n, 1) * resize_n + noresize_n
        resize = resize.to(x.device)
        return resize * x


def get_transforms(augment_para: dict, augmenters: List[str]):
    aug_dict = {'random_flip': RandomFlip,
                'noise': Noise,
                'random_collapse': RandomCollapse,
                'shuffle': Shuffle,
                'random_resize': RandomResize,
                'erasing': Erasing,
                'random_erasing': RandomErasing}
    if augmenters == []:
        return None

    augment_list = []
    for augment_name in augmenters:
        augment = aug_dict[augment_name](**augment_para[augment_name])
        augment_list.append(augment)
    return Transform(augment_list)


if __name__ == "__main__":
    x = torch.randn(5, 5)
    a = {'p': 0.5, 'erasing_rate': 0.4}
    da = RandomErasing(**a)
    print(x)
    print(da(x))
