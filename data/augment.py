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

    def __call__(self, x, get_noise=False):
        # Parameters
        no, dim = x.shape
        # gen mask
        me = torch.zeros((dim))
        cov = torch.eye(dim) * self.eps
        mn = torch.distributions.MultivariateNormal(me, cov)
        x_bar = mn.sample((no,)).to(x.device)
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
        flip_n = torch.bernoulli(torch.ones(n, 1) * self.p)
        noflip_n = torch.ones(n, 1) - flip_n
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
                'random_resize': RandomResize}
    if augmenters == []:
        return None

    augment_list = []
    for augment_name in augmenters:
        augment = aug_dict[augment_name](augment_para[augment_name])
        augment_list.append(augment)
    return Transform(augment_list)


if __name__ == "__main__":
    x = torch.randn(5, 5)
    da = RandomFlip(0.5)
    print(x)
    print(da(x))
