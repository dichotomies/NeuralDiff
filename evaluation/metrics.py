import torch


def mse(image_pred, image_gt):
    value = (image_pred - image_gt) ** 2
    return torch.mean(value)


def psnr(image_pred, image_gt):
    return -10 * torch.log10(mse(image_pred, image_gt))
