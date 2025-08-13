# src/ml/datamodules/cifar10.py

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

class Cutout(object):
    def __init__(self, size=8):
        self.size = size
    def __call__(self, img):
        if self.size <= 0:
            return img
        h, w = img.size[1], img.size[0]
        y = torch.randint(0, h, (1,)).item()
        x = torch.randint(0, w, (1,)).item()
        y1 = max(0, y - self.size // 2); y2 = min(h, y + self.size // 2)
        x1 = max(0, x - self.size // 2); x2 = min(w, x + self.size // 2)
        img = transforms.functional.to_tensor(img)
        img[:, y1:y2, x1:x2] = 0
        return transforms.functional.to_pil_image(img)

def train_transform(img_size: int, randaug_m: int, randaug_n: int, cutout_size: int):
    tf = [
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if randaug_n > 0 and randaug_m > 0:
        tf.append(transforms.TrivialAugmentWide())  # 경량 대체 (속도↑), 필요 시 RandAugment로 교체 가능
        # from torchvision.transforms import RandAugment
        # tf.append(RandAugment(num_ops=randaug_n, magnitude=randaug_m))
    if cutout_size > 0:
        tf.append(Cutout(cutout_size))
    tf.extend([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return transforms.Compose(tf)

def test_transform(img_size: int):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

def make_dataloaders(root: str, batch_size: int, num_workers: int, img_size: int,
                     randaug_m: int, randaug_n: int, cutout_size: int, persistent_workers: bool = True):
    tr_tf = train_transform(img_size, randaug_m, randaug_n, cutout_size)
    te_tf = test_transform(img_size)
    train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=tr_tf)
    test_ds  = datasets.CIFAR10(root=root, train=False, download=True, transform=te_tf)
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers
    )
    test_dl  = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers
    )
    return train_dl, test_dl
