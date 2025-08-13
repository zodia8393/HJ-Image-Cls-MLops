# src/scripts/download_data.py

import os
from pathlib import Path
from torchvision import datasets
from torchvision.transforms import ToTensor

def download_cifar10(root_dir: str):
    """
    CIFAR-10 공개 데이터셋 다운로드 및 저장
    """
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    
    # 다운받기만 하면 자동으로 raw 내부에 저장됨
    train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=ToTensor())
    test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=ToTensor())

    print(f"CIFAR-10 dataset downloaded to: {root.resolve()}")

def main():
    raw_data_path = os.path.join("src", "data", "raw")
    download_cifar10(raw_data_path)

if __name__ == "__main__":
    main()
