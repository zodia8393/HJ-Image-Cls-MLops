# src/ml/models/build_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# WideResNet-28-10 (CIFAR-10 전용) 구현
# 참고 구조: 28 layers, widen factor 10
# 간결 구현으로 유지(서빙/학습 일관성)
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = None if self.equalInOut else nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
            out = self.conv1(x)
        else:
            out = self.relu1(self.bn1(x))
            out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return out + (x if self.equalInOut else self.convShortcut(x))

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate):
        super().__init__()
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, drop_rate))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes, drop_rate=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0, "depth should be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nStages[0], nStages[1], BasicBlock, 1, drop_rate)
        self.block2 = NetworkBlock(n, nStages[1], nStages[2], BasicBlock, 2, drop_rate)
        self.block3 = NetworkBlock(n, nStages[2], nStages[3], BasicBlock, 2, drop_rate)
        self.bn = nn.BatchNorm2d(nStages[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nStages[3], num_classes)
        self.nStages = nStages

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)
        return self.fc(out)

def build_model(arch: str, num_classes: int, timm_name: str = None, pretrained: bool = False):
    # CIFAR-10 전용 WRN
    if arch == "wrn28_10_cifar":
        # WRN-28-10 + CIFAR-10: dropout은 보통 0~0.3 범위, 기본 0.0로 시작
        model = WideResNet(depth=28, widen_factor=10, num_classes=num_classes, drop_rate=0.0)
        return model

    # 기존 사용 아키텍처 fallback (원하면 유지)
    if arch == "resnet18":
        # torchvision resnet18를 32x32에 쓰려면 첫 conv/kernel/stride 조정 필요 → 추천하지 않음
        raise ValueError("Use wrn28_10_cifar for CIFAR-10. resnet18 is not configured for 32x32 here.")
    raise ValueError(f"Unsupported arch: {arch}")
