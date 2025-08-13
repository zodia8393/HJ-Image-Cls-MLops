# src/scripts/train.py

import os, json, random, numpy as np, yaml
from pathlib import Path
import torch
from torch import nn
from torch.optim import SGD
from torch.cuda.amp import GradScaler, autocast
from src.ml.datamodules.cifar10 import make_dataloaders
from src.ml.models.build_model import build_model
from src.ml.utils.mlflow_utils import set_tracking, log_params, log_metrics, register_model
import mlflow, mlflow.pytorch

def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def accuracy(output, target):
    pred = output.argmax(dim=1)
    return (pred == target).float().mean().item()

class WarmupCosine:
    def __init__(self, optimizer, total_steps, warmup_steps, base_lr, min_lr):
        self.opt = optimizer
        self.total = total_steps
        self.warmup = warmup_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.t = 0
    def step(self):
        self.t += 1
        if self.t < self.warmup:
            lr = self.base_lr * self.t / max(1, self.warmup)
        else:
            progress = (self.t - self.warmup) / max(1, self.total - self.warmup)
            lr = self.min_lr + 0.5*(self.base_lr - self.min_lr)*(1+np.cos(np.pi*progress))
        for pg in self.opt.param_groups:
            pg['lr'] = lr

def main():
    with open("params.yaml","r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    set_seed(int(params["seed"]))

    data_cfg = params["data"]; train_cfg = params["train"]; model_cfg = params["model"]; aug_cfg = params["aug"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_tracking()

    with mlflow.start_run():
        log_params({**train_cfg, **model_cfg, **aug_cfg, "dataset": data_cfg["name"], "device": device})

        train_dl, val_dl = make_dataloaders(
            root=data_cfg["root"]+"/raw",
            batch_size=train_cfg["batch_size"],
            num_workers=data_cfg["num_workers"],
            img_size=model_cfg["img_size"],
            randaug_m=aug_cfg["randaug_m"],
            randaug_n=aug_cfg["randaug_n"],
            cutout_size=aug_cfg["cutout"],
            persistent_workers=bool(data_cfg.get("persistent_workers", True))
        )

        model = build_model(
            arch=model_cfg["arch"],
            num_classes=model_cfg["num_classes"],
            timm_name=model_cfg.get("timm_name"),
            pretrained=bool(model_cfg.get("pretrained", False))
        ).to(device)

        # Label smoothing
        ls = float(train_cfg.get("label_smoothing", 0.0))
        criterion = nn.CrossEntropyLoss(label_smoothing=ls) if ls > 0 else nn.CrossEntropyLoss()

        optim = SGD(
            model.parameters(),
            lr=float(train_cfg["lr"]),
            momentum=float(train_cfg.get("momentum", 0.9)),
            weight_decay=float(train_cfg["weight_decay"]),
            nesterov=bool(train_cfg.get("nesterov", True))
        )

        epochs = int(train_cfg["epochs"])
        iters_per_epoch = len(train_dl)
        total_steps = epochs * iters_per_epoch
        warmup_steps = int(train_cfg.get("warmup_epochs", 0)) * iters_per_epoch
        scheduler = WarmupCosine(
            optim,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            base_lr=float(train_cfg["lr"]),
            min_lr=float(train_cfg["lr"])*1e-2
        )

        scaler = GradScaler(enabled=bool(train_cfg.get("amp", True)))
        best_acc = 0.0
        step = 0

        for epoch in range(epochs):
            model.train()
            tr_loss, tr_acc_sum, n = 0.0, 0.0, 0
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optim.zero_grad()
                with autocast(enabled=bool(train_cfg.get("amp", True))):
                    out = model(xb)
                    loss = criterion(out, yb)
                scaler.scale(loss).backward()
                if train_cfg.get("grad_clip", 0.0) > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
                scaler.step(optim)
                scaler.update()
                scheduler.step()
                bs = xb.size(0)
                tr_loss += loss.item() * bs
                tr_acc_sum += accuracy(out, yb) * bs
                n += bs
                step += 1

            tr_loss /= n
            tr_acc = tr_acc_sum / n

            model.eval()
            val_acc, m = 0.0, 0
            with torch.no_grad(), autocast(enabled=bool(train_cfg.get("amp", True))):
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    val_acc += accuracy(out, yb) * xb.size(0)
                    m += xb.size(0)
            val_acc /= m

            log_metrics({"train_loss": tr_loss, "train_acc": tr_acc, "val_acc": val_acc}, step=epoch)
            if val_acc > best_acc: best_acc = val_acc

        out_dir = Path("outputs"); out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), (out_dir/"model.pth").as_posix())
        with open(out_dir/"metrics.json","w") as f:
            json.dump({"best_val_acc": best_acc}, f)

        pip_requirements = [
            "torch==2.3.1+cu118",
            "torchvision==0.18.1+cu118",
            "numpy==2.1.2",
            "pillow==11.0.0",
            "mlflow==2.14.1",
        ]
        mlflow.log_artifact((out_dir/"model.pth").as_posix(), artifact_path="artifacts")
        mlflow.pytorch.log_model(model, artifact_path="model", pip_requirements=pip_requirements)

        mv = register_model(mlflow.active_run().info.run_id, "model", os.getenv("MODEL_NAME","imgcls-resnet"))
        print("Registered model:", mv.name, "version:", mv.version)

if __name__ == "__main__":
    main()
