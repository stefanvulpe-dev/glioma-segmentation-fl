import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl
import copy
import segmentation_models_pytorch as smp
import random

from seg_data import (
    ImageOnlyGliomaDataset,
    SubsetByPIDs,
    image_only_collate_fn,
    calc_metrics,
    DATA_ROOT,
    METADATA_DF_PATH,
    USE_ATLAS,
)

CLIENT_DIR = "client"

BATCH_SIZE = 8
NUM_WORKERS = 2
SEED = 42

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DEFAULT_MODEL_NAME = "unet"
DEFAULT_ENCODER_NAME = "timm-mobilenetv3_small_100"
DEFAULT_ENCODER_WEIGHTS = "imagenet"


def _run_name(model_name: str, encoder_name: str) -> str:
    return f"{model_name}__{encoder_name}".replace("/", "-")


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_model(
    model_name=DEFAULT_MODEL_NAME,
    encoder_name=DEFAULT_ENCODER_NAME,
    encoder_weights=DEFAULT_ENCODER_WEIGHTS,
):
    in_ch = 2 if USE_ATLAS else 1
    mn = model_name.lower()

    if mn == "unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_ch,
            classes=1,
        )
    elif mn in ["deeplabv3plus", "deeplabv3+", "dlv3p"]:
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_ch,
            classes=1,
        )
    else:
        raise ValueError(f"Unknown model_name={model_name}. Use 'unet' or 'deeplabv3plus'.")

    return model.to(DEVICE)


def get_loaders(cid: int, base_seed: int):
    full = ImageOnlyGliomaDataset(
        METADATA_DF_PATH,
        DATA_ROOT,
        use_atlas=USE_ATLAS,
        exclude_ids=["PatientID_0191"],
    )

    with open(os.path.join(CLIENT_DIR, f"client_{cid}", "train_pids.json")) as f:
        tr_p = json.load(f)
    with open(os.path.join(CLIENT_DIR, f"client_{cid}", "val_pids.json")) as f:
        va_p = json.load(f)

    ds_tr = SubsetByPIDs(full, tr_p)
    ds_va = SubsetByPIDs(full, va_p)

    g_tr = torch.Generator().manual_seed(base_seed + 12345)
    g_va = torch.Generator().manual_seed(base_seed + 67890)

    ld_tr = DataLoader(
        ds_tr,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=lambda b: image_only_collate_fn(b, use_atlas=USE_ATLAS),
        worker_init_fn=seed_worker,
        generator=g_tr,
        persistent_workers=(NUM_WORKERS > 0),
    )

    ld_va = DataLoader(
        ds_va,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=lambda b: image_only_collate_fn(b, use_atlas=USE_ATLAS),
        worker_init_fn=seed_worker,
        generator=g_va,
        persistent_workers=(NUM_WORKERS > 0),
    )

    return ld_tr, ld_va, len(ds_tr), len(ds_va)


def get_parameters(model):
    return [p.detach().cpu().numpy() for _, p in model.state_dict().items()]


def set_parameters(model, params):
    sd = model.state_dict()
    for k, v in zip(sd.keys(), params):
        sd[k] = torch.tensor(v)
    model.load_state_dict(sd, strict=True)


bce = nn.BCEWithLogitsLoss()
dice_loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)


def criterion(pred, y):
    return 0.5 * bce(pred, y) + 0.5 * dice_loss(pred, y)


def fedprox_prox_term(model: torch.nn.Module, global_params: list[torch.Tensor]) -> torch.Tensor:
    prox = 0.0
    for p, g in zip(model.parameters(), global_params):
        prox = prox + torch.sum((p - g) ** 2)
    return prox


def maybe_save_best(run_dir, cid, val_loss, val_dice, best_epoch, rnd, model):
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_json = os.path.join(ckpt_dir, f"client_{cid}_best.json")
    best_pt = os.path.join(ckpt_dir, f"client_{cid}_best.pt")

    prev = {"val_loss": float("inf"), "val_dice": -1.0}
    if os.path.isfile(best_json):
        try:
            with open(best_json, "r") as f:
                prev = json.load(f)
        except Exception:
            pass

    improved = (val_loss < prev.get("val_loss", float("inf"))) and (val_dice > prev.get("val_dice", -1.0))
    if improved:
        torch.save(model.state_dict(), best_pt)
        with open(best_json, "w") as f:
            json.dump(
                {
                    "round": int(rnd),
                    "epoch": int(best_epoch),
                    "val_loss": float(val_loss),
                    "val_dice": float(val_dice),
                },
                f,
            )


class SegClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: int,
        model_name=DEFAULT_MODEL_NAME,
        encoder_name=DEFAULT_ENCODER_NAME,
        encoder_weights=DEFAULT_ENCODER_WEIGHTS,
    ):
        self.cid = int(cid)
        self.model_name = model_name
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights

        self.base_seed = SEED + self.cid
        seed_everything(self.base_seed, deterministic=True)

        self.run_name = _run_name(model_name, encoder_name)
        self.run_dir = os.path.join("AITDM", self.run_name)

        self.model = get_model(model_name, encoder_name, encoder_weights)
        self.train_loader, self.val_loader, self.ntr, self.nva = get_loaders(self.cid, self.base_seed)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)

        epochs = int(config.get("local_epochs", 1))
        lr = float(config.get("lr", 1e-3))
        rnd = int(config.get("round", 0))
        mu = float(config.get("mu", 0.0))

        opt = optim.AdamW(self.model.parameters(), lr=lr)
        scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

        global_params = [p.detach().clone().to(DEVICE) for p in self.model.parameters()]

        best_state = None
        best_val_loss = float("inf")
        best_val_dice = -1.0
        best_epoch_idx = -1
        epoch_logs = []

        for epoch_idx in range(1, epochs + 1):
            self.model.train()
            tot_tr_loss = tot_tr_d = tot_tr_i = tot_tr_a = 0.0
            nb_tr = 0

            for batch in self.train_loader:
                x = batch["x"].to(DEVICE)
                y = batch["y"].to(DEVICE)
                opt.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    pred = self.model(x)
                    base_loss = criterion(pred, y)
                    if mu > 0.0:
                        prox = 0.5 * mu * fedprox_prox_term(self.model, global_params)
                        loss = base_loss + prox
                    else:
                        loss = base_loss

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                with torch.no_grad():
                    y_hat = (torch.sigmoid(pred).detach().cpu().numpy() > 0.5).astype(np.uint8)
                    y_np = (y.detach().cpu().numpy() > 0.5).astype(np.uint8)
                    d, i, a = calc_metrics(y_np, y_hat)

                tot_tr_loss += float(loss.item())
                tot_tr_d += d
                tot_tr_i += i
                tot_tr_a += a
                nb_tr += 1

            nb_tr = max(nb_tr, 1)
            epoch_tr_loss = tot_tr_loss / nb_tr
            epoch_tr_dice = tot_tr_d / nb_tr
            epoch_tr_iou = tot_tr_i / nb_tr
            epoch_tr_acc = tot_tr_a / nb_tr

            self.model.eval()
            tot_val_loss = tot_val_d = tot_val_i = tot_val_a = 0.0
            nb_val = 0

            with torch.no_grad():
                for batch in self.val_loader:
                    x = batch["x"].to(DEVICE)
                    y = batch["y"].to(DEVICE)

                    pred = self.model(x)
                    v_loss = float(criterion(pred, y).item())
                    y_hat = (torch.sigmoid(pred).cpu().numpy() > 0.5).astype(np.uint8)
                    y_np = (y.cpu().numpy() > 0.5).astype(np.uint8)
                    d, i, a = calc_metrics(y_np, y_hat)

                    tot_val_loss += v_loss
                    tot_val_d += d
                    tot_val_i += i
                    tot_val_a += a
                    nb_val += 1

            nb_val = max(nb_val, 1)
            epoch_val_loss = tot_val_loss / nb_val
            epoch_val_dice = tot_val_d / nb_val
            epoch_val_iou = tot_val_i / nb_val
            epoch_val_acc = tot_val_a / nb_val

            epoch_logs.append(
                {
                    "epoch": int(epoch_idx),
                    "train_loss": float(epoch_tr_loss),
                    "train_dice": float(epoch_tr_dice),
                    "train_iou": float(epoch_tr_iou),
                    "train_acc": float(epoch_tr_acc),
                    "val_loss": float(epoch_val_loss),
                    "val_dice": float(epoch_val_dice),
                    "val_iou": float(epoch_val_iou),
                    "val_acc": float(epoch_val_acc),
                }
            )

            if (epoch_val_loss < best_val_loss) and (epoch_val_dice > best_val_dice):
                best_val_loss = epoch_val_loss
                best_val_dice = epoch_val_dice
                best_state = copy.deepcopy(self.model.state_dict())
                best_epoch_idx = epoch_idx

        if best_state is not None:
            self.model.load_state_dict(best_state)

        for ep in epoch_logs:
            ep["best_epoch"] = (ep["epoch"] == best_epoch_idx)

        train_metrics = {
            "cid": int(self.cid),
            "best_epoch": int(best_epoch_idx),
            "best_val_loss": float(best_val_loss),
            "best_val_dice": float(best_val_dice),
            "per_epoch": json.dumps(epoch_logs),
            "run_name": self.run_name,
            "model_name": self.model_name,
            "encoder_name": self.encoder_name,
            "mu": float(mu),
        }

        maybe_save_best(self.run_dir, self.cid, best_val_loss, best_val_dice, best_epoch_idx, rnd, self.model)

        return get_parameters(self.model), self.ntr, train_metrics

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.eval()

        tot_loss = tot_d = tot_i = tot_a = 0.0
        nb = 0

        with torch.no_grad():
            for batch in self.val_loader:
                x = batch["x"].to(DEVICE)
                y = batch["y"].to(DEVICE)
                pred = self.model(x)

                loss = float(criterion(pred, y).item())
                y_hat = (torch.sigmoid(pred).cpu().numpy() > 0.5).astype(np.uint8)
                y_np = (y.cpu().numpy() > 0.5).astype(np.uint8)
                d, i, a = calc_metrics(y_np, y_hat)

                tot_loss += loss
                tot_d += d
                tot_i += i
                tot_a += a
                nb += 1

        nb = max(nb, 1)
        metrics = {
            "loss": tot_loss / nb,
            "dice": tot_d / nb,
            "iou": tot_i / nb,
            "acc": tot_a / nb,
            "cid": int(self.cid),
            "run_name": self.run_name,
        }

        return metrics["loss"], self.nva, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True)
    parser.add_argument("--server", default="0.0.0.0:8080")
    args = parser.parse_args()

    fl.client.start_numpy_client(
        server_address=args.server,
        client=SegClient(args.cid),
    )
