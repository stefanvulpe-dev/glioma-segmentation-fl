import os, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import logging

from seg_data import (
    ImageOnlyGliomaDataset,
    SubsetByPIDs,
    image_only_collate_fn,
    DATA_ROOT,
    METADATA_DF_PATH,
    USE_ATLAS,
    calc_metrics,
)

logging.getLogger("timm.models._builder").setLevel(logging.ERROR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
THRESHOLD = 0.5

# CLIENT_DIR = "/content/client"
CLIENT_DIR = "client"
BATCH_SIZE = 8
NUM_WORKERS = 2

ENSEMBLE_CFGS = [
    ("unet", "resnet50"),
    ("unet", "mit_b3"),
    ("deeplabv3plus", "timm-mobilenetv3_small_100"),
]

WEIGHT_MODE = "power"
WEIGHT_POWER = 8.0
WEIGHT_EPS = 1e-6


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

bce = nn.BCEWithLogitsLoss()
dice_loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)


def criterion(logits, y):
    return 0.5 * bce(logits, y) + 0.5 * dice_loss(logits, y)


def get_val_loader(cid: int):
    full = ImageOnlyGliomaDataset(
        METADATA_DF_PATH, DATA_ROOT, use_atlas=USE_ATLAS, exclude_ids=["PatientID_0191"]
    )
    with open(os.path.join(CLIENT_DIR, f"client_{cid}", "val_pids.json")) as f:
        va_p = json.load(f)

    ds_va = SubsetByPIDs(full, va_p)
    g = torch.Generator().manual_seed(SEED)

    ld_va = torch.utils.data.DataLoader(
        ds_va,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=lambda b: image_only_collate_fn(b, use_atlas=USE_ATLAS),
        generator=g,
    )
    return ld_va, len(ds_va)


def run_name(model_name: str, encoder_name: str) -> str:
    return f"{model_name}__{encoder_name}".replace("/", "-")


def build_model(model_name: str, encoder_name: str, encoder_weights="imagenet"):
    in_ch = 2 if USE_ATLAS else 1
    mn = model_name.lower()

    if mn == "unet":
        m = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_ch,
            classes=1,
        )
    elif mn in ["deeplabv3plus", "deeplabv3+", "dlv3p"]:
        m = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_ch,
            classes=1,
        )
    else:
        raise ValueError(f"Unknown model_name={model_name}")

    return m.to(DEVICE)


def ckpt_path(model_name: str, encoder_name: str, cid: int) -> str:
    rn = run_name(model_name, encoder_name)
    return os.path.join("AITDM", rn, "checkpoints", f"client_{cid}_best.pt")


def best_json_path(model_name: str, encoder_name: str, cid: int) -> str:
    rn = run_name(model_name, encoder_name)
    return os.path.join("AITDM", rn, "checkpoints", f"client_{cid}_best.json")


def load_model(model_name: str, encoder_name: str, cid: int):
    path = ckpt_path(model_name, encoder_name, cid)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    m = build_model(model_name, encoder_name).to(DEVICE)
    sd = torch.load(path, map_location="cpu")
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m


def load_best_val_dice(model_name: str, encoder_name: str, cid: int) -> float:
    p = best_json_path(model_name, encoder_name, cid)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Missing best json: {p}")
    with open(p, "r") as f:
        j = json.load(f)
    return float(j.get("val_dice", 0.0))


def get_client_weights(cid: int, cfgs, mode="power", power=2.0, eps=1e-6):
    dices = [load_best_val_dice(mn, enc, cid) for (mn, enc) in cfgs]
    d = np.array(dices, dtype=np.float32)

    if mode == "linear":
        raw = np.clip(d, 0.0, None)
    elif mode == "power":
        raw = np.power(np.clip(d, 0.0, None), power)
    else:
        raise ValueError("mode must be 'linear' or 'power'")

    raw = raw + eps
    w = raw / raw.sum()
    return w.tolist(), dices


@torch.no_grad()
def ensemble_forward_logits(x, models, weights, target_hw=None):
    w = np.array(weights, dtype=np.float32)
    w = w / (w.sum() + 1e-8)

    logits_sum = None
    for mi, wi in zip(models, w):
        li = mi(x)
        if target_hw is not None and li.shape[-2:] != target_hw:
            li = F.interpolate(li, size=target_hw, mode="bilinear", align_corners=False)
        logits_sum = li * float(wi) if logits_sum is None else logits_sum + li * float(wi)
    return logits_sum


@torch.no_grad()
def eval_ensemble_on_client(cid: int, threshold=0.5):
    val_loader, nva = get_val_loader(cid)
    models = [load_model(mn, enc, cid) for (mn, enc) in ENSEMBLE_CFGS]

    weights, best_dices = get_client_weights(
        cid, ENSEMBLE_CFGS, mode=WEIGHT_MODE, power=WEIGHT_POWER, eps=WEIGHT_EPS
    )

    tot_loss = tot_d = tot_i = tot_a = 0.0
    nb = 0

    for batch in val_loader:
        x = batch["x"].to(DEVICE)
        y = batch["y"].to(DEVICE)

        logits = ensemble_forward_logits(x, models=models, weights=weights, target_hw=y.shape[-2:])
        loss = float(criterion(logits, y).item())

        preds_bin = (torch.sigmoid(logits).cpu().numpy() > threshold).astype(np.uint8)
        y_np = (y.cpu().numpy() > 0.5).astype(np.uint8)

        d, i, a = calc_metrics(y_np, preds_bin)

        tot_loss += loss
        tot_d += d
        tot_i += i
        tot_a += a
        nb += 1

    nb = max(nb, 1)
    return {
        "cid": int(cid),
        "nva": int(nva),
        "loss": tot_loss / nb,
        "dice": tot_d / nb,
        "iou": tot_i / nb,
        "acc": tot_a / nb,
        "weights": weights,
        "best_dices": best_dices,
    }


def main():
    print(f"[Ensemble-3 | threshold={THRESHOLD} | weight_mode={WEIGHT_MODE} | power={WEIGHT_POWER}]\n")
    for cid in [0, 1, 2]:
        r = eval_ensemble_on_client(cid, threshold=THRESHOLD)
        bd = r["best_dices"]
        ww = r["weights"]
        print(
            f"Client {r['cid']} (n={r['nva']}): "
            f"loss={r['loss']:.4f} dice={r['dice']:.4f} iou={r['iou']:.4f} acc={r['acc']:.4f}\n"
            f"  best_dice={['%.4f' % x for x in bd]} -> weights={['%.3f' % x for x in ww]}\n"
        )


if __name__ == "__main__":
    main()
