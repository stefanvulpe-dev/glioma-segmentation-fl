import os, json, pickle, random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import torch
from torch.utils.data import Dataset, DataLoader

# -------------------- CONFIG --------------------
DATA_ROOT = "Preprocessed-Data"
METADATA_DF_PATH = "cleaned_df.pkl"
OUT_BASE = "AITDM"
CLIENT_DIR = os.path.join(OUT_BASE, "client")
os.makedirs(CLIENT_DIR, exist_ok=True)

USE_ATLAS = True
N_CLIENTS = 3
VAL_FRAC_PER_CLIENT = 0.2
BATCH_SIZE = 8
NUM_WORKERS = 2
SEED = 42

# -------------------- REPRO --------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(SEED)

# -------------------- DATASET --------------------
class ImageOnlyGliomaDataset(Dataset):
    def __init__(self, metadata_df_path, data_root, use_atlas=True, exclude_ids=None, transform=None):
        with open(metadata_df_path, "rb") as f:
            df = pickle.load(f)
        if exclude_ids is None:
            exclude_ids = ["PatientID_0191"]
        self.df = df[~df["Patient_ID"].isin(exclude_ids)].reset_index(drop=True)
        self.data_root = data_root
        self.use_atlas = use_atlas
        self.transform = transform
        self.patient_ids = []
        for pid in sorted(self.df["Patient_ID"].tolist()):
            mri_p = os.path.join(self.data_root, pid, f"{pid}_mri.npy")
            tumor_p = os.path.join(self.data_root, pid, f"{pid}_tumor.npy")
            if self.use_atlas:
                reg_p = os.path.join(self.data_root, pid, f"{pid}_regions.npy")
                if os.path.isfile(mri_p) and os.path.isfile(tumor_p) and os.path.isfile(reg_p):
                    self.patient_ids.append(pid)
            else:
                if os.path.isfile(mri_p) and os.path.isfile(tumor_p):
                    self.patient_ids.append(pid)

    def __len__(self): return len(self.patient_ids)

    @staticmethod
    def _minmax(x):
        x = x.astype(np.float32)
        mn, mx = np.min(x), np.max(x)
        return (x - mn) / (mx - mn) if mx > mn else np.zeros_like(x, dtype=np.float32)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        base = os.path.join(self.data_root, pid)
        mri = np.load(os.path.join(base, f"{pid}_mri.npy")).astype(np.float32)
        tumor = np.load(os.path.join(base, f"{pid}_tumor.npy")).astype(np.float32)
        mri = self._minmax(mri)
        tumor = (tumor > 0.5).astype(np.float32)
        sample = {"patient_id": pid, "mri": mri, "tumor": tumor}
        if self.use_atlas:
            regions = np.load(os.path.join(base, f"{pid}_regions.npy")).astype(np.float32)
            regions = self._minmax(regions)
            sample["regions"] = regions
        return sample

def image_only_collate_fn(batch, use_atlas=True):
    mri = torch.stack([torch.tensor(it["mri"]) for it in batch]).unsqueeze(1)
    y = torch.stack([torch.tensor(it["tumor"]) for it in batch]).unsqueeze(1)
    if use_atlas:
        regs = torch.stack([torch.tensor(it["regions"]) for it in batch]).unsqueeze(1)
        x = torch.cat([mri.float(), regs.float()], dim=1)
    else:
        x = mri.float()
    pids = [it["patient_id"] for it in batch]
    return {"x": x, "y": y.float(), "pid": pids}

# -------------------- STRAT LABELS (from images+atlas) --------------------
def patient_meta(pid: str) -> Tuple[int, float, int]:
    base = os.path.join(DATA_ROOT, pid)
    tumor = np.load(os.path.join(base, f"{pid}_tumor.npy")).astype(np.float32)
    mask = tumor > 0.5
    area = float(mask.sum())
    has_tumor = 1 if area >= 1 else 0

    dom_region = -1
    reg_path = os.path.join(base, f"{pid}_regions.npy")
    if USE_ATLAS and os.path.isfile(reg_path) and mask.any():
        regs = np.load(reg_path).astype(np.int32)
        vals, counts = np.unique(regs[mask], return_counts=True)
        if len(vals) > 0:
            dom_region = int(vals[np.argmax(counts)])
    return has_tumor, area, dom_region

def build_meta_for(dataset: ImageOnlyGliomaDataset) -> pd.DataFrame:
    rows = []
    for pid in dataset.patient_ids:
        try:
            ht, area, dom = patient_meta(pid)
            rows.append({"pid": pid, "has_tumor": ht, "area": area, "dom": dom})
        except Exception:
            pass
    meta = pd.DataFrame(rows)
    meta["size_bin"] = 0
    m = meta["has_tumor"] == 1
    if m.any():
        areas = meta.loc[m, "area"].values
        qs = np.quantile(areas, np.linspace(0, 1, 4))
        qs = np.unique(qs)
        bins = np.digitize(areas, qs[1:-1], right=True) if len(qs) > 2 else np.zeros_like(areas, dtype=int)
        meta.loc[m, "size_bin"] = bins.astype(int)
    # default fine-grained label (may be relaxed below)
    meta["strat_label"] = [f"{int(ht)}_{int(dom)}_{int(sb)}"
                           for ht, dom, sb in zip(meta["has_tumor"], meta["dom"], meta["size_bin"])]
    return meta

# -------------------- ROBUST STRATIFICATION HELPERS --------------------
def build_label(meta_df: pd.DataFrame, level: str) -> np.ndarray:
    if level == "full":      # has_tumor + dom + size
        return np.array([f"{int(ht)}_{int(dom)}_{int(sb)}"
                         for ht, dom, sb in zip(meta_df["has_tumor"], meta_df["dom"], meta_df["size_bin"])])
    if level == "ht_dom":    # has_tumor + dom
        return np.array([f"{int(ht)}_{int(dom)}"
                         for ht, dom in zip(meta_df["has_tumor"], meta_df["dom"])])
    if level == "ht_size":   # has_tumor + size
        return np.array([f"{int(ht)}_{int(sb)}"
                         for ht, sb in zip(meta_df["has_tumor"], meta_df["size_bin"])])
    if level == "ht":        # has_tumor only
        return meta_df["has_tumor"].astype(str).values
    raise ValueError(level)

def pick_strat_labels_for_kfold(meta_df: pd.DataFrame, n_splits: int):
    for level in ["full", "ht_dom", "ht_size", "ht"]:
        y = build_label(meta_df, level)
        counts = pd.Series(y).value_counts()
        if (counts >= n_splits).all():
            print(f"[INFO] Using K-Fold stratification level: {level}")
            return y, level
        else:
            rare = counts[counts < n_splits]
            print(f"[WARN] Level '{level}' has rare classes (<{n_splits}): {rare.to_dict()} -> trying coarser...")
    print("[WARN] Falling back to NON-stratified K-Fold (insufficient counts at all levels).")
    return None, "none"

def pick_strat_labels_for_client(meta_client_df: pd.DataFrame, min_per_class: int = 2):
    for level in ["full", "ht_dom", "ht_size", "ht"]:
        y = build_label(meta_client_df, level)
        counts = pd.Series(y).value_counts()
        if (counts >= min_per_class).all():
            print(f"[INFO] Using client train/val stratification level: {level}")
            return y, level
        else:
            rare = counts[counts < min_per_class]
            print(f"[WARN] Client level '{level}' rare classes (<{min_per_class}): {rare.to_dict()} -> trying coarser...")
    print("[WARN] Falling back to NON-stratified train/val (insufficient counts at all levels).")
    return None, "none"

# -------------------- MAKE 3 STRATIFIED CLIENTS, THEN STRAT TRAIN/VAL --------------------
dataset = ImageOnlyGliomaDataset(METADATA_DF_PATH, DATA_ROOT, use_atlas=USE_ATLAS, exclude_ids=["PatientID_0191"])
meta = build_meta_for(dataset)

X_all = meta["pid"].values
y_all_full, kfold_level = pick_strat_labels_for_kfold(meta, n_splits=N_CLIENTS)

clients: List[Dict[str, List[str]]] = []
client_metas = []

if kfold_level != "none":
    skf = StratifiedKFold(n_splits=N_CLIENTS, shuffle=True, random_state=SEED)
    for split_idx, (_, idx) in enumerate(skf.split(X_all, y_all_full)):
        pids_client = X_all[idx]
        meta_c = meta[meta["pid"].isin(pids_client)].reset_index(drop=True)

        y_client, tv_level = pick_strat_labels_for_client(meta_c, min_per_class=2)
        Xc = meta_c["pid"].values

        if tv_level != "none":
            sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRAC_PER_CLIENT, random_state=SEED)
            (tr_idx, va_idx), = sss.split(Xc, y_client)
            train_pids = Xc[tr_idx].tolist()
            val_pids   = Xc[va_idx].tolist()
        else:
            rng = np.random.RandomState(SEED)
            perm = rng.permutation(len(Xc))
            split_at = int((1.0 - VAL_FRAC_PER_CLIENT) * len(Xc))
            train_pids = Xc[perm[:split_at]].tolist()
            val_pids   = Xc[perm[split_at:]].tolist()

        cdir = os.path.join(CLIENT_DIR, f"client_{split_idx}")
        os.makedirs(cdir, exist_ok=True)
        with open(os.path.join(cdir, "train_pids.json"), "w") as f:
            json.dump(train_pids, f, indent=2)
        with open(os.path.join(cdir, "val_pids.json"), "w") as f:
            json.dump(val_pids, f, indent=2)

        clients.append({"train": train_pids, "val": val_pids})
        client_metas.append(meta_c)
else:
    print("[INFO] Non-stratified 3-way split (deterministic) for clients.")
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(X_all))
    sizes = [len(X_all)//N_CLIENTS]*N_CLIENTS
    sizes[-1] += len(X_all) - sum(sizes)
    start = 0
    for split_idx, sz in enumerate(sizes):
        pids_client = X_all[perm[start:start+sz]]
        start += sz

        meta_c = meta[meta["pid"].isin(pids_client)].reset_index(drop=True)
        Xc = meta_c["pid"].values

        perm_c = rng.permutation(len(Xc))
        split_at = int((1.0 - VAL_FRAC_PER_CLIENT) * len(Xc))
        train_pids = Xc[perm_c[:split_at]].tolist()
        val_pids   = Xc[perm_c[split_at:]].tolist()

        cdir = os.path.join(CLIENT_DIR, f"client_{split_idx}")
        os.makedirs(cdir, exist_ok=True)
        with open(os.path.join(cdir, "train_pids.json"), "w") as f:
            json.dump(train_pids, f, indent=2)
        with open(os.path.join(cdir, "val_pids.json"), "w") as f:
            json.dump(val_pids, f, indent=2)

        clients.append({"train": train_pids, "val": val_pids})
        client_metas.append(meta_c)

# also save a manifest with seed and settings + chosen levels
manifest = {
    "seed": SEED,
    "use_atlas": USE_ATLAS,
    "n_clients": N_CLIENTS,
    "val_frac_per_client": VAL_FRAC_PER_CLIENT,
    "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "kfold_level": kfold_level
}
with open(os.path.join(CLIENT_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

# -------------------- PREVIEW DATALOADERS (OPTIONAL) --------------------
class SubsetByPIDs(Dataset):
    def __init__(self, full_dataset: ImageOnlyGliomaDataset, pid_list: List[str]):
        self.ds = full_dataset
        pid_to_idx = {pid: i for i, pid in enumerate(self.ds.patient_ids)}
        self.indices = [pid_to_idx[p] for p in pid_list if p in pid_to_idx]
    def __len__(self): return len(self.indices)
    def __getitem__(self, i): return self.ds[self.indices[i]]

def make_loader(ds, shuffle):
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=lambda b: image_only_collate_fn(b, use_atlas=USE_ATLAS),
        generator=torch.Generator().manual_seed(SEED),
    )

preview_loaders = []
for cid, cl in enumerate(clients):
    ds_tr = SubsetByPIDs(dataset, cl["train"])
    ds_va = SubsetByPIDs(dataset, cl["val"])
    ld_tr = make_loader(ds_tr, shuffle=True)
    ld_va = make_loader(ds_va, shuffle=False)
    preview_loaders.append((ld_tr, ld_va))

# -------------------- PRINT DISTRIBUTION STATS --------------------
def summarize(meta_df: pd.DataFrame, name: str) -> Dict:
    s = {"name": name, "n": int(len(meta_df))}
    s["has_tumor_counts"] = meta_df["has_tumor"].value_counts().to_dict()
    s["size_bin_counts"] = meta_df["size_bin"].value_counts().to_dict()
    s["dom_region_top5"] = meta_df["dom"].value_counts().head(5).to_dict()
    return s

summary_all = summarize(meta, "ALL")
print("\n=== Global summary ===")
print(summary_all)

per_client_summaries = []
for cid, cl in enumerate(clients):
    meta_c = client_metas[cid]
    tr = meta_c[meta_c["pid"].isin(cl["train"])]
    va = meta_c[meta_c["pid"].isin(cl["val"])]
    s_client = {
        "client": cid,
        "train": summarize(tr, f"client_{cid}_train"),
        "val": summarize(va, f"client_{cid}_val"),
    }
    per_client_summaries.append(s_client)
    print(f"\n=== Client {cid} ===")
    print(s_client)

with open(os.path.join(CLIENT_DIR, "summary.json"), "w") as f:
    json.dump({"global": summary_all, "per_client": per_client_summaries}, f, indent=2)

print(f"\nSaved 3 client splits and summaries to: {CLIENT_DIR}")
