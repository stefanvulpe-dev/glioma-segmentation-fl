import os, json, pickle, random
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple

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

# Dirichlet parameter: lower = stronger non-IID
DIRICHLET_ALPHA = 0.3  

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

# -------------------- META FROM IMAGES+ATLAS --------------------
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
    return meta

# -------------------- MAKE NON-IID CLIENTS (Dirichlet Split) --------------------
dataset = ImageOnlyGliomaDataset(METADATA_DF_PATH, DATA_ROOT, use_atlas=USE_ATLAS, exclude_ids=["PatientID_0191"])
meta = build_meta_for(dataset)

rng = np.random.RandomState(SEED)

# group patients by label (using size_bin as proxy label)
class_to_pids = defaultdict(list)
for _, row in meta.iterrows():
    label = row["size_bin"]  # can use dom or other label for stronger skew
    class_to_pids[label].append(row["pid"])

# initialize empty clients
client_pid_sets = [[] for _ in range(N_CLIENTS)]

# Dirichlet distribution-based non-IID assignment
for label, pids in class_to_pids.items():
    pids = rng.permutation(pids)
    proportions = rng.dirichlet([DIRICHLET_ALPHA] * N_CLIENTS)
    counts = (proportions * len(pids)).astype(int)
    # adjust rounding errors
    while counts.sum() < len(pids):
        counts[rng.randint(0, N_CLIENTS)] += 1
    start = 0
    for cid, count in enumerate(counts):
        subset = pids[start:start + count]
        client_pid_sets[cid].extend(subset)
        start += count

# perform train/val split for each client
clients: List[Dict[str, List[str]]] = []
client_metas = []
for cid in range(N_CLIENTS):
    Xc = np.array(client_pid_sets[cid])
    rng_c = np.random.RandomState(SEED + cid)
    perm_c = rng_c.permutation(len(Xc))
    split_at = max(1, int((1.0 - VAL_FRAC_PER_CLIENT) * len(Xc)))
    split_at = min(split_at, len(Xc) - 1)
    train_pids = Xc[perm_c[:split_at]].tolist()
    val_pids = Xc[perm_c[split_at:]].tolist()

    meta_c = meta[meta["pid"].isin(Xc)].reset_index(drop=True)

    cdir = os.path.join(CLIENT_DIR, f"client_{cid}")
    os.makedirs(cdir, exist_ok=True)
    with open(os.path.join(cdir, "train_pids.json"), "w") as f:
        json.dump(train_pids, f, indent=2)
    with open(os.path.join(cdir, "val_pids.json"), "w") as f:
        json.dump(val_pids, f, indent=2)

    clients.append({"train": train_pids, "val": val_pids})
    client_metas.append(meta_c)

print(f"[INFO] Created {N_CLIENTS} non-IID clients using Dirichlet split (alpha={DIRICHLET_ALPHA}).")

# save manifest
manifest = {
    "seed": SEED,
    "use_atlas": USE_ATLAS,
    "n_clients": N_CLIENTS,
    "val_frac_per_client": VAL_FRAC_PER_CLIENT,
    "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "split_type": "non-IID_Dirichlet",
    "dirichlet_alpha": DIRICHLET_ALPHA
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

print(f"\nSaved {N_CLIENTS} non-IID client splits and summaries to: {CLIENT_DIR}")
