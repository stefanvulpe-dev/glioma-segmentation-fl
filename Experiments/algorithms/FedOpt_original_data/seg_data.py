import os, pickle, numpy as np, torch
from typing import List
from torch.utils.data import Dataset, DataLoader

# Global paths and configuration
# DATA_ROOT = "/content/Preprocessed-Data"
DATA_ROOT = "Preprocessed-Data"
METADATA_DF_PATH = "cleaned_df.pkl"
USE_ATLAS = True
EXCLUDE_IDS = ["PatientID_0191"]

# Dataset that loads MRI, tumor mask and optional atlas for each patient
class ImageOnlyGliomaDataset(Dataset):
    def __init__(
        self,
        metadata_df_path=METADATA_DF_PATH,
        data_root=DATA_ROOT,
        use_atlas=USE_ATLAS,
        exclude_ids=None,
        transform=None,
    ):
        with open(metadata_df_path, "rb") as f:
            df = pickle.load(f)

        # Optionally exclude specific patients
        if exclude_ids is None:
            exclude_ids = EXCLUDE_IDS

        # Keep only non-excluded patient rows
        self.df = df[~df["Patient_ID"].isin(exclude_ids)].reset_index(drop=True)
        self.data_root = data_root
        self.use_atlas = use_atlas
        self.transform = transform

        # Collect patient IDs that have all required .npy files
        self.patient_ids = []
        for pid in sorted(self.df["Patient_ID"].tolist()):
            base = os.path.join(self.data_root, pid)
            mri_p = os.path.join(base, f"{pid}_mri.npy")
            tumor_p = os.path.join(base, f"{pid}_tumor.npy")
            if self.use_atlas:
                reg_p = os.path.join(base, f"{pid}_regions.npy")
                ok = os.path.isfile(mri_p) and os.path.isfile(tumor_p) and os.path.isfile(reg_p)
            else:
                ok = os.path.isfile(mri_p) and os.path.isfile(tumor_p)
            if ok:
                self.patient_ids.append(pid)

    def __len__(self):
        return len(self.patient_ids)

    # Simple min–max normalization
    @staticmethod
    def _minmax(x):
        x = x.astype(np.float32)
        mn, mx = np.min(x), np.max(x)
        return (x - mn) / (mx - mn) if mx > mn else np.zeros_like(x, dtype=np.float32)

    # Load a single sample: MRI, tumor mask, and optional atlas
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

# Collate function to build batched tensors and patient ID list
def image_only_collate_fn(batch, use_atlas=USE_ATLAS):
    mri = torch.stack([torch.tensor(it["mri"]) for it in batch]).unsqueeze(1)
    y = torch.stack([torch.tensor(it["tumor"]) for it in batch]).unsqueeze(1)

    if use_atlas:
        regs = torch.stack([torch.tensor(it["regions"]) for it in batch]).unsqueeze(1)
        x = torch.cat([mri.float(), regs.float()], dim=1)
    else:
        x = mri.float()

    return {"x": x, "y": y.float(), "pid": [it["patient_id"] for it in batch]}

# Dataset wrapper that restricts to a subset of patient IDs
class SubsetByPIDs(Dataset):
    def __init__(self, full_dataset: ImageOnlyGliomaDataset, pid_list: List[str]):
        self.ds = full_dataset
        pid_to_idx = {pid: i for i, pid in enumerate(self.ds.patient_ids)}
        self.indices = [pid_to_idx[p] for p in pid_list if p in pid_to_idx]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.ds[self.indices[i]]

# Compute Dice, IoU and accuracy for binary masks
def calc_metrics(y_true, y_pred):
    y_true = y_true.astype(np.uint8).reshape(-1)
    y_pred = y_pred.astype(np.uint8).reshape(-1)

    inter = (y_true & y_pred).sum()
    dice = (2.0 * inter) / (y_true.sum() + y_pred.sum() + 1e-8)
    union = y_true.sum() + y_pred.sum() - inter + 1e-8
    iou = inter / union
    acc = (y_true == y_pred).mean()

    return float(dice), float(iou), float(acc)
