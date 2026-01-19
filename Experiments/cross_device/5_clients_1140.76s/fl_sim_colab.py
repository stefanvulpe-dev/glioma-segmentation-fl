import os
import csv
import json
import time
from datetime import datetime

import torch
import flwr as fl
from flwr.common import FitIns

from fl_client import SegClient


def ensure_csv(path: str, header: list[str]):
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)


def append_row(path: str, row: list):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def fmt_seconds(sec: float) -> str:
    sec = float(sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def save_summary_txt(base_dir: str, num_rounds: int, local_epochs: int, elapsed_seconds: float):
    os.makedirs(base_dir, exist_ok=True)
    out_path = os.path.join(base_dir, "experiment_summary.txt")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path_ts = os.path.join(base_dir, f"experiment_summary_{ts}.txt")

    txt = (
        "=== Experiment summary ===\n"
        f"Timestamp: {datetime.now().isoformat(timespec='seconds')}\n\n"
        f"Rounds: {int(num_rounds)}\n"
        f"Local epochs per round: {int(local_epochs)}\n\n"
        f"Total duration (seconds): {float(elapsed_seconds):.4f}\n"
        f"Total duration (hh:mm:ss): {fmt_seconds(elapsed_seconds)}\n"
    )

    for p in (out_path, out_path_ts):
        with open(p, "w", encoding="utf-8") as f:
            f.write(txt)


class PerClientLoggingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, metrics_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.metrics_dir = metrics_dir
        self.header = [
            "round", "epoch", "train_loss", "train_dice", "train_iou", "train_acc",
            "val_loss", "val_dice", "val_iou", "val_acc", "best_epoch",
        ]
        self.round_fit_times = []
        self.round_eval_times = []
        self.round_gpu_alloc = []
        self.round_rss_delta = []
        self.last_model_bytes = None
        self.last_n_params = None

    def configure_fit(self, server_round, parameters, client_manager):
        items = super().configure_fit(server_round, parameters, client_manager)
        out = []
        for it in items:
            if isinstance(it, tuple):
                client, fitins = it
            else:
                client, fitins = None, it
            cfg = dict(fitins.config)
            cfg["round"] = server_round
            new_fitins = FitIns(fitins.parameters, cfg)
            out.append((client, new_fitins) if client is not None else new_fitins)
        return out

    def aggregate_fit(self, rnd, results, failures):
        agg = super().aggregate_fit(rnd, results, failures)

        for client_proxy, fit_res in results:
            m = fit_res.metrics or {}
            cid = str(m.get("cid", client_proxy.cid))

            ft = m.get("fit_time_sec", None)
            if ft is not None:
                self.round_fit_times.append(float(ft))

            ga = m.get("gpu_max_alloc_bytes", None)
            if ga is not None and int(ga) >= 0:
                self.round_gpu_alloc.append(int(ga))

            rd = m.get("rss_delta_bytes", None)
            if rd is not None and int(rd) >= 0:
                self.round_rss_delta.append(int(rd))

            mb = m.get("model_bytes", None)
            np_ = m.get("n_params", None)
            if mb is not None:
                self.last_model_bytes = int(mb)
            if np_ is not None:
                self.last_n_params = int(np_)

            client_csv = os.path.join(self.metrics_dir, f"metrics_client_{cid}.csv")
            ensure_csv(client_csv, self.header)

            best_epoch = int(m.get("best_epoch", -1))
            per_epoch_raw = m.get("per_epoch", "[]")

            try:
                per_epoch = json.loads(per_epoch_raw)
            except Exception:
                per_epoch = []

            for ep in per_epoch:
                epoch = ep.get("epoch", "")
                row = [
                    rnd,
                    epoch,
                    ep.get("train_loss", ""),
                    ep.get("train_dice", ""),
                    ep.get("train_iou", ""),
                    ep.get("train_acc", ""),
                    ep.get("val_loss", ""),
                    ep.get("val_dice", ""),
                    ep.get("val_iou", ""),
                    ep.get("val_acc", ""),
                    "x" if int(epoch) == best_epoch else "",
                ]
                append_row(client_csv, row)

        return agg

    def aggregate_evaluate(self, rnd, results, failures):
        agg = super().aggregate_evaluate(rnd, results, failures)
        for client_proxy, eval_res in results:
            m = eval_res.metrics or {}
            et = m.get("eval_time_sec", None)
            if et is not None:
                self.round_eval_times.append(float(et))
        return agg


def run_one_experiment(model_name: str, encoder_name: str, num_rounds=5, local_epochs=5, lr=1e-3):
    run_name = f"{model_name}__{encoder_name}".replace("/", "-")
    base_dir = os.path.join("AITDM", run_name)
    metrics_dir = os.path.join(base_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    exp_csv = os.path.join(base_dir, "experiment_metrics.csv")
    ensure_csv(exp_csv, [
        "run_name", "model_name", "encoder_name",
        "num_rounds", "local_epochs", "lr",
        "elapsed_sec",
        "model_bytes", "n_params",
        "est_total_comm_mb",
        "avg_fit_time_sec_per_round_per_client",
        "avg_eval_time_sec_per_round_per_client",
        "avg_gpu_max_alloc_mb",
        "avg_rss_delta_mb",
    ])

    t0 = time.perf_counter()

    def client_fn(cid: str):
        return SegClient(int(cid), model_name=model_name, encoder_name=encoder_name).to_client()

    strategy = PerClientLoggingFedAvg(
        metrics_dir=metrics_dir,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
        on_fit_config_fn=lambda rnd: {"local_epochs": local_epochs, "lr": lr},
    )

    use_gpu = torch.cuda.is_available()
    client_resources = {"num_cpus": 1, "num_gpus": 1.0 if use_gpu else 0.0}

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=5,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args={"include_dashboard": False},
    )

    elapsed = time.perf_counter() - t0

    model_bytes = strategy.last_model_bytes if strategy.last_model_bytes is not None else -1
    n_params = strategy.last_n_params if strategy.last_n_params is not None else -1

    num_clients = 5
    if model_bytes > 0:
        est_total_comm_mb = (num_rounds * num_clients * 2.0 * model_bytes) / (1024**2)
    else:
        est_total_comm_mb = -1.0

    def _avg(xs):
        return float(sum(xs) / max(len(xs), 1)) if xs else -1.0

    avg_fit = _avg(strategy.round_fit_times)
    avg_eval = _avg(strategy.round_eval_times)
    avg_gpu_mb = _avg([b / (1024**2) for b in strategy.round_gpu_alloc]) if strategy.round_gpu_alloc else -1.0
    avg_rss_mb = _avg([b / (1024**2) for b in strategy.round_rss_delta]) if strategy.round_rss_delta else -1.0

    append_row(exp_csv, [
        run_name, model_name, encoder_name,
        int(num_rounds), int(local_epochs), float(lr),
        float(elapsed),
        int(model_bytes), int(n_params),
        float(est_total_comm_mb),
        float(avg_fit),
        float(avg_eval),
        float(avg_gpu_mb),
        float(avg_rss_mb),
    ])

    save_summary_txt(base_dir=base_dir, num_rounds=num_rounds, local_epochs=local_epochs, elapsed_seconds=elapsed)
    print(f"[{run_name}] total duration: {fmt_seconds(elapsed)} ({elapsed:.2f}s)")


if __name__ == "__main__":
    t_all0 = time.perf_counter()

    experiments = [
        ("unet", "resnet50"),
        ("unet", "mit_b3"),
        ("deeplabv3plus", "timm-mobilenetv3_small_100"),
    ]

    for model_name, encoder_name in experiments:
        print(f"\n=== Running: {model_name} + {encoder_name} ===")
        run_one_experiment(model_name, encoder_name, num_rounds=7, local_epochs=7, lr=1e-3)

    total_all = time.perf_counter() - t_all0
    print(f"\n=== TOTAL (all experiments) === {fmt_seconds(total_all)} ({total_all:.2f}s)")
