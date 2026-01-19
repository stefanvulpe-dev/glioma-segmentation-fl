import os
import csv
import json
import time
from datetime import datetime
from typing import List

import torch
import flwr as fl
from flwr.common import FitIns, Parameters

try:
    from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
except Exception:
    parameters_to_ndarrays = None
    ndarrays_to_parameters = None

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


def _params_to_ndarrays(p: Parameters):
    if parameters_to_ndarrays is None:
        raise RuntimeError("parameters_to_ndarrays is not available in this flwr version.")
    return parameters_to_ndarrays(p)


def _ndarrays_to_params(arrs):
    if ndarrays_to_parameters is None:
        raise RuntimeError("ndarrays_to_parameters is not available in this flwr version.")
    return ndarrays_to_parameters(arrs)


class ServerSGD:
    def __init__(self, lr: float = 1e-2, momentum: float = 0.0):
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.v = None

    def step(self, weights, grads):
        if self.v is None:
            self.v = [0.0 * w for w in weights]
        mu = self.momentum
        lr = self.lr

        new_w = []
        for i in range(len(weights)):
            self.v[i] = mu * self.v[i] + grads[i]
            new_w.append(weights[i] - lr * self.v[i])
        return new_w


class FedOptSGDWithLogging(fl.server.strategy.FedAvg):
    def __init__(self, metrics_dir: str, server_lr: float = 1e-2, server_momentum: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.metrics_dir = metrics_dir
        self.header = [
            "round", "epoch", "train_loss", "train_dice", "train_iou", "train_acc",
            "val_loss", "val_dice", "val_iou", "val_acc", "best_epoch",
        ]
        self._current_weights = None
        self.opt = ServerSGD(lr=server_lr, momentum=server_momentum)

    def configure_fit(self, server_round, parameters, client_manager):
        self._current_weights = _params_to_ndarrays(parameters)
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
        if not results:
            return None, {}

        if self._current_weights is None:
            return super().aggregate_fit(rnd, results, failures)

        total_n = 0
        client_items = []
        for client_proxy, fit_res in results:
            n_i = int(getattr(fit_res, "num_examples", 0) or 0)
            total_n += n_i
            w_i = _params_to_ndarrays(fit_res.parameters)
            client_items.append((client_proxy, fit_res, n_i, w_i))

        if total_n <= 0:
            return super().aggregate_fit(rnd, results, failures)

        w_avg = [0.0 * w for w in self._current_weights]
        for _, _, n_i, w_i in client_items:
            alpha = n_i / total_n
            for j in range(len(w_avg)):
                w_avg[j] = w_avg[j] + alpha * w_i[j]

        grads = [self._current_weights[j] - w_avg[j] for j in range(len(w_avg))]
        new_weights = self.opt.step(self._current_weights, grads)
        self._current_weights = new_weights
        aggregated_parameters = _ndarrays_to_params(new_weights)

        for client_proxy, fit_res, _, _ in client_items:
            m = fit_res.metrics or {}
            cid = str(m.get("cid", client_proxy.cid))

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

        return aggregated_parameters, {}


def run_one_experiment(
    model_name: str,
    encoder_name: str,
    num_rounds=5,
    local_epochs=5,
    lr=1e-3,
    server_lr=1e-2,
    server_momentum=0.0,
):
    run_name = f"{model_name}__{encoder_name}".replace("/", "-")
    base_dir = os.path.join("AITDM", run_name)
    metrics_dir = os.path.join(base_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    t0 = time.perf_counter()

    def client_fn(cid: str):
        return SegClient(int(cid), model_name=model_name, encoder_name=encoder_name).to_client()

    strategy = FedOptSGDWithLogging(
        metrics_dir=metrics_dir,
        server_lr=server_lr,
        server_momentum=server_momentum,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        on_fit_config_fn=lambda rnd: {"local_epochs": local_epochs, "lr": lr},
    )

    use_gpu = torch.cuda.is_available()
    client_resources = {"num_cpus": 1, "num_gpus": 1.0 if use_gpu else 0.0}

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=3,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args={"include_dashboard": False},
    )

    elapsed = time.perf_counter() - t0
    save_summary_txt(base_dir=base_dir, num_rounds=num_rounds, local_epochs=local_epochs, elapsed_seconds=elapsed)
    print(f"[{run_name}] total duration: {fmt_seconds(elapsed)} ({elapsed:.2f}s)")


if __name__ == "__main__":
    experiments = [
        ("unet", "resnet50"),
        ("unet", "mit_b3"),
        ("deeplabv3plus", "timm-mobilenetv3_small_100"),
    ]

    for model_name, encoder_name in experiments:
        print(f"\n=== Running: {model_name} + {encoder_name} ===")
        run_one_experiment(
            model_name,
            encoder_name,
            num_rounds=5,
            local_epochs=5,
            lr=1e-3,
            server_lr=1e-2,
            server_momentum=0.0,
        )
