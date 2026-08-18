"""
fl_client.py - ONE CLIENT IN THE FL SYSTEM (Flower NumPyClient)
===============================================================
A malicious client runs NO different code - the only difference is WHICH DATA FILE IT
READS (clean or poisoned). That is exactly what data poisoning means: the attack lives in
the DATA, not in the client logic. This makes the simulation faithful to reality - the
server cannot tell malicious clients apart by inspecting their source code.
"""

import os
import pandas as pd
import flwr as fl
from flwr.common import NDArrays
from typing import Dict, Tuple

import config
from src import model as M


class EmailClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, data_path: str):
        self.client_id = client_id
        self.model = M.build_model()
        df = pd.read_csv(data_path)
        self.n_samples = len(df)
        self.dataloader = M.make_dataloader(df, shuffle=True)

        n_pois = int(df["is_poisoned"].sum()) if "is_poisoned" in df.columns else 0
        tag = "  <== MALICIOUS" if n_pois > 0 else ""
        print(f"    [Client {client_id}] {self.n_samples} samples, {n_pois} poisoned"
              f" ({os.path.basename(data_path)}){tag}")

    def get_parameters(self, config_dict) -> NDArrays:
        return M.get_weights(self.model)

    def set_parameters(self, parameters: NDArrays):
        M.set_weights(self.model, parameters)

    def fit(self, parameters, config_dict) -> Tuple[NDArrays, int, Dict]:
        self.set_parameters(parameters)
        loss = 0.0
        for _ in range(config.LOCAL_EPOCHS):
            loss = M.train_one_epoch(self.model, self.dataloader)
        # IMPORTANT: the client REPORTS ITS OWN INDEX in the metrics.
        # Reason: in recent Flower versions the server-side ClientProxy.cid is a RANDOM
        # HASH (node_id, e.g. 2465052526735391746), not "0".."9". If the server relied on
        # cid to know who is who, every diagnostic (which client was excluded, the
        # malicious group's update norms, ...) would be wrong. Passing client_id through
        # the metrics is the only reliable way for the server to map results back.
        return (self.get_parameters(config_dict), self.n_samples,
                {"loss": float(loss), "client_id": int(self.client_id)})

    def evaluate(self, parameters, config_dict) -> Tuple[float, int, Dict]:
        # The OFFICIAL evaluation is performed by the server on its own held-out test set
        # (centralised evaluation).
        return 0.0, self.n_samples, {}


def make_client_fn(data_path_map: Dict[int, str]):
    def client_fn(cid: str) -> fl.client.Client:
        return EmailClient(int(cid), data_path_map[int(cid)]).to_client()
    return client_fn
