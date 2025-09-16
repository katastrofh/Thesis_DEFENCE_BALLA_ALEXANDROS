# microchain/roles/client.py
from __future__ import annotations
import threading
import time
import hashlib
from typing import Callable
try:
    import numpy as np
except Exception:
    np = None

from ..core.state import State
from ..storage.ipfs_stub import IPFS

class Client(threading.Thread):
    def __init__(self, acct: str, state: State, ipfs: IPFS, submit_tx: Callable[[dict], None],
                 model_id: str, data_seed: int = 0):
        super().__init__(daemon=True)
        self.acct = acct
        self.state = state
        self.ipfs = ipfs
        self.submit_tx = submit_tx
        self.model_id = model_id
        self.data_seed = data_seed
        self.running = True

    def _current_round(self) -> int:
        with self.state.lock():
            return self.state.models[self.model_id].round_index

    def _deposit(self) -> int:
        with self.state.lock():
            return self.state.models[self.model_id].rules.deposit

    def _train_bytes(self) -> bytes:
        """
        Fallback: deterministic bytes (no ML).
        """
        r = self._current_round()
        payload = f"{self.acct}-{self.data_seed}-r{r}".encode()
        return hashlib.sha256(payload).digest()

    def _maybe_ml_delta_csv(self) -> str | None:
        """
        If an ML task is registered and numpy is available, produce a tiny CSV
        delta vector deterministically seeded by (acct, seed, round).
        """
        if np is None:
            return None
        with self.state.lock():
            task = self.state.ml_tasks.get(self.model_id)
        if not task:
            return None
        D = int(task["D"])
        r = self._current_round()
        # deterministic per-(acct, round, seed)
        seed = abs(hash((self.acct, self.data_seed, r))) % (2**32)
        rng = np.random.default_rng(seed)
        delta = rng.normal(loc=0.0, scale=0.05, size=D)
        return ",".join(f"{x:.6f}" for x in delta)

    def _my_tickets(self, round_: int) -> list[int]:
        with self.state.lock():
            return [t.ticket_id for t in self.state.tickets_for(self.model_id, round_)
                    if t.owner == self.acct]

    def _maybe_claim(self):
        with self.state.lock():
            elig = self.state.entitlement_get(self.acct, self.model_id)
            nonce = self.state.claim_nonce.get(self.acct, 0)
        if elig <= 0:
            return
        txc = {"type": "ClaimTx", "acct": self.acct, "model_id": self.model_id,
               "amount": elig, "nonce": nonce, "fee": 2}
        self.submit_tx(txc)

        deadline = time.time() + 2.0
        while time.time() < deadline:
            with self.state.lock():
                now = self.state.entitlement_get(self.acct, self.model_id)
            if now < elig:
                break
            time.sleep(0.05)

    def run(self):
        while True:
            with self.state.lock():
                if self.model_id in self.state.models:
                    break
            time.sleep(0.1)

        while self.running:
            self._maybe_claim()

            r = self._current_round()
            if self._my_tickets(r):
                time.sleep(0.2)
                continue

            dep = self._deposit()
            txi = {"type": "UpdateIntentTx", "client": self.acct, "model_id": self.model_id,
                   "round": r, "deposit": dep, "fee": 1}
            self.submit_tx(txi)
            time.sleep(0.4)

            tids = self._my_tickets(r)
            if not tids:
                time.sleep(0.2)
                continue

            tid = tids[-1]
            ml_csv = self._maybe_ml_delta_csv()
            if ml_csv is not None:
                update_cid = self.ipfs.put(ml_csv.encode("utf-8"), tag="update")
            else:
                update_cid = self.ipfs.put(self._train_bytes(), tag="update")

            txp = {"type": "UpdatePublishTx", "client": self.acct, "ticket_id": tid,
                   "update_cid": update_cid, "fee": 1}
            self.submit_tx(txp)

            deadline = time.time() + 3.0
            while time.time() < deadline:
                with self.state.lock():
                    t = self.state.tickets.get(tid)
                    if t and t.revealed:
                        break
                time.sleep(0.05)

            time.sleep(0.3)

    def stop(self):
        self.running = False
