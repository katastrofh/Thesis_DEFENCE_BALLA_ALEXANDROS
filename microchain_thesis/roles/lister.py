# microchain/roles/lister.py
from __future__ import annotations
from typing import List, Dict, Any
import threading
import time
import random
try:
    import numpy as np
except Exception:
    np = None

from ..core.state import State, Rules, Ticket
from ..storage.ipfs_stub import IPFS

class Lister(threading.Thread):
    def __init__(
        self,
        acct: str,
        state: State,
        ipfs: IPFS,
        submit_tx,
        model_id: str = "mnist",
        init_bytes: bytes = b"init",
        rules: Rules = None,
        fund_amount: int = 1000,
    ):
        super().__init__(daemon=True)
        self.acct = acct
        self.state = state
        self.ipfs = ipfs
        self.submit_tx = submit_tx
        self.model_id = model_id
        self.init_bytes = init_bytes
        self.rules = rules or Rules(
            deposit=20,
            gamma=1.3,
            cap_n=1000,
            intent_cap=3,
            refund_policy="treasury",
            round_deadline_slots=8,
            round_reward=fund_amount,
        )
        self.fund_amount = fund_amount
        self.running = True
        self._round_start_slot = 0     # track when each round began
        self._last_round_seen = -1
        self._last_finalized = -1

    


    def create_listing(self):
        cid = self.ipfs.put(self.init_bytes, tag="init")
        listing_tx = {
            "type": "ListingTx",
            "lister": self.acct,
            "model_id": self.model_id,
            "init_cid": cid,
            "rules": self.rules,
            "fee": 3,  # give listing a higher priority
        }
        self.submit_tx(listing_tx)

        # Wait until the model is visible to everyone
        while True:
            with self.state.lock():
                if self.model_id in self.state.models:
                    # mark round start slot for round 0
                    self._round_start_slot = self.state.slot
                    self._last_round_seen = 0
                    break
            time.sleep(0.05)

        if self.fund_amount > 0:
            funding_tx = {
                "type": "FundingTx",
                "lister": self.acct,
                "model_id": self.model_id,
                "amount": self.fund_amount,
                "fee": 3,
            }
            self.submit_tx(funding_tx)

    def collect_revealed(self, round_: int) -> List[Ticket]:
        with self.state.lock():
            tickets = self.state.tickets_for(self.model_id, round_)
            return [t for t in tickets if t.revealed]

    
    def aggregate(self, tickets: List[Ticket]) -> Dict[str, Any]:
        # If ML task active and NumPy present, do FedAvg-style aggregation
        with self.state.lock():
            task = self.state.ml_tasks.get(self.model_id)
        if task and np is not None:
            D = int(task["D"])
            vectors = []
            refunds = []
            for t in tickets:
                ok, buf = self.ipfs.get(t.update_cid)
                if not ok or not buf:
                    continue
                try:
                    vec = np.fromstring(buf.decode("utf-8"), sep=",", dtype=float)
                    if vec.size == D:
                        vectors.append(vec)
                        refunds.append(t.ticket_id)  # refundable if retrievable & well-formed
                except Exception:
                    continue

            included = [t.ticket_id for t in tickets]
            if not vectors:
                # no usable updates; keep the old "empty" finalize path
                global_cid = self.ipfs.put(b"global-empty", tag="global")
                return {"included": included, "scores": [0.0]*len(included), "refunds": refunds, "global_cid": global_cid}

            delta_mean = np.mean(vectors, axis=0)
            with self.state.lock():
                w_before = task["w"].copy()
                Xv, yv = task["val"]

            w_after = w_before + delta_mean
            acc_b = self._val_acc(w_before, Xv, yv)
            acc_a = self._val_acc(w_after, Xv, yv)
            impr = max(0.0, acc_a - acc_b)

            # commit the new global weights and log the round ML metrics
            with self.state.lock():
                task["w"] = w_after
                # determine current round index to log
                r = self.state.models[self.model_id].round_index
                task["history"].append({
                    "round": int(r),
                    "acc_before": float(acc_b),
                    "acc_after": float(acc_a),
                    "impr": float(impr),
                    "k": int(len(vectors)),
                })

            # score: share pool proportional to the round's overall improvement (equal split per included)
            # (You can refine to per-ticket contribution later.)
            score_per = float(impr) if len(included) > 0 else 0.0
            scores = [score_per for _ in included]
            global_bytes = f"global-ml-r{int(r)}-{random.randint(0, 1_000_000)}".encode()
            global_cid = self.ipfs.put(global_bytes, tag="global")
            return {"included": included, "scores": scores, "refunds": refunds, "global_cid": global_cid}

        # -------- Fallback path (no NumPy or ML disabled): keep your deterministic scoring ----------
        included = []
        scores = []
        refunds = []
        for t in tickets:
            ok, _data = self.ipfs.get(t.update_cid)
            if ok:
                refunds.append(t.ticket_id)  # refundable if retrievable

            # synthesize metrics from CID for determinism
            h = int(t.update_cid[:8], 16)
            n = (h % 2000) + 1
            delta = ((h // 13) % 200) / 1000.0 - 0.05  # ~[-0.05 .. 0.145]
            w_ = min(n, self.rules.cap_n)
            s = max(0.0, delta) * w_
            included.append(t.ticket_id)
            scores.append(s)

        global_bytes = b"global-" + str(random.randint(0, 1_000_000)).encode()
        global_cid = self.ipfs.put(global_bytes, tag="global")
        return {"included": included, "scores": scores, "refunds": refunds, "global_cid": global_cid}


    def finalize(self, round_, payload):
        tx = {
            "type": "GlobalUpdateTx",
            "lister": self.acct,
            "model_id": self.model_id,
            "round": round_,
            "global_cid": payload["global_cid"],
            "included": payload["included"],
            "scores": payload["scores"],
            "refunds": payload["refunds"],
            "fee": 2,  # prioritize finalize a bit
        }
        self.submit_tx(tx)

    def _maybe_roll_round_start(self):
        with self.state.lock():
            r = self.state.models[self.model_id].round_index
            s = self.state.slot
        if r != self._last_round_seen:
            self._last_round_seen = r
            self._round_start_slot = s

    def run(self):
        self.create_listing()
        time.sleep(0.6)  # small slack for inclusion

        while self.running:
            self._maybe_roll_round_start()

            with self.state.lock():
                r = self.state.models[self.model_id].round_index
                now = self.state.slot
                deadline = self._round_start_slot + self.rules.round_deadline_slots

            # only attempt once per round
            if r == self._last_finalized:
                time.sleep(0.1)
                continue

            revealed = self.collect_revealed(round_=r)

            # Case A: we have revealed tickets -> aggregate & finalize
            if revealed:
                payload = self.aggregate(revealed)
                self.finalize(r, payload)
                self._last_finalized = r

            # Case B: no reveals but deadline passed -> finalize empty payload
            elif now >= deadline:
                empty_global = self.ipfs.put(b"global-empty", tag="global")
                payload = {"included": [], "scores": [], "refunds": [], "global_cid": empty_global}
                self.finalize(r, payload)
                self._last_finalized = r

            time.sleep(0.1)

    def stop(self):
        self.running = False

    @staticmethod
    def _val_acc(w: "np.ndarray", Xv: "np.ndarray", yv: "np.ndarray") -> float:
        # Ensure arrays and compatible shapes
        w  = np.asarray(w)
        Xv = np.asarray(Xv)
        yv = np.asarray(yv)

        # Support w as (d,) or (d,1); always flatten the logits to 1-D
        z = (Xv @ w).reshape(-1)          # <- critical: avoid (n,1) vs (n,) broadcasting

        # Stable sigmoid (avoid overflow on large |z|)
        # Clip z so np.exp doesn't overflow; keeps results identical in [~1e-22, 1-1e-22]
        z = np.clip(z, -50.0, 50.0)
        p = 1.0 / (1.0 + np.exp(-z))

        # Labels can be {0,1} or {-1,1}; normalize to {0,1}
        if yv.ndim > 1:
            yv = yv.reshape(-1)
        if yv.min() < 0:                  # treat {-1,1} as {0,1}
            yv = ((yv > 0).astype(np.int8))

        preds = (p >= 0.5).astype(np.int8)

        # Now preds and yv are both shape (n,), so this is O(n) as intended
        return float((preds == yv).mean())
