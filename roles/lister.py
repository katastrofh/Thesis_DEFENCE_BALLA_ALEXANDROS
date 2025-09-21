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
        self._round_start_slot = 0
        self._last_round_seen = -1
        self._last_finalized = -1

    def create_listing(self):
        t0 = time.perf_counter()
        cid = self.ipfs.put(self.init_bytes, tag="init")
        t1 = time.perf_counter()
        with self.state.lock():
            self.state.record_ipfs_put_ms((t1 - t0) * 1000.0)

        with self.state.lock(): created_slot = self.state.slot
        listing_tx = {
            "type": "ListingTx",
            "lister": self.acct,
            "model_id": self.model_id,
            "init_cid": cid,
            "rules": self.rules,
            "fee": 3,
            "created_slot": created_slot,
        }
        self.submit_tx(listing_tx)

        while True:
            with self.state.lock():
                if self.model_id in self.state.models:
                    self._round_start_slot = self.state.slot
                    self._last_round_seen = 0
                    self.state.last_round_start_slot = self._round_start_slot
                    break
            time.sleep(0.05)

        if self.fund_amount > 0:
            with self.state.lock(): created_slot_f = self.state.slot
            funding_tx = {
                "type": "FundingTx",
                "lister": self.acct,
                "model_id": self.model_id,
                "amount": self.fund_amount,
                "fee": 3,
                "created_slot": created_slot_f,
            }
            self.submit_tx(funding_tx)

    def collect_revealed(self, round_: int) -> List[Ticket]:
        with self.state.lock():
            tickets = self.state.tickets_for(self.model_id, round_)
            return [t for t in tickets if t.revealed]

    def aggregate(self, tickets: List[Ticket]) -> Dict[str, Any]:
        with self.state.lock():
            task = self.state.ml_tasks.get(self.model_id)

        # Measure aggregation wall time (includes IPFS gets)
        agg_t0 = time.perf_counter()

        if task and np is not None:
            D = int(task["D"])
            vectors = []
            refunds = []
            for t in tickets:
                g0 = time.perf_counter()
                ok, buf = self.ipfs.get(t.update_cid)
                g1 = time.perf_counter()
                with self.state.lock():
                    self.state.record_ipfs_get_ms((g1 - g0) * 1000.0)
                if not ok or not buf:
                    continue
                try:
                    vec = np.fromstring(buf.decode("utf-8"), sep=",", dtype=float)
                    if vec.size == D:
                        vectors.append(vec)
                        refunds.append(t.ticket_id)
                except Exception:
                    continue

            included = [t.ticket_id for t in tickets]
            if not vectors:
                gb0 = time.perf_counter()
                global_cid = self.ipfs.put(b"global-empty", tag="global")
                gb1 = time.perf_counter()
                with self.state.lock():
                    self.state.record_ipfs_put_ms((gb1 - gb0) * 1000.0)
                agg_t1 = time.perf_counter()
                with self.state.lock():
                    self.state.record_aggregate_ms((agg_t1 - agg_t0) * 1000.0)
                return {"included": included, "scores": [0.0]*len(included), "refunds": refunds, "global_cid": global_cid}

            with self.state.lock():
                w_before = task["w"].copy()
                Xv, yv = task["val"]

            delta_mean = np.mean(vectors, axis=0)
            w_after = w_before + delta_mean

            acc_b = self._val_acc(w_before, Xv, yv)
            acc_a = self._val_acc(w_after, Xv, yv)
            impr = max(0.0, acc_a - acc_b)

            with self.state.lock():
                task["w"] = w_after
                r = self.state.models[self.model_id].round_index
                task["history"].append({
                    "round": int(r),
                    "acc_before": float(acc_b),
                    "acc_after": float(acc_a),
                    "impr": float(impr),
                    "k": int(len(vectors)),
                })

            score_per = float(impr) if len(included) > 0 else 0.0
            scores = [score_per for _ in included]
            gb0 = time.perf_counter()
            global_bytes = f"global-ml-r{int(r)}-{random.randint(0, 1_000_000)}".encode()
            global_cid = self.ipfs.put(global_bytes, tag="global")
            gb1 = time.perf_counter()
            with self.state.lock():
                self.state.record_ipfs_put_ms((gb1 - gb0) * 1000.0)

            agg_t1 = time.perf_counter()
            with self.state.lock():
                self.state.record_aggregate_ms((agg_t1 - agg_t0) * 1000.0)
            return {"included": included, "scores": scores, "refunds": refunds, "global_cid": global_cid}

        # Fallback deterministic path
        included = []
        scores = []
        refunds = []
        for t in tickets:
            g0 = time.perf_counter()
            ok, _data = self.ipfs.get(t.update_cid)
            g1 = time.perf_counter()
            with self.state.lock():
                self.state.record_ipfs_get_ms((g1 - g0) * 1000.0)
            if ok:
                refunds.append(t.ticket_id)
            h = int(t.update_cid[:8], 16)
            n = (h % 2000) + 1
            delta = ((h // 13) % 200) / 1000.0 - 0.05
            w_ = min(n, self.rules.cap_n)
            s = max(0.0, delta) * w_
            included.append(t.ticket_id)
            scores.append(s)

        gb0 = time.perf_counter()
        global_bytes = b"global-" + str(random.randint(0, 1_000_000)).encode()
        global_cid = self.ipfs.put(global_bytes, tag="global")
        gb1 = time.perf_counter()
        with self.state.lock():
            self.state.record_ipfs_put_ms((gb1 - gb0) * 1000.0)

        agg_t1 = time.perf_counter()
        with self.state.lock():
            self.state.record_aggregate_ms((agg_t1 - agg_t0) * 1000.0)

        return {"included": included, "scores": scores, "refunds": refunds, "global_cid": global_cid}

    def finalize(self, round_, payload):
        with self.state.lock():
            created_slot = self.state.slot
            round_start_slot = self._round_start_slot
        tx = {
            "type": "GlobalUpdateTx",
            "lister": self.acct,
            "model_id": self.model_id,
            "round": round_,
            "global_cid": payload["global_cid"],
            "included": payload["included"],
            "scores": payload["scores"],
            "refunds": payload["refunds"],
            "fee": 2,
            "created_slot": created_slot,
            "round_start_slot": round_start_slot,
        }
        self.submit_tx(tx)

    def _maybe_roll_round_start(self):
        with self.state.lock():
            r = self.state.models[self.model_id].round_index
            s = self.state.slot
        if r != self._last_round_seen:
            self._last_round_seen = r
            self._round_start_slot = s
            with self.state.lock():
                self.state.last_round_start_slot = self._round_start_slot

    def run(self):
        self.create_listing()
        time.sleep(0.6)

        while self.running:
            self._maybe_roll_round_start()

            with self.state.lock():
                r = self.state.models[self.model_id].round_index
                now = self.state.slot
                deadline = self._round_start_slot + self.rules.round_deadline_slots

            if r == self._last_finalized:
                time.sleep(0.1)
                continue

            revealed = self.collect_revealed(round_=r)

            if revealed:
                payload = self.aggregate(revealed)
                self.finalize(r, payload)
                self._last_finalized = r

            elif now >= deadline:
                gb0 = time.perf_counter()
                empty_global = self.ipfs.put(b"global-empty", tag="global")
                gb1 = time.perf_counter()
                with self.state.lock():
                    self.state.record_ipfs_put_ms((gb1 - gb0) * 1000.0)
                payload = {"included": [], "scores": [], "refunds": [], "global_cid": empty_global}
                self.finalize(r, payload)
                self._last_finalized = r

            time.sleep(0.1)

    def stop(self):
        self.running = False

    @staticmethod
    def _val_acc(w: "np.ndarray", Xv: "np.ndarray", yv: "np.ndarray") -> float:
        w  = np.asarray(w)
        Xv = np.asarray(Xv)
        yv = np.asarray(yv)
        z = (Xv @ w).reshape(-1)
        z = np.clip(z, -50.0, 50.0)
        p = 1.0 / (1.0 + np.exp(-z))
        if yv.ndim > 1:
            yv = yv.reshape(-1)
        if yv.min() < 0:
            yv = ((yv > 0).astype(np.int8))
        preds = (p >= 0.5).astype(np.int8)
        return float((preds == yv).mean())

