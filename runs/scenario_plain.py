#microchain/runs/scenario_plain.py
from __future__ import annotations
import time, argparse, csv, os, random
from typing import List, Dict, Optional
try:
    import numpy as np
except Exception:
    np = None

from ..storage.ipfs_stub import IPFS

def _median(xs: List[float]) -> float:
    if not xs: return 0.0
    s = sorted(xs); n = len(s); m = n // 2
    if n % 2 == 1: return float(s[m])
    return float(0.5 * (s[m-1] + s[m]))

class PlainState:
    def __init__(self, slot_duration_sec: float = 0.4):
        self.slot = 0
        self.slot_duration_sec = slot_duration_sec
        self.round_index = 0
        self.metrics = {
            "round_dur_slots": [],
            "ipfs_put_ms": [],
            "ipfs_get_ms": [],
            "aggregate_ms": [],
            "clients_used_per_round": [],
        }

class PlainClient:
    def __init__(self, acct: str, model_id: str, state: PlainState, data_seed: int = 0):
        self.acct = acct
        self.model_id = model_id
        self.state = state
        self.data_seed = data_seed

    def ml_delta_csv(self, D: int) -> str:
        if np is None:
            # deterministic fallback vector
            rnd = random.Random(hash((self.acct, self.data_seed, self.state.round_index)) & 0xffffffff)
            return ",".join(f"{(rnd.random()-0.5)*0.1:.6f}" for _ in range(D))
        r = self.state.round_index
        seed = abs(hash((self.acct, self.data_seed, r))) % (2**32)
        rng = np.random.default_rng(seed)
        delta = rng.normal(loc=0.0, scale=0.05, size=D)
        return ",".join(f"{x:.6f}" for x in delta)

class PlainLister:
    def __init__(self, state: PlainState, ipfs: Optional[IPFS], model_id: str, ml_dim: int, ml_val: int, seed: Optional[int]):
        self.state = state
        self.ipfs = ipfs
        self.model_id = model_id
        self.ml_dim = ml_dim
        self.ml_val = ml_val
        self.rng = np.random.default_rng(seed ^ 0xA5A5A5A5) if (np is not None and seed is not None) else None
        if np is not None:
            if self.rng is None:
                self.rng = np.random.default_rng()
            true_w = self.rng.normal(0.0, 1.0, size=ml_dim)
            Xv = self.rng.normal(0.0, 1.0, size=(ml_val, ml_dim))
            z = Xv @ true_w
            z = np.clip(z, -50.0, 50.0)
            p = 1.0 / (1.0 + np.exp(-z))
            yv = (self.rng.random(ml_val) < p).astype(np.int8)
            self.w = np.zeros(ml_dim, dtype=float)
            self.Xv = Xv; self.yv = yv
        else:
            self.w = None; self.Xv = None; self.yv = None

    def _val_acc(self, w, Xv, yv) -> float:
        if np is None:
            return 0.5
        z = (Xv @ w).reshape(-1)
        z = np.clip(z, -50.0, 50.0)
        p = 1.0 / (1.0 + np.exp(-z))
        preds = (p >= 0.5).astype(np.int8)
        return float((preds == yv).mean())

    def aggregate_and_finalize(self, deltas: List[str], cap_collect_slots: int = 4):
        start_slot = int(self.state.slot)
        agg_t0 = time.perf_counter()

        vectors = []
        refunds = 0
        if self.ipfs is None:
            for d in deltas:
                try:
                    vec = np.fromstring(d, sep=",", dtype=float) if np is not None else None
                    if np is None or vec.size > 0:
                        vectors.append(vec if np is not None else None)
                        refunds += 1
                except Exception:
                    pass
        else:
            for d in deltas:
                p0 = time.perf_counter()
                cid = self.ipfs.put(d.encode("utf-8"), tag="update")
                p1 = time.perf_counter()
                self.state.metrics["ipfs_put_ms"].append((p1 - p0) * 1000.0)
                g0 = time.perf_counter()
                ok, buf = self.ipfs.get(cid)
                g1 = time.perf_counter()
                self.state.metrics["ipfs_get_ms"].append((g1 - g0) * 1000.0)
                if ok and buf:
                    if np is not None:
                        vec = np.fromstring(buf.decode("utf-8"), sep=",", dtype=float)
                        if vec.size == self.ml_dim:
                            vectors.append(vec)
                            refunds += 1
                    else:
                        vectors.append(None)
                        refunds += 1

        clients_used = len(vectors)

        if np is not None and clients_used > 0:
            delta_mean = np.mean(vectors, axis=0)
            acc_b = self._val_acc(self.w, self.Xv, self.yv)
            self.w = self.w + delta_mean
            acc_a = self._val_acc(self.w, self.Xv, self.yv)
        else:
            acc_b = acc_a = 0.5

        if self.ipfs is not None:
            p0 = time.perf_counter()
            _cid = self.ipfs.put(b"global-plain", tag="global")
            p1 = time.perf_counter()
            self.state.metrics["ipfs_put_ms"].append((p1 - p0) * 1000.0)

        agg_t1 = time.perf_counter()
        self.state.metrics["aggregate_ms"].append((agg_t1 - agg_t0) * 1000.0)

        finalize_slot = int(start_slot + cap_collect_slots)  # emulate fixed window close
        dur_slots = max(0, finalize_slot - start_slot)
        self.state.metrics["round_dur_slots"].append(dur_slots)
        self.state.metrics["clients_used_per_round"].append(clients_used)
        self.state.round_index += 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--runtime", type=float, default=8.0)
    ap.add_argument("--slot_duration_sec", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ml_enable", action="store_true")
    ap.add_argument("--ml_dim", type=int, default=32)
    ap.add_argument("--ml_val", type=int, default=512)
    ap.add_argument("--use_ipfs", action="store_true", help="Plain+IPFS if set; else Plain+Mem")
    ap.add_argument("--ipfs_fail_rate", type=float, default=0.0)
    ap.add_argument("--csv_out", type=str, default=None)
    args = ap.parse_args()

    random.seed(args.seed or 0)
    state = PlainState(slot_duration_sec=args.slot_duration_sec)
    ipfs = IPFS(fail_rate=args.ipfs_fail_rate) if args.use_ipfs else None
    model_id = "toy"

    clients = [PlainClient(acct=f"C{i+1}", model_id=model_id, state=state, data_seed=i+1)
               for i in range(args.clients)]
    lister = PlainLister(state=state, ipfs=ipfs, model_id=model_id,
                         ml_dim=args.ml_dim, ml_val=args.ml_val, seed=args.seed)

    t0 = time.time()
    try:
        while time.time() - t0 < args.runtime:
            # emulate one collection window per (few) slots
            deltas = []
            for c in clients:
                if args.ml_enable:
                    deltas.append(c.ml_delta_csv(args.ml_dim))
                else:
                    deltas.append("0.0")
            lister.aggregate_and_finalize(deltas, cap_collect_slots=4)
            # advance a few slots
            for _ in range(4):
                state.slot += 1
                time.sleep(state.slot_duration_sec)
    finally:
        pass

    dur_slots_med = _median(state.metrics["round_dur_slots"])
    dur_secs_med = dur_slots_med * state.slot_duration_sec
    ipfs_put_ms_med = _median(state.metrics["ipfs_put_ms"])
    ipfs_get_ms_med = _median(state.metrics["ipfs_get_ms"])
    aggregate_ms_med = _median(state.metrics["aggregate_ms"])

    print("\n=== PLAIN SUMMARY ===")
    print(f"round_index={state.round_index}")
    print(f"round_dur_slots_median={dur_slots_med} round_dur_secs_median={dur_secs_med:.6f}")
    print(f"ipfs_put_ms_median={ipfs_put_ms_med:.3f} ipfs_get_ms_median={ipfs_get_ms_med:.3f} aggregate_ms_median={aggregate_ms_med:.3f}")

    if args.csv_out:
        row = {
            "mode": "plain_ipfs" if args.use_ipfs else "plain_mem",
            "clients": args.clients,
            "runtime": args.runtime,
            "slot_duration_sec": args.slot_duration_sec,
            "seed": args.seed if args.seed is not None else -1,
            "ipfs_fail_rate": args.ipfs_fail_rate if args.use_ipfs else 0.0,
            "ml_enable": int(bool(args.ml_enable)),
            "ml_dim": args.ml_dim,
            "ml_val": args.ml_val,
            "round_index": state.round_index,
            "round_dur_slots_median": int(dur_slots_med),
            "round_dur_secs_median": f"{dur_secs_med:.6f}",
            "ipfs_put_ms_median": f"{ipfs_put_ms_med:.3f}",
            "ipfs_get_ms_median": f"{ipfs_get_ms_med:.3f}",
            "aggregate_ms_median": f"{aggregate_ms_med:.3f}",
        }
        exists = os.path.exists(args.csv_out)
        with open(args.csv_out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                w.writeheader()
            w.writerow(row)

if __name__ == "__main__":
    main()
