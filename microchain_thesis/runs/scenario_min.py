# microchain/runs/scenario_min.py
from __future__ import annotations
import time, argparse, random, csv, os
from typing import List, Dict
try:
    import numpy as np
except Exception:
    np = None

from ..core.state import State, Rules
from ..roles.validator import Validator, InProcNetwork
from ..roles.lister import Lister
from ..roles.client import Client
from ..storage.ipfs_stub import IPFS

def give_faucet(st: State, who: str, amt: int):
    with st.lock():
        st.credit(who, amt)

def _seed_everything(state: State, seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    if np is not None:
        try:
            np.random.seed(seed)
        except Exception:
            pass
    with state.lock():
        state.epoch_randomness = (str(seed) + "|genesis").encode()
        state.epoch_randomness = __import__("hashlib").sha256(state.epoch_randomness).digest()

def _init_synth_ml(state: State, model_id: str, D: int, n_val: int, seed: int | None):
    """
    Register a tiny logistic-regression synthetic validation task.
    """
    if np is None:
        return False
    if seed is not None:
        rng = np.random.default_rng(seed ^ 0xA5A5A5A5)
    else:
        rng = np.random.default_rng()

    # true weights used to make labels; model starts at zeros
    true_w = rng.normal(0.0, 1.0, size=D)
    Xv = rng.normal(0.0, 1.0, size=(n_val, D))
    z = Xv @ true_w
    z = np.clip(z, -50.0, 50.0)
    p = 1.0 / (1.0 + np.exp(-z))
    yv = (rng.random(n_val) < p).astype(np.int8)

    with state.lock():
        state.ml_tasks[model_id] = {
            "D": int(D),
            "w": np.zeros(D, dtype=float),
            "val": (Xv, yv),
            "history": [],  # appended in Lister.aggregate()
        }
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validators", type=int, default=3)
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--deposit", type=int, default=20)
    ap.add_argument("--reward", type=int, default=1000)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--runtime", type=float, default=8.0, help="seconds to run")
    ap.add_argument("--remainder_sink", choices=["treasury","model_pool","burn"], default="treasury")
    ap.add_argument("--rand_period", type=int, default=16, help="rotate randomness every K slots")
    # NEW knobs
    ap.add_argument("--seed", type=int, default=None, help="deterministic seed")
    ap.add_argument("--ipfs_fail_rate", type=float, default=0.0)
    ap.add_argument("--intent_cap", type=int, default=3)
    ap.add_argument("--fee_policy", choices=["proposer","treasury","split"], default="proposer")
    ap.add_argument("--csv_out", type=str, default=None, help="append one-line CSV of metrics to this path")
    # NEW: ML toggles (off by default)
    ap.add_argument("--ml_enable", action="store_true", help="enable tiny synthetic ML task")
    ap.add_argument("--ml_dim", type=int, default=32)
    ap.add_argument("--ml_val", type=int, default=512)
    args = ap.parse_args()

    state = State()
    state.rand_period_slots = max(1, args.rand_period)
    state.fee_policy = args.fee_policy
    _seed_everything(state, args.seed)

    net = InProcNetwork(state, tau=args.tau)
    ipfs = IPFS(fail_rate=args.ipfs_fail_rate)

    # seed validators
    for i in range(args.validators):
        vid = f"V{i+1}"
        give_faucet(state, vid, 0)
        with state.lock():
            state.register_validator(vid, stake=100)

    validators: List[Validator] = []
    for i in range(args.validators):
        v = Validator(vid=f"V{i+1}", secret=f"sk{i+1}".encode(), state=state, net=net)
        v.start(); validators.append(v)

    def submit_tx(tx: dict):
        validators[0].submit_tx(tx)

    # Faucets
    lister_fee_budget = 10
    minted_total = 0
    give_faucet(state, "Lister", args.reward + lister_fee_budget); minted_total += (args.reward + lister_fee_budget)
    for c in range(args.clients):
        give_faucet(state, f"C{c+1}", 200); minted_total += 200

    # Optional ML task
    model_id = "toy"
    if args.ml_enable:
        ok = _init_synth_ml(state, model_id=model_id, D=args.ml_dim, n_val=args.ml_val, seed=args.seed)
        if not ok:
            print("[ML] NumPy not available; running without ML.")
    else:
        # ensure attribute exists to avoid AttributeError even when ML disabled
        with state.lock():
            state.ml_tasks.setdefault(model_id, {})

    rules = Rules(
        deposit=args.deposit, gamma=1.3, cap_n=1000, intent_cap=args.intent_cap,
        refund_policy="treasury", round_deadline_slots=8, round_reward=0,
        remainder_sink=args.remainder_sink,
    )
    lister = Lister(acct="Lister", state=state, ipfs=ipfs,
                    submit_tx=submit_tx, model_id=model_id, rules=rules,
                    fund_amount=args.reward)
    lister.start()

    clients: List[Client] = []
    for c in range(args.clients):
        cl = Client(acct=f"C{c+1}", state=state, ipfs=ipfs,
                    submit_tx=submit_tx, model_id=model_id, data_seed=c+1)
        cl.start(); clients.append(cl)

    # clock
    t0 = time.time()
    try:
        while time.time() - t0 < args.runtime:
            with state.lock():
                state.slot += 1
                if state.slot % state.rand_period_slots == 0:
                    state.rotate_randomness()
            time.sleep(state.slot_duration_sec)
    finally:
        for cl in clients: cl.stop()
        lister.stop()
        for v in validators: v.stop()
        net.stop()

    # final claims (zero-fee claim okay)
    time.sleep(0.8)
    with state.lock():
        for i in range(args.clients):
            acct = f"C{i+1}"
            elig = state.entitlement_get(acct, model_id)
            if elig > 0:
                nonce = state.claim_nonce.get(acct, 0)
                validators[0].submit_tx({"type":"ClaimTx","acct":acct,"model_id":model_id,
                                         "amount":elig,"nonce":nonce,"fee":0})
    time.sleep(0.6)

    # --- Summary ---
    with state.lock():
        print("\n=== SUMMARY ===")
        model = state.models.get(model_id)
        if model is None:
            print(f"Model '{model_id}' was not created (lister may have crashed).")
            return
        print("Round index:", model.round_index)
        for c in range(args.clients):
            acct = f"C{c+1}"
            bal = state.balances.get(acct, 0)
            ent = state.entitlement_get(acct, model_id)
            print(f"{acct}: balance={bal}, entitlement={ent}")
        print("Lister balance:", state.balances.get("Lister", 0))
        print("Treasury:", state.treasury)

        total_balances = sum(state.balances.values())
        total_ents     = sum(state.entitlements.values())
        total_pools    = sum(state.pools.values())
        locked_depos   = sum(t.deposit for t in state.tickets.values() if not t.settled)

        rounds = len(state.round_log)
        alloc_total = sum(r["allocated"] for r in state.round_log)
        refund_total = sum(r["refund_total"] for r in state.round_log)
        forfeit_total = sum(r["forfeit_total"] for r in state.round_log)
        remainder_total = sum(r["remainder"] for r in state.round_log)
        included_total = sum(r["included"] for r in state.round_log)
        refunds_count_total = sum(r["refunds"] for r in state.round_log)

        print("Money minted       :", minted_total)
        print("Stake total        :", sum(state.validator_stake.values()))
        print("--- LEDGER BREAKDOWN ---")
        print("Balances sum       :", total_balances)
        print("Entitlements sum   :", total_ents)
        print("Pools sum          :", total_pools)
        print("Locked deposits    :", locked_depos)
        print("Treasury           :", state.treasury)
        conserved = total_balances + total_ents + total_pools + locked_depos + state.treasury
        print("Conserved money    :", conserved)

        # VRF / Production metrics
        elig_counts: Dict[str,int] = {}
        for slot, vids in state.vrf_eligibility.items():
            for v in set(vids):
                elig_counts[v] = elig_counts.get(v, 0) + 1

        produced_counts: Dict[str,int] = {}
        for ev in state.block_log:
            produced_counts[ev["proposer"]] = produced_counts.get(ev["proposer"], 0) + 1

        print("\n--- VRF / PRODUCTION ---")
        for v in sorted(state.validator_stake.keys()):
            e = elig_counts.get(v, 0)
            p = produced_counts.get(v, 0)
            print(f"{v}: eligible_slots={e}, blocks_accepted={p}")

        if state.block_log:
            last_slots = sorted({ev["slot"] for ev in state.block_log})[-5:]
            print("\nPer-slot (eligible -> proposer):")
            for s in last_slots:
                elig = state.vrf_eligibility.get(s, [])
                proposers = [ev["proposer"] for ev in state.block_log if ev["slot"] == s]
                who = proposers[0] if proposers else "-"
                txs = sum(ev["txs"] for ev in state.block_log if ev["slot"] == s)
                fees = sum(ev["fees_paid"] for ev in state.block_log if ev["slot"] == s)
                print(f"  slot {s}: eligible={sorted(set(elig))} -> proposer={who} (txs={txs}, fees={fees})")

        last = state.block_log[-10:]
        if last:
            print("\nLast blocks (slot,height,proposer,txs,fees):")
            for ev in last:
                print(f"  s{ev['slot']}, h{ev['height']}, {ev['proposer']}, txs={ev['txs']}, fees={ev['fees_paid']}")

        if state.round_log:
            print("\n--- ROUND EVENTS ---")
            for r in state.round_log[-5:]:
                print(f"  model={r['model']} round={r['round']} pool={r['pool']} "
                      f"included={r['included']} refunds={r['refunds']} "
                      f"refund_total={r['refund_total']} forfeit_total={r['forfeit_total']} "
                      f"allocated={r['allocated']} remainder={r['remainder']}->{r['remainder_sink']}")

        print("\n--- FORKS ---")
        print("forks_observed:", state.forks_observed)
        if state.fork_log:
            for fk in state.fork_log[-5:]:
                print(f"  side-block: s{fk['slot']} by {fk['proposer']} parent={fk['parent']} head_at_recv={fk['head_at_receive']}")

        print("\n--- FEES ---")
        print(f"policy={state.fee_policy} total_charged={state.total_fees_charged} total_distributed={state.total_fees_distributed}")
        def _fmt(kv): return ", ".join(f"{k}:{v}" for k,v in sorted(kv.items()))
        print("fees_paid_by_acct   :", _fmt(state.fees_paid_by_acct))
        print("fees_earned_by_acct :", _fmt(state.fees_earned_by_acct))

        blocks_applied = len(state.block_log)
        txs_applied = sum(ev["txs"] for ev in state.block_log)
        avg_txs_per_block = (txs_applied / blocks_applied) if blocks_applied else 0.0
        avg_fee_per_tx = (state.total_fees_charged / txs_applied) if txs_applied else 0.0
        unique_elig = sum(len(set(v)) for v in state.vrf_eligibility.values())
        missed_proposals = max(0, unique_elig - blocks_applied)

        print("\n--- DERIVED ---")
        print(f"blocks_applied={blocks_applied} txs_applied={txs_applied} avg_txs/block={avg_txs_per_block:.2f} avg_fee/tx={avg_fee_per_tx:.2f}")
        print(f"unique_eligibilities={unique_elig} missed_proposals={missed_proposals}")

        # --- ML SUMMARY (if enabled) ---
        task = state.ml_tasks.get(model_id)
        if task and isinstance(task, dict) and task.get("history"):
            print("\n--- ML ---")
            for h in task["history"][-5:]:
                print(f"  round={h['round']} acc_before={h['acc_before']:.3f} acc_after={h['acc_after']:.3f} "
                      f"impr={h['impr']:.3f} clients_used={h['k']}")
        ml_rounds = 0; ml_acc_before = 0.0; ml_acc_after = 0.0; ml_impr_sum = 0.0; ml_clients_used = 0
        task = getattr(state, "ml_tasks", {}).get("toy") if hasattr(state, "ml_tasks") else None
        if task and task.get("history"):
            ml_rounds = len(task["history"])
            ml_acc_before = task["history"][-1]["acc_before"]
            ml_acc_after  = task["history"][-1]["acc_after"]
            ml_impr_sum   = sum(h["impr"] for h in task["history"])
            ml_clients_used = sum(h["k"] for h in task["history"])

        # optional CSV
        if args.csv_out:
            row = {
                "validators": args.validators,
                "clients": args.clients,
                "deposit": args.deposit,
                "reward": args.reward,
                "tau": args.tau,
                "runtime": args.runtime,
                "remainder_sink": args.remainder_sink,
                "rand_period": args.rand_period,
                "seed": args.seed if args.seed is not None else -1,
                "ipfs_fail_rate": args.ipfs_fail_rate,
                "intent_cap": args.intent_cap,
                "fee_policy": args.fee_policy,
                "round_index": model.round_index,
                "balances_sum": total_balances,
                "entitlements_sum": total_ents,
                "pools_sum": total_pools,
                "locked_deposits": locked_depos,
                "treasury": state.treasury,
                "conserved_money": conserved,
                "blocks_applied": blocks_applied,
                "txs_applied": txs_applied,
                "avg_txs_per_block": f"{avg_txs_per_block:.4f}",
                "avg_fee_per_tx": f"{avg_fee_per_tx:.6f}",
                "unique_eligibilities": unique_elig,
                "missed_proposals": missed_proposals,
                "forks_observed": state.forks_observed,
                "fees_charged": state.total_fees_charged,
                "fees_distributed": state.total_fees_distributed,
                "rounds": rounds,
                "alloc_total": alloc_total,
                "refund_total": refund_total,
                "forfeit_total": forfeit_total,
                "remainder_total": remainder_total,
                "included_total": included_total,
                "refunds_count_total": refunds_count_total,
                "ml_rounds": ml_rounds,
                "ml_last_acc_before": f"{ml_acc_before:.3f}",
                "ml_last_acc_after":  f"{ml_acc_after:.3f}",
                "ml_impr_sum": f"{ml_impr_sum:.4f}",
                "ml_clients_used_sum": ml_clients_used,
            }
            exists = os.path.exists(args.csv_out)
            with open(args.csv_out, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not exists:
                    w.writeheader()
                w.writerow(row)

if __name__ == "__main__":
    main()
