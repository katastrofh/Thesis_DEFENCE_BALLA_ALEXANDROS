# microchain/core/apply.py
from __future__ import annotations
import hashlib
from dataclasses import is_dataclass, asdict
from typing import Dict, Any, List
from .state import State, Rules, Ticket

def _norm_for_hash(x):
    if is_dataclass(x): x = asdict(x)
    if isinstance(x, dict):  return {k: _norm_for_hash(v) for k, v in sorted(x.items())}
    if isinstance(x, list):  return [_norm_for_hash(v) for v in x]
    if isinstance(x, tuple): return tuple(_norm_for_hash(v) for v in x)
    return x

def _tx_hash(tx: Dict[str, Any]) -> str:
    s = repr(_norm_for_hash(tx))
    return hashlib.sha256(s.encode()).hexdigest()

def _charge_fee(st: State, payer: str, fee: int) -> bool:
    if fee <= 0: return True
    ok = st.debit(payer, fee)
    if not ok: return False
    # fees are pooled to treasury first; proposer distribution happens in validator
    st.treasury += fee
    st.record_fee_paid(payer, fee)
    return True

def TICKET_CREATE(st: State, client: str, model_id: str, round_: int, deposit: int, expiry_slot: int) -> Ticket:
    ok = st.debit(client, deposit)
    if not ok: return None
    return st.new_ticket(owner=client, model_id=model_id, round_=round_, deposit=deposit, expiry_slot=expiry_slot)

def TICKET_REVEAL(st: State, ticket_id: int, update_cid: str):
    t = st.tickets.get(ticket_id)
    if not t: return
    t.revealed = True
    t.update_cid = update_cid

def ENTITLEMENT_TRANSFER(st: State, acct: str, model_id: str, amount: int):
    elig = st.entitlement_get(acct, model_id)
    if amount <= 0 or amount > elig:
        return
    st.entitlement_sub(acct, model_id, amount)
    st.credit(acct, amount)

def ADVANCE_ROUND(st: State, model_id: str, round_: int, global_cid: str,
                  included: List[int], scores: List[float], refunds: List[int],
                  start_slot: int, finalize_slot: int) -> bool:
    cur = st.get_round_index(model_id)
    if cur != round_:
        return False

    pool_key = (model_id, round_)
    R = st.pools.get(pool_key, 0)
    if R <= 0:
        return False

    st.set_global_cid(model_id, global_cid)
    rules = st.models[model_id].rules

    # payouts
    gamma = rules.gamma
    weights = [max(0.0, s) ** gamma for s in scores]
    S = sum(weights)
    allocated = 0
    if S > 0 and R > 0:
        for tid, w in zip(included, weights):
            if w <= 0: continue
            t = st.tickets.get(tid)
            if not t or t.model_id != model_id or t.round != round_:
                continue
            acct = t.owner
            pay = int(R * (w / S))
            if pay > 0:
                st.entitlement_add(acct, model_id, pay)
                allocated += pay

    remainder_carry = 0
    remainder = max(0, R - allocated)
    if remainder > 0:
        sink = getattr(rules, "remainder_sink", "treasury")
        if sink == "model_pool":
            remainder_carry = remainder
        elif sink == "burn":
            pass
        else:
            st.treasury += remainder

    # refunds
    refundable_set = set(refunds)
    refund_total = 0
    for tid in refundable_set:
        t = st.tickets.get(tid)
        if not t or t.settled: continue
        st.entitlement_add(t.owner, model_id, t.deposit)
        refund_total += t.deposit
        t.refundable = True; t.refunded = True; t.settled = True

    # forfeits
    forfeit_total = 0
    for t in st.tickets_for(model_id, round_):
        if t.settled: continue
        t.forfeited = True; t.settled = True
        if rules.refund_policy == "treasury":
            st.treasury += t.deposit
        elif rules.refund_policy == "model_pool":
            st.pool_add(model_id, t.deposit)
        forfeit_total += t.deposit

    # consume pool R
    consume = min(R, st.pools.get(pool_key, 0))
    if consume > 0:
        st.pool_consume(model_id, consume)

    # round event + latency
    round_dur_slots = max(0, int(finalize_slot) - int(start_slot))
    st.record_round_dur_slots(round_dur_slots)
    st.round_log.append({
        "model": model_id,
        "round": round_,
        "pool": R,
        "included": len(included),
        "refunds": len(refundable_set),
        "refund_total": refund_total,
        "forfeit_total": forfeit_total,
        "allocated": allocated,
        "remainder": remainder,
        "remainder_sink": getattr(rules, "remainder_sink", "treasury"),
        "global_cid": global_cid,
        "start_slot": int(start_slot),
        "finalize_slot": int(finalize_slot),
        "round_dur_slots": round_dur_slots,
    })

    st.inc_round(model_id)
    if remainder_carry > 0:
        new_r = st.get_round_index(model_id)
        st.pool_add_at(model_id, new_r, remainder_carry)

    return True

def apply_tx(st: State, tx: Dict[str, Any]) -> bool:
    """
    Return True iff the tx mutated state.
    (We add tx hash to st.applied_txs only after success so retries aren’t poisoned.)
    """
    h = _tx_hash(tx)
    if h in st.applied_txs:
        return False

    t = tx["type"]
    fee = int(tx.get("fee", 0))

    if t == "ListingTx":
        if not _charge_fee(st, tx["lister"], fee): return False
        mid = tx["model_id"]
        if mid in st.models: return False
        st.add_model(model_id=mid, init_cid=tx["init_cid"], rules=tx["rules"])
        st.applied_txs.add(h)
        return True

    elif t == "FundingTx":
        if not _charge_fee(st, tx["lister"], fee): return False
        if tx["model_id"] not in st.models: return False
        if not st.debit(tx["lister"], tx["amount"]): return False
        st.pool_add(tx["model_id"], tx["amount"])
        st.applied_txs.add(h)
        return True

    elif t == "UpdateIntentTx":
        if not _charge_fee(st, tx["client"], fee): return False
        mid = tx["model_id"]
        if mid not in st.models: return False
        expiry = st.now_slot() + st.models[mid].rules.round_deadline_slots
        tkt = TICKET_CREATE(st, tx["client"], mid, tx["round"], tx["deposit"], expiry)
        if not tkt: return False
        st.applied_txs.add(h)
        return True

    elif t == "UpdatePublishTx":
        if not _charge_fee(st, tx["client"], fee): return False
        TICKET_REVEAL(st, tx["ticket_id"], tx["update_cid"])
        st.applied_txs.add(h)
        return True

    elif t == "GlobalUpdateTx":
        if not _charge_fee(st, tx["lister"], fee): return False
        start_slot = int(tx.get("round_start_slot", getattr(st, "last_round_start_slot", st.slot)))
        finalize_slot = int(st.slot)
        ok = ADVANCE_ROUND(st, tx["model_id"], tx["round"], tx["global_cid"],
                           tx["included"], tx["scores"], tx["refunds"],
                           start_slot=start_slot, finalize_slot=finalize_slot)
        if ok:
            st.applied_txs.add(h)
        return ok

    elif t == "ClaimTx":
        before = st.entitlement_get(tx["acct"], tx["model_id"])
        ENTITLEMENT_TRANSFER(st, tx["acct"], tx["model_id"], tx["amount"])
        after = st.entitlement_get(tx["acct"], tx["model_id"])
        if after < before:
            if not _charge_fee(st, tx["acct"], fee):
                return False
            st.claim_next_nonce(tx["acct"])
            st.applied_txs.add(h)
            return True
        return False

    else:
        return False
