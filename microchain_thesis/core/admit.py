# microchain/core/admit.py
from __future__ import annotations
from typing import Dict, Any
from .state import State, Ticket

###############################
# Admission helper primitives
###############################

def CHECK_DEPOSIT(st: State, acct: str, amount: int) -> bool:
    with st.lock():
        return st.balances.get(acct, 0) >= amount > 0

def WITHIN_EXPIRY(st: State, ticket_id: int) -> bool:
    with st.lock():
        t = st.tickets.get(ticket_id)
        if not t:
            return False
        return st.now_slot() <= t.expiry_slot

def CHECK_TICKET(st: State, ticket_id: int, acct: str) -> bool:
    with st.lock():
        t = st.tickets.get(ticket_id)
        if not t:
            return False
        return t.owner == acct

def ENFORCE_RATELIMITS(st: State, acct: str, model_id: str, round_: int) -> bool:
    with st.lock():
        rules = st.models[model_id].rules
        cap = rules.intent_cap
        if cap <= 0:
            return True
        count = 0
        for t in st.tickets_for(model_id, round_):
            if t.owner == acct:
                count += 1
        return count < cap

###############################
# Admission for each tx type
###############################

def admit_listing(st: State, tx: Dict[str, Any]) -> bool:
    with st.lock():
        if tx["model_id"] in st.models:
            return False
        cid = tx["init_cid"]
        if not isinstance(cid, str) or len(cid) < 16:
            return False
        fee = int(tx.get("fee", 0))
        return CHECK_DEPOSIT(st, tx["lister"], fee) if fee > 0 else True

def admit_funding(st: State, tx: Dict[str, Any]) -> bool:
    with st.lock():
        mid = tx["model_id"]
        if mid not in st.models:
            return False
        r = st.get_round_index(mid)
        if st.pools.get((mid, r), 0) > 0:
            return False
        fee = int(tx.get("fee", 0))
        need = tx["amount"] + max(0, fee)
        return CHECK_DEPOSIT(st, tx["lister"], need)

def admit_intent(st: State, tx: Dict[str, Any]) -> bool:
    with st.lock():
        mid = tx["model_id"]
        if mid not in st.models: return False
        if st.get_round_index(mid) != tx["round"]: return False
        if not ENFORCE_RATELIMITS(st, tx["client"], mid, tx["round"]): return False
        fee = int(tx.get("fee", 0))
        need = tx["deposit"] + max(0, fee)
        return CHECK_DEPOSIT(st, tx["client"], need)

def admit_publish(st: State, tx: Dict[str, Any]) -> bool:
    with st.lock():
        if not CHECK_TICKET(st, tx["ticket_id"], tx["client"]):
            return False
        if not WITHIN_EXPIRY(st, tx["ticket_id"]):
            return False
        cid = tx["update_cid"]
        fee = int(tx.get("fee", 0))
        if fee > 0 and not CHECK_DEPOSIT(st, tx["client"], fee):
            return False
        return isinstance(cid, str) and len(cid) >= 16

def admit_global(st: State, tx: Dict[str, Any]) -> bool:
    with st.lock():
        mid = tx["model_id"]; rnd = tx["round"]
        if mid not in st.models: return False
        if st.get_round_index(mid) != rnd: return False
        if st.pools.get((mid, rnd), 0) <= 0: return False
        inc = tx["included"]; sc = tx["scores"]
        if len(inc) != len(sc) or len(set(inc)) != len(inc): return False
        for s in sc:
            try:
                if s != s or s < 0:  # NaN or negative
                    return False
            except Exception:
                return False
        for tid in inc + tx["refunds"]:
            t = st.tickets.get(tid)
            if not t or not (t.model_id == mid and t.round == rnd and t.revealed):
                return False
        fee = int(tx.get("fee", 0))
        return CHECK_DEPOSIT(st, tx["lister"], fee) if fee > 0 else True

def admit_claim(st: State, tx: Dict[str, Any]) -> bool:
    with st.lock():
        amt = tx["amount"]
        if amt <= 0: return False
        elig = st.entitlement_get(tx["acct"], tx["model_id"])
        if amt > elig: return False
        nextn = st.claim_nonce.get(tx["acct"], 0)
        # Allow zero-fee claims or require fee balance when present.
        fee = int(tx.get("fee", 0))
        if fee > 0 and not CHECK_DEPOSIT(st, tx["acct"], fee):
            return False
        return tx.get("nonce", -1) == nextn
