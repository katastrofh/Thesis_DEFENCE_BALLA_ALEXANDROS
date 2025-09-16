below is the full code for every file you listed. copy these into files with the same paths on your Ubuntu box, then follow the run steps at the end.

⸻

microchain/core/state.py

# microchain/core/state.py
from _future_ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List, Set
from threading import RLock
import time

########################
# Canonical state types
########################

@dataclass
class Rules:
    deposit: int                      # per-ticket deposit
    gamma: float                      # curvature for payout
    cap_n: int                        # cap for declared sample size in scoring
    intent_cap: int                   # max intents per account per round
    refund_policy: str = "treasury"   # 'burn'|'treasury'|'model_pool'
    round_deadline_slots: int = 10    # tickets expire after this many slots
    round_reward: int = 0             # initial pool for current round

@dataclass
class Ticket:
    ticket_id: int
    owner: str
    model_id: str
    round: int
    deposit: int
    expiry_slot: int
    revealed: bool = False
    update_cid: Optional[str] = None
    refundable: bool = False   # set by lister via refunds[]
    refunded: bool = False     # marked on ADVANCE_ROUND / claim path
    forfeited: bool = False    # marked when deposit burned/redistributed
    settled: bool = False      # true iff refunded or forfeited processed

@dataclass
class Model:
    model_id: str
    init_cid: str
    latest_global_cid: str
    round_index: int
    rules: Rules

@dataclass
class BlockHeader:
    parent: Optional[str]
    slot: int
    height: int
    proposer: str
    vrf: str
    hash: str

@dataclass
class Block:
    header: BlockHeader
    txs: List[dict]

class State:
    """
    A thread-safe canonical state (single copy for the sim).
    """
    def __init__(self):
        self._lock = RLock()
        # balances & staking
        self.balances: Dict[str, int] = {}
        self.treasury: int = 0
        self.validator_stake: Dict[str, int] = {}
        self.active_validators: Set[str] = set()

        # models and rules
        self.models: Dict[str, Model] = {}
        # per (model,round) reward pool
        self.pools: Dict[Tuple[str, int], int] = {}

        # tickets
        self.tickets: Dict[int, Ticket] = {}
        self.tickets_by_mr: Dict[Tuple[str, int], List[int]] = {}
        self.next_ticket_id = 1

        # entitlements: withdrawable earned funds by (acct, model)
        self.entitlements: Dict[Tuple[str, str], int] = {}

        # claim nonces for idempotency
        self.claim_nonce: Dict[str, int] = {}

        # chain
        self.blocks: Dict[str, Block] = {}
        self.head_hash: Optional[str] = None
        self.height: int = 0

        # time-ish
        self.genesis_time = time.time()
        self.slot: int = 0
        self.slot_duration_sec: float = 0.4

        # randomness per epoch
        self.epoch: int = 0
        self.epoch_randomness: bytes = b"genesis-R"

    #######################
    # Helpers (thread-safe)
    #######################
    def lock(self):
        return self._lock

    def credit(self, acct: str, amount: int):
        if amount <= 0: return
        self.balances[acct] = self.balances.get(acct, 0) + amount

    def debit(self, acct: str, amount: int) -> bool:
        if amount <= 0: return True
        bal = self.balances.get(acct, 0)
        if bal < amount:
            return False
        self.balances[acct] = bal - amount
        return True

    def add_model(self, model_id: str, init_cid: str, rules: Rules):
        assert model_id not in self.models
        self.models[model_id] = Model(model_id=model_id,
                                      init_cid=init_cid,
                                      latest_global_cid=init_cid,
                                      round_index=0,
                                      rules=rules)
        # seed pool for round 0 from rules (if provided)
        if rules.round_reward > 0:
            self.pools[(model_id, 0)] = rules.round_reward

    def get_round_index(self, model_id: str) -> int:
        return self.models[model_id].round_index

    def inc_round(self, model_id: str):
        self.models[model_id].round_index += 1

    def set_global_cid(self, model_id: str, cid: str):
        self.models[model_id].latest_global_cid = cid

    def pool_add(self, model_id: str, amount: int):
        r = self.get_round_index(model_id)
        key = (model_id, r)
        self.pools[key] = self.pools.get(key, 0) + amount

    def pool_consume(self, model_id: str, amount: int):
        r = self.get_round_index(model_id)
        key = (model_id, r)
        current = self.pools.get(key, 0)
        assert current >= amount, "Pool underflow"
        self.pools[key] = current - amount

    def entitlement_add(self, acct: str, model_id: str, amount: int):
        if amount <= 0: return
        key = (acct, model_id)
        self.entitlements[key] = self.entitlements.get(key, 0) + amount

    def entitlement_get(self, acct: str, model_id: str) -> int:
        return self.entitlements.get((acct, model_id), 0)

    def entitlement_sub(self, acct: str, model_id: str, amount: int):
        key = (acct, model_id)
        val = self.entitlements.get(key, 0)
        assert val >= amount > 0
        self.entitlements[key] = val - amount

    def claim_next_nonce(self, acct: str) -> int:
        n = self.claim_nonce.get(acct, 0)
        self.claim_nonce[acct] = n + 1
        return n

    def register_validator(self, vid: str, stake: int):
        self.validator_stake[vid] = stake
        self.active_validators.add(vid)

    def total_stake(self) -> int:
        return sum(self.validator_stake.get(v, 0) for v in self.active_validators)

    def slash(self, vid: str, fraction: float):
        stake = self.validator_stake.get(vid, 0)
        penalty = int(stake * fraction)
        if penalty <= 0: return
        self.validator_stake[vid] = max(0, stake - penalty)
        # send slashed funds to treasury
        self.treasury += penalty
        # optionally remove from active set
        if self.validator_stake[vid] == 0:
            self.active_validators.discard(vid)

    def new_ticket(self, owner: str, model_id: str, round_: int, deposit: int, expiry_slot: int) -> Ticket:
        tid = self.next_ticket_id
        self.next_ticket_id += 1
        t = Ticket(ticket_id=tid, owner=owner, model_id=model_id,
                   round=round_, deposit=deposit, expiry_slot=expiry_slot)
        self.tickets[tid] = t
        self.tickets_by_mr.setdefault((model_id, round_), []).append(tid)
        return t

    def tickets_for(self, model_id: str, round_: int) -> List[Ticket]:
        ids = self.tickets_by_mr.get((model_id, round_), [])
        return [self.tickets[i] for i in ids]

    def now_slot(self) -> int:
        return self.slot


⸻

microchain/core/consensus.py

# microchain/core/consensus.py
from _future_ import annotations
import hmac, hashlib
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from .state import State, Block, BlockHeader

def vrf_bytes(secret: bytes, randomness: bytes, slot: int) -> bytes:
    msg = randomness + slot.to_bytes(8, 'big')
    return hmac.new(secret, msg, hashlib.sha256).digest()

def vrf_value(secret: bytes, randomness: bytes, slot: int) -> float:
    digest = vrf_bytes(secret, randomness, slot)
    # turn 256-bit int into [0,1)
    x = int.from_bytes(digest, 'big')
    return x / (2**256)

def proposer_eligible(stake_i: int, total_stake: int, tau: float, vrf_val: float) -> bool:
    if total_stake <= 0 or stake_i <= 0:
        return False
    threshold = tau * (stake_i / total_stake)
    return vrf_val < threshold

@dataclass
class EquivocationEvidence:
    slot: int
    proposer: str
    block_hashes: Tuple[str, str]

class ForkChoice:
    """Longest-chain with lexicographic tiebreaker on head hash."""
    def __init__(self):
        self.children: Dict[Optional[str], List[str]] = {}
        self.blocks: Dict[str, Block] = {}

    def add_block(self, block: Block):
        h = block.header
        self.blocks[h.hash] = block
        self.children.setdefault(h.parent, []).append(h.hash)

    def head(self) -> Optional[str]:
        if not self.blocks:
            return None
        # compute heights from roots
        heights = {h: self.blocks[h].header.height for h in self.blocks}
        # choose max height, then min hash for tie-break
        max_h = max(heights.values())
        cands = [h for h, v in heights.items() if v == max_h]
        return sorted(cands)[0]

def detect_equivocation(slot: int, proposer: str, seen: Dict[Tuple[int, str], str], new_hash: str) -> Optional[EquivocationEvidence]:
    key = (slot, proposer)
    if key in seen and seen[key] != new_hash:
        return EquivocationEvidence(slot=slot, proposer=proposer, block_hashes=(seen[key], new_hash))
    return None


⸻

microchain/core/admit.py

# microchain/core/admit.py
from _future_ import annotations
from typing import Dict, Any
from .state import State, Ticket

###############################
# Admission helper primitives
###############################

def CHECK_DEPOSIT(st: State, acct: str, amount: int) -> bool:
    """Ensure acct has at least amount balance."""
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
    """Per-account per-round cap taken from rules.intent_cap."""
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
    """
    tx = {type:'ListingTx', lister, model_id, init_cid, rules: Rules-like dict, fee}
    Fee ignored in demo; must not already exist.
    """
    with st.lock():
        if tx["model_id"] in st.models:
            return False
        # naive CID check: hex length
        cid = tx["init_cid"]
        if not isinstance(cid, str) or len(cid) < 16:
            return False
        return True

def admit_funding(st: State, tx: Dict[str, Any]) -> bool:
    # tx = {type:'FundingTx', lister, model_id, amount}
    with st.lock():
        if tx["model_id"] not in st.models:
            return False
        return CHECK_DEPOSIT(st, tx["lister"], tx["amount"])

def admit_intent(st: State, tx: Dict[str, Any]) -> bool:
    # tx = {type:'UpdateIntentTx', client, model_id, round, deposit}
    with st.lock():
        mid = tx["model_id"]
        if mid not in st.models: return False
        if st.get_round_index(mid) != tx["round"]: return False
        if not ENFORCE_RATELIMITS(st, tx["client"], mid, tx["round"]): return False
        return CHECK_DEPOSIT(st, tx["client"], tx["deposit"])

def admit_publish(st: State, tx: Dict[str, Any]) -> bool:
    # tx = {type:'UpdatePublishTx', client, ticket_id, update_cid}
    with st.lock():
        if not CHECK_TICKET(st, tx["ticket_id"], tx["client"]):
            return False
        if not WITHIN_EXPIRY(st, tx["ticket_id"]):
            return False
        cid = tx["update_cid"]
        return isinstance(cid, str) and len(cid) >= 16

def admit_global(st: State, tx: Dict[str, Any]) -> bool:
    # tx = {type:'GlobalUpdateTx', lister, model_id, round, global_cid, included:list[int], refunds:list[int], scores:list[float]}
    with st.lock():
        mid = tx["model_id"]
        if mid not in st.models: return False
        if st.get_round_index(mid) != tx["round"]: return False
        inc = tx["included"]
        sc = tx["scores"]
        if len(inc) != len(sc): return False
        # included and refunds must be revealed tickets for (m,r)
        for tid in inc + tx["refunds"]:
            t = st.tickets.get(tid)
            if not t: return False
            if not (t.model_id == mid and t.round == tx["round"] and t.revealed):
                return False
        gcid = tx["global_cid"]
        return isinstance(gcid, str) and len(gcid) >= 16

def admit_claim(st: State, tx: Dict[str, Any]) -> bool:
    # tx = {type:'ClaimTx', acct, model_id, amount, nonce}
    with st.lock():
        amt = tx["amount"]
        if amt <= 0: return False
        elig = st.entitlement_get(tx["acct"], tx["model_id"])
        if amt > elig: return False
        # idempotency: nonce must be exactly next expected
        nextn = st.claim_nonce.get(tx["acct"], 0)
        return tx.get("nonce", -1) == nextn


⸻

microchain/core/apply.py

# microchain/core/apply.py
from _future_ import annotations
from typing import Dict, Any, List
from .state import State, Rules, Ticket

def TICKET_CREATE(st: State, client: str, model_id: str, round_: int, deposit: int, expiry_slot: int) -> Ticket:
    """Debit deposit and create on-chain ticket."""
    ok = st.debit(client, deposit)
    assert ok, "deposit debit failed (race)"
    t = st.new_ticket(owner=client, model_id=model_id, round_=round_, deposit=deposit, expiry_slot=expiry_slot)
    return t

def TICKET_REVEAL(st: State, ticket_id: int, update_cid: str):
    t = st.tickets[ticket_id]
    t.revealed = True
    t.update_cid = update_cid

def ENTITLEMENT_TRANSFER(st: State, acct: str, model_id: str, amount: int):
    st.entitlement_sub(acct, model_id, amount)
    st.credit(acct, amount)

def ADVANCE_ROUND(st: State, model_id: str, round_: int, global_cid: str, included: List[int], scores: List[float], refunds: List[int]):
    assert st.get_round_index(model_id) == round_, "round mismatch"
    st.set_global_cid(model_id, global_cid)
    rules = st.models[model_id].rules
    R = st.pools.get((model_id, round_), 0)

    # payouts
    gamma = rules.gamma
    weights = [max(0.0, s) ** gamma for s in scores]
    S = sum(weights)
    if S > 0 and R > 0:
        for tid, w in zip(included, weights):
            if w <= 0: 
                continue
            acct = st.tickets[tid].owner
            pay = int(R * (w / S))
            if pay > 0:
                st.entitlement_add(acct, model_id, pay)

    # deposit refunds for refundable list
    refundable_set = set(refunds)
    for tid in refundable_set:
        t = st.tickets[tid]
        if not t.settled:
            st.entitlement_add(t.owner, model_id, t.deposit)
            t.refundable = True
            t.refunded = True
            t.settled = True

    # forfeits for others that were not refunded
    for t in st.tickets_for(model_id, round_):
        if not t.settled:
            # unrevealed or non-refundable revealed => forfeit
            t.forfeited = True
            t.settled = True
            sink = rules.refund_policy
            if sink == "burn":
                # do nothing (deposit disappears)
                pass
            elif sink == "treasury":
                st.treasury += t.deposit
            elif sink == "model_pool":
                st.pool_add(model_id, t.deposit)
            else:
                st.treasury += t.deposit

    # consume full intended round reward (cap by available)
    consume = min(R, st.pools.get((model_id, round_), 0))
    st.pool_consume(model_id, consume)

    # advance round
    st.inc_round(model_id)

def apply_tx(st: State, tx: Dict[str, Any]):
    t = tx["type"]
    if t == "ListingTx":
        st.add_model(model_id=tx["model_id"],
                     init_cid=tx["init_cid"],
                     rules=tx["rules"])
    elif t == "FundingTx":
        ok = st.debit(tx["lister"], tx["amount"])
        assert ok
        st.pool_add(tx["model_id"], tx["amount"])
    elif t == "UpdateIntentTx":
        mid = tx["model_id"]
        expiry = st.now_slot() + st.models[mid].rules.round_deadline_slots
        TICKET_CREATE(st, tx["client"], mid, tx["round"], tx["deposit"], expiry)
    elif t == "UpdatePublishTx":
        TICKET_REVEAL(st, tx["ticket_id"], tx["update_cid"])
    elif t == "GlobalUpdateTx":
        ADVANCE_ROUND(st, tx["model_id"], tx["round"], tx["global_cid"], tx["included"], tx["scores"], tx["refunds"])
    elif t == "ClaimTx":
        ENTITLEMENT_TRANSFER(st, tx["acct"], tx["model_id"], tx["amount"])
        # increment nonce after successful claim
        st.claim_next_nonce(tx["acct"])
    else:
        raise ValueError(f"Unknown tx type {t}")


⸻

microchain/roles/validator.py

# microchain/roles/validator.py
from _future_ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib
import threading
import queue
import time

from ..core.state import State, Block, BlockHeader
from ..core.consensus import vrf_value, proposer_eligible, detect_equivocation
from ..core.admit import (
    admit_listing, admit_funding, admit_intent, admit_publish, admit_global, admit_claim
)
from ..core.apply import apply_tx

def tx_hash(tx: Dict[str, Any]) -> str:
    h = hashlib.sha256(repr(sorted(tx.items())).encode()).hexdigest()
    return h

@dataclass
class NetMsg:
    kind: str  # 'tx'|'block'
    data: Any

class InProcNetwork:
    """Very small pubsub + chain clock + randomness holder."""
    def __init__(self, state: State, tau: float = 0.8):
        self.state = state
        self.tau = tau
        self.nodes: Dict[str, "Validator"] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()

    def register(self, v: "Validator"):
        with self._lock:
            self.nodes[v.vid] = v

    def broadcast(self, msg: NetMsg):
        with self._lock:
            for v in self.nodes.values():
                v.inbox.put(msg)

    def stop(self):
        self._stop.set()

    def stopped(self) -> bool:
        return self._stop.is_set()

class Validator(threading.Thread):
    """Validator with local mempool; proposes blocks when eligible."""
    def __init__(self, vid: str, secret: bytes, state: State, net: InProcNetwork):
        super().__init__(daemon=True)
        self.vid = vid
        self.secret = secret
        self.state = state
        self.net = net
        self.mempool: List[Dict[str, Any]] = []
        self.inbox: "queue.Queue[NetMsg]" = queue.Queue()
        self.seen_tx: set[str] = set()
        self.seen_blocks: set[str] = set()
        self.seen_slot_block_by_me: Dict[tuple, str] = {}  # (slot, vid)->hash
        self.running = True

        net.register(self)

    def _admit(self, tx: Dict[str, Any]) -> bool:
        t = tx["type"]
        if t == "ListingTx":
            return admit_listing(self.state, tx)
        if t == "FundingTx":
            return admit_funding(self.state, tx)
        if t == "UpdateIntentTx":
            return admit_intent(self.state, tx)
        if t == "UpdatePublishTx":
            return admit_publish(self.state, tx)
        if t == "GlobalUpdateTx":
            return admit_global(self.state, tx)
        if t == "ClaimTx":
            return admit_claim(self.state, tx)
        return False

    def _maybe_propose(self):
        with self.state.lock():
            slot = self.state.slot
            R = self.state.epoch_randomness
            s_i = self.state.validator_stake.get(self.vid, 0)
            S = self.state.total_stake()
        val = vrf_value(self.secret, R, slot)
        if not proposer_eligible(s_i, S, self.net.tau, val):
            return

        # Build a block
        parent = self.state.head_hash
        height = (self.state.height + 1)
        txs = []
        # include up to 200 txs
        for tx in list(self.mempool)[:200]:
            if self._admit(tx):
                txs.append(tx)

        hr = f"{parent}|{slot}|{height}|{self.vid}|{val}"
        bhash = hashlib.sha256(hr.encode()).hexdigest()
        hdr = BlockHeader(parent=parent, slot=slot, height=height,
                          proposer=self.vid, vrf=str(val), hash=bhash)
        blk = Block(header=hdr, txs=txs)

        # equivocation detection (same slot, two blocks)
        ev = detect_equivocation(slot, self.vid, self.seen_slot_block_by_me, bhash)
        if ev:
            # slash myself (demo)
            with self.state.lock():
                self.state.slash(self.vid, 0.5)
        else:
            self.seen_slot_block_by_me[(slot, self.vid)] = bhash

        self.net.broadcast(NetMsg(kind="block", data=blk))

    def _apply_block(self, blk: Block):
        with self.state.lock():
            # sanity: parent is current head OR chain is empty
            # (We do not implement full reorg; demo keeps a single canonical chain.)
            for tx in blk.txs:
                apply_tx(self.state, tx)
            self.state.height = blk.header.height
            self.state.head_hash = blk.header.hash

    def run(self):
        while self.running and not self.net.stopped():
            try:
                # non-blocking drain
                while True:
                    msg = self.inbox.get_nowait()
                    if msg.kind == "tx":
                        tx = msg.data
                        h = tx_hash(tx)
                        if h in self.seen_tx:
                            continue
                        self.seen_tx.add(h)
                        if self._admit(tx):
                            self.mempool.append(tx)
                    elif msg.kind == "block":
                        blk: Block = msg.data
                        if blk.header.hash in self.seen_blocks:
                            continue
                        self.seen_blocks.add(blk.header.hash)
                        # apply immediately
                        self._apply_block(blk)
                        # drop included txs from mempool
                        mt = set(tx_hash(t) for t in blk.txs)
                        self.mempool = [t for t in self.mempool if tx_hash(t) not in mt]
            except queue.Empty:
                pass

            # Maybe propose (once per slot window we attempt)
            self._maybe_propose()

            time.sleep(0.02)  # allow other threads to run

    def submit_tx(self, tx: Dict[str, Any]):
        self.net.broadcast(NetMsg(kind="tx", data=tx))

    def stop(self):
        self.running = False


⸻

microchain/roles/lister.py

# microchain/roles/lister.py
from _future_ import annotations
from typing import List, Dict, Any
import threading
import time
import random

from ..core.state import State, Rules, Ticket
from ..storage.ipfs_stub import IPFS

class Lister(threading.Thread):
    def __init__(self, acct: str, state: State, ipfs: IPFS, submit_tx, model_id: str = "mnist", init_bytes: bytes = b"init", rules: Rules = None, fund_amount: int = 1000):
        super().__init__(daemon=True)
        self.acct = acct
        self.state = state
        self.ipfs = ipfs
        self.submit_tx = submit_tx
        self.model_id = model_id
        self.init_bytes = init_bytes
        self.rules = rules or Rules(deposit=20, gamma=1.3, cap_n=1000, intent_cap=3,
                                    refund_policy="treasury", round_deadline_slots=8, round_reward=fund_amount)
        self.fund_amount = fund_amount
        self.running = True

    def create_listing(self):
        cid = self.ipfs.put(self.init_bytes, tag="init")
        tx = {
            "type": "ListingTx",
            "lister": self.acct,
            "model_id": self.model_id,
            "init_cid": cid,
            "rules": self.rules,
            "fee": 0
        }
        self.submit_tx(tx)
        time.sleep(0.2)
        # fund reward pool
        if self.fund_amount > 0:
            tx2 = {"type":"FundingTx","lister":self.acct,"model_id":self.model_id,"amount":self.fund_amount}
            self.submit_tx(tx2)

    def collect_revealed(self, round_: int) -> List[Ticket]:
        with self.state.lock():
            tickets = self.state.tickets_for(self.model_id, round_)
            return [t for t in tickets if t.revealed]

    def aggregate(self, tickets: List[Ticket]) -> Dict[str, Any]:
        included = []
        scores = []
        refunds = []
        # simple scoring: s_j = max(0, delta)*min(n, cap)
        for t in tickets:
            # attempt to fetch update blob
            ok, _data = self.ipfs.get(t.update_cid)
            if ok:
                refunds.append(t.ticket_id)  # refundable if retrievable
            # metrics are unknown in this minimal demo; synthesize from CID hash digits for determinism
            # emulate declared n and delta by hashing
            h = int(t.update_cid[:8], 16)
            n = (h % 2000) + 1
            delta = ((h // 13) % 200) / 1000.0 - 0.05  # ~[-0.05 .. 0.145]
            w = min(n, self.rules.cap_n)
            s = max(0.0, delta) * w
            included.append(t.ticket_id)
            scores.append(s)
        # global bytes
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
        }
        self.submit_tx(tx)

    def run(self):
        # create listing & fund
        self.create_listing()
        # wait one slot for validators to include
        time.sleep(0.5)
        while self.running:
            with self.state.lock():
                r = self.state.models[self.model_id].round_index
            # collect revealed tickets for current round
            revealed = self.collect_revealed(round_=r)
            # finalize when some revealed (or when deadline is near)
            if revealed:
                payload = self.aggregate(revealed)
                self.finalize(r, payload)
                # let clients claim rewards/deposits next round
                time.sleep(0.5)
            time.sleep(0.2)

    def stop(self):
        self.running = False


⸻

microchain/roles/client.py

# microchain/roles/client.py
from _future_ import annotations
import threading
import time
import os
import hashlib
from typing import Callable
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
        # toy "training": hash of (acct, seed, time) to get deterministic-ish bytes
        payload = f"{self.acct}-{self.data_seed}-{time.time_ns()}".encode()
        return hashlib.sha256(payload).digest()

    def my_tickets(self, round) -> list[int]:
        with self.state.lock():
            tids = []
            for t in self.state.tickets_for(self.model_id, round_):
                if t.owner == self.acct:
                    tids.append(t.ticket_id)
            return tids

    def run(self):
        # wait until model exists
        while self.model_id not in self.state.models:
            time.sleep(0.1)

        while self.running:
            r = self._current_round()
            dep = self._deposit()
            # send intent
            txi = {"type":"UpdateIntentTx","client":self.acct,"model_id":self.model_id,"round":r,"deposit":dep}
            self.submit_tx(txi)
            # wait a bit for ticket creation in block
            time.sleep(0.4)

            # find my ticket
            tids = self._my_tickets(r)
            if not tids:
                time.sleep(0.2)
                continue
            tid = tids[-1]

            # upload update to IPFS
            update_bytes = self._train_bytes()
            update_cid = self.ipfs.put(update_bytes, tag="update")

            # publish
            txp = {"type":"UpdatePublishTx","client":self.acct,"ticket_id":tid,"update_cid":update_cid}
            self.submit_tx(txp)

            # wait for round finalize and then claim (next round)
            time.sleep(1.0)
            with self.state.lock():
                elig = self.state.entitlement_get(self.acct, self.model_id)
                if elig > 0:
                    nonce = self.state.claim_nonce.get(self.acct, 0)
                    txc = {"type":"ClaimTx","acct":self.acct,"model_id":self.model_id,"amount":elig,"nonce":nonce}
                    self.submit_tx(txc)

            # don't spam too fast
            time.sleep(0.5)

    def stop(self):
        self.running = False


⸻

microchain/storage/ipfs_stub.py

# microchain/storage/ipfs_stub.py
from _future_ import annotations
from typing import Tuple, Dict
import hashlib
import random

class IPFS:
    def __init__(self, fail_rate: float = 0.0):
        self.store: Dict[str, bytes] = {}
        self.pins: Dict[str, int] = {}
        self.fail_rate = fail_rate

    def put(self, data: bytes, tag: str = "") -> str:
        # CID := sha256 hex
        cid = hashlib.sha256((tag.encode() + b"|" + data)).hexdigest()
        self.store[cid] = data
        self.pins[cid] = self.pins.get(cid, 0) + 1
        return cid

    def get(self, cid: str) -> Tuple[bool, bytes]:
        if random.random() < self.fail_rate:
            return False, b""
        data = self.store.get(cid)
        if data is None:
            return False, b""
        # verify
        return True, data

    def pin(self, cid: str):
        self.pins[cid] = self.pins.get(cid, 0) + 1


⸻

microchain/runs/scenario_min.py

# microchain/runs/scenario_min.py
from _future_ import annotations
import time
import argparse
from typing import List
from ..core.state import State, Rules
from ..roles.validator import Validator, InProcNetwork
from ..roles.lister import Lister
from ..roles.client import Client
from ..storage.ipfs_stub import IPFS

def give_faucet(st: State, who: str, amt: int):
    with st.lock():
        st.credit(who, amt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validators", type=int, default=3)
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--deposit", type=int, default=20)
    ap.add_argument("--reward", type=int, default=1000)
    ap.add_argument("--tau", type=float, default=0.😎
    ap.add_argument("--runtime", type=float, default=8.0, help="seconds to run")
    args = ap.parse_args()

    state = State()
    net = InProcNetwork(state, tau=args.tau)
    ipfs = IPFS(fail_rate=0.0)

    # seed balances & stakes
    for i in range(args.validators):
        vid = f"V{i+1}"
        give_faucet(state, vid, 0)
        # fixed stake for demo
        with state.lock():
            state.register_validator(vid, stake=100)

    # bring up validators
    validators: List[Validator] = []
    for i in range(args.validators):
        v = Validator(vid=f"V{i+1}", secret=f"sk{i+1}".encode(), state=state, net=net)
        v.start()
        validators.append(v)

    def submit_tx(tx: dict):
        # any validator can inject (we broadcast through net)
        validators[0].submit_tx(tx)

    # fund lister and clients
    give_faucet(state, "Lister", args.reward + 1000)
    for c in range(args.clients):
        give_faucet(state, f"C{c+1}", 200)

    # create lister
    rules = Rules(deposit=args.deposit, gamma=1.3, cap_n=1000, intent_cap=3,
                  refund_policy="treasury", round_deadline_slots=8, round_reward=args.reward)
    lister = Lister(acct="Lister", state=state, ipfs=ipfs, submit_tx=submit_tx, model_id="toy", rules=rules, fund_amount=args.reward)
    lister.start()

    # create clients
    clients: List[Client] = []
    for c in range(args.clients):
        cl = Client(acct=f"C{c+1}", state=state, ipfs=ipfs, submit_tx=submit_tx, model_id="toy", data_seed=c+1)
        cl.start()
        clients.append(cl)

    # run clock
    t0 = time.time()
    try:
        while time.time() - t0 < args.runtime:
            with state.lock():
                state.slot += 1
            time.sleep(state.slot_duration_sec)
    finally:
        for cl in clients: cl.stop()
        lister.stop()
        for v in validators: v.stop()
        net.stop()

    # Summary
    with state.lock():
        print("\n=== SUMMARY ===")
        print("Round index:", state.models["toy"].round_index)
        for c in range(args.clients):
            acct = f"C{c+1}"
            bal = state.balances.get(acct, 0)
            ent = state.entitlement_get(acct, "toy")
            print(f"{acct}: balance={bal}, entitlement={ent}")
        print("Lister balance:", state.balances.get("Lister", 0))
        print("Treasury:", state.treasury)

if _name_ == "_main_":
    main()


⸻

microchain/runs/exp_latency.py

# microchain/runs/exp_latency.py
from _future_ import annotations
import time
from .scenario_min import State, InProcNetwork, IPFS, Validator, Lister, Client, Rules, give_faucet

def run(validators=3, clients=5, runtime=8.0, slot_sec=0.2):
    state = State()
    state.slot_duration_sec = slot_sec
    net = InProcNetwork(state, tau=0.😎
    ipfs = IPFS(fail_rate=0.0)

    for i in range(validators):
        with state.lock():
            state.register_validator(f"V{i+1}", 100)

    vals = []
    for i in range(validators):
        v = Validator(f"V{i+1}".encode().decode(), f"sk{i+1}".encode(), state, net)
        v.start()
        vals.append(v)

    def submit_tx(tx: dict):
        vals[0].submit_tx(tx)

    give_faucet(state, "Lister", 2000)
    for c in range(clients):
        give_faucet(state, f"C{c+1}", 200)

    rules = Rules(deposit=20, gamma=1.2, cap_n=1000, intent_cap=3, refund_policy="treasury", round_deadline_slots=8, round_reward=1000)
    l = Lister("Lister", state, ipfs, submit_tx, model_id="toy", rules=rules, fund_amount=1000)
    l.start()

    cs = []
    for c in range(clients):
        cl = Client(f"C{c+1}", state, ipfs, submit_tx, "toy", data_seed=c+1)
        cl.start()
        cs.append(cl)

    t0 = time.time()
    while time.time() - t0 < runtime:
        with state.lock():
            state.slot += 1
        time.sleep(state.slot_duration_sec)

    for c in cs: c.stop(); 
    l.stop()
    for v in vals: v.stop()
    net.stop()

    # report
    with state.lock():
        return {
            "round": state.models["toy"].round_index,
            "balances": {f"C{i+1}": state.balances.get(f"C{i+1}",0) for i in range(clients)},
            "treasury": state.treasury,
        }

if _name_ == "_main_":
    out = run()
    print(out)


⸻

microchain/runs/exp_deposit_spam.py

# microchain/runs/exp_deposit_spam.py
from _future_ import annotations
import time
from .scenario_min import State, InProcNetwork, IPFS, Validator, Lister, Client, Rules, give_faucet

def run(spammer_intent_cap=1, clients=1, runtime=6.0):
    state = State()
    net = InProcNetwork(state, tau=0.8)
    ipfs = IPFS(fail_rate=0.0)

    with state.lock():
        state.register_validator("V1", 100)
    v = Validator("V1", b"sk1", state, net)
    v.start()

    def submit_tx(tx: dict):
        v.submit_tx(tx)

    # one lister, one spammer client
    give_faucet(state, "Lister", 2000)
    give_faucet(state, "SPAM", 1000)

    rules = Rules(deposit=10, gamma=1.0, cap_n=1000, intent_cap=spammer_intent_cap, refund_policy="treasury", round_deadline_slots=6, round_reward=200)
    l = Lister("Lister", state, ipfs, submit_tx, "toy", rules, fund_amount=200)
    l.start()

    spam = Client("SPAM", state, ipfs, submit_tx, "toy", data_seed=42)
    spam.start()

    t0 = time.time()
    while time.time() - t0 < runtime:
        with state.lock(): state.slot += 1
        time.sleep(state.slot_duration_sec)

    spam.stop(); l.stop(); v.stop(); net.stop()

    with state.lock():
        return {
            "spam_balance": state.balances.get("SPAM",0),
            "spam_entitlement": state.entitlement_get("SPAM", "toy"),
            "treasury": state.treasury
        }

if _name_ == "_main_":
    print(run())


⸻

microchain/runs/exp_refund_avail.py

# microchain/runs/exp_refund_avail.py
from _future_ import annotations
import time
from .scenario_min import State, InProcNetwork, IPFS, Validator, Lister, Client, Rules, give_faucet

def run(fail_rate=0.5, clients=3, runtime=8.0):
    state = State()
    net = InProcNetwork(state, tau=0.8)
    ipfs = IPFS(fail_rate=fail_rate)

    with state.lock():
        state.register_validator("V1", 100)
    v = Validator("V1", b"sk1", state, net)
    v.start()

    def submit_tx(tx: dict):
        v.submit_tx(tx)

    give_faucet(state, "Lister", 2000)
    for i in range(clients): give_faucet(state, f"C{i+1}", 200)

    rules = Rules(deposit=20, gamma=1.2, cap_n=1000, intent_cap=3, refund_policy="treasury", round_deadline_slots=8, round_reward=600)
    l = Lister("Lister", state, ipfs, submit_tx, "toy", rules, fund_amount=600)
    l.start()

    cs = []
    for i in range(clients):
        c = Client(f"C{i+1}", state, ipfs, submit_tx, "toy", data_seed=i+1)
        c.start(); cs.append(c)

    t0 = time.time()
    while time.time() - t0 < runtime:
        with state.lock(): state.slot += 1
        time.sleep(state.slot_duration_sec)

    for c in cs: c.stop()
    l.stop(); v.stop(); net.stop()

    with state.lock():
        info = {f"C{i+1}": dict(balance=state.balances.get(f"C{i+1}",0),
                                 ent=state.entitlement_get(f"C{i+1}","toy"))
                for i in range(clients)}
        return {"clients": info, "treasury": state.treasury}

if _name_ == "_main_":
    print(run())


⸻

microchain/runs/exp_gamma_rewards.py

# microchain/runs/exp_gamma_rewards.py
from _future_ import annotations
import time
from .scenario_min import State, InProcNetwork, IPFS, Validator, Lister, Client, Rules, give_faucet

def run(gamma=1.0, clients=5, runtime=8.0):
    state = State()
    net = InProcNetwork(state, tau=0.😎
    ipfs = IPFS(fail_rate=0.0)

    with state.lock():
        state.register_validator("V1", 100)
    v = Validator("V1", b"sk1", state, net)
    v.start()

    def submit_tx(tx: dict):
        v.submit_tx(tx)

    give_faucet(state, "Lister", 2000)
    for i in range(clients): give_faucet(state, f"C{i+1}", 200)

    rules = Rules(deposit=20, gamma=gamma, cap_n=1000, intent_cap=3, refund_policy="treasury", round_deadline_slots=8, round_reward=1000)
    l = Lister("Lister", state, ipfs, submit_tx, "toy", rules, fund_amount=1000)
    l.start()

    cs = []
    for i in range(clients):
        c = Client(f"C{i+1}", state, ipfs, submit_tx, "toy", data_seed=i+1)
        c.start(); cs.append(c)

    t0 = time.time()
    while time.time() - t0 < runtime:
        with state.lock(): state.slot += 1
        time.sleep(state.slot_duration_sec)

    for c in cs: c.stop()
    l.stop(); v.stop(); net.stop()

    with state.lock():
        return {f"C{i+1}": state.balances.get(f"C{i+1}",0) for i in range(clients)}

if _name_ == "_main_":
    print(run())


⸻

microchain/runs/exp_reorg.py

# microchain/runs/exp_reorg.py
# Note: the minimal validator does not implement reorg; this file shows the limitation and returns a note.
def run():
    return {"note":"The minimal demo keeps a single canonical head without reorg simulation."}

if _name_ == "_main_":