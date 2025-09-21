# microchain/core/state.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Set
from threading import RLock
import time
import hashlib

########################
# Canonical state types
########################

@dataclass
class Rules:
    deposit: int
    gamma: float
    cap_n: int
    intent_cap: int
    refund_policy: str = "treasury"   # 'burn'|'treasury'|'model_pool'
    round_deadline_slots: int = 10
    round_reward: int = 0
    remainder_sink: str = "treasury"  # 'treasury' | 'model_pool' | 'burn'

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
    refundable: bool = False
    refunded: bool = False
    forfeited: bool = False
    settled: bool = False

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

        # accounts / balances / treasury
        self.balances: Dict[str, int] = {}
        self.treasury: int = 0

        # validator stake
        self.validator_stake: Dict[str, int] = {}
        self.active_validators: Set[str] = set()

        # models / pools
        self.models: Dict[str, Model] = {}
        self.pools: Dict[Tuple[str, int], int] = {}

        # tickets
        self.tickets: Dict[int, Ticket] = {}
        self.tickets_by_mr: Dict[Tuple[str, int], List[int]] = {}
        self.next_ticket_id = 1

        # entitlements & claim nonces
        self.entitlements: Dict[Tuple[str, str], int] = {}
        self.claim_nonce: Dict[str, int] = {}

        # tx idempotency guard
        self.applied_txs: Set[str] = set()

        # equivocation detection
        self.equivocation_seen: Set[Tuple[int, str]] = set()

        # metrics & logs
        self.vrf_eligibility: Dict[int, List[str]] = {}  # slot -> eligible vids
        self.block_log: List[dict] = []                  # applied-on-head blocks
        self.round_log: List[dict] = []                  # finalize events
        self.slashing_log: List[dict] = []               # slashing events
        self.fork_log: List[dict] = []                   # side-branch blocks
        self._fork_seen: Set[str] = set()

        # fee ledger & policy
        self.fee_policy: str = "proposer"                # 'proposer'|'treasury'|'split'
        self.fees_paid_by_acct: Dict[str, int] = {}
        self.fees_earned_by_acct: Dict[str, int] = {}
        self.total_fees_charged: int = 0
        self.total_fees_distributed: int = 0

        # chain
        self.blocks: Dict[str, Block] = {}
        self.children: Dict[Optional[str], List[str]] = {}
        self.head_hash: Optional[str] = None
        self.height: int = 0
        self.forks_observed: int = 0

        # time-ish
        self.genesis_time = time.time()
        self.slot: int = 0
        self.slot_duration_sec: float = 0.4

        # randomness
        self.epoch: int = 0
        self.epoch_randomness: bytes = b"genesis-R"
        self.rand_period_slots: int = 16

        # ML task registry
        self.ml_tasks: Dict[str, dict] = {}

        # round start mirror (set by lister on new round)
        self.last_round_start_slot: int = 0

        # metrics store
        self.metrics: Dict[str, any] = {
            "tx_lat_slots_by_type": {
                "ListingTx": [], "FundingTx": [], "UpdateIntentTx": [],
                "UpdatePublishTx": [], "GlobalUpdateTx": [], "ClaimTx": []
            },
            "ipfs_put_ms": [],
            "ipfs_get_ms": [],
            "aggregate_ms": [],
            "round_dur_slots": []
        }

    #######################
    # Helpers (thread-safe)
    #######################

    def lock(self):
        return self._lock

    # randomness
    def rotate_randomness(self):
        self.epoch += 1
        payload = self.epoch_randomness + self.slot.to_bytes(8, "big")
        self.epoch_randomness = hashlib.sha256(payload).digest()

    # fees
    def record_fee_paid(self, acct: str, amount: int):
        if amount <= 0: return
        self.fees_paid_by_acct[acct] = self.fees_paid_by_acct.get(acct, 0) + amount
        self.total_fees_charged += amount

    def record_fee_earned(self, acct: str, amount: int):
        if amount <= 0: return
        self.fees_earned_by_acct[acct] = self.fees_earned_by_acct.get(acct, 0) + amount
        self.total_fees_distributed += amount

    # idempotency (convenience)
    def mark_tx_applied(self, txh: str) -> bool:
        if txh in self.applied_txs:
            return False
        self.applied_txs.add(txh)
        return True

    # balances
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

    # models/pools
    def add_model(self, model_id: str, init_cid: str, rules: Rules):
        assert model_id not in self.models
        self.models[model_id] = Model(model_id=model_id,
                                      init_cid=init_cid,
                                      latest_global_cid=init_cid,
                                      round_index=0,
                                      rules=rules)

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
    
    def pool_add_at(self, model_id: str, round_index: int, amount: int):
        key = (model_id, round_index)
        self.pools[key] = self.pools.get(key, 0) + amount

    def pool_consume(self, model_id: str, amount: int):
        r = self.get_round_index(model_id)
        key = (model_id, r)
        current = self.pools.get(key, 0)
        assert current >= amount, "Pool underflow"
        self.pools[key] = current - amount

    # entitlements / claims
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

    # validators
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
        self.treasury += penalty
        if self.validator_stake[vid] == 0:
            self.active_validators.discard(vid)

    def record_equivocation_and_slash(self, slot: int, proposer: str, fraction: float = 0.5):
        key = (slot, proposer)
        if key in self.equivocation_seen:
            return
        self.equivocation_seen.add(key)
        before = self.validator_stake.get(proposer, 0)
        self.slash(proposer, fraction)
        after = self.validator_stake.get(proposer, 0)
        self.slashing_log.append({
            "slot": slot, "proposer": proposer,
            "penalty_before": before, "penalty_after": after, "fraction": fraction
        })

    # tickets
    def new_ticket(self, owner: str, model_id: str, round_: int, deposit: int, expiry_slot: int) -> Ticket:
        tid = self.next_ticket_id; self.next_ticket_id += 1
        t = Ticket(tid, owner, model_id, round_, deposit, expiry_slot)
        self.tickets[tid] = t
        self.tickets_by_mr.setdefault((model_id, round_), []).append(tid)
        return t

    def tickets_for(self, model_id: str, round_: int) -> List[Ticket]:
        ids = self.tickets_by_mr.get((model_id, round_), [])
        return [self.tickets[i] for i in ids]

    def now_slot(self) -> int:
        return self.slot

    # ---- metrics helpers ----
    def record_tx_latency(self, tx_type: str, lat_slots: int):
        arr = self.metrics["tx_lat_slots_by_type"].get(tx_type)
        if arr is not None:
            arr.append(int(lat_slots))

    def record_ipfs_put_ms(self, ms: float):
        self.metrics["ipfs_put_ms"].append(float(ms))

    def record_ipfs_get_ms(self, ms: float):
        self.metrics["ipfs_get_ms"].append(float(ms))

    def record_aggregate_ms(self, ms: float):
        self.metrics["aggregate_ms"].append(float(ms))

    def record_round_dur_slots(self, dur_slots: int):
        self.metrics["round_dur_slots"].append(int(dur_slots))

