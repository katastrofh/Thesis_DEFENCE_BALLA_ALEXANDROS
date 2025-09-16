# microchain/roles/validator.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import hashlib, threading, queue, time

from ..core.state import State, Block, BlockHeader
from ..core.consensus import vrf_value, proposer_eligible, detect_equivocation
from ..core.admit import (
    admit_listing, admit_funding, admit_intent, admit_publish, admit_global, admit_claim
)
from ..core.apply import apply_tx

PRUNE_EVERY_SEC = 0.5
MAX_MEMPOOL = 2000

def tx_hash(tx: Dict[str, Any]) -> str:
    return hashlib.sha256(repr(sorted(tx.items())).encode()).hexdigest()

@dataclass
class NetMsg:
    kind: str  # 'tx'|'block'
    data: Any

class InProcNetwork:
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
    def __init__(self, vid: str, secret: bytes, state: State, net: InProcNetwork):
        super().__init__(daemon=True)
        self.vid = vid; self.secret = secret; self.state = state; self.net = net
        self.mempool: List[Dict[str, Any]] = []
        self.inbox: "queue.Queue[NetMsg]" = queue.Queue()
        self.seen_tx: set[str] = set()
        self.seen_blocks: set[str] = set()
        self.seen_slot_block_by_me: Dict[tuple, str] = {}
        self.seen_slot_block_by_proposer: Dict[Tuple[int, str], str] = {}
        self.running = True
        self.last_proposed_slot = -1
        self.last_prune = time.time()
        net.register(self)

    def _admit(self, tx: Dict[str, Any]) -> bool:
        t = tx["type"]
        if t == "ListingTx":       return admit_listing(self.state, tx)
        if t == "FundingTx":       return admit_funding(self.state, tx)
        if t == "UpdateIntentTx":  return admit_intent(self.state, tx)
        if t == "UpdatePublishTx": return admit_publish(self.state, tx)
        if t == "GlobalUpdateTx":  return admit_global(self.state, tx)
        if t == "ClaimTx":         return admit_claim(self.state, tx)
        return False

    def _prune_mempool(self):
        now = time.time()
        if now - self.last_prune < PRUNE_EVERY_SEC:
            return
        self.last_prune = now
        fresh = []
        with self.state.lock(): cur_slot = self.state.slot
        for tx in self.mempool:
            t = tx.get("type")
            if not self._admit(tx): continue
            if t == "UpdateIntentTx":
                with self.state.lock():
                    mid = tx["model_id"]
                    if self.state.get_round_index(mid) != tx["round"]:
                        continue
            elif t == "UpdatePublishTx":
                tid = tx.get("ticket_id")
                with self.state.lock():
                    tkt = self.state.tickets.get(tid)
                    if not tkt: continue
                    if tkt.revealed or cur_slot > tkt.expiry_slot: continue
            elif t == "GlobalUpdateTx":
                with self.state.lock():
                    mid = tx["model_id"]
                    if self.state.get_round_index(mid) != tx["round"]:
                        continue
            elif t == "ClaimTx":
                with self.state.lock():
                    acct = tx["acct"]; mid = tx["model_id"]
                    elig = self.state.entitlement_get(acct, mid)
                    nonce = self.state.claim_nonce.get(acct, 0)
                    if elig <= 0 or tx.get("nonce") != nonce:
                        continue
            fresh.append(tx)
        if len(fresh) > MAX_MEMPOOL:
            fresh = fresh[-MAX_MEMPOOL:]
        self.mempool = fresh

    def _maybe_propose(self):
        with self.state.lock():
            slot = self.state.slot
            R = self.state.epoch_randomness
            s_i = self.state.validator_stake.get(self.vid, 0)
            S = self.state.total_stake()
        val = vrf_value(self.secret, R, slot)
        eligible = proposer_eligible(s_i, S, self.net.tau, val)
        if eligible:
            with self.state.lock():
                self.state.vrf_eligibility.setdefault(slot, []).append(self.vid)
        if not eligible or slot == self.last_proposed_slot:
            return

        # fee-prioritized selection
        candidates = sorted(self.mempool, key=lambda tx: tx.get("fee", 0), reverse=True)

        parent = self.state.head_hash
        height = (self.state.height + 1)
        txs = []
        for tx in candidates[:200]:
            if self._admit(tx):
                txs.append(tx)

        hr = f"{parent}|{slot}|{height}|{self.vid}|{val}"
        bhash = hashlib.sha256(hr.encode()).hexdigest()
        hdr = BlockHeader(parent=parent, slot=slot, height=height,
                          proposer=self.vid, vrf=str(val), hash=bhash)
        blk = Block(header=hdr, txs=txs)

        # self-equivocation detection
        ev = detect_equivocation(slot, self.vid, self.seen_slot_block_by_me, bhash)
        if ev:
            with self.state.lock():
                self.state.record_equivocation_and_slash(slot, self.vid, 0.5)
        else:
            self.seen_slot_block_by_me[(slot, self.vid)] = bhash

        self.net.broadcast(NetMsg(kind="block", data=blk))
        self.last_proposed_slot = slot

    def _apply_block(self, blk):
        with self.state.lock():
            key = (blk.header.slot, blk.header.proposer)
            prev = self.seen_slot_block_by_proposer.get(key)
            if prev is not None and prev != blk.header.hash:
                self.state.record_equivocation_and_slash(blk.header.slot, blk.header.proposer, 0.5)
                return

            parent = blk.header.parent
            on_head = (self.state.head_hash is None) or (parent == self.state.head_hash)

            # --- fork-tolerance: store known-side-branch blocks, count forks
            if not on_head:
                # if the parent exists (side branch), store & count fork; do not apply txs
                if parent in self.state.blocks:
                    if blk.header.hash not in self.state._fork_seen:
                        self.state._fork_seen.add(blk.header.hash)
                        self.state.blocks[blk.header.hash] = blk
                        self.state.children.setdefault(parent, []).append(blk.header.hash)
                        self.state.forks_observed += 1
                        self.state.fork_log.append({
                            "slot": blk.header.slot,
                            "proposer": blk.header.proposer,
                            "parent": parent,
                            "head_at_receive": self.state.head_hash,
                            "height": blk.header.height,
                            "hash": blk.header.hash,
                        })
                    return
                # if parent is unknown, just ignore (orphan)
                return

            # apply txs on the head-extended block
            applied = []
            fee_sum = 0
            for tx in blk.txs:
                try:
                    ok = apply_tx(self.state, tx)
                    if ok:
                        applied.append(tx)
                        fee_sum += int(tx.get("fee", 0))
                except AssertionError as e:
                    print(f"[{self.vid}] drop invalid tx in block: {e}")

            # distribute fees according to policy
            if fee_sum > 0:
                pol = self.state.fee_policy
                if pol == "proposer":
                    self.state.treasury = max(0, self.state.treasury - fee_sum)
                    self.state.credit(blk.header.proposer, fee_sum)
                    self.state.record_fee_earned(blk.header.proposer, fee_sum)
                elif pol == "split":
                    to_prop = fee_sum // 2
                    self.state.treasury = max(0, self.state.treasury - to_prop)
                    self.state.credit(blk.header.proposer, to_prop)
                    self.state.record_fee_earned(blk.header.proposer, to_prop)
                else:  # 'treasury' keeps everything; nothing to credit

                    pass

            # update head
            self.state.height = blk.header.height
            self.state.head_hash = blk.header.hash
            self.seen_slot_block_by_proposer[key] = blk.header.hash

            # store in block tree
            self.state.blocks[blk.header.hash] = blk
            self.state.children.setdefault(parent, []).append(blk.header.hash)

            # block log (applied-on-head only)
            self.state.block_log.append({
                "slot": blk.header.slot,
                "height": blk.header.height,
                "proposer": blk.header.proposer,
                "hash": blk.header.hash,
                "parent": blk.header.parent,
                "txs": len(applied),
                "fees_paid": fee_sum,
            })

            # prune mempool: drop only applied
            applied_hashes = {tx_hash(t) for t in applied}
            self.mempool = [t for t in self.mempool if tx_hash(t) not in applied_hashes]

    def run(self):
        while self.running and not self.net.stopped():
            try:
                while True:
                    msg = self.inbox.get_nowait()
                    if msg.kind == "tx":
                        tx = msg.data
                        h = tx_hash(tx)
                        if h in self.seen_tx: continue
                        self.seen_tx.add(h)
                        if self._admit(tx):
                            self.mempool.append(tx)
                    elif msg.kind == "block":
                        blk: Block = msg.data
                        if blk.header.hash in self.seen_blocks: continue
                        self.seen_blocks.add(blk.header.hash)
                        self._apply_block(blk)
            except queue.Empty:
                pass
            self._prune_mempool()
            self._maybe_propose()
            time.sleep(0.02)

    def submit_tx(self, tx: Dict[str, Any]):
        self.net.broadcast(NetMsg(kind="tx", data=tx))

    def stop(self):
        self.running = False
