# microchain/core/consensus.py
from __future__ import annotations
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
