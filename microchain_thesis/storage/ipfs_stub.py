# microchain/storage/ipfs_stub.py
from __future__ import annotations
from typing import Tuple, Dict
import hashlib
import random

class IPFS:
    def __init__(self, fail_rate: float = 0.0):
        self.store: Dict[str, bytes] = {}
        self.pins: Dict[str, int] = {}
        self.fail_rate = fail_rate

    def put(self, data: bytes, tag: str = "") -> str:
        if not isinstance(data, (bytes, bytearray)):
            data = str(data).encode("utf-8")
        payload = tag.encode() + b"|" + bytes(data)
        cid = hashlib.sha256(payload).hexdigest()
        self.store[cid] = bytes(data)
        self.pins[cid] = self.pins.get(cid, 0) + 1
        return cid

    def get(self, cid: str) -> Tuple[bool, bytes]:
        if random.random() < self.fail_rate:
            return False, b""
        data = self.store.get(cid)
        if data is None:
            return False, b""
        return True, data

    def pin(self, cid: str):
        self.pins[cid] = self.pins.get(cid, 0) + 1


