# microchain/runs/exp_refund_avail.py
from __future__ import annotations
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
    l = Lister("Lister", state, ipfs, submit_tx, "toy", rules=rules, fund_amount=600)
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

if __name__ == "__main__":
    print(run())
