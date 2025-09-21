# microchain/runs/exp_latency.py
from __future__ import annotations
import time
from .scenario_min import State, InProcNetwork, IPFS, Validator, Lister, Client, Rules, give_faucet

def run(validators=3, clients=5, runtime=8.0, slot_sec=0.2):
    state = State()
    state.slot_duration_sec = slot_sec
    net = InProcNetwork(state, tau=0.8)
    ipfs = IPFS(fail_rate=0.0)

    for i in range(validators):
        with state.lock():
            state.register_validator(f"V{i+1}", 100)

    vals = []
    for i in range(validators):
        v = Validator(f"V{i+1}", f"sk{i+1}".encode(), state, net)
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

if __name__ == "__main__":
    out = run()
    print(out)
