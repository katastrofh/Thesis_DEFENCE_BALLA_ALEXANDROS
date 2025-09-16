# microchain/runs/exp_deposit_spam.py
from __future__ import annotations
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
    l = Lister("Lister", state, ipfs, submit_tx, "toy", rules=rules, fund_amount=200)
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

if __name__ == "__main__":
    print(run())
