# Thesis_DEFENSE_BALLA_ALEXANDROS

**Support Decentralized Federated Learning via PoS Blockchains**

This repository contains the code and artifacts used in my thesis defense. It implements a minimal Proof-of-Stake blockchain (“microchain”) that coordinates Federated Learning (FL) rounds with on-chain tickets, deposits, refunds, and finalization. It also includes a plain (non-blockchain) FL baseline so you can compare end-to-end round latency.

---

## TL;DR (one-minute demo)

From the repo root, run:

```bash
# BC-FL (blockchain-backed FL)
python -m microchain.runs.scenario_min \
  --validators 3 --clients 5 --tau 0.8 --runtime 20 \
  --ml_enable --ml_dim 32 --ml_val 512 --seed 1 \
  --csv_out bc_once.csv

# Plain FL + IPFS stub (storage comparable to BC-FL)
python -m microchain.runs.scenario_plain \
  --clients 5 --runtime 20 --ml_enable --ml_dim 32 --ml_val 512 \
  --seed 1 --slot_duration_sec 0.4 --use_ipfs \
  --csv_out plain_ipfs_once.csv

# Plain FL in-memory (idealized baseline)
python -m microchain.runs.scenario_plain \
  --clients 5 --runtime 20 --ml_enable --ml_dim 32 --ml_val 512 \
  --seed 1 --slot_duration_sec 0.4 \
  --csv_out plain_mem_once.csv

# Print a tiny comparison (+ overhead)
python - <<'PY'
import csv
def last_row(p):
    rows=list(csv.DictReader(open(p)))
    print(f"[{p}] last row:", rows[-1])
    return rows[-1]
bc  = last_row("bc_once.csv")
pip = last_row("plain_ipfs_once.csv")
pm  = last_row("plain_mem_once.csv")
L_bc  = float(bc["round_dur_secs_median"])
L_pip = float(pip["round_dur_secs_median"])
L_pm  = float(pm["round_dur_secs_median"])
pct = lambda a,b: 100*(a-b)/b
print("\n=== FL vs BC-FL (medians) ===")
print(f"Plain FL (mem):  {L_pm:.3f}s")
print(f"Plain FL (IPFS): {L_pip:.3f}s")
print(f"BC-FL:           {L_bc:.3f}s")
print(f"Overhead vs Plain+IPFS: {pct(L_bc,L_pip):.1f}%")
print(f"Overhead vs Plain+Mem:  {pct(L_bc,L_pm):.1f}%")
print("\nBC-FL stage medians (s) assuming 0.4s slots:")
slot=0.4
def gi(k):
    try: return int(bc.get(k,0))
    except: return 0
intent  = gi("intent_lat_slots_median")*slot
publish = gi("publish_lat_slots_median")*slot
finalz  = gi("finalize_lat_slots_median")*slot
agg     = float(bc.get("aggregate_ms_median","0"))/1000.0
ipfs    = (float(bc.get("ipfs_put_ms_median","0"))+float(bc.get("ipfs_get_ms_median","0")))/1000.0
print(f"intent={intent:.3f}, publish={publish:.3f}, finalize={finalz:.3f}, aggregate={agg:.3f}, ipfs={ipfs:.3f}")
PY
```

**What to expect (typical on a single machine):**
- Plain FL (mem or IPFS): **~1.6 s** median round latency  
- BC-FL (on-chain): **~2.0 s** median → **~+25%** overhead  
- Overhead is from consensus scheduling (intent/publish/finalize), not ML compute.

---

## Requirements

- **Python** 3.11+ (tested with 3.12)
- **pip** / **venv** recommended
- No external services required (IPFS is an in-process stub)

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies (if a requirements.txt exists; otherwise this is a no-op)
pip install -r requirements.txt || true

# Ensure Python can import from the repo root
export PYTHONPATH=.
```

---

## Repository layout (high level)

```
microchain/
  core/
    state.py        # canonical state (rules, tickets, balances, pools, logs)
    admit.py        # per-transaction admission checks
    apply.py        # deterministic state transitions for each tx type
    consensus.py    # VRF-based proposer eligibility and fork-choice helpers
  roles/
    validator.py    # proposer/validator loop, mempool, fee policy, block apply
    lister.py       # model listing, aggregation, round finalization
    client.py       # clients reserve tickets, publish updates, claim rewards
  storage/
    ipfs_stub.py    # simple in-memory content-addressed storage
  runs/
    scenario_min.py   # BC-FL (blockchain) scenario
    scenario_plain.py # Plain FL baseline (mem or IPFS)
```
---

## How it works (in one paragraph)

- The **Lister** lists a model (with rules). **Clients** obtain **tickets** via `UpdateIntentTx` (deposit + rate limit).  
- Clients train locally, upload tiny updates to the IPFS stub, then **publish** the update by revealing the ticket.  
- The Lister **aggregates** revealed updates deterministically and **finalizes** the round via `GlobalUpdateTx` containing `included`, `scores`, and `refunds`.  
- **Validators** run a VRF-based leader election, build blocks, and apply state transitions deterministically. Fees are handled per policy.  
- Large artifacts are referenced by CIDs; consensus only carries small metadata, making the system auditable and replayable.

---

## Running the scenarios

### BC-FL (blockchain) mode

```bash
python -m microchain.runs.scenario_min \
  --validators 3 --clients 5 --tau 0.8 --runtime 20 \
  --ml_enable --ml_dim 32 --ml_val 512 --seed 1 \
  --csv_out results_bc.csv
```

**Useful flags**
- `--clients N` : number of clients  
- `--validators V` : number of validators  
- `--tau X` : proposer eligibility (higher → more contention/forks)  
- `--runtime SECS` : wall-clock runtime  
- `--ml_enable --ml_dim D --ml_val NVAL` : synthetic logistic regression task  
- `--ipfs_fail_rate P` : inject IPFS retrieval failures  
- `--fee_policy {proposer|treasury|split}` : fee distribution policy  
- `--csv_out PATH` : append a one-line CSV summary for the run

**Console output:** balances, fees, fork stats, and round events (start/finalize slots and duration).  
**CSV:** the same run’s key medians/metrics captured in one row.

### Plain FL baseline (no blockchain)

With IPFS stub:
```bash
python -m microchain.runs.scenario_plain \
  --clients 5 --runtime 20 --ml_enable --ml_dim 32 --ml_val 512 \
  --seed 1 --slot_duration_sec 0.4 --use_ipfs \
  --csv_out results_plain_ipfs.csv
```

In-memory (idealized):
```bash
python -m microchain.runs.scenario_plain \
  --clients 5 --runtime 20 --ml_enable --ml_dim 32 --ml_val 512 \
  --seed 1 --slot_duration_sec 0.4 \
  --csv_out results_plain_mem.csv
```

**Console output:** median round duration and per-component medians (`aggregate_ms_median`, `ipfs_put_ms_median`, `ipfs_get_ms_median`).

---

## Minimal comparison (what to cite)

- Plain FL (mem or IPFS): **~1.6 s** median per round  
- BC-FL: **~2.0 s** median per round  
- **Overhead** vs Plain FL: **~+25%**, dominated by consensus scheduling (intent → publish → finalize).  
- Aggregation and IPFS I/O are negligible in this single-host setup.

Use the script in the **TL;DR** section to print these from your own run CSVs.

---

## Key metrics (quick reference)

- **`round_dur_secs_median`** — primary end-to-end round latency.  
- **BC-FL stage medians** (if present):  
  - `intent_lat_slots_median`, `publish_lat_slots_median`, `finalize_lat_slots_median` → multiply by slot duration for seconds.  
  - `aggregate_ms_median`, `ipfs_*_ms_median` → milliseconds.  
- **Consensus health:** `forks_observed`, `unique_eligibilities`, `avg_txs_per_block`.  
- **Economics:** `fees_charged`, `fees_distributed`, and their ratio (should be ~1.0).

---

## Troubleshooting

- **`ModuleNotFoundError: microchain`**  
  Run commands from the repo root and set `PYTHONPATH=.`

- **No CSV file written**  
  Ensure `--csv_out path.csv` is provided and the program is allowed to finish.

- **Numbers differ from README**  
  That’s normal; latency depends on machine load. Trend should hold: BC-FL ≈ +25% over plain FL in this small setup.

- **Want non-trivial aggregation/IPFS times**  
  Increase model size (`--ml_dim`), add clients, or introduce `--ipfs_fail_rate`.

---

## Reproducibility notes

- Single-machine, in-process networking, stubbed IPFS.  
- Use `--seed` to keep the synthetic ML task and leader selection reproducible.  
- Each run appends one CSV line with configuration and measured medians.

---

## Citation

> Alexandros Balla, *Support Federated Learning via PoS Blockchains*, Thesis Defense Repository, 2025. Repository: `Thesis_DEFENSE_BALLA_ALEXANDROS`.

---
