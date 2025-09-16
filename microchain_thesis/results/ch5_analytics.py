#!/usr/bin/env python3
import argparse, os, sys, textwrap
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Helpers
# ----------------------------
def load_csv(path):
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        # normalize common column name variants
        ren = {
            "avg_txs/block": "avg_txs_per_block",
            "avg_txs_per_block": "avg_txs_per_block",
            "avg_fee/tx": "avg_fee_per_tx",
            "avg_fee_per_tx": "avg_fee_per_tx",
        }
        df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
        return df
    except Exception as e:
        print(f"[warn] failed to read {path}: {e}", file=sys.stderr)
        return None

def safe_cols(df, cols):
    return [c for c in cols if c in df.columns]

def summarize(df, by=None):
    """Return mean±std for key metrics; grouped if by is given."""
    if df is None or df.empty:
        return pd.DataFrame()
    metrics = [c for c in [
        "txs_applied","blocks_applied","avg_txs_per_block","avg_fee_per_tx",
        "fees_charged","fees_distributed","forks_observed","missed_proposals",
        "unique_eligibilities","alloc_total","refund_total","forfeit_total",
        "remainder_total","treasury","ml_rounds","ml_last_acc_before",
        "ml_last_acc_after","ml_impr_sum","ml_clients_used_sum"
    ] if c in df.columns]

    if not metrics:
        return pd.DataFrame()

    if by is None:
        g = df.assign(_k=1).groupby("_k")
    else:
        by = [b for b in ([by] if isinstance(by, str) else list(by)) if b in df.columns]
        if not by:
            g = df.assign(_k=1).groupby("_k")
        else:
            g = df.groupby(by, dropna=False)

    agg = {}
    for m in metrics:
        agg[m] = ["mean","std","min","max","count"]
    out = g.agg(agg)
    out.columns = ["_".join(col).strip() for col in out.columns.values]
    out = out.reset_index()
    if "fees_charged_mean" in out.columns and "fees_distributed_mean" in out.columns:
        out["fees_distribution_ratio"] = out["fees_distributed_mean"] / out["fees_charged_mean"].replace(0, np.nan)
    return out

def lineplot(df, x, y, hue=None, title="", out=None):
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        return
    plt.figure(figsize=(6,4))
    if hue and hue in df.columns:
        for k, sub in df.sort_values(x).groupby(hue):
            plt.plot(sub[x], sub[y], marker="o", label=str(k))
        plt.legend()
    else:
        sub = df.sort_values(x)
        plt.plot(sub[x], sub[y], marker="o")
    plt.xlabel(x); plt.ylabel(y); plt.title(title); plt.grid(True, alpha=.3)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()

def barplot(df, x, y, hue=None, title="", out=None, rotate=False):
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        return
    plt.figure(figsize=(7,4))
    if hue and hue in df.columns:
        # grouped bars
        piv = df.pivot_table(index=x, columns=hue, values=y, aggfunc="mean")
        piv.plot(kind="bar")
        plt.legend(title=hue)
    else:
        df.plot(kind="bar", x=x, y=y)
    plt.title(title); plt.ylabel(y)
    if rotate:
        plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", alpha=.3)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()

def save_table(df, outpath):
    if df is None or df.empty:
        return
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outpath, index=False)

def section(md, title):
    md.append(f"\n## {title}\n")

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default=".", help="input directory with CSVs")
    ap.add_argument("--out", dest="outdir", default="./ch5_outputs", help="output directory")
    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    figs = outdir / "figures"
    tabs = outdir / "tables"
    outdir.mkdir(parents=True, exist_ok=True)

    files = {
        "baseline": "baseline.csv",
        "tau_sweep": "tau_sweep.csv",
        "fees_vs_sink": "fees_vs_sink.csv",
        "ipfs_fail": "ipfs_fail.csv",
        "intent_cap": "intent_cap.csv",
        "scale_validators": "scale_validators.csv",
        "scale_clients": "scale_clients.csv",
        "ml_clean": "ml_clean.csv",
        "ml_fail": "ml_fail.csv",
        "rand_period": "rand_period.csv",
    }

    # Load
    dfs = {k: load_csv(indir / v) for k, v in files.items()}

    # Global baseline summary
    base_sum = summarize(dfs.get("baseline"))
    save_table(base_sum, tabs / "baseline_summary.csv")

    # τ sweep
    tau_df = dfs.get("tau_sweep")
    if tau_df is not None and not tau_df.empty and "tau" in tau_df.columns:
        tau_sum = summarize(tau_df, by="tau")
        save_table(tau_sum, tabs / "tau_summary.csv")
        for y in ["txs_applied_mean","avg_txs_per_block_mean","forks_observed_mean","fees_charged_mean","fees_distribution_ratio"]:
            if y in tau_sum.columns:
                lineplot(tau_sum, x="tau", y=y, title=f"{y} vs tau", out=figs / f"tau_{y}.png")

    # Fees policy × remainder sink
    fr_df = dfs.get("fees_vs_sink")
    if fr_df is not None and not fr_df.empty:
        by = [c for c in ["fee_policy","remainder_sink"] if c in fr_df.columns]
        fr_sum = summarize(fr_df, by=by or None)
        save_table(fr_sum, tabs / "fees_vs_sink_summary.csv")
        # plots
        ycand = [c for c in fr_sum.columns if c.endswith("_mean") and any(m in c for m in ["txs_applied","forks_observed","fees_charged","fees_distribution_ratio"])]
        for y in ycand:
            if "fee_policy" in fr_sum.columns and "remainder_sink" in fr_sum.columns:
                # one figure per metric, grouped by fee_policy with remainder_sink on x
                for pol, sub in fr_sum.groupby("fee_policy"):
                    fname = f"fees_vs_sink_{y}_{pol}.png"
                    barplot(sub, x="remainder_sink", y=y, title=f"{y} by sink ({pol})", out=figs / fname, rotate=True)

    # IPFS fail rate sweep
    ipfs_df = dfs.get("ipfs_fail")
    if ipfs_df is not None and not ipfs_df.empty and "ipfs_fail_rate" in ipfs_df.columns:
        ipfs_sum = summarize(ipfs_df, by="ipfs_fail_rate")
        save_table(ipfs_sum, tabs / "ipfs_fail_summary.csv")
        for y in ["txs_applied_mean","forks_observed_mean","avg_txs_per_block_mean"]:
            if y in ipfs_sum.columns:
                lineplot(ipfs_sum, x="ipfs_fail_rate", y=y, title=f"{y} vs ipfs_fail_rate", out=figs / f"ipfs_{y}.png")

    # Intent cap
    cap_df = dfs.get("intent_cap")
    if cap_df is not None and not cap_df.empty and "intent_cap" in cap_df.columns:
        cap_sum = summarize(cap_df, by="intent_cap")
        save_table(cap_sum, tabs / "intent_cap_summary.csv")
        for y in ["txs_applied_mean","avg_txs_per_block_mean","fees_charged_mean"]:
            if y in cap_sum.columns:
                lineplot(cap_sum, x="intent_cap", y=y, title=f"{y} vs intent_cap", out=figs / f"intent_cap_{y}.png")

    # Scale validators
    val_df = dfs.get("scale_validators")
    if val_df is not None and not val_df.empty and "validators" in val_df.columns:
        val_sum = summarize(val_df, by="validators")
        save_table(val_sum, tabs / "scale_validators_summary.csv")
        for y in ["txs_applied_mean","forks_observed_mean","avg_txs_per_block_mean","unique_eligibilities_mean"]:
            if y in val_sum.columns:
                lineplot(val_sum, x="validators", y=y, title=f"{y} vs validators", out=figs / f"validators_{y}.png")

    # Scale clients
    cli_df = dfs.get("scale_clients")
    if cli_df is not None and not cli_df.empty and "clients" in cli_df.columns:
        cli_sum = summarize(cli_df, by="clients")
        save_table(cli_sum, tabs / "scale_clients_summary.csv")
        for y in ["txs_applied_mean","forks_observed_mean","avg_txs_per_block_mean"]:
            if y in cli_sum.columns:
                lineplot(cli_sum, x="clients", y=y, title=f"{y} vs clients", out=figs / f"clients_{y}.png")

    # Random period sweep (if present)
    rp_df = dfs.get("rand_period")
    if rp_df is not None and not rp_df.empty and "rand_period" in rp_df.columns:
        rp_sum = summarize(rp_df, by="rand_period")
        save_table(rp_sum, tabs / "rand_period_summary.csv")
        for y in ["txs_applied_mean","forks_observed_mean","avg_txs_per_block_mean"]:
            if y in rp_sum.columns:
                lineplot(rp_sum, x="rand_period", y=y, title=f"{y} vs rand_period", out=figs / f"rand_period_{y}.png")

    # ML experiment summaries (optional)
    for k in ["ml_clean","ml_fail"]:
        mdf = dfs.get(k)
        if mdf is not None and not mdf.empty:
            msum = summarize(mdf)
            save_table(msum, tabs / f"{k}_summary.csv")
            # plot improvement if we have it
            if "ml_rounds_mean" in msum.columns and "ml_impr_sum_mean" in msum.columns:
                # use raw df for round-wise if present
                if "ml_rounds" in mdf.columns and "ml_impr_sum" in mdf.columns:
                    lineplot(mdf.sort_values("ml_rounds"), x="ml_rounds", y="ml_impr_sum",
                             title=f"{k}: cumulative ML improvement vs rounds",
                             out=figs / f"{k}_ml_improvement_vs_rounds.png")

    # --------- Write a short markdown report ----------
    md = []
    md.append("# Chapter 5: Experiment Metrics Summary\n")
    section(md, "Overview")
    md.append("- This report aggregates per-run CSV outputs from scenario sweeps.\n")
    md.append("- Metrics include throughput, fees, consensus health, and ML outcomes where applicable.\n")

    if base_sum is not None and not base_sum.empty:
        section(md, "Baseline")
        md.append("See `tables/baseline_summary.csv`.\n")

    if tau_df is not None and not tau_df.empty:
        section(md, "τ Sweep")
        md.append("Figures: `figures/tau_*` and table `tables/tau_summary.csv`.\n")

    if fr_df is not None and not fr_df.empty:
        section(md, "Fee Policy × Remainder Sink")
        md.append("Figures: `figures/fees_vs_sink_*` and `tables/fees_vs_sink_summary.csv`.\n")

    if ipfs_df is not None and not ipfs_df.empty:
        section(md, "IPFS Failure Sensitivity")
        md.append("Figures: `figures/ipfs_*` and `tables/ipfs_fail_summary.csv`.\n")

    if cap_df is not None and not cap_df.empty:
        section(md, "Intent Cap")
        md.append("Figures: `figures/intent_cap_*` and `tables/intent_cap_summary.csv`.\n")

    if val_df is not None and not val_df.empty:
        section(md, "Scaling Validators")
        md.append("Figures: `figures/validators_*` and `tables/scale_validators_summary.csv`.\n")

    if cli_df is not None and not cli_df.empty:
        section(md, "Scaling Clients")
        md.append("Figures: `figures/clients_*` and `tables/scale_clients_summary.csv`.\n")

    if rp_df is not None and not rp_df.empty:
        section(md, "Random Period")
        md.append("Figures: `figures/rand_period_*` and `tables/rand_period_summary.csv`.\n")

    if any(dfs.get(k) is not None and not dfs[k].empty for k in ["ml_clean","ml_fail"]):
        section(md, "ML Outcomes")
        md.append("See `tables/ml_clean_summary.csv`, `tables/ml_fail_summary.csv` and `figures/*ml_improvement_vs_rounds.png`.\n")

    (outdir / "ch5_report.md").write_text("".join(md))
    print(f"[ok] Wrote outputs to: {outdir}")

if __name__ == "__main__":
    main()

