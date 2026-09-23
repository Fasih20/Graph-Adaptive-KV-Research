from __future__ import annotations

import argparse
import io
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


POLICY_ORDER_SYNC = [
    "no_prefetch", "cosine", "graph_fixed",
    "adaptive_offline_global", "adaptive_online_warm",
]
POLICY_ORDER_ASYNC = [
    "no_prefetch", "cosine", "semantic_only", "structure",
    "two_hop", "offline", "online",
]
LABELS = {
    "no_prefetch": "No prefetch",
    "cosine": "Cosine",
    "graph_fixed": "Fixed graph",
    "adaptive_offline_global": "Offline adaptive",
    "adaptive_online_warm": "Online adaptive",
    "semantic_only": "Semantic-only",
    "structure": "+ Structure",
    "two_hop": "+ Two-hop",
    "offline": "+ Offline",
    "online": "+ Online",
}
COLORS = {
    "no_prefetch": "#6B7280",
    "cosine": "#4C78A8",
    "graph_fixed": "#F58518",
    "adaptive_offline_global": "#54A24B",
    "adaptive_online_warm": "#B279A2",
    "semantic_only": "#72B7B2",
    "structure": "#ECA82C",
    "two_hop": "#E45756",
    "offline": "#54A24B",
    "online": "#B279A2",
}


def first(root: Path, pattern: str) -> Path:
    matches = list(root.rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern}, found {len(matches)}")
    return matches[0]


def find_run(root: Path, dataset: str, model_fragment: str) -> Path:
    matches = []
    for manifest in root.rglob("run_manifest.json"):
        data = json.loads(manifest.read_text())
        cfg = data.get("config", {})
        if cfg.get("dataset") == dataset and model_fragment in cfg.get("model", ""):
            matches.append(manifest.parent)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one run for {dataset}/{model_fragment}, found {len(matches)}"
        )
    return matches[0]


def savefig(fig: plt.Figure, out: Path, stem: str) -> None:
    fig.tight_layout()
    for suffix, dpi in [("png", 320), ("pdf", None), ("svg", None)]:
        # Render in memory and commit the complete byte stream in one write.
        # This avoids leaving a zero-byte/truncated figure if a notebook-backed
        # filesystem briefly interrupts a direct Matplotlib file write.
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format=suffix,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        payload = buffer.getvalue()
        if not payload:
            raise RuntimeError(f"Matplotlib produced an empty {suffix} for {stem}")
        (out / f"{stem}.{suffix}").write_bytes(payload)
    plt.close(fig)


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.12)
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 320,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_event_frame(run_root: Path) -> pd.DataFrame:
    fields = [
        "policy", "repetition", "event_id", "document_id", "query_type",
        "prediction_hit", "shadow_residency_hit", "ready_cache_ttft_ms",
        "end_to_end_ttft_ms", "request_success", "retry_count",
        "prompt_token_hash", "target_chunk_id", "output_text_hash",
    ]
    rows = []
    paths = sorted(run_root.glob("arms/*/k=*/rep=*/attempt_*/events.jsonl"))
    for path in paths:
        with path.open() as handle:
            for line in handle:
                record = json.loads(line)
                rows.append({field: record.get(field) for field in fields})
    return pd.DataFrame(rows)


def document_cluster_bootstrap(
    frame: pd.DataFrame,
    policy: str,
    baseline: str,
    metric: str,
    draws: int = 10000,
    seed: int = 20260921,
) -> tuple[float, float, float]:
    sub = frame[frame.policy.isin([policy, baseline])]
    wide = sub.pivot(
        index=["document_id", "repetition", "event_id"],
        columns="policy",
        values=metric,
    ).dropna()
    wide["delta"] = wide[policy] - wide[baseline]
    by_doc = wide.groupby(level="document_id").delta.agg(["sum", "size"])
    sums = by_doc["sum"].to_numpy()
    sizes = by_doc["size"].to_numpy()
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(by_doc), size=(draws, len(by_doc)))
    boot = sums[picks].sum(axis=1) / sizes[picks].sum(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(wide.delta.mean()), float(lo), float(hi)


def audit_sync_run(run_root: Path, name: str) -> dict:
    manifest = json.loads((run_root / "run_manifest.json").read_text())
    cfg = manifest["config"]
    sanity = json.loads((run_root / "sanity/sanity_report.json").read_text())
    post = json.loads((run_root / "postrun_protocol_report.json").read_text())
    completes = list(run_root.glob("arms/*/k=*/rep=*/COMPLETE.json"))
    expected = len(cfg["policies"]) * len(cfg["k_values"]) * cfg["repetitions"]
    events = [json.loads(p.read_text()).get("events") for p in completes]
    return {
        "run": name,
        "dataset": cfg["dataset"],
        "model": cfg["model"],
        "k_values": ";".join(map(str, cfg["k_values"])),
        "top_m": cfg["top_m"],
        "unique_events": cfg["events"],
        "repetitions": cfg["repetitions"],
        "complete_arms": len(completes),
        "expected_arms": expected,
        "events_per_complete_arm_min": min(events) if events else np.nan,
        "sanity_critical_passed": bool(sanity.get("critical_passed")),
        "publication_controls_passed": bool(post.get("publication_controls_passed")),
        "order_independence_status": post.get("order_independence_status"),
        "status": "authoritative" if (
            len(completes) == expected
            and all(x == cfg["events"] for x in events)
            and sanity.get("critical_passed")
            and post.get("publication_controls_passed")
        ) else "review",
    }


def build(input_zip: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    tables = output_dir / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    setup_style()

    with tempfile.TemporaryDirectory(prefix="graphkv_figures_") as tmp:
        extracted = Path(tmp)
        with zipfile.ZipFile(input_zip) as archive:
            archive.extractall(extracted)
        root = first(extracted, "New_Suggestions_Results")

        llama_hotpot = find_run(root, "hotpot", "Llama-3.2-1B")
        llama_2wiki = find_run(root, "2wiki", "Llama-3.2-1B")
        qwen_2wiki = find_run(root, "2wiki", "Qwen2.5-1.5B")
        async_root = first(root, "revision_manifest.json").parent
        sensitivity_root = first(root, "development_m_selection.csv").parent

        # ---------- Audit ----------
        audit_rows = [
            audit_sync_run(llama_hotpot, "Llama–Hotpot synchronous"),
            audit_sync_run(llama_2wiki, "Llama–2Wiki synchronous"),
            audit_sync_run(qwen_2wiki, "Qwen–2Wiki synchronous"),
        ]
        async_manifest = json.loads((async_root / "revision_manifest.json").read_text())
        async_complete = json.loads((async_root / "COMPLETE.json").read_text())
        native_review = json.loads((async_root / "gate/native_equivalence_review.json").read_text())
        audit_rows.append({
            "run": "Qwen–Hotpot asynchronous",
            "dataset": "hotpot",
            "model": async_manifest["config"]["model"],
            "k_values": "10",
            "top_m": 20,
            "unique_events": 600,
            "repetitions": 1,
            "complete_arms": async_complete.get("blocks"),
            "expected_arms": 42,
            "events_per_complete_arm_min": 100,
            "sanity_critical_passed": native_review.get("validated", native_review.get("passed")),
            "publication_controls_passed": native_review.get("validated", native_review.get("passed")),
            "order_independence_status": "randomized blocks",
            "status": "authoritative" if async_complete.get("blocks") == 42 else "review",
        })
        audit = pd.DataFrame(audit_rows)
        audit.to_csv(tables / "audit_inventory.csv", index=False)

        # ---------- Llama Hotpot K sweep ----------
        lh = pd.read_csv(llama_hotpot / "headline_mean_results.csv")
        lh.to_csv(tables / "llama_hotpot_k_sweep.csv", index=False)

        fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7))
        for policy in POLICY_ORDER_SYNC[1:]:
            g = lh[lh.policy.eq(policy)].sort_values("k")
            axes[0].plot(g.k, g.prediction_recall * 100, marker="o", linewidth=2.2,
                         label=LABELS[policy], color=COLORS[policy])
        axes[0].set(title="Prediction recall", xlabel="Prefetch budget K", ylabel="Recall (%)",
                    xticks=sorted(lh.k.unique()), ylim=(60, 101))
        for policy in POLICY_ORDER_SYNC:
            g = lh[lh.policy.eq(policy)].sort_values("k")
            axes[1].plot(g.k, g.mean_ready_cache_ttft_ms, marker="o", linewidth=2.2,
                         label=LABELS[policy], color=COLORS[policy])
        axes[1].set(title="Ready-cache target TTFT", xlabel="Prefetch budget K",
                    ylabel="TTFT (ms)", xticks=sorted(lh.k.unique()))
        axes[0].legend(ncol=2, fontsize=9)
        savefig(fig, figures, "fig01_llama_hotpot_k_sweep")

        fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7))
        for policy in POLICY_ORDER_SYNC:
            g = lh[lh.policy.eq(policy)].sort_values("k")
            axes[0].plot(g.k, g.mean_ready_cache_ttft_ms, marker="o", linewidth=2.1,
                         label=LABELS[policy], color=COLORS[policy])
            axes[1].plot(g.k, g.mean_end_to_end_ttft_ms, marker="o", linewidth=2.1,
                         label=LABELS[policy], color=COLORS[policy])
        axes[0].set(title="Service begins after synchronous population", xlabel="K",
                    ylabel="Ready-cache TTFT (ms)", xticks=sorted(lh.k.unique()))
        axes[1].set(title="Population cost included", xlabel="K",
                    ylabel="Synchronous E2E TTFT (ms)", xticks=sorted(lh.k.unique()))
        axes[1].legend(ncol=2, fontsize=8.5)
        savefig(fig, figures, "fig02_llama_hotpot_latency_boundary")

        # Llama Hotpot paired recall deltas vs cosine.
        ldel = pd.read_csv(llama_hotpot / "analysis/paired_policy_deltas.csv")
        lrec = ldel[(ldel.baseline == "cosine") & (ldel.query_type == "all")
                    & (ldel.metric == "prediction_hit")
                    & ldel.policy.isin(POLICY_ORDER_SYNC[2:])].copy()
        lrec.to_csv(tables / "llama_hotpot_paired_recall_vs_cosine.csv", index=False)
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        offsets = {"graph_fixed": -0.20, "adaptive_offline_global": 0,
                   "adaptive_online_warm": 0.20}
        for policy in POLICY_ORDER_SYNC[2:]:
            g = lrec[lrec.policy.eq(policy)].sort_values("k")
            x = g.k.to_numpy() + offsets[policy]
            y = g.mean_delta.to_numpy() * 100
            lo = g.cluster_bootstrap_ci_low.to_numpy() * 100
            hi = g.cluster_bootstrap_ci_high.to_numpy() * 100
            ax.errorbar(x, y, yerr=[y - lo, hi - y], marker="o", capsize=4,
                        linewidth=1.8, label=LABELS[policy], color=COLORS[policy])
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set(title="Llama–Hotpot paired recall gain over cosine",
               xlabel="Prefetch budget K", ylabel="Recall difference (percentage points)",
               xticks=sorted(lh.k.unique()))
        ax.legend(ncol=3, fontsize=8.5)
        savefig(fig, figures, "fig03_llama_hotpot_paired_recall")

        # ---------- 2Wiki cross-model ----------
        qh = pd.read_csv(qwen_2wiki / "headline_mean_results.csv")
        ll = pd.read_csv(llama_2wiki / "headline_mean_results.csv")
        qh["model_short"] = "Qwen-2.5-1.5B"
        ll["model_short"] = "Llama-3.2-1B"
        cross = pd.concat([qh, ll], ignore_index=True)
        cross.to_csv(tables / "2wiki_cross_model_headline.csv", index=False)

        metrics = [
            ("prediction_recall", "Prediction recall (%)", 100),
            ("shadow_residency_rate", "Resident at demand (%)", 100),
            ("mean_ready_cache_ttft_ms", "Ready-cache TTFT (ms)", 1),
            ("mean_end_to_end_ttft_ms", "Synchronous E2E TTFT (ms)", 1),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
        for ax, (metric, title, scale) in zip(axes.flat, metrics):
            draw = cross.copy()
            draw["value"] = draw[metric] * scale
            policies = POLICY_ORDER_SYNC if metric != "prediction_recall" else POLICY_ORDER_SYNC[1:]
            draw = draw[draw.policy.isin(policies)]
            sns.barplot(data=draw, x="model_short", y="value", hue="policy",
                        hue_order=policies, palette=COLORS, ax=ax, errorbar=None)
            ax.set(title=title, xlabel="", ylabel=title)
            if scale == 100:
                ax.set_ylim(0, 102)
            handles, _ = ax.get_legend_handles_labels()
            ax.legend_.remove()
        legend_labels = [LABELS[p] for p in POLICY_ORDER_SYNC]
        fig.legend(handles, legend_labels[:len(handles)], loc="upper center",
                   ncol=5, bbox_to_anchor=(0.5, 1.015), fontsize=9)
        savefig(fig, figures, "fig04_2wiki_cross_model")

        # Event-level 2Wiki audit and paired clustered CIs.
        qevents = load_event_frame(qwen_2wiki)
        levents = load_event_frame(llama_2wiki)
        paired_rows = []
        integrity_rows = []
        for model, events in [("Qwen-2.5-1.5B", qevents), ("Llama-3.2-1B", levents)]:
            groups = events.groupby(["repetition", "event_id"])
            integrity_rows.append({
                "model": model,
                "records": len(events),
                "unique_events": events.event_id.nunique(),
                "repetitions": events.repetition.nunique(),
                "request_failures": int((events.request_success != 1).sum()),
                "retries": int(events.retry_count.sum()),
                "prompt_parity_failures": int((groups.prompt_token_hash.nunique(dropna=False) != 1).sum()),
                "target_parity_failures": int((groups.target_chunk_id.nunique(dropna=False) != 1).sum()),
                "output_hash_disagreement_groups": int((groups.output_text_hash.nunique(dropna=False) != 1).sum()),
                "paired_groups": groups.ngroups,
            })
            for policy in POLICY_ORDER_SYNC[2:]:
                for metric in ["prediction_hit", "shadow_residency_hit", "ready_cache_ttft_ms"]:
                    point, lo, hi = document_cluster_bootstrap(events, policy, "cosine", metric)
                    paired_rows.append({
                        "model": model, "policy": policy, "baseline": "cosine",
                        "metric": metric, "mean_delta": point,
                        "ci_low": lo, "ci_high": hi,
                        "unique_events": 600, "system_measurements": 1200,
                        "document_clusters": events.document_id.nunique(),
                    })
        paired = pd.DataFrame(paired_rows)
        paired.to_csv(tables / "2wiki_document_clustered_paired_deltas.csv", index=False)
        pd.DataFrame(integrity_rows).to_csv(tables / "2wiki_event_integrity.csv", index=False)

        fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))
        for ax, metric, title, scale in [
            (axes[0], "prediction_hit", "Prediction recall gain over cosine", 100),
            (axes[1], "ready_cache_ttft_ms", "Ready-cache TTFT change vs cosine", 1),
        ]:
            sub = paired[paired.metric.eq(metric)].copy()
            labels = []
            ypos = []
            for i, (_, row) in enumerate(sub.iterrows()):
                y = len(sub) - 1 - i
                point, lo, hi = row.mean_delta * scale, row.ci_low * scale, row.ci_high * scale
                ax.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt="o", capsize=4,
                            color=COLORS[row.policy], markersize=6)
                labels.append(f"{row.model} · {LABELS[row.policy]}")
                ypos.append(y)
            ax.axvline(0, color="#333333", linewidth=1)
            ax.set_yticks(ypos, labels)
            ax.set_title(title)
            ax.set_xlabel("Percentage points" if scale == 100 else "Milliseconds (negative is faster)")
        savefig(fig, figures, "fig05_2wiki_paired_deltas")

        # Query-type recall, pooled correctly over repetitions.
        qsum = pd.read_csv(qwen_2wiki / "summary_by_query_type.csv")
        lsum = pd.read_csv(llama_2wiki / "summary_by_query_type.csv")
        qsum["model_short"] = "Qwen-2.5-1.5B"
        lsum["model_short"] = "Llama-3.2-1B"
        qtypes = pd.concat([qsum, lsum], ignore_index=True)
        qtypes = qtypes[qtypes.query_type.ne("all")]
        pooled = (qtypes.groupby(["model_short", "policy", "query_type"], as_index=False)
                  .apply(lambda g: pd.Series({
                      "prediction_recall": np.average(g.mean_prediction_hit, weights=g.events),
                      "ready_cache_ttft_ms": np.average(g.mean_ready_cache_ttft_ms, weights=g.events),
                  }), include_groups=False))
        pooled.to_csv(tables / "2wiki_query_type_pooled.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), sharey=True)
        for ax, model in zip(axes, ["Qwen-2.5-1.5B", "Llama-3.2-1B"]):
            draw = pooled[(pooled.model_short == model) & pooled.policy.isin(POLICY_ORDER_SYNC[1:])].copy()
            draw["recall_pct"] = draw.prediction_recall * 100
            sns.barplot(data=draw, x="query_type", y="recall_pct", hue="policy",
                        hue_order=POLICY_ORDER_SYNC[1:], palette=COLORS, ax=ax, errorbar=None)
            ax.set(title=model, xlabel="Controlled query type", ylabel="Prediction recall (%)", ylim=(0, 102))
            ax.legend_.remove()
        handles, _ = axes[0].get_legend_handles_labels()
        fig.legend(handles, [LABELS[p] for p in POLICY_ORDER_SYNC[1:]],
                   loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04))
        savefig(fig, figures, "fig06_2wiki_query_type_recall")

        # Learned offline weights.
        weights_rows = []
        for label, path in [
            ("Llama–Hotpot", llama_hotpot / "fitted_policy.json"),
            ("Qwen–2Wiki", qwen_2wiki / "fitted_policy.json"),
            ("Llama–2Wiki", llama_2wiki / "fitted_policy.json"),
        ]:
            fit = json.loads(path.read_text())
            weights_rows.append({"run": label, "semantic": fit["offline_weights"][0],
                                 "structural": fit["offline_weights"][1], "eta": fit["eta"]})
        weights = pd.DataFrame(weights_rows)
        weights.to_csv(tables / "learned_offline_weights.csv", index=False)
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        ax.bar(weights.run, weights.semantic * 100, label="Semantic", color="#4C78A8")
        ax.bar(weights.run, weights.structural * 100, bottom=weights.semantic * 100,
               label="Structural", color="#F58518")
        ax.set(title="Leakage-safe offline weights", ylabel="Weight (%)", ylim=(0, 100))
        ax.legend(ncol=2)
        for i, row in weights.iterrows():
            ax.text(i, row.semantic * 50, f"{row.semantic:.2f}", ha="center", va="center", color="white")
            ax.text(i, (row.semantic + row.structural / 2) * 100,
                    f"{row.structural:.2f}", ha="center", va="center", color="white")
        savefig(fig, figures, "fig07_learned_offline_weights")

        # ---------- Asynchronous ablation ----------
        ah = pd.read_csv(async_root / "headline_results.csv")
        ah["policy"] = pd.Categorical(ah.policy, categories=POLICY_ORDER_ASYNC, ordered=True)
        ah = ah.sort_values("policy")
        ah.to_csv(tables / "async_hotpot_ablation.csv", index=False)
        fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0))
        async_metrics = [
            ("prediction_hit", "Prediction recall (%)", 100),
            ("target_prepared_before_arrival", "Target ready at arrival (%)", 100),
            ("target_service_ttft_ms", "Target-service TTFT (ms)", 1),
            ("application_to_first_token_ms", "Application-to-first-token (ms)", 1),
        ]
        for ax, (metric, title, scale) in zip(axes.flat, async_metrics):
            vals = ah[metric] * scale
            bars = ax.bar([LABELS[str(p)] for p in ah.policy], vals,
                          color=[COLORS[str(p)] for p in ah.policy])
            ax.set(title=title, ylabel=title)
            ax.tick_params(axis="x", rotation=28)
            if scale == 100:
                ax.set_ylim(0, 103)
            ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
        savefig(fig, figures, "fig08_async_hotpot_ablation")

        ad = pd.read_csv(async_root / "paired_deltas.csv")
        ad.to_csv(tables / "async_paired_deltas.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
        for ax, metric, title in [
            (axes[0], "target_service_ttft_ms", "Target-service TTFT vs cosine"),
            (axes[1], "application_to_first_token_ms", "Application TTFT vs cosine"),
        ]:
            sub = ad[(ad.control == "cosine") & (ad.metric == metric)].copy()
            sub["order"] = sub.policy.map({p: i for i, p in enumerate(POLICY_ORDER_ASYNC)})
            sub = sub.sort_values("order")
            y = np.arange(len(sub))[::-1]
            points = sub.mean_delta.to_numpy()
            ax.errorbar(points, y,
                        xerr=[points - sub.ci_low.to_numpy(), sub.ci_high.to_numpy() - points],
                        fmt="o", capsize=4, color="#2F4B7C")
            ax.axvline(0, color="#333333", linewidth=1)
            ax.set_yticks(y, [LABELS[p] for p in sub.policy.astype(str)])
            ax.set(title=title, xlabel="Paired difference (ms; negative is faster)")
        savefig(fig, figures, "fig09_async_paired_latency_vs_cosine")

        # ---------- M/K sensitivity ----------
        dev_m = pd.read_csv(sensitivity_root / "development_m_selection.csv").sort_values("m")
        sens = pd.read_csv(sensitivity_root / "summary_by_m_k_policy.csv")
        sens.to_csv(tables / "mk_sensitivity_all.csv", index=False)
        offline_test = sens[(sens.split == "test") & (sens.query_type == "all")
                            & (sens.policy == "offline")]
        matrix = offline_test.pivot(index="m", columns="k", values="prediction_recall") * 100
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
        axes[0].plot(dev_m.m, dev_m.mean_dev_recall * 100, marker="o", linewidth=2.3, color="#4C78A8")
        axes[0].axvline(20, color="#777777", linestyle="--", label="Frozen systems M=20")
        axes[0].axvline(30, color="#D62728", linestyle=":", label="Development-selected M=30")
        axes[0].set(title="Development selection across M", xlabel="Top-M semantic neighbors",
                    ylabel="Mean development recall (%)", xticks=dev_m.m)
        axes[0].legend(fontsize=8.5)
        sns.heatmap(matrix, annot=True, fmt=".1f", cmap="YlGnBu", ax=axes[1],
                    cbar_kws={"label": "Test recall (%)"})
        axes[1].set(title="Offline adaptive held-out test recall", xlabel="Prefetch budget K",
                    ylabel="Top-M semantic neighbors")
        savefig(fig, figures, "fig10_mk_sensitivity")

        # Policy curves at development-selected M=30.
        chosen = sens[(sens.split == "test") & (sens.query_type == "all") & (sens.m == 30)
                      & sens.policy.isin(POLICY_ORDER_ASYNC[1:])]
        fig, ax = plt.subplots(figsize=(8.8, 5.0))
        for policy in POLICY_ORDER_ASYNC[1:]:
            g = chosen[chosen.policy.eq(policy)].sort_values("k")
            if g.empty:
                continue
            ax.plot(g.k, g.prediction_recall * 100, marker="o", linewidth=2,
                    label=LABELS[policy], color=COLORS[policy])
        ax.set(title="Held-out policy sensitivity at development-selected M=30",
               xlabel="Prefetch budget K", ylabel="Prediction recall (%)",
               xticks=sorted(chosen.k.unique()), ylim=(60, 101))
        ax.legend(ncol=3, fontsize=8.5)
        savefig(fig, figures, "fig11_selected_m_policy_recall")

        # ---------- Findings ----------
        dev20 = float(dev_m.loc[dev_m.m.eq(20), "mean_dev_recall"].iloc[0])
        dev30 = float(dev_m.loc[dev_m.m.eq(30), "mean_dev_recall"].iloc[0])
        q_off = cross[(cross.model_short == "Qwen-2.5-1.5B")
                      & (cross.policy == "adaptive_offline_global")].iloc[0]
        q_cos = cross[(cross.model_short == "Qwen-2.5-1.5B") & (cross.policy == "cosine")].iloc[0]
        l_off = cross[(cross.model_short == "Llama-3.2-1B")
                      & (cross.policy == "adaptive_offline_global")].iloc[0]
        l_cos = cross[(cross.model_short == "Llama-3.2-1B") & (cross.policy == "cosine")].iloc[0]
        findings = f"""# GraphKV new-results analysis

## Scope

This bundle covers only the new experiments supplied in `New_Suggestions_Results`:

- Llama-3.2-1B on HotpotQA, K = 6, 10, 16, two repetitions.
- Qwen2.5-1.5B and Llama-3.2-1B on 2Wiki, K = 10, two repetitions.
- Qwen2.5-1.5B HotpotQA asynchronous ablation at M = 20, K = 10, 750 ms lead.
- CPU-only HotpotQA M/K prediction sensitivity.

Older results already present in the paper are intentionally not duplicated.

## Main decisions

1. **Use the synchronous runs for cross-model and cross-dataset prediction/service claims.**
   On 2Wiki, offline adaptation improves recall over cosine from {q_cos.prediction_recall*100:.2f}% to {q_off.prediction_recall*100:.2f}% for Qwen and from {l_cos.prediction_recall*100:.2f}% to {l_off.prediction_recall*100:.2f}% for Llama.

2. **Keep asynchronous results in a separate systems subsection.**
   At 750 ms lead, adaptive policies improve prediction and target-service TTFT relative to cosine, but no-prefetch remains faster in application-observed latency.

3. **Do not retrofit M=30 into completed systems runs.**
   Development selection chose M=30 ({dev30*100:.2f}% mean recall), while frozen M=20 achieved {dev20*100:.2f}%, a difference of only {(dev30-dev20)*100:.2f} percentage points. Report a broad M=20–30 plateau and retain M=20 for the already completed systems experiments.

4. **Offline adaptation is the most reliable adaptive method.**
   Online learning remains competitive but does not consistently exceed the leakage-safe offline fit.

5. **Separate service benefit from total cost.**
   Synchronous prefetch improves ready-cache target service but population dominates total E2E latency. The asynchronous run tests whether lead time hides that cost and establishes the tested T4 boundary condition.

## Statistical wording

- Describe each synchronous run as **600 unique access events measured in two system repetitions (1,200 measurements)**.
- Use document-clustered bootstrap intervals for synchronous event comparisons.
- Use the provided equal-block bootstrap intervals for the asynchronous run and identify six 100-event blocks.
- The CPU sensitivity experiment measures prediction only; it does not measure vLLM latency, LMCache transfer, answer F1, or FLOPs.

## Integrity note

The authoritative synchronous runs completed every configured arm and passed their critical sanity and publication controls. The asynchronous bundle completed all 42 blocks and has a validated native-equivalence review. The original strict reuse gate remains a documented conservative failure; native-cache equivalence is the evidence used for the approved exact-prefix benchmark scope.
"""
        (output_dir / "KEY_FINDINGS.md").write_text(findings, encoding="utf-8")

        # Machine-readable figure index.
        index_rows = []
        descriptions = {
            "fig01_llama_hotpot_k_sweep": "Llama Hotpot prediction recall and ready-cache TTFT across K.",
            "fig02_llama_hotpot_latency_boundary": "Ready-cache versus synchronous E2E latency boundary.",
            "fig03_llama_hotpot_paired_recall": "Document-clustered paired recall gains over cosine.",
            "fig04_2wiki_cross_model": "Qwen/Llama 2Wiki headline comparison.",
            "fig05_2wiki_paired_deltas": "Document-clustered 2Wiki paired deltas versus cosine.",
            "fig06_2wiki_query_type_recall": "2Wiki recall by controlled query type.",
            "fig07_learned_offline_weights": "Leakage-safe learned semantic/structural weights.",
            "fig08_async_hotpot_ablation": "Full asynchronous component ablation.",
            "fig09_async_paired_latency_vs_cosine": "Asynchronous paired latency changes versus cosine.",
            "fig10_mk_sensitivity": "Development M selection and held-out offline recall heatmap.",
            "fig11_selected_m_policy_recall": "Held-out policy curves at development-selected M=30.",
        }
        for stem, description in descriptions.items():
            index_rows.append({"figure": stem, "description": description,
                               "formats": "png;pdf;svg"})
        pd.DataFrame(index_rows).to_csv(output_dir / "FIGURE_INDEX.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_zip", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    build(args.input_zip.resolve(), args.output_dir.resolve())
    shutil.make_archive(str(args.output_dir), "zip", root_dir=args.output_dir)
    print(f"Created: {args.output_dir}")
    print(f"Archive: {args.output_dir}.zip")


if __name__ == "__main__":
    main()
