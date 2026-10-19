"""The report's result tables and figures from completed runs (no training; data/runs/ is not changed).

    python -m experiments.results final_r2 tune_connectome_confirm tune_ppo_mlp_confirm tune_dqn_mlp_confirm
"""
from __future__ import annotations

import argparse
import csv
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np

from rl.evaluation import GREEDY, INFORMED_RULE, ODOR_RULE, RANDOM, SAMPLED, evaluate_rules
from rl.storage import RESULTS, load_runs, read_json, write_json

AGENT_NAMES = {"ppo_mlp": "PPO · MLP", "ppo_bio": "PPO · Bio connectome",
               "ppo_random_graph": "PPO · Random connectome", "dqn_mlp": "DQN · MLP"}
AGENT_ORDER = list(AGENT_NAMES)
# Validated categorical slots (light surface), fixed per agent so colors never depend on
# which agents are present. Line styles are the secondary encoding.
AGENT_STYLE = {"ppo_mlp": ("#2a78d6", "-"), "ppo_bio": ("#eb6834", "-"),
               "ppo_random_graph": ("#1baf7a", "--"), "dqn_mlp": ("#eda100", ":")}
EXTRA_STYLES = [("#e87ba4", "-."), ("#008300", (0, (5, 1))), ("#4a3aa7", (0, (1, 2))), ("#e34948", (0, (3, 1, 1, 1)))]
# Reference agents: key, label, line style, color (hues no agent condition uses).
REFERENCES = ((RANDOM, "Random agent", (0, (2, 2)), "#8b5a2b"), (ODOR_RULE, "Odor rule", (0, (6, 2)), "#0fa3c2"),
              (INFORMED_RULE, "Informed rule", (0, (1, 1)), "#8f8f00"))
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e5e4e0", "#ffffff"


# Loading

def condition_of(run: dict, manifest: dict) -> str:
    """Display name of the condition a run belongs to; seeds are pooled within it."""
    # A condition is an agent plus any --set overrides (the tuned configurations),
    # e.g. "PPO · MLP · ent_coef=0.002 · learning_rate=0.00034 · n_steps=512".
    name = AGENT_NAMES.get(run["agent"], run["agent"])
    parts = []
    for key, value in sorted((manifest.get("overrides") or {}).items()):
        parts.append("no readout norm" if (key, value) == ("readout_norm", False) else f"{key}={value}")
    return " · ".join([name, *parts])


def short_label(run: dict, manifest: dict) -> str:
    """Figure label: the agent name, plus "tuned" for tuned configurations (full values are in summary.md)."""
    name = AGENT_NAMES.get(run["agent"], run["agent"])
    overrides = dict(manifest.get("overrides") or {})
    if overrides.pop("readout_norm", True) is False:
        name += " · no readout norm"
    return f"{name} tuned" if overrides else name


def training_curve(directory: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per training episode: end timestep, fruit reached, poisoned, timed out (neither)."""
    steps, fruit, poison = [], [], []
    with (directory / "episodes.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            steps.append(int(row["timesteps"]))
            fruit.append(row["success"] == "True")
            poison.append(row.get("poisoned") == "True")
    fruit, poison = np.array(fruit, dtype=float), np.array(poison, dtype=float)
    return np.array(steps), fruit, poison, 1.0 - np.maximum(fruit, poison)


def steps_to_threshold(steps: np.ndarray, fruit: np.ndarray, threshold: float, window: int) -> int | None:
    """First timestep at which the success rate of the last ``window`` training episodes reaches threshold."""
    if len(fruit) < window:
        return None
    # Mean over each window of `window` consecutive training episodes; the first window
    # reaching the threshold ends at episode hits[0] + window - 1.
    rolling = np.convolve(fruit, np.ones(window) / window, mode="valid")
    hits = np.flatnonzero(rolling >= threshold)
    return int(steps[hits[0] + window - 1]) if len(hits) else None


def load(experiments: list[str], rules: bool) -> tuple[dict, list[dict], list[dict]]:
    """Conditions with their completed runs, reference results per seed, and skipped runs."""
    conditions: dict[str, dict] = {}
    skipped, environments = [], {}
    for run in load_runs():
        if run["experiment"] not in experiments:
            continue
        if run["status"] != "completed" or not run["evaluation"]:
            skipped.append(run)   # listed at the end of summary.md, never counted
            continue
        manifest = read_json(run["directory"] / "config.json")
        label = condition_of(run, manifest)
        condition = conditions.setdefault(label, {"label": label, "short": short_label(run, manifest),
                                                  "agent": run["agent"], "runs": []})
        condition["runs"].append({**run, "environment": manifest["environment"],
                                  "budget": manifest["training"]["total_timesteps"]})
        # One arena per seed: the reference agents are re-played in it (same test episodes).
        environments.setdefault(run["seed"], (manifest["environment"], run["evaluation"][RANDOM]["episodes"]))
    check_environments(conditions)
    references = reference_results(conditions, environments, rules)
    return conditions, references, skipped


def check_environments(conditions: dict) -> None:
    """Stop when runs were trained in different arenas (seed-tied fields aside).

    Seeds index the reference agents and the paired tests, so mixing arenas would
    silently compare different tasks. Analyse each arena's experiments separately."""
    seen = {}
    for condition in conditions.values():
        for run in condition["runs"]:
            arena = {k: v for k, v in run["environment"].items() if k not in ("seed", "odor_profile_seed")}
            seen.setdefault(tuple(sorted(arena.items())), []).append(condition["label"])
    if len(seen) > 1:
        base = dict(next(iter(seen)))
        lines = ["These runs use different arenas; analyse each arena's experiments separately:"]
        for arena, labels in seen.items():
            differs = {k: v for k, v in dict(arena).items() if base.get(k) != v} or "(first arena)"
            lines.append(f"  {sorted(set(labels))}: {differs}")
        raise SystemExit("\n".join(lines))


def reference_results(conditions: dict, environments: dict, rules: bool) -> list[dict]:
    """Per seed: the random agent (from the runs) and the rules (re-played on the same episodes)."""
    randoms = defaultdict(list)
    for condition in conditions.values():
        for run in condition["runs"]:
            randoms[run["seed"]].append(run["evaluation"][RANDOM])
    rows = []
    for seed, (environment, episodes) in sorted(environments.items()):
        row = {"seed": seed, RANDOM: randoms[seed][0]}   # identical across conditions: same episodes
        # The rules are not stored in any run, so they are played now on the same test episodes.
        if rules:
            row.update(evaluate_rules(environment, seed, episodes))
        rows.append(row)
    return rows


# Statistics

def mean_sd(values) -> tuple[float, float]:
    # Mean and sample standard deviation over seeds; NaN where undefined (shown as "–").
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else float("nan")


def permutation_p(a: list[float], b: list[float]) -> float:
    """Exact two-sided permutation test on the difference of means (unpaired)."""
    # Every way of splitting the pooled values into groups of the original sizes;
    # p = share of splits whose mean difference is at least the observed one.
    pooled, n = np.array(a + b), len(a)
    observed = abs(np.mean(a) - np.mean(b))
    diffs = [abs(pooled[list(idx)].mean() - np.delete(pooled, list(idx)).mean())
             for idx in itertools.combinations(range(len(pooled)), n)]
    return float(np.mean(np.array(diffs) >= observed - 1e-12))


def sign_flip_p(a: dict, b: dict) -> tuple[float, int] | None:
    """Exact paired test over shared seeds (same odor pair and test episodes): sign-flip of differences."""
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return None
    # Under "no difference" each per-seed difference is equally likely to have either sign,
    # so compare the observed mean difference with all 2^n sign patterns.
    d = np.array([a[s] - b[s] for s in seeds])
    observed = abs(d.mean())
    flips = [abs((d * signs).mean()) for signs in itertools.product((1, -1), repeat=len(d))]
    return float(np.mean(np.array(flips) >= observed - 1e-12)), len(seeds)


def summarize(conditions: dict, references: list[dict], threshold: float, window: int) -> dict:
    # Per condition: held-out fruit, poison and timeout rates and steps to the fruit, for
    # sampled and greedy actions (mean ± sd over seeds); and learning speed, the training
    # steps until the rolling training success first reaches `threshold`.
    summary = {"threshold": threshold, "rolling_window": window, "conditions": [], "references": [], "tests": []}
    for condition in ordered(conditions):
        runs = sorted(condition["runs"], key=lambda r: r["seed"])
        row = {"label": condition["label"], "agent": condition["agent"], "seeds": [r["seed"] for r in runs],
               "budget": sorted({r["budget"] for r in runs})}
        for mode in (SAMPLED, GREEDY):
            evals = [r["evaluation"][mode] for r in runs]
            for metric in ("success_rate", "poison_rate", "timeout_rate", "successful_mean_steps"):
                row[f"{mode}.{metric}"] = mean_sd(e.get(metric) for e in evals)
        reached = []
        for run in runs:
            steps, fruit = training_curve(run["directory"])[:2]
            reached.append(steps_to_threshold(steps, fruit, threshold, window))
        row["steps_to_threshold"] = reached
        row["per_seed_sampled"] = {r["seed"]: r["evaluation"][SAMPLED]["success_rate"] for r in runs}
        summary["conditions"].append(row)
    # The same rates for the reference agents (random agent and both rules).
    for key, label, _, _ in REFERENCES:
        evals = [ref[key] for ref in references if key in ref]
        if evals:
            summary["references"].append({"label": label, "key": key, "seeds": len(evals), **{
                metric: mean_sd(e.get(metric) for e in evals)
                for metric in ("success_rate", "poison_rate", "timeout_rate", "successful_mean_steps")}})
    # Pairwise tests between conditions on the per-seed sampled fruit rates.
    rows = [row for row in summary["conditions"] if len(row["seeds"]) > 1]   # a test needs a spread
    for a, b in itertools.combinations(rows, 2):
        x, y = a["per_seed_sampled"], b["per_seed_sampled"]
        paired = sign_flip_p(x, y)
        summary["tests"].append({"a": a["label"], "b": b["label"],
                                 "difference": float(np.mean(list(x.values())) - np.mean(list(y.values()))),
                                 "unpaired_p": permutation_p(list(x.values()), list(y.values())),
                                 "paired_p": paired[0] if paired else None, "paired_seeds": paired[1] if paired else 0})
    return summary


def ordered(conditions: dict) -> list[dict]:
    # Report order: by agent (MLP, Bio, Random, DQN), the default before its tuned variant.
    rank = lambda c: (AGENT_ORDER.index(c["agent"]) if c["agent"] in AGENT_ORDER else len(AGENT_ORDER), c["label"])
    return sorted(conditions.values(), key=rank)


# Tables

def pct(value: tuple[float, float]) -> str:
    mean, sd = value
    if np.isnan(mean):
        return "–"
    return f"{mean:.0%}" if np.isnan(sd) else f"{mean:.0%} ± {sd:.0%}"


def num(value: tuple[float, float]) -> str:
    mean, sd = value
    if np.isnan(mean):
        return "–"
    return f"{mean:.0f}" if np.isnan(sd) else f"{mean:.0f} ± {sd:.0f}"


def speed(reached: list) -> str:
    hits = [r for r in reached if r is not None]
    if not hits:
        return f"never (0/{len(reached)})"
    return f"{np.median(hits) / 1000:.0f}k ({len(hits)}/{len(reached)})"



# Summarize all results in a markdown file
def markdown(summary: dict, skipped: list[dict], experiments: list[str]) -> str:
    t, w = summary["threshold"], summary["rolling_window"]
    lines = [f"# Results: {', '.join(experiments)}", "",
             "Held-out: 100 fresh test episodes per seed, never seen in training. Mean ± standard deviation "
             "over seeds. Sampled = PPO sampling (DQN: epsilon-greedy); greedy = most-likely action.", "",
             "| Condition | Seeds | Fruit (sampled) | Poison (sampled) | Timeout (sampled) | Fruit (greedy) "
             f"| Poison (greedy) | Steps to fruit (sampled) | Steps to {t:.0%} training success* |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in summary["conditions"]:
        lines.append(f"| {row['label']} | {len(row['seeds'])} | {pct(row['ppo_stochastic.success_rate'])} "
                     f"| {pct(row['ppo_stochastic.poison_rate'])} | {pct(row['ppo_stochastic.timeout_rate'])} "
                     f"| {pct(row['ppo.success_rate'])} | {pct(row['ppo.poison_rate'])} "
                     f"| {num(row['ppo_stochastic.successful_mean_steps'])} | {speed(row['steps_to_threshold'])} |")
    for ref in summary["references"]:
        lines.append(f"| *{ref['label']}* (reference) | {ref['seeds']} | {pct(ref['success_rate'])} "
                     f"| {pct(ref['poison_rate'])} | {pct(ref['timeout_rate'])} | – | – "
                     f"| {num(ref['successful_mean_steps'])} | – |")
    lines += ["", f"\\* Median training steps until the success rate over the last {w} training episodes first "
              f"reached {t:.0%}, over the seeds that reached it (count in brackets).", "",
              "## Comparisons (sampled fruit rate per seed)", "",
              "| A | B | A − B | Unpaired permutation p | Paired sign-flip p (seeds) |", "|---|---|---|---|---|"]
    for test in summary["tests"]:
        paired = f"{test['paired_p']:.3f} ({test['paired_seeds']})" if test["paired_p"] is not None else "–"
        lines.append(f"| {test['a']} | {test['b']} | {test['difference']:+.0%} | {test['unpaired_p']:.3f} | {paired} |")
    lines += ["", "Exact tests over all permutations. The paired test matches seeds (same odor pair and test "
              "episodes); with n seeds its smallest possible p is 2/2^n (0.0625 for 5 seeds), so read effect "
              "sizes and per-seed values, not only p.", "", "## Per-seed sampled fruit rate", ""]
    seeds = sorted({s for row in summary["conditions"] for s in row["per_seed_sampled"]})
    lines += ["| Condition | " + " | ".join(str(s) for s in seeds) + " |", "|---|" + "---|" * len(seeds)]
    for row in summary["conditions"]:
        cells = [f"{row['per_seed_sampled'][s]:.0%}" if s in row["per_seed_sampled"] else "–" for s in seeds]
        lines.append(f"| {row['label']} | " + " | ".join(cells) + " |")
    if skipped:
        lines += ["", "## Not included (not completed)", ""]
        lines += [f"- {r['id'][:8]} {r['agent']} seed {r['seed']}: {r['status']} "
                  f"({r['timesteps']:,}/{r['budget'] or 0:,} steps)" for r in skipped]
    return "\n".join(lines) + "\n"


# Figures

def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=9, length=3)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def styles(conditions: list[dict]) -> dict[str, tuple]:
    """Agent conditions keep their fixed style; variants (ablations) take the extra slots in order."""
    result, extras = {}, iter(EXTRA_STYLES)
    for condition in conditions:
        is_base = condition["label"] == AGENT_NAMES.get(condition["agent"])
        result[condition["label"]] = AGENT_STYLE[condition["agent"]] if is_base and condition["agent"] in AGENT_STYLE \
            else next(extras, ("#52514e", "-"))
    return result


def spread_labels(ends: list[tuple[float, float]], gap: float) -> list[tuple[float, float, float]]:
    """(value, label y, x) with label positions pushed apart by at least ``gap`` so end labels never overlap."""
    placed = []
    for value, x in sorted(ends):
        y = value if not placed or value - placed[-1][1] >= gap else placed[-1][1] + gap
        placed.append((value, y, x))
    return placed


def binned_rate(steps, values, edges) -> np.ndarray:
    index = np.digitize(steps, edges) - 1
    return np.array([values[index == i].mean() if np.any(index == i) else np.nan for i in range(len(edges) - 1)])


def learning_curves(conditions: list[dict], summary: dict, path: Path, bin_steps: int,
                    panels: list[tuple[str, int, str]] | None = None) -> None:
    """Training rates per panel: (title, training_curve column, held-out reference key)."""
    import matplotlib.pyplot as plt

    if panels is None:
        has_poison = any(r["environment"].get("poison_enabled") for c in conditions for r in c["runs"])
        panels = [("Fruit reached", 1, "success_rate")] + ([("Poison eaten", 2, "poison_rate")] if has_poison else [])
    # At least two panels wide, so the shared legend below always fits.
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * max(len(panels), 2), 4.6), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    look = styles(conditions)
    budget = max(r["budget"] for c in conditions for r in c["runs"])
    edges = np.arange(0, budget + bin_steps, bin_steps)
    centres = (edges[:-1] + edges[1:]) / 2 / 1000
    for ax, (title, column, key) in zip(axes[0], panels):
        style_axes(ax)
        ends, highest = [], 0.0
        for condition in conditions:
            color, dash = look[condition["label"]]
            curves = []
            for run in condition["runs"]:
                data = training_curve(run["directory"])
                curves.append(binned_rate(data[0], data[column], edges))
            curves = np.array(curves)
            mean = np.nanmean(curves, axis=0)
            ax.fill_between(centres, np.nanmin(curves, axis=0), np.nanmax(curves, axis=0),
                            color=color, alpha=0.12, linewidth=0)
            highest = max(highest, float(np.nanmax(curves)))
            ax.plot(centres, mean, color=color, linestyle=dash, linewidth=1.8, label=condition["short"])
            last = np.flatnonzero(~np.isnan(mean))
            if len(last):
                ends.append((mean[last[-1]], centres[last[-1]]))
        lines ={rkey: (dash, color) for rkey, _, dash, color in REFERENCES}
        for ref in summary["references"]:
            level = ref[key][0]
            if not np.isnan(level):
                dash, color = lines[ref["key"]]
                ax.axhline(level, color=color, linestyle=dash, linewidth=1.2,
                           label=f"{ref['label']} (held-out)")
                highest = max(highest, level)
        # Poison rates are small: fit that axis to the data instead of 0-100%.
        top = min(1.0, max(0.2, 1.15 * highest)) if key == "poison_rate" else 1.0
        for value, label_y, x in spread_labels(ends, gap=0.045 * top):
            ax.annotate(f"{value:.0%}", (x, label_y), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK_2)
        ax.set_ylim(-0.02 * top, 1.05 * top)
        ax.set_xlim(0, budget / 1000 * 1.06)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_xlabel("Training steps (thousands)", color=INK, fontsize=9)
        ax.set_title(f"{title} during training", color=INK, fontsize=10.5, loc="left")
    axes[0][0].set_ylabel(f"Share of training episodes ({bin_steps // 1000}k-step bins)", color=INK, fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.5, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def heldout_figure(conditions: list[dict], references: list[dict], summary: dict, path: Path) -> None:
    """Per-seed held-out rates (dots) and their mean (tick), sampled actions, one row per agent."""
    import matplotlib.pyplot as plt

    has_poison = any(r["environment"].get("poison_enabled") for c in conditions for r in c["runs"])
    metrics = [("success_rate", "Fruit reached")] + ([("poison_rate", "Poison eaten")] if has_poison else [])
    look = styles(conditions)
    rows = [(c["short"], [r["evaluation"][SAMPLED] for r in sorted(c["runs"], key=lambda r: r["seed"])],
             look[c["label"]][0]) for c in conditions]
    rows += [(label, [ref[key] for ref in references if key in ref], color)
             for key, label, _, color in REFERENCES if any(key in ref for ref in references)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics) + 2.2, 0.45 * len(rows) + 1.3),
                             sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, (metric, title) in zip(axes[0], metrics):
        style_axes(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        for y, (label, evals, color) in enumerate(rows):
            values = np.array([e[metric] for e in evals if e.get(metric) is not None])
            if not len(values):
                continue
            jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.zeros(1)
            ax.plot([values.mean()] * 2, [y - 0.22, y + 0.22], color=INK, linewidth=1.2, zorder=2)
            ax.scatter(values, y + jitter, s=34, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3)
            ax.annotate(f"{values.mean():.0%}", (values.mean(), y - 0.22), xytext=(0, 1), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, color=INK)   # y axis is inverted: above the row
        ax.set_xlim(-0.03, 1.03)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title(f"{title} (held-out)", color=INK, fontsize=10.5, loc="left")
    axes[0][0].set_yticks(range(len(rows)), [label for label, _, _ in rows], fontsize=9, color=INK)
    axes[0][0].invert_yaxis()
    fig.text(0.01, 0.01, "Dots: one seed (100 test episodes each); black tick and label: mean over seeds. "
             "Trained agents use sampled actions (DQN: epsilon-greedy).", fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# Main

def jsonable(value):
    """NaN (an undefined mean or spread) becomes null; tuples become lists."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="*", default=["final_r2"], help="Experiment labels to include")
    parser.add_argument("--threshold", type=float, default=0.8, help="Training success for the learning-speed metric")
    parser.add_argument("--rolling", type=int, default=50, help="Training episodes in the rolling success window")
    parser.add_argument("--bin-steps", type=int, default=25_000, help="Training steps per learning-curve bin")
    parser.add_argument("--no-rules", action="store_true", help="Skip re-playing the hand-coded rules")
    parser.add_argument("--out", type=Path, help="Output directory (default data/results/<experiments>)")
    args = parser.parse_args()

    # Writes summary.md, summary.json, learning_curves.png, timeouts.png and heldout.png to
    # data/results/<experiment labels joined by "+">/.
    conditions, references, skipped = load(args.experiments, rules=not args.no_rules)
    if not conditions:
        raise SystemExit(f"No completed runs in {args.experiments}. See: python -m experiments.runs")
    out = args.out or RESULTS / "+".join(args.experiments)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(conditions, references, args.threshold, args.rolling)
    text = markdown(summary, skipped, args.experiments)
    (out / "summary.md").write_text(text, encoding="utf-8")
    write_json(out / "summary.json", jsonable({**summary, "experiments": args.experiments,
                                               "references_per_seed": references}))
    ordered_conditions = ordered(conditions)
    learning_curves(ordered_conditions, summary, out / "learning_curves.png", args.bin_steps)
    learning_curves(ordered_conditions, summary, out / "timeouts.png", args.bin_steps,
                    panels=[("Timed out", 3, "timeout_rate")])
    heldout_figure(ordered_conditions, references, summary, out / "heldout.png")
    print(text)
    print(f"Written to {out}: summary.md, summary.json, learning_curves.png, timeouts.png, heldout.png")


if __name__ == "__main__":
    main()
