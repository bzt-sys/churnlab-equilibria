
import os
from collections import Counter
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import pandas as pd
import json
import csv
import matplotlib.gridspec as gridspec
from config import *
from utils.tree_arrays import *
from utils.game_tree import *
from scipy.ndimage import uniform_filter1d
from datetime import datetime
from matplotlib.ticker import FuncFormatter, MaxNLocator
from viz.exporters import export_demo_bundle_csv
from viz.viz_prep import *

# ---- Slide-ready defaults ----

def _save_fig(fig, path_base: str, facecolor="white"):
    fig.set_size_inches(*SLIDE_FIGSIZE)
    fig.savefig(path_base + ".png", dpi=SLIDE_DPI, bbox_inches="tight", pad_inches=PAD_INCHES, facecolor=facecolor)
    #fig.savefig(path_base + ".svg", bbox_inches="tight", pad_inches=PAD_INCHES, facecolor=facecolor)

def _add_footer(ax, text= f" Seed #{SEED} github.com/divinecomedy-labs/ChurnLab"):
    # optional: call before saving if you want a footer link
    ax.figure.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=8, color="#000000", alpha=0.25)

# --- Formatting helpers for pitch decks ---
def _add_dual_line(ax, series, label, color, window=10, linestyle="-"):
    """
    Plots both the raw jagged line and a smoothed version for readability.
    """
    series = np.asarray(series, dtype=float)
    ax.plot(series, label=label + " (raw)", color=color, alpha=0.4, linewidth=1.2, linestyle=linestyle)

    if len(series) > window:
        smooth = uniform_filter1d(series, size=window, mode="nearest")
        ax.plot(smooth, label=label + " (smoothed)", color=color, linewidth=2.2, linestyle=linestyle)


def _style_axes(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=14)


def to_cpu_array(x):
    """Ensure array is on CPU (NumPy) before plotting."""
    try:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except ImportError:
        pass

    import numpy as np
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)):
        return np.array(x)
    return np.array([x])  # fallback wrap scalar


def _abbr_currency_formatter():
    # $K / $M / $B tick formatter (no scientific notation)
    def _fmt(x, pos):
        v = abs(x)
        if v >= 1e9:
            val, suf = x/1e9, "B"
        elif v >= 1e6:
            val, suf = x/1e6, "M"
        elif v >= 1e3:
            val, suf = x/1e3, "K"
        else:
            val, suf = x, ""
        if suf and abs(val) < 10:
            return f"${val:,.1f}{suf}"
        return f"${val:,.0f}{suf}"
    return FuncFormatter(_fmt)

def _format_currency_axis(ax):
    ax.yaxis.set_major_formatter(_abbr_currency_formatter())
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

def _format_percent_axis(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y*100:.0f}%"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

def _annotate_line_end(ax, series, label, color="#222", offset=(12,0.02)):
    # Adds an unobtrusive label near the last point of a line
    try:
        y = float(series[-1])
        x = len(series) - 1
        ylim = ax.get_ylim()
        y_off = y + (ylim[1]-ylim[0]) * offset[1]
        if abs(y) >= 1000:
            text_val = f"{y:,.0f}"
        else:
            text_val = f"{y:,.2f}"
        ax.annotate(f"{label}: {text_val}",
                    xy=(x, y), xytext=(x+offset[0], y_off),
                    textcoords='data', fontsize=10, color=color,
                    arrowprops=dict(arrowstyle='-', lw=0.6, color=color, alpha=0.5))
    except Exception:
        pass

def _safe_div(a, b):
    """Safe division helper that returns None if denominator is zero or invalid."""
    try:
        b = float(b)
        if abs(b) < 1e-9:
            return None
        return float(a) / b
    except Exception:
        return None


def plot_content_event_severity(content_real, content_base, save_path=None):
        """
        Comparative scatterplot of content event severity for Virgil vs Baseline.
        Virgil = circles, Baseline = triangles.
        Positive events (virality) green, negative (backlash) red.
        """
        att_v, nov_v, resp_v = content_real["attention"], content_real["novelty"], content_real["response"]
        att_b, nov_b, resp_b = content_base["attention"], content_base["novelty"], content_base["response"]

        batches = list(range(max(len(att_v), len(att_b))))

        def compute_points(att, nov, resp):
            signed = np.array(att) + np.array(nov) + np.array(resp)
            severity = np.abs(np.array(att)) + np.abs(np.array(nov)) + np.abs(np.array(resp))
            colors = ["green" if val >= 0 else "red" for val in signed]
            return severity, colors

        sev_v, colors_v = compute_points(att_v, nov_v, resp_v)
        sev_b, colors_b = compute_points(att_b, nov_b, resp_b)

        plt.figure(figsize=(10, 4))
        plt.scatter(range(len(sev_v)), sev_v, c=colors_v, s=50+sev_v*200, alpha=0.6,
                    edgecolor="k", marker="o", label="Virgil")
        plt.scatter(range(len(sev_b)), sev_b, c=colors_b, s=50+sev_b*200, alpha=0.6,
                    edgecolor="k", marker="^", label="Baseline")

        plt.title("Content Event Severity Over Time")
        plt.xlabel("Batch")
        plt.ylabel("Severity (|attention|+|novelty|+|response|)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

def plot_population_size(challenger_metrics, baseline_metrics, save_path=None):
    """
    Comparative line chart of alive population size (and churned) for Virgil vs Baseline.
    """
    alive_v = challenger_metrics.get("alive_users", [])
    alive_b = baseline_metrics.get("alive_users", [])
    churned_v = challenger_metrics.get("churned_users", [])
    churned_b = baseline_metrics.get("churned_users", [])

    batches = range(max(len(alive_v), len(alive_b)))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))

    # Alive users
    plt.plot(alive_v, label="Virgil Alive", color="blue", linewidth=2)
    plt.plot(alive_b, label="Baseline Alive", color="orange", linewidth=2, linestyle="--")

    # Churned users (secondary lines)
    if churned_v:
        plt.plot(churned_v, label="Virgil Churned", color="blue", linestyle=":")
    if churned_b:
        plt.plot(churned_b, label="Baseline Churned", color="orange", linestyle=":")

    plt.title("Population Size Over Time")
    plt.xlabel("Batch")
    plt.ylabel("Users")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

def get_archetype_counts(user_states):
    counts = {}
    for st in user_states.values():
        a = st["archetype"]
        counts[a] = counts.get(a, 0) + 1
    return counts



def plot_archetype_distributions(challenger_init, challenger_final,
                                 baseline_init, baseline_final,
                                 save_path=None):
    """
    Compare archetype distributions at initialization vs end of sim.
    Side-by-side bar charts for Virgil vs Baseline.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Collect archetypes
    archetypes = sorted(set(list(challenger_init.keys())
                            + list(challenger_final.keys())
                            + list(baseline_init.keys())
                            + list(baseline_final.keys())))

    x = np.arange(len(archetypes))  # positions

    # Extract counts (0 if not present)
    c_init = [challenger_init.get(a, 0) for a in archetypes]
    c_final = [challenger_final.get(a, 0) for a in archetypes]
    b_init = [baseline_init.get(a, 0) for a in archetypes]
    b_final = [baseline_final.get(a, 0) for a in archetypes]

    width = 0.18

    plt.figure(figsize=(14, 6))

    # Virgil
    plt.bar(x - width*1.5, c_init, width, label="Virgil Init", color="blue", alpha=0.6)
    plt.bar(x - width*0.5, c_final, width, label="Virgil Final", color="blue", alpha=0.9)

    # Baseline
    plt.bar(x + width*0.5, b_init, width, label="Baseline Init", color="orange", alpha=0.6)
    plt.bar(x + width*1.5, b_final, width, label="Baseline Final", color="orange", alpha=0.9)

    plt.xticks(x, archetypes, rotation=30, ha="right")
    plt.ylabel("Users")
    plt.title("Archetype Distribution: Initial vs Final (Virgil vs Baseline)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_value_tier_distributions(challenger_init, challenger_final,
                                  baseline_init, baseline_final,
                                  save_path=None):
    """
    Compare value tier distributions at initialization vs end of sim.
    Side-by-side bar charts for Virgil vs Baseline.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    tiers = sorted(set(list(challenger_init.keys())
                       + list(challenger_final.keys())
                       + list(baseline_init.keys())
                       + list(baseline_final.keys())))

    x = np.arange(len(tiers))

    c_init = [challenger_init.get(t, 0) for t in tiers]
    c_final = [challenger_final.get(t, 0) for t in tiers]
    b_init = [baseline_init.get(t, 0) for t in tiers]
    b_final = [baseline_final.get(t, 0) for t in tiers]

    width = 0.18

    plt.figure(figsize=(10, 5))

    # Virgil
    plt.bar(x - width*1.5, c_init, width, label="Virgil Init", color="blue", alpha=0.6)
    plt.bar(x - width*0.5, c_final, width, label="Virgil Final", color="blue", alpha=0.9)

    # Baseline
    plt.bar(x + width*0.5, b_init, width, label="Baseline Init", color="orange", alpha=0.6)
    plt.bar(x + width*1.5, b_final, width, label="Baseline Final", color="orange", alpha=0.9)

    plt.xticks(x, tiers, rotation=0)
    plt.ylabel("Users")
    plt.title("Value Tier Distribution: Initial vs Final (Virgil vs Baseline)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

def plot_growth(challenger_metrics, baseline_metrics, initial_users, save_path=None):
    """
    Plot active user population growth over time for Virgil vs Baseline,
    compared against a 17% annual growth reference line.
    """
    alive_v = challenger_metrics.get("alive_users", [])
    alive_b = baseline_metrics.get("alive_users", [])

    if not alive_v and not alive_b:
        return  # nothing to plot

    # Convert batch index to days
    batches = max(len(alive_v), len(alive_b))
    days = np.arange(batches) / BATCHES_PER_DAY

    # Reference: 17% annual growth from initial user base
    target = initial_users * (1.17 ** (days / 365.0))

    fig, ax = plt.subplots(figsize=(10, 6))
    if alive_v:
        ax.plot(days, alive_v, label="Virgil (Population)", color="royalblue")
    if alive_b:
        ax.plot(days, alive_b, label="Baseline (Population)", color="darkorange", linestyle="--")
    ax.plot(days, target, linestyle=":", color="gray", label="17% Annual Growth Target")

    _style_axes(ax, "Population Growth Over Time", "Days", "Active Users")
    ax.legend()

    if save_path:
        _save_fig(fig, save_path)
    else:
        plt.show()
    plt.close(fig)



def generate_pitch_charts(challenger_metrics, baseline_metrics, challenger_strat_counts,
    base_strat_counts, challenger_state_counts, base_state_counts, save=True, prefix="chart", dashboard=True, seed=None):
    """
    Generate all charts (Energy, Net Revenue, Health/Engagement, Churn, Cumulative Rev,
    and Content Events). Takes the metrics dicts from challenger & baseline.
    """
    os.makedirs("output", exist_ok=True)

    # --- unpack metrics ---
    real_energy = challenger_metrics.get("energy", [])
    base_energy = baseline_metrics.get("energy", [])
    net_revenue_real = challenger_metrics.get("net_revenue", [])
    net_revenue_base = baseline_metrics.get("net_revenue", [])
    real_churn = challenger_metrics.get("rolling_churn", [])
    base_churn = baseline_metrics.get("rolling_churn", [])
    mean_health_real = challenger_metrics.get("mean_health", None)
    mean_health_base = baseline_metrics.get("mean_health", None)
    mean_engage_real = challenger_metrics.get("mean_engagement", None)
    mean_engage_base = baseline_metrics.get("mean_engagement", None)
    churned_users = challenger_metrics.get("churned_users", [])
    base_churn_users = baseline_metrics.get("churned_users", [])
    annotations = challenger_metrics.get("annotations", None)
    challenger_strat_counts = challenger_strat_counts
    base_strat_counts = base_strat_counts
    # for content events
    content_real = dict(
        attention=challenger_metrics.get("content_attention", []),
        novelty=challenger_metrics.get("content_novelty", []),
        response=challenger_metrics.get("content_response", []),
    )
    content_base = dict(
        attention=baseline_metrics.get("content_attention", []),
        novelty=baseline_metrics.get("content_novelty", []),
        response=baseline_metrics.get("content_response", []),
    )

    # ... existing chart logic uses these vars as before ...


    """
    Generate *all* individual metric charts (Energy, ARR, Churn, Cumulative ARR)
    and a 2x2 dashboard, every time this function is called.
    Function signature, filenames, and data exports are preserved.
    """
    SEED = seed
    os.makedirs("output", exist_ok=True)
    os.makedirs(f"output/seed_{SEED}", exist_ok=True)
   
    # --- Energy ---
    # --- Energy (Infrastructure Usage) ---
    fig1, ax1 = plt.subplots(constrained_layout=True)
    _add_dual_line(ax1, real_energy, "Virgil Energy (kWh)", color="royalblue")
    _add_dual_line(ax1, base_energy, "Baseline Energy (kWh)", color="darkorange", linestyle="--")
    _style_axes(ax1, f"Infrastructure Energy Usage per Batch, seed: {SEED}", "Batch", "kWh")
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))  # cleaner ticks
    ax1.legend()
    base = os.path.join(f"output/seed_{SEED}", f"{prefix}_energy_seed{SEED}_{timestamp}")
    _save_fig(fig1, base)
    plt.close(fig1)


    # --- ARR Retention ---
    # --- Net Revenue per Batch ---
    fig2, ax2 = plt.subplots(constrained_layout=True)
    _add_dual_line(ax2, net_revenue_real, label="Virgil", color='royalblue')
    _add_dual_line(ax2, net_revenue_base, label="Baseline", linestyle="--", color='orange')
    _style_axes(ax2, f"Net Revenue per Batch, seed: {SEED}", "Batch", "Revenue ($)")
    _format_currency_axis(ax2)
    ax2.legend()
    base = os.path.join(f"output/seed_{SEED}", f"{prefix}_netrev_seed{SEED}_{timestamp}")
    _save_fig(fig2, base)
    plt.close(fig2)


    # --- Health & Engagement (Population Quality) ---
    if mean_health_real is not None and mean_engage_real is not None:
        figH, axH = plt.subplots(constrained_layout=True)
        _add_dual_line(axH, mean_health_real, label="Virgil Health", color='royalblue')
        _add_dual_line(axH, mean_engage_real, label="Virgil Engagement", color='navy', linestyle=":")
        if mean_health_base is not None:
            _add_dual_line(axH, mean_health_base, label="Baseline Health", color='orange')
        if mean_engage_base is not None:
            _add_dual_line(axH, mean_engage_base, label="Baseline Engagement", color='darkred', linestyle=":")
        _style_axes(axH, f"Population Health & Engagement Over Time, seed: {SEED}", "Batch", "Normalized Value")
        axH.legend()
        base = os.path.join(f"output/seed_{SEED}", f"{prefix}_health_engage_seed{SEED}_{timestamp}")
        _save_fig(figH, base)
        plt.close(figH)

    # --- Rolling Churn Rate ---
    fig3, ax3 = plt.subplots(constrained_layout=True)
    _add_dual_line(ax3, real_churn, "Virgil Rolling Churn", color="royalblue")
    _add_dual_line(ax3, base_churn, "Baseline Rolling Churn", color="darkorange", linestyle="--")
    _style_axes(ax3, f"Rolling Churn Rate, seed: {SEED}", "Batch", "Churn (%)")
    _format_percent_axis(ax3)

    try:
        lines = ax3.get_lines()
        if len(lines) >= 2:
            _annotate_line_end(ax3, lines[0].get_ydata(), "Virgil", lines[0].get_color())
            _annotate_line_end(ax3, lines[1].get_ydata(), "Baseline", lines[1].get_color())
    except Exception:
        pass
    ax3.legend()
    base = os.path.join(f"output/seed_{SEED}", f"{prefix}_rollingchurn_seed{SEED}_{timestamp}")
    _save_fig(fig3, base)
    plt.close(fig3)

    # --- Cumulative Revenue (Profit after Costs) ---
    netrev_real_np = np.asarray(net_revenue_real, dtype=float)
    netrev_base_np = np.asarray(net_revenue_base, dtype=float)
    cum_real = np.cumsum(netrev_real_np)
    cum_base = np.cumsum(netrev_base_np)

    fig4, ax4 = plt.subplots(constrained_layout=True)
    ax4.plot(cum_real, label="Virgil (Cumulative Revenue)", linewidth=2.2)
    ax4.plot(cum_base, label="Baseline (Cumulative Revenue)", linestyle="--", linewidth=2.2)
    _style_axes(ax4, f"Cumulative Revenue (Simulated), seed: {SEED}", "Batch", "Cumulative Revenue ($)")
    _format_currency_axis(ax4)

    # Annotate uplift and end-of-line values
    try:
        uplift = float(cum_real[-1] - cum_base[-1])
        ratio = (float(cum_real[-1]) / max(1e-9, float(cum_base[-1]))) if float(cum_base[-1]) != 0 else None
        ylim = ax4.get_ylim()
        ax4.annotate((f"Revenue Difference = ${uplift:,.0f}" + (f"  ({ratio:.2f}×)" if ratio else "")),
                    xy=(len(cum_real)*0.65, ylim[0] + 0.1*(ylim[1]-ylim[0])),
                    fontsize=12, color="#444")
        _annotate_line_end(ax4, cum_real, "Virgil", color=ax4.get_lines()[0].get_color())
        _annotate_line_end(ax4, cum_base, "Baseline", color=ax4.get_lines()[1].get_color())
    except Exception:
        pass

    ax4.legend()
    base = os.path.join(f"output/seed_{SEED}", f"{prefix}_cumrev_seed{SEED}_{timestamp}")
    _save_fig(fig4, base)
    plt.close(fig4)


    ts_path = os.path.join(f"output/seed_{SEED}", f"{prefix}_timeseries_seed{SEED}_{timestamp}.csv")
    headers = [
        "batch",
        "virgil_energy", "baseline_energy",
        "netrev_virgil", "netrev_baseline",
        "rolling_churn_virgil", "rolling_churn_baseline",
        "cumrev_virgil", "cumrev_baseline",
        "health_virgil", "health_baseline",
        "engage_virgil", "engage_baseline"
    ]
    N = max(
        len(real_energy), len(base_energy),
        len(net_revenue_real), len(net_revenue_base),
        len(real_churn), len(base_churn),
        len(mean_health_real or []), len(mean_health_base or []),
        len(mean_engage_real or []), len(mean_engage_base or [])
    )
    with open(ts_path, "w", newline="") as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(headers)
        for i in range(N):
            def _get(seq, idx):
                try: return seq[idx]
                except Exception: return ""
            w.writerow([
                i,
                _get(real_energy, i), _get(base_energy, i),
                _get(net_revenue_real, i), _get(net_revenue_base, i),
                _get(real_churn, i), _get(base_churn, i),
                _get(np.cumsum(net_revenue_real), i), _get(np.cumsum(net_revenue_base), i),
                _get(mean_health_real or [], i), _get(mean_health_base or [], i),
                _get(mean_engage_real or [], i), _get(mean_engage_base or [], i),
            ])


    last_n = max(50, int(N*0.05)) if N else 0
    import numpy as _np
    energy_v = float(_np.nansum(_np.asarray(real_energy, dtype=float))) if N else 0.0
    energy_b = float(_np.nansum(_np.asarray(base_energy, dtype=float))) if N else 0.0
    cum_v = float(cum_real[-1]) if len(cum_real) else 0.0
    cum_b = float(cum_base[-1]) if len(cum_base) else 0.0
    churn_v = float(_np.nanmedian(_np.asarray(real_churn[-last_n:], dtype=float))) if last_n else None
    churn_b = float(_np.nanmedian(_np.asarray(base_churn[-last_n:], dtype=float))) if last_n else None
    metrics = {
        "batches": int(N),
        "cumrev_virgil": float(np.cumsum(np.asarray(net_revenue_real, dtype=float))[-1]) if net_revenue_real else 0.0,
        "cumrev_baseline": float(np.cumsum(np.asarray(net_revenue_base, dtype=float))[-1]) if net_revenue_base else 0.0,
        "cumrev_uplift_abs": float(
            (np.cumsum(np.asarray(net_revenue_real, dtype=float))[-1]) -
            (np.cumsum(np.asarray(net_revenue_base, dtype=float))[-1])
        ),
        "cumrev_uplift_pct": float(
            (np.cumsum(np.asarray(net_revenue_real, dtype=float))[-1] -
            np.cumsum(np.asarray(net_revenue_base, dtype=float))[-1]) /
            (np.cumsum(np.asarray(net_revenue_base, dtype=float))[-1] + 1e-9)
        ) if net_revenue_base else None,
        "final_rolling_churn_virgil": float(np.nanmedian(np.asarray(real_churn[-last_n:], dtype=float))) if last_n else None,
        "final_rolling_churn_baseline": float(np.nanmedian(np.asarray(base_churn[-last_n:], dtype=float))) if last_n else None,
        "energy_total_virgil": float(np.nansum(np.asarray(real_energy, dtype=float))),
        "energy_total_baseline": float(np.nansum(np.asarray(base_energy, dtype=float))),
        "kwh_per_1k_revenue_virgil": _safe_div(
            np.nansum(np.asarray(real_energy, dtype=float)),
            (np.cumsum(np.asarray(net_revenue_real, dtype=float))[-1] / 1000.0)
        ) if net_revenue_real else None,
        "kwh_per_1k_revenue_baseline": _safe_div(
            np.nansum(np.asarray(base_energy, dtype=float)),
            (np.cumsum(np.asarray(net_revenue_base, dtype=float))[-1] / 1000.0)
        ) if net_revenue_base else None,
        "notes": "last_n used for final rolling churn is max(50, 5% of batches)"
    }

    with open(os.path.join(f"output/seed_{SEED}" f"{prefix}_metrics_seed{SEED}_{timestamp}.json"), "w", encoding="utf-8") as jf:
        json.dump(metrics, jf, indent=2)
    
# --- 2x2 Dashboard (Energy | Net Revenue / Rolling Churn | Cumulative Revenue) ---
    if dashboard:
        dash = plt.figure(constrained_layout=True)
        dash.set_size_inches(*SLIDE_FIGSIZE)
        gs = mpl.gridspec.GridSpec(2, 2, figure=dash)
        axE = dash.add_subplot(gs[0,0])
        axN = dash.add_subplot(gs[0,1])
        axC = dash.add_subplot(gs[1,0])
        axR = dash.add_subplot(gs[1,1])

        # Energy
        _add_dual_line(axE, real_energy, label="Virgil Energy", color='royalblue')
        _add_dual_line(axE, base_energy, label="Baseline Energy", linestyle="--", color='orange')
        _style_axes(axE, "Energy Usage per Batch", "Batch", "kWh")
        axE.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axE.legend(fontsize=10)

        # Net Revenue
        _add_dual_line(axN, net_revenue_real, label="Virgil", color='royalblue')
        _add_dual_line(axN, net_revenue_base, label="Baseline", linestyle="--", color='orange')
        _style_axes(axN, "Net Revenue per Batch", "Batch", "Revenue ($)")
        _format_currency_axis(axN)
        axN.legend(fontsize=10)

        # Rolling Churn
        _add_dual_line(axC, real_churn, label="Virgil Rolling Churn", color='royalblue')
        _add_dual_line(axC, base_churn, label="Baseline Rolling Churn", linestyle="--", color='orange')
        if annotations:
            for batch_idx, label in annotations:
                axC.axvline(x=batch_idx, color="gray", linestyle=":", alpha=0.6)
        _style_axes(axC, "Rolling Churn Rate", "Batch", "Churn (%)")
        _format_percent_axis(axC)
        axC.legend(fontsize=10)

        # Cumulative Revenue
        cum_real = np.cumsum(np.asarray(net_revenue_real, dtype=float))
        cum_base = np.cumsum(np.asarray(net_revenue_base, dtype=float))
        axR.plot(cum_real, label="Virgil (Cumulative Revenue)", linewidth=2)
        axR.plot(cum_base, label="Baseline (Cumulative Revenue)", linestyle="--", linewidth=2)
        _style_axes(axR, "Cumulative Revenue", "Batch", "Revenue ($)")
        _format_currency_axis(axR)
        axR.legend(fontsize=10)

        _add_footer(dash)
        base = os.path.join(f"output/seed_{SEED}", f"{prefix}_dashboard_seed{SEED}_{timestamp}")
        _save_fig(dash, base)
        plt.close(dash)

    # Content Event Severity Chart
    plot_content_event_severity(content_real, content_base, save_path=os.path.join("output", f"chart_content_severity_seed{SEED}.png"))

    plot_population_size(
        challenger_metrics,
        baseline_metrics,
        save_path=os.path.join(f"output/seed_{SEED}", f"chart_population_size_seed{SEED}.png")
    )

        # --- Growth Chart ---
    plot_growth(
        challenger_metrics,
        baseline_metrics,
        initial_users=NUM_USERS,
        save_path=os.path.join(f"output/seed_{SEED}", f"{prefix}_growth_seed{SEED}_{timestamp}")
    )

# ====================== NEW: Strategy Usage Export & Animation ======================

# at the end of generate_pitch_charts(...) in viz_tools.py
# ...
    demo_csv = export_demo_bundle_csv(
        challenger_metrics, baseline_metrics,
        virgil_strat_counts_per_batch=challenger_strat_counts,
        base_strat_counts_per_batch=base_strat_counts,
        challenger_state_counts_per_batch=challenger_state_counts,
        base_state_counts_per_batch=base_state_counts,
        seed=SEED
    )
    print("Demo bundle CSV:", demo_csv)
