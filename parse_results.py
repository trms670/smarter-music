#!/usr/bin/env python3
"""
parse_results.py
----------------
Parses HMM terminal output from note_detector.py and produces alignment
statistics and plots for thesis results.

Usage
-----
  # Capture a run:
  python note_detector.py 2>&1 | tee run1.txt

  # Analyse one or more runs:
  python parse_results.py run1.txt run2.txt run3.txt

  # Analyse a jump/recovery test (separate experiment):
  python parse_results.py --jump jump_test.txt
  python parse_results.py --recovery recovery_test.txt

Output
------
  Prints a summary table to stdout.
  Saves confidence_plot.pdf and bpm_plot.pdf in the current directory.
"""

import argparse
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Line format emitted by note_detector.py (HMM mode) ───────────────────────
# "  [LOCK]  m 1  beat 1.0  │  G4   → G4    │  ██████████ 100%  │  ~100 BPM"
LINE_RE = re.compile(
    r"\[(LOCK|srch)\]"          
    r"\s+m\s*(\d+)"           
    r"\s+beat\s+([\d.]+)"      
    r".+?"                     
    r"(\S+)\s*→\s*(\S+)"     
    r".+?"                  
    r"(\d+)%"            
    r".+?~([\d.]+)\s*BPM",  
    re.UNICODE,
)

A_SECTION_MAX_MEASURE = 16  
DISPLAY_CONF_THRESHOLD = 55 


def parse_log(path: str) -> list[dict]:
    """Return a list of per-note-event records parsed from a log file."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                records.append({
                    "locked":     m.group(1) == "LOCK",
                    "measure":    int(m.group(2)),
                    "beat":       float(m.group(3)),
                    "expected":   m.group(4).strip(),
                    "heard":      m.group(5).strip(),
                    "confidence": int(m.group(6)),
                    "bpm":        float(m.group(7)),
                })
    return records


def _pitch_class(note_name: str) -> str:
    """'F#4' → 'F#',  'Bb3' → 'Bb',  'G4' → 'G'."""
    for i, ch in enumerate(note_name):
        if ch.isdigit() or ch == "-":
            return note_name[:i]
    return note_name


def exact_match(exp: str, heard: str) -> bool:
    """True if heard note (including octave) matches expected."""
    return exp == heard


def pitch_class_match(exp: str, heard: str) -> bool:
    """True if heard pitch class matches expected, regardless of octave.
    Catches YIN octave errors while still counting as a correct pitch."""
    return _pitch_class(exp) == _pitch_class(heard)



def _section_stats(records: list[dict], label: str) -> dict:
    n = len(records)
    if n == 0:
        return {"section": label, "n": 0}
    exact   = sum(exact_match(r["expected"], r["heard"]) for r in records)
    pc      = sum(pitch_class_match(r["expected"], r["heard"]) for r in records)
    locked  = sum(r["locked"] for r in records)
    confs   = [r["confidence"] for r in records]
    return {
        "section":       label,
        "n":             n,
        "exact_acc":     exact / n,
        "pc_acc":        pc / n,
        "lock_rate":     locked / n,
        "avg_conf":      float(np.mean(confs)),
        "median_conf":   float(np.median(confs)),
    }


def compute_stats(records: list[dict]) -> dict:
    a = [r for r in records if r["measure"] <= A_SECTION_MAX_MEASURE]
    b = [r for r in records if r["measure"] >  A_SECTION_MAX_MEASURE]
    return {
        "overall":   _section_stats(records, "Overall"),
        "a_section": _section_stats(a,       "A-section (m1–16)"),
        "b_section": _section_stats(b,       "B-section (m17–32)"),
    }


def notes_to_first_lock(records: list[dict]):
    """Return the 0-based index of the first LOCK event, or None if never locked."""
    for i, r in enumerate(records):
        if r["locked"]:
            return i
    return None


def detect_resets(records: list[dict]) -> list[int]:
    """Detect indices where a new attempt begins after a hard reset."""
    jump_indices = []
    for i in range(1, len(records)):
        prev, curr = records[i - 1], records[i]
        # hard reset
        if prev["locked"] and prev["confidence"] > 70 \
                and not curr["locked"] and curr["confidence"] < 55:
            jump_indices.append(i)
            continue
        # large backwards measure jump
        if records[i]["measure"] - records[i - 1]["measure"] < -4:
            jump_indices.append(i)
    return jump_indices



def print_summary(stats: dict) -> None:
    hdr = f"{'Section':<22} {'N':>5} {'Exact':>8} {'PitchCl':>8} {'Locked':>8} {'AvgConf':>9} {'MedConf':>9}"
    print(hdr)
    print("─" * len(hdr))
    for key in ("overall", "a_section", "b_section"):
        s = stats[key]
        if s["n"] == 0:
            print(f"  {s['section']:<20}  (no data)")
            continue
        print(
            f"  {s['section']:<20}"
            f"  {s['n']:>5}"
            f"  {s['exact_acc']*100:>7.1f}%"
            f"  {s['pc_acc']*100:>7.1f}%"
            f"  {s['lock_rate']*100:>7.1f}%"
            f"  {s['avg_conf']:>8.1f}%"
            f"  {s['median_conf']:>8.1f}%"
        )
    print()


def print_jump_stats(records: list[dict], label: str = "") -> None:
    """For a jump/recovery log: split on detected resets and report
    notes-to-first-lock for each attempt."""
    reset_indices = detect_resets(records)
    boundaries = [0] + [i for i in reset_indices] + [len(records)]
    attempts = [records[boundaries[i]:boundaries[i + 1]]
                for i in range(len(boundaries) - 1)
                if boundaries[i + 1] > boundaries[i]]

    print(f"  Jump/recovery analysis  {label}")
    print(f"  {'Attempt':>8}  {'Start m':>8}  {'NtoLock':>9}  {'AvgConf':>9}")
    print("  " + "─" * 44)
    ntl_values = []
    for i, attempt in enumerate(attempts):
        ntl = notes_to_first_lock(attempt)
        start_m = attempt[0]["measure"] if attempt else "–"
        avg_c   = np.mean([r["confidence"] for r in attempt]) if attempt else 0.0
        ntl_str = str(ntl + 1) if ntl is not None else "never"
        ntl_values.append(ntl + 1 if ntl is not None else None)
        print(f"  {i+1:>8}  {start_m:>8}  {ntl_str:>9}  {avg_c:>8.1f}%")

    valid = [v for v in ntl_values if v is not None]
    if valid:
        print(f"\n  Mean notes-to-lock: {np.mean(valid):.1f}  "
              f"(median {np.median(valid):.0f},  max {max(valid)})")
    print()



def plot_confidence(records: list[dict], out: str = "confidence_plot.pdf",
                    title: str = "HMM Alignment Confidence Over Performance") -> None:
    fig, ax = plt.subplots(figsize=(13, 4))

    xs    = list(range(len(records)))
    confs = [r["confidence"] for r in records]

    # Section shading
    a_end = next((i for i, r in enumerate(records) if r["measure"] > A_SECTION_MAX_MEASURE),
                 len(records))
    ax.axvspan(0,     a_end,        alpha=0.07, color="royalblue", label="A-section (m1–16)")
    ax.axvspan(a_end, len(records), alpha=0.07, color="seagreen",  label="B-section (m17–32)")

    # Confidence line
    ax.plot(xs, confs, color="steelblue", linewidth=0.9, alpha=0.85)

    # Scatter: LOCK vs searching
    lx = [i for i in xs if records[i]["locked"]]
    sx = [i for i in xs if not records[i]["locked"]]
    ax.scatter(lx, [confs[i] for i in lx], color="steelblue", s=14, zorder=4, label="Locked")
    ax.scatter(sx, [confs[i] for i in sx], color="tomato",    s=14, zorder=4, label="Searching")

    # Display threshold line
    ax.axhline(DISPLAY_CONF_THRESHOLD, color="dimgray", linestyle="--",
               linewidth=0.8, label=f"Display threshold ({DISPLAY_CONF_THRESHOLD}%)")

    ax.set_xlabel("Note event index")
    ax.set_ylabel("Beam confidence (%)")
    ax.set_title(title)
    ax.set_xlim(0, len(records) - 1)
    ax.set_ylim(0, 108)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close()


def plot_bpm(records: list[dict], out: str = "bpm_plot.pdf",
             title: str = "Adaptive Tempo Estimate Over Performance") -> None:
    fig, ax = plt.subplots(figsize=(13, 3))
    xs   = list(range(len(records)))
    bpms = [r["bpm"] for r in records]
    ax.plot(xs, bpms, color="darkorange", linewidth=1.2)
    ax.set_xlabel("Note event index")
    ax.set_ylabel("Estimated BPM")
    ax.set_title(title)
    ax.set_xlim(0, len(records) - 1)
    ax.set_ylim(40, 220)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close()


def plot_navigation(records: list[dict], out: str = "navigation_plot.pdf",
                    title: str = "Score Position Tracking During Free Navigation") -> None:
    """Plot measure number vs note index, coloured by confidence.
    Shows the system tracking position as the performer jumps around the score."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    xs      = list(range(len(records)))
    measures = [r["measure"] for r in records]
    confs    = [r["confidence"] / 100.0 for r in records]
    locked   = [r["locked"] for r in records]

    # Top panel: measure number coloured by confidence
    sc = ax1.scatter(xs, measures, c=confs, cmap="RdYlGn", vmin=0.0, vmax=1.0,
                     s=30, zorder=3)
    ax1.plot(xs, measures, color="gray", linewidth=0.5, alpha=0.4, zorder=2)

    # Shade A and B sections
    a_end = next((i for i, r in enumerate(records)
                  if r["measure"] > A_SECTION_MAX_MEASURE), len(records))
    ax1.axvspan(0,     a_end,        alpha=0.06, color="royalblue")
    ax1.axvspan(a_end, len(records), alpha=0.06, color="seagreen")

    # Mark searching points with an 'x'
    sx = [i for i in xs if not locked[i]]
    ax1.scatter(sx, [measures[i] for i in sx], marker="x", color="tomato",
                s=25, linewidths=1.0, zorder=4, label="Searching")

    cbar = fig.colorbar(sc, ax=ax1, pad=0.01)
    cbar.set_label("Confidence", fontsize=8)
    ax1.set_ylabel("Estimated measure")
    ax1.set_title(title)
    ax1.set_ylim(0, 34)
    ax1.set_yticks([1, 8, 16, 17, 24, 32])
    ax1.axhline(16.5, color="steelblue", linestyle=":", linewidth=0.8, alpha=0.6)
    ax1.text(2, 17.3, "B-section →", fontsize=7, color="steelblue", alpha=0.8)
    ax1.legend(fontsize=8, loc="upper right")

    # Bottom panel: confidence over time
    ax2.plot(xs, [c * 100 for c in confs], color="steelblue", linewidth=0.9)
    ax2.axhline(DISPLAY_CONF_THRESHOLD, color="dimgray", linestyle="--",
                linewidth=0.8, label=f"Display threshold ({DISPLAY_CONF_THRESHOLD}%)")
    ax2.set_xlabel("Note event index")
    ax2.set_ylabel("Confidence (%)")
    ax2.set_ylim(0, 108)
    ax2.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close()


def plot_confidence_multi(all_records: list[list[dict]],
                          out: str = "confidence_multi.pdf") -> None:
    """Overlay confidence curves from multiple runs for visual consistency check."""
    fig, ax = plt.subplots(figsize=(13, 4))
    colors = plt.cm.tab10.colors
    for i, records in enumerate(all_records):
        xs    = list(range(len(records)))
        confs = [r["confidence"] for r in records]
        ax.plot(xs, confs, color=colors[i % 10], linewidth=0.9,
                alpha=0.75, label=f"Run {i+1}")
    ax.axhline(DISPLAY_CONF_THRESHOLD, color="dimgray", linestyle="--",
               linewidth=0.8, label=f"Display threshold ({DISPLAY_CONF_THRESHOLD}%)")
    ax.set_xlabel("Note event index")
    ax.set_ylabel("Beam confidence (%)")
    ax.set_title("Confidence Across Multiple Runs")
    ax.set_ylim(0, 108)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse note_detector HMM output")
    parser.add_argument("logs", nargs="+", metavar="FILE",
                        help="One or more terminal log files to analyse")
    parser.add_argument("--jump", action="store_true",
                        help="Treat logs as cold-start jump tests; report notes-to-lock")
    parser.add_argument("--recovery", action="store_true",
                        help="Treat logs as recovery tests; report notes-to-lock after resets")
    parser.add_argument("--nav", action="store_true",
                        help="Treat log as a free-navigation demo; produce a navigation plot")
    args = parser.parse_args()

    all_records = []
    for path in args.logs:
        recs = parse_log(path)
        if not recs:
            print(f"  WARNING: no HMM lines found in {path!r} — check the file.")
            continue
        print(f"  {path}: {len(recs)} note events")
        all_records.append(recs)

    if not all_records:
        print("No data to analyse.")
        sys.exit(1)

    print()

    if args.jump or args.recovery:
        for i, recs in enumerate(all_records):
            print_jump_stats(recs, label=f"({args.logs[i]})")
        return

    if args.nav:
        for i, recs in enumerate(all_records):
            out = f"navigation_plot_{i+1}.pdf" if len(all_records) > 1 else "navigation_plot.pdf"
            plot_navigation(recs, out=out)
            print_summary(compute_stats(recs))
        return

    # Standard accuracy analysis
    for i, recs in enumerate(all_records):
        print(f"── Run {i + 1}  ({args.logs[i]}) ──────────────────────────")
        print_summary(compute_stats(recs))

    # Pooled stats if multiple runs
    if len(all_records) > 1:
        pooled = [r for recs in all_records for r in recs]
        print(f"── Pooled ({len(all_records)} runs, {len(pooled)} total notes) ─────────────")
        print_summary(compute_stats(pooled))

    # Plots
    plot_confidence(all_records[0], "confidence_plot.pdf")
    plot_bpm(all_records[0],        "bpm_plot.pdf")
    if len(all_records) > 1:
        plot_confidence_multi(all_records, "confidence_multi.pdf")


if __name__ == "__main__":
    main()
