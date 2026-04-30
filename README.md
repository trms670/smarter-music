# SmarterMusic — Real-Time Score Follower for Solo Violin

A real-time audio-to-score alignment system for solo violin practice. The system
listens to a live performance through a microphone, estimates the performer's
current position in a digital score using an online Hidden Markov Model, and
displays a synchronized cursor on a browser-rendered score.

Evaluated on Bach's Minuet in G (32 measures, 3/4 time, 127 note events).

---

## Requirements

- Python 3.9+
- A microphone connected to your computer
- A modern web browser (Chrome or Firefox recommended)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the System

### Live violin mode (recommended)

```bash
python score_display.py
```

This starts the Flask server, opens the browser automatically, and begins
listening on your default microphone. Start playing the piece and the cursor
will follow your position in real time.

Options:

```bash
python score_display.py --port 5050        # use a different port (default: 5000)
python score_display.py --no-browser       # don't auto-open the browser
```

### Simulation mode (no microphone needed)

Steps through the score automatically at a specified tempo, useful for
testing the display and SSE pipeline without a live instrument.

```bash
python score_display.py --simulate --bpm 100
```

### Terminal-only mode (no browser)

Runs the HMM score follower and prints alignment output to the terminal
without starting the web server. Useful for data collection and debugging.

```bash
python -u note_detector.py
```

To capture output for analysis:

```bash
python -u note_detector.py 2>&1 | tee run1.txt
```

---

## Changing the Input Device

By default the system uses device index `0`. To list available devices:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

Then set `DEVICE` at the top of `note_detector.py` to the desired index or
device name.

---

## Analysing Results

`parse_results.py` parses terminal output captured from `note_detector.py`
and produces accuracy statistics and plots.

```bash
# Analyse one or more full run-throughs:
python parse_results.py run1.txt run2.txt run3.txt

# Cold-start jump test (performer starts at an arbitrary measure):
python parse_results.py --jump jump_test.txt

# Free navigation demonstration (performer jumps around the piece):
python parse_results.py --nav nav_test.txt
```

Output files: `confidence_plot.pdf`, `bpm_plot.pdf`,
`confidence_multi.pdf` (multi-run), `navigation_plot.pdf` (nav mode).

---
