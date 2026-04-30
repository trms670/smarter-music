"""
Usage
-----
  # Live violin mode (default):
  python score_display.py

  # Simulation mode (steps through score automatically at a given BPM):
  python score_display.py --simulate --bpm 100

  # Custom port:
  python score_display.py --port 5050

  The server prints a URL — open it in any browser to see the score display.
  Press Ctrl+C to stop.

"""

import argparse
import json
import queue
import threading
import time
import types
import webbrowser
from pathlib import Path

from flask import Flask, Response, render_template, send_file

# ── Import project modules ────────────────────────────────────────────────────
from score import expected
from score_follower import follower

# ── Pub-sub broadcast state ───────────────────────────────────────────────────
# Each SSE connection registers its own queue here.  Every follower result is
# copied into ALL registered queues so no client ever races another for events.

_client_queues: list = []
_client_queues_lock = threading.Lock()

_last_result: dict = {
    "idx": 0, "measure": 1, "beat": 1.0,
    "confidence": 0.0, "locked": False, "bpm": 100.0,
    "note_name": expected[0]["pitches_name"][0],
}


def _broadcast(result: dict) -> None:
    """Push a result to every connected SSE client."""
    with _client_queues_lock:
        for q in _client_queues:
            q.put(result)


# ── Monkey-patch follower.observe() ───────────────────────────────────────────

_original_observe = follower.observe.__func__  # unbound method


def _capturing_observe(self, midi_est: float, timestamp: float) -> dict:
    result = _original_observe(self, midi_est, timestamp)
    result["note_name"] = expected[result["idx"]]["pitches_name"][0]
    _last_result.update(result)
    _broadcast(dict(result))
    return result


follower.observe = types.MethodType(_capturing_observe, follower)


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates")
_MXL_PATH = Path(__file__).parent / "Bach_Minuet.mxl"


@app.route("/")
def index():
    return render_template("score.html", total_measures=32, total_notes=len(expected))


@app.route("/score.mxl")
def serve_mxl():
    """Serve the MusicXML file so OSMD can fetch it from the browser."""
    return send_file(_MXL_PATH, mimetype="application/vnd.recordare.musicxml+xml")


@app.route("/events")
def sse_stream():
    """
    Server-Sent Events endpoint — one queue per connection (pub-sub).
    Every follower result is broadcast to all connected clients so multiple
    tabs and SSE reconnections never steal events from each other.
    """
    my_queue: queue.Queue = queue.Queue()

    with _client_queues_lock:
        _client_queues.append(my_queue)

    def generate():
        last_heartbeat = time.time()
        try:
            while True:
                try:
                    result = my_queue.get(timeout=1.0)
                    yield f"data: {json.dumps(result)}\n\n"
                    last_heartbeat = time.time()
                except queue.Empty:
                    if time.time() - last_heartbeat >= 1.0:
                        yield ": heartbeat\n\n"
                        last_heartbeat = time.time()
        finally:
            # Clean up when the browser disconnects
            with _client_queues_lock:
                try:
                    _client_queues.remove(my_queue)
                except ValueError:
                    pass

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/state")
def current_state():
    """Return the latest follower state as JSON (for page-load init)."""
    return json.dumps(_last_result)


# ── Audio pipeline thread ─────────────────────────────────────────────────────

def _run_audio_pipeline():
    """
    Import and start note_detector's audio stream in a background thread.
    The import triggers all aubio/sounddevice initialisation at module level.
    """
    try:
        import sounddevice as sd
        import note_detector as nd

        device_info = sd.query_devices(nd.DEVICE) if nd.DEVICE is not None \
                      else sd.query_devices(kind="input")
        print(f"\n  Audio   : {device_info['name']}")
        print( "  Mode    : HMM beam-search  —  play from any measure\n")

        with sd.InputStream(
            device=nd.DEVICE,
            channels=1,
            samplerate=nd.SAMPLE_RATE,
            blocksize=nd.HOP_SIZE,
            callback=nd.audio_callback,
        ):
            while True:
                time.sleep(0.1)

    except Exception as exc:
        print(f"\n  [audio] Failed to start audio pipeline: {exc}")
        print( "  [audio] Run with --simulate to test without a microphone.\n")


# ── Simulation thread ─────────────────────────────────────────────────────────

def _run_simulation(bpm: float):
    """
    Step through `expected` automatically at `bpm` quarter-notes per second,
    calling follower.observe() as if a real violinist were playing.
    Loops the score continuously.
    """
    qps = bpm / 60.0
    print(f"\n  Simulating score playback at {bpm:.0f} BPM …\n")
    time.sleep(1.5)  # give Flask a moment to start

    while True:
        t = time.time()
        for ev in expected:
            # Compute how long this note should last at current tempo
            dur_s = ev["duration_q"] / qps
            follower.observe(float(ev["pitches_midi"][0]), t)
            t += dur_s
            time.sleep(max(0.0, dur_s - 0.003))  # small fudge for sleep overhead
        time.sleep(2.0)  # pause between repeats


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Smarter Music — real-time visual score follower")
    parser.add_argument("--simulate", action="store_true",
                        help="Step through the score automatically (no mic needed)")
    parser.add_argument("--bpm", type=float, default=100.0,
                        help="Simulation tempo in BPM (default: 100)")
    parser.add_argument("--port", type=int, default=5000,
                        help="Flask port (default: 5000)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't auto-open the browser")
    args = parser.parse_args()

    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║      Smarter Music — Visual Score Display        ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print()
    print(f"  Piece   : Bach Minuet in G  ({len(expected)} notes, 32 measures)")
    print(f"  Server  : http://127.0.0.1:{args.port}")

    # Start the backend worker thread
    if args.simulate:
        print(f"  Mode    : Simulation @ {args.bpm:.0f} BPM  (no microphone)")
        worker = threading.Thread(target=_run_simulation, args=(args.bpm,),
                                  daemon=True)
    else:
        print( "  Mode    : Live violin — start playing after the browser opens")
        worker = threading.Thread(target=_run_audio_pipeline, daemon=True)

    worker.start()

    # Auto-open browser after a short delay
    if not args.no_browser:
        def _open():
            time.sleep(1.2)
            webbrowser.open(f"http://127.0.0.1:{args.port}")
        threading.Thread(target=_open, daemon=True).start()

    # Run Flask (blocks until Ctrl+C)
    app.run(host="127.0.0.1", port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
