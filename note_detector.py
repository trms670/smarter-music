import numpy as np
import sounddevice as sd
import aubio
import math
import time
from collections import deque

from score import expected

# ------------------------
# Audio params
# ------------------------
SAMPLE_RATE = 44100
HOP_SIZE    = 512
BUFFER_SIZE = 2048

# ------------------------
# Pitch detector (YIN)
# ------------------------
pitch_o = aubio.pitch("yin", BUFFER_SIZE, HOP_SIZE, SAMPLE_RATE)
pitch_o.set_unit("Hz")
pitch_o.set_silence(-40)

# ------------------------
# Onset detector
# ------------------------
onset_o = aubio.onset("complex", BUFFER_SIZE, HOP_SIZE, SAMPLE_RATE)
onset_o.set_threshold(0.3)
onset_o.set_minioi_ms(200)  # suppress re-triggers within 200 ms (covers secondary bow energy peaks)

# ------------------------
# Detection thresholds
# ------------------------
PITCH_TOL_SEMITONES = 1.0   # wider than before since we're trusting onset timing more
MIN_CONFIDENCE      = 0.75
MIN_FREQ            = 180   # violin low G ~196 Hz; a little headroom below
MAX_FREQ            = 1400  # ~E6

# Frames to wait after an onset before reading pitch (lets the attack transient settle)
PITCH_LOOKAHEAD_FRAMES = 2

# Pitch-change detection (catches slurred notes with no onset spike)
PITCH_CHANGE_THRESHOLD  = 1.2   # semitones — above vibrato range (~0.5 st), below a half step
PITCH_STABLE_FRAMES     = 4     # frames the new pitch must hold before confirming a slurred note

# Cooldown after a note event fires — blocks both detectors to prevent double-triggers
# 10 frames * (512 / 44100) ≈ 116 ms; fast enough for quick passages, long enough to debounce
COOLDOWN_FRAMES = 15

# ------------------------
# Utilities
# ------------------------
def hz_to_midi(f_hz: float) -> float:
    return 69.0 + 12.0 * math.log2(f_hz / 440.0)

def midi_to_name(m: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[m % 12]}{m // 12 - 1}"

# ------------------------
# State
# ------------------------
idx                = 0        # current position in expected note list
frames_since_onset = None     # None = not waiting; int = frames elapsed since last onset
pending_onset_time = None     # wall-clock time of the last onset (for future tempo use)

# Ring buffer of recent (f0, confidence) readings to average across lookahead window
pitch_history = deque(maxlen=PITCH_LOOKAHEAD_FRAMES + 1)

# Pitch-change detection state (for slurred notes)
current_semitone   = None     # semitone we're currently "in"
pitch_change_count = 0        # consecutive frames at a different semitone

# Cooldown state
cooldown_remaining = 0

def on_note_event(midi_est: float, onset_time: float):
    """Called once per detected onset with a stable pitch estimate."""
    global idx

    if idx >= len(expected):
        return

    midi_round = int(round(midi_est))
    exp        = expected[idx]
    exp_midi   = int(exp["pitches_midi"][0])
    exp_name   = exp["pitches_name"][0]
    diff       = abs(midi_est - exp_midi)

    heard_name = midi_to_name(midi_round)

    if diff <= PITCH_TOL_SEMITONES:
        print(
            f"  MATCH  idx={idx:3d}  measure={exp['measure']}  beat={exp['beat']}"
            f"  expected={exp_name:<4s}  heard={heard_name:<4s}"
        )
        idx += 1
    else:
        print(
            f"  MISS   idx={idx:3d}  measure={exp['measure']}  beat={exp['beat']}"
            f"  expected={exp_name:<4s}  heard={heard_name:<4s}"
        )

# ------------------------
# Audio callback
# ------------------------
def audio_callback(indata, frames, time_info, status):
    global idx, frames_since_onset, pending_onset_time, current_semitone, pitch_change_count, cooldown_remaining

    if status:
        return

    x = indata[:, 0].astype(np.float32)

    # Always feed aubio every frame to keep its internal state current —
    # skipping frames causes stale state and spurious onsets after cooldown.
    f0   = float(pitch_o(x)[0])
    conf = float(pitch_o.get_confidence())
    pitch_history.append((f0, conf))
    is_onset = onset_o(x)[0]

    # --- Cooldown: ignore detection results briefly after each note event ---
    if cooldown_remaining > 0:
        cooldown_remaining -= 1
        return

    # --- Pitch-change detection (slurred notes) ---
    # Only runs when we're not already in a post-onset lookahead window,
    # so onset-detected and slur-detected notes don't double-trigger.
    if frames_since_onset is None and f0 > 0 and conf >= MIN_CONFIDENCE \
            and MIN_FREQ <= f0 <= MAX_FREQ:
        semitone = int(round(hz_to_midi(f0)))
        if current_semitone is None:
            current_semitone = semitone
            pitch_change_count = 0
        elif semitone != current_semitone:
            pitch_change_count += 1
            if pitch_change_count >= PITCH_STABLE_FRAMES:
                print(f"  [slur]  pitch shift {midi_to_name(current_semitone)} -> {midi_to_name(semitone)}")
                current_semitone = semitone
                pitch_change_count = 0
                frames_since_onset = 0
                pending_onset_time = time.time()
        else:
            pitch_change_count = 0  # pitch is stable — reset drift counter

    if is_onset:
        # Reject low-confidence onsets when we're already mid-note — these are
        # typically bow pressure artifacts, not genuine new notes.
        if conf < MIN_CONFIDENCE and current_semitone is not None:
            print(f"  [onset] suppressed (low conf={conf:.2f} while on {midi_to_name(current_semitone)})")
        else:
            frames_since_onset = 0
            pending_onset_time = time.time()
            current_semitone = None
            pitch_change_count = 0
            print(f"  [onset] detected  f0={f0:.1f} Hz  conf={conf:.2f}")

    # --- After PITCH_LOOKAHEAD_FRAMES frames post-onset, read stable pitch ---
    if frames_since_onset is not None:
        frames_since_onset += 1

        if frames_since_onset >= PITCH_LOOKAHEAD_FRAMES:
            # Average the valid pitch readings in the lookahead window
            valid = [(f, c) for f, c in pitch_history if f > 0 and c >= MIN_CONFIDENCE
                     and MIN_FREQ <= f <= MAX_FREQ]

            frames_since_onset = None  # reset; done processing this onset

            if not valid:
                return  # no reliable pitch detected after onset — skip

            avg_f0 = np.mean([f for f, _ in valid])
            current_semitone = int(round(hz_to_midi(avg_f0)))  # anchor slur tracker to reliable pitch
            pitch_change_count = 0
            on_note_event(hz_to_midi(avg_f0), pending_onset_time)
            cooldown_remaining = COOLDOWN_FRAMES

# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    print(f"Score loaded: {len(expected)} note events\n")
    print("Starting onset-based pitch detection + sequential alignment.")
    print("Play the melody from the beginning. Press Ctrl+C to stop.\n")

    try:
        with sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=HOP_SIZE,
            callback=audio_callback,
        ):
            while idx < len(expected):
                time.sleep(0.05)

        print("\nDone — reached end of expected note list.")

    except KeyboardInterrupt:
        print(f"\nStopped at note index {idx} / {len(expected)}.")
