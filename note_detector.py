import numpy as np
import sounddevice as sd
import aubio
import math
import time
from collections import deque

from score import expected
from score_follower import follower

# "hmm" — HMM beam-search follower (robust, any start position)
# "sequential" — strict left-to-right index matcher
MODE = "hmm"


# False — clean demo output only
# True  — show all debug lines: [onset], [slur], [pitch], [dbl?], [harm?]
VERBOSE = False


# Audio params
SAMPLE_RATE = 44100
HOP_SIZE    = 512
BUFFER_SIZE = 2048


# Pitch detector (YIN)
pitch_o = aubio.pitch("yin", BUFFER_SIZE, HOP_SIZE, SAMPLE_RATE)
pitch_o.set_unit("Hz")
pitch_o.set_silence(-40)

# Onset detector
onset_o = aubio.onset("complex", BUFFER_SIZE, HOP_SIZE, SAMPLE_RATE)
onset_o.set_threshold(0.3)
onset_o.set_minioi_ms(250)  # minimum gap between aubio onsets

# Input device
# Set to None to use the system default, or set to a device name/index.
# Run: python -c "import sounddevice as sd; print(sd.query_devices())" to list devices.
DEVICE = 0

# Detection thresholds
PITCH_TOL_SEMITONES = 1.0
MIN_CONFIDENCE      = 0.75
MIN_FREQ            = 180
MAX_FREQ            = 1400

# Frames to wait after an onset before reading pitch
PITCH_LOOKAHEAD_FRAMES = 3

# Within the lookahead window, reject the note if pitch readings are still spread out
PITCH_STABILITY_TOL = 0.8 

# If the detected pitch is an octave off from the last confirmed note, print a warning
HARMONIC_WARN = True

# Pitch-change detection
PITCH_CHANGE_THRESHOLD  = 1.2
PITCH_STABLE_FRAMES     = 6

# Cooldown after a note event fires
COOLDOWN_FRAMES = 18

# Same-pitch time gate
SAME_PITCH_GATE_S = 0.28

def hz_to_midi(f_hz: float) -> float:
    return 69.0 + 12.0 * math.log2(f_hz / 440.0)

def midi_to_name(m: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[m % 12]}{m // 12 - 1}"

idx                  = 0        
frames_since_onset   = None     
pending_onset_time   = None    
pending_onset_conf   = None    
last_confirmed_midi  = None  
last_confirmed_time  = None  

# Ring buffer of recent (f0, confidence) readings to average across lookahead window
pitch_history = deque(maxlen=PITCH_LOOKAHEAD_FRAMES + 1)


current_semitone   = None     
pitch_change_count = 0   

# Cooldown state
cooldown_remaining = 0

def on_note_event(midi_est: float, onset_time: float):
    """Called once per detected onset with a stable pitch estimate."""
    global idx

    heard = midi_to_name(int(round(midi_est)))

    if MODE == "hmm":
        result   = follower.observe(midi_est, onset_time)
        exp_name = expected[result["idx"]]["pitches_name"][0]
        conf_pct = int(result["confidence"] * 100)
        conf_bar = "█" * (conf_pct // 10) + "░" * (10 - conf_pct // 10)
        status   = "LOCK" if result["locked"] else "srch"
        print(
            f"  [{status}]  m{result['measure']:2d}  beat {result['beat']:.1f}"
            f"  │  {exp_name:<4s} → {heard:<4s}"
            f"  │  {conf_bar} {conf_pct:2d}%"
            f"  │  ~{result['bpm']:.0f} BPM"
        )

    else:  # sequential
        if idx >= len(expected):
            return
        exp      = expected[idx]
        exp_midi = int(exp["pitches_midi"][0])
        exp_name = exp["pitches_name"][0]
        diff     = abs(midi_est - exp_midi)
        if diff <= PITCH_TOL_SEMITONES:
            print(f"  ✓  m{exp['measure']:2d}  beat {exp['beat']:.1f}  │  {exp_name:<4s} → {heard:<4s}")
            idx += 1
        else:
            print(f"  ✗  m{exp['measure']:2d}  beat {exp['beat']:.1f}  │  {exp_name:<4s} → {heard:<4s}  (wrong)")


def audio_callback(indata, frames, time_info, status):
    global idx, frames_since_onset, pending_onset_time, pending_onset_conf, \
           last_confirmed_midi, last_confirmed_time, \
           current_semitone, pitch_change_count, cooldown_remaining

    if status:
        return

    x = indata[:, 0].astype(np.float32)

    f0   = float(pitch_o(x)[0])
    conf = float(pitch_o.get_confidence())
    pitch_history.append((f0, conf))
    is_onset = onset_o(x)[0]

    if cooldown_remaining > 0:
        cooldown_remaining -= 1
        return

    # Pitch-change detection (slurred notes)
    if frames_since_onset is None and f0 > 0 and conf >= MIN_CONFIDENCE \
            and MIN_FREQ <= f0 <= MAX_FREQ:
        semitone = int(round(hz_to_midi(f0)))
        if current_semitone is None:
            current_semitone = semitone
            pitch_change_count = 0
        elif semitone != current_semitone:
            pitch_change_count += 1
            if pitch_change_count >= PITCH_STABLE_FRAMES:
                if VERBOSE:
                    print(f"  [slur]  pitch shift {midi_to_name(current_semitone)} -> {midi_to_name(semitone)}")
                current_semitone = semitone
                pitch_change_count = 0
                frames_since_onset = 0
                pending_onset_time = time.time()
        else:
            pitch_change_count = 0  # pitch is stable

    if is_onset:
        # Reject low-confidence onsets when we're already mid-note
        if conf < MIN_CONFIDENCE and current_semitone is not None:
            if VERBOSE:
                print(f"  [onset] suppressed (low conf={conf:.2f} while on {midi_to_name(current_semitone)})")
        else:
            frames_since_onset = 0
            pending_onset_time = time.time()
            pending_onset_conf = conf
            current_semitone   = None
            pitch_change_count = 0
            if VERBOSE:
                print(f"  [onset] detected  f0={f0:.1f} Hz  conf={conf:.2f}")

    # read stable pitch
    if frames_since_onset is not None:
        frames_since_onset += 1

        if frames_since_onset >= PITCH_LOOKAHEAD_FRAMES:
            valid = [(f, c) for f, c in pitch_history if f > 0 and c >= MIN_CONFIDENCE
                     and MIN_FREQ <= f <= MAX_FREQ]

            frames_since_onset = None  # reset; done processing this onset

            if not valid:
                return 

            # Pitch stability gate
            midis = [hz_to_midi(f) for f, _ in valid]
            if len(midis) >= 2 and float(np.std(midis)) > PITCH_STABILITY_TOL:
                if VERBOSE:
                    print(f"  [pitch] unstable (std={np.std(midis):.2f} st) — skipping onset")
                return

            # Weight toward later frames
            weights = [c * (i + 1) for i, (_, c) in enumerate(valid)]
            avg_f0  = float(np.average([f for f, _ in valid], weights=weights))
            new_midi = hz_to_midi(avg_f0)

            # Same-pitch time gate
            now = time.time()
            if (SAME_PITCH_GATE_S > 0
                    and last_confirmed_midi is not None
                    and last_confirmed_time is not None
                    and abs(new_midi - last_confirmed_midi) < 0.8
                    and (now - last_confirmed_time) < SAME_PITCH_GATE_S):
                if VERBOSE:
                    print(f"  [dbl?] {midi_to_name(int(round(new_midi)))} same as last note, "
                          f"only {(now - last_confirmed_time)*1000:.0f} ms later — skipping")
                return

            # Harmonic warning
            if VERBOSE and HARMONIC_WARN and last_confirmed_midi is not None:
                for oct_shift in (12, -12):
                    if abs(new_midi - (last_confirmed_midi + oct_shift)) < 1.5:
                        print(f"  [harm?] {midi_to_name(int(round(new_midi)))} may be "
                              f"{oct_shift:+d} st octave of last note "
                              f"{midi_to_name(int(round(last_confirmed_midi)))}")

            last_confirmed_midi  = new_midi
            last_confirmed_time  = now
            current_semitone     = int(round(new_midi))
            pitch_change_count   = 0
            on_note_event(new_midi, pending_onset_time)
            cooldown_remaining = COOLDOWN_FRAMES

if __name__ == "__main__":
    device_info = sd.query_devices(DEVICE) if DEVICE is not None else sd.query_devices(kind="input")

    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║          Smarter Music — Score Follower          ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print()
    print(f"  Piece   : Bach Minuet in G  ({len(expected)} note events, 32 measures)")
    print(f"  Mic     : {device_info['name']}")
    if MODE == "hmm":
        print(f"  Mode    : HMM beam search  —  play from any measure, jump freely")
        print(f"  Output  : [status]  measure  beat  │  expected → heard  │  confidence  │  tempo")
    else:
        print(f"  Mode    : Sequential  —  play from the beginning in order")
        print(f"  Output  : ✓/✗  measure  beat  │  expected → heard")
    print(f"  Verbose : {'on  (showing raw onset/slur/debug events)' if VERBOSE else 'off (Ctrl+C to stop)'}")
    print()
    print("  " + "─" * 50)
    print()

    try:
        with sd.InputStream(
            device=DEVICE,
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=HOP_SIZE,
            callback=audio_callback,
        ):
            if MODE == "sequential":
                while idx < len(expected):
                    time.sleep(0.05)
                print("\nDone — reached end of score.")
            else:
                while True:
                    time.sleep(0.05)

    except KeyboardInterrupt:
        if MODE == "sequential":
            print(f"\nStopped at note index {idx} / {len(expected)}.")
        else:
            print("\nStopped.")
