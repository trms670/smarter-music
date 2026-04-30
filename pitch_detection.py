import numpy as np
import sounddevice as sd
import aubio
import math
import time

from score import expected


def hz_to_midi(f_hz: float) -> float:
    return 69.0 + 12.0 * math.log2(f_hz / 440.0)

def midi_to_name(m: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[m % 12]}{m // 12 - 1}"


PITCH_TOL_SEMITONES = 0.5     
HOLD_CONFIRM_FRAMES = 3       
MIN_CONFIDENCE = 0.80         
MIN_FREQ = 80                 
MAX_FREQ = 1400               


SAMPLE_RATE = 44100
HOP_SIZE = 512
BUFFER_SIZE = 2048

pitch_o = aubio.pitch("yin", BUFFER_SIZE, HOP_SIZE, SAMPLE_RATE)
pitch_o.set_unit("Hz")
pitch_o.set_silence(-40)      # dB threshold


idx = 0
confirm = 0

def expected_midi_at(i: int) -> int:
    return int(expected[i]["pitches_midi"][0])

print("Starting real-time pitch + simple alignment...")
print("Play the melody clearly, one note at a time to start.")
print("Press Ctrl+C to stop.\n")

def audio_callback(indata, frames, time_info, status):
    global idx, confirm

    if status:
        return

    if idx >= len(expected):
        return

    # Convert audio block to aubio float array (mono)
    x = indata[:, 0].astype(np.float32)

    # aubio wants a vector length hop_size; sounddevice gives exactly that if blocksize=HOP_SIZE
    f0 = float(pitch_o(x)[0])
    conf = float(pitch_o.get_confidence())

    if f0 <= 0 or conf < MIN_CONFIDENCE or f0 < MIN_FREQ or f0 > MAX_FREQ:
        confirm = 0
        return

    midi_est = hz_to_midi(f0)
    midi_round = int(round(midi_est))

    exp_midi = expected_midi_at(idx)
    diff = abs(midi_est - exp_midi)

    # Simple match
    if diff <= PITCH_TOL_SEMITONES:
        confirm += 1
        if confirm >= HOLD_CONFIRM_FRAMES:
            # Advance to next expected note
            exp_name = expected[idx]["pitches_name"][0]
            print(f"Matched idx={idx}: expected {exp_name} | heard {midi_to_name(midi_round)} (Hz={f0:.1f}, conf={conf:.2f})")
            idx += 1
            confirm = 0
    else:
        confirm = 0

# Stream with small blocksize for low latency
try:
    with sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=HOP_SIZE,
        callback=audio_callback,
    ):
        while idx < len(expected):
            time.sleep(0.05)

    print("\nDone! Reached end of expected note list.")

except KeyboardInterrupt:
    print("\nStopped.")