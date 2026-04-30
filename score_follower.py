"""
score_follower.py
-----------------
Real-time HMM-based score follower using online beam search.

Each detected note (MIDI pitch + wall-clock timestamp) is fed to
`follower.observe()`, which returns the current estimated score position.

Algorithm
---------
States      : one per note in `expected` (N states total)
Emission    : Gaussian over MIDI semitone difference
              + Gaussian over inter-onset interval (IOI) at current tempo
Transitions : forward-biased; allow small backward jumps and note skips
Search      : beam search — maintain K hypotheses, prune to K after each step

Robustness
----------
* Cold start      : first pitch scores all N states, keeps top K → any start measure
* Locked display  : reported position only changes when confidence ≥ DISPLAY_MIN_CONF,
                    suppressing single-note jitter
* Pause (soft)    : gap ≥ PAUSE_SOFT_RESET_S → inject global hypotheses + skip IOI,
                    keeping current beams but opening the model to a new location
* Pause (hard)    : gap ≥ SILENCE_RESET_S → full reset, re-initialise on next note
* Low-conf rescue : inject global hypotheses when all beams become unlikely
* Fixed tempo     : set FIXED_BPM to lock tempo; tightens IOI emission significantly
                    Set to None to fall back to adaptive EMA tempo tracking
"""

import threading
from collections import deque
from typing import Optional

import numpy as np

from score import expected

# Tuning constants
BEAM_K          = 25 

SIGMA_PITCH     = 1.0 

USE_IOI         = True    
IOI_WEIGHT      = 0.6    
SIGMA_IOI_REL   = 0.35    
SIGMA_IOI_FLOOR = 0.07  

# Tempo
FIXED_BPM       = None    
INIT_BPM        = 100.0  
TEMPO_ALPHA     = 0.20 

PAUSE_SOFT_RESET_S = 1.5  
SILENCE_RESET_S    = 5.0  

# Display stability
DISPLAY_MIN_CONF = 0.55

LOG_CONF_THRESH  = -20.0 

# N-gram context window
CONTEXT_SIZE   = 3  
CONTEXT_DECAY  = 0.5 
CONTEXT_WEIGHT = 0.8  

# ransition model

_RAW_TRANS = [
    (-3, 0.002),
    (-2, 0.005),
    (-1, 0.015),
    ( 0, 0.080),   # stay: repeated note or pitch-detector glitch
    ( 1, 0.620),   # advance one note: normal playing
    ( 2, 0.150),   # skip one note: missed detection
    ( 3, 0.070),   
    ( 4, 0.030),   
    ( 5, 0.018),   
    ( 6, 0.010),   
]
_W_TOT       = sum(w for _, w in _RAW_TRANS)
_TRANSITIONS = [(d, np.log(w / _W_TOT)) for d, w in _RAW_TRANS]

_N    = len(expected)
_MIDI = np.array([e["pitches_midi"][0] for e in expected], dtype=np.float64)
_DUR  = np.array([e["duration_q"]      for e in expected], dtype=np.float64)

_NO_PREV = -1   

# Emission helpers
def _pitch_lp_state(midi_obs: float, s: int) -> float:
    diff = abs(midi_obs - _MIDI[s])
    return -(diff * diff) / (2.0 * SIGMA_PITCH * SIGMA_PITCH)


def _pitch_lp_all(midi_obs: float) -> np.ndarray:
    diff = np.abs(midi_obs - _MIDI)
    return -(diff * diff) / (2.0 * SIGMA_PITCH * SIGMA_PITCH)


# Adaptive tempo tracker
class _Tempo:
    def __init__(self, bpm: float) -> None:
        self.qps: float = bpm / 60.0   # quarter-notes per second

    def update(self, obs_ioi: float, dur_q: float) -> None:
        if obs_ioi <= 0.0 or dur_q <= 0.0:
            return
        new_qps = float(np.clip(dur_q / obs_ioi, 40.0 / 60.0, 220.0 / 60.0))
        self.qps = (1.0 - TEMPO_ALPHA) * self.qps + TEMPO_ALPHA * new_qps

    @property
    def bpm(self) -> float:
        return self.qps * 60.0


# ScoreFollower
class ScoreFollower:

    def __init__(self, score_events: list, beam_k: int = BEAM_K) -> None:
        self._score  = score_events
        self._n      = len(score_events)
        self._k      = beam_k
        self._lock   = threading.Lock()

        init_bpm = FIXED_BPM if FIXED_BPM is not None else INIT_BPM
        self._tempo        = _Tempo(init_bpm)
        self._tempo_locked = FIXED_BPM is not None

        # Each beam: [state_idx, log_prob, prev_state]
        self._beams: list  = []
        self._ready: bool  = False
        self._last_ts: Optional[float] = None

        # Confidence-gated display state
        self._display_idx: int = _NO_PREV   # -1 until first confident lock

        # Sliding window of recent MIDI observations for N-gram context scoring
        self._obs_buf: deque = deque(maxlen=CONTEXT_SIZE)


    def reset(self) -> None:
        """Discard all hypotheses and reinitialise on the next observation."""
        with self._lock:
            self._reset_state()

    def observe(self, midi_est: float, timestamp: float) -> dict:
        """
        Update the position estimate with a new note observation.

        Parameters
        ----------
        midi_est  : fractional MIDI pitch (e.g. 69.3 for A4-ish)
        timestamp : wall-clock seconds (time.time())

        Returns
        -------
        dict with keys: idx, measure, beat, confidence, locked, bpm
        """
        with self._lock:
            return self._observe_locked(midi_est, timestamp)


    def _observe_locked(self, midi_est: float, timestamp: float) -> dict:
        obs_ioi   = 0.0
        soft_reset = False

        if self._last_ts is not None:
            gap = timestamp - self._last_ts
            if gap >= SILENCE_RESET_S:
                self._reset_state()
            elif gap >= PAUSE_SOFT_RESET_S:
                # Performer probably paused and may have jumped — widen the beam
                # by injecting globally-scored states, and skip IOI for this note.
                soft_reset = True
            else:
                obs_ioi = gap
        self._last_ts = timestamp

        # Cold start
        if not self._ready:
            log_emit    = _pitch_lp_all(midi_est)
            top_idx     = np.argsort(log_emit)[-self._k:][::-1]
            self._beams = [[int(i), float(log_emit[i]), _NO_PREV] for i in top_idx]
            self._ready = True
            self._obs_buf.append(midi_est)
            return self._build_report()

        # Soft reset
        if soft_reset:
            self._obs_buf.clear()
            self._inject_global(midi_est)

        # Add current observation to the context buffer (after clearing if needed)
        self._obs_buf.append(midi_est)

        # Beam expansion
        # candidates[next_state] = [best_total_log_prob, best_prev_state]
        candidates: dict = {}

        for state, lp, _prev in self._beams:
            for delta, trans_lp in _TRANSITIONS:
                ns = state + delta
                if not (0 <= ns < self._n):
                    continue
                p_lp   = _pitch_lp_state(midi_est, ns)
                ctx_lp = self._context_lp(ns)
                # Skip IOI after a soft reset: the gap was a pause, not a note duration
                i_lp   = (self._ioi_lp(obs_ioi, state, delta)
                          if (USE_IOI and obs_ioi > 0 and not soft_reset) else 0.0)
                total  = lp + trans_lp + p_lp + CONTEXT_WEIGHT * ctx_lp + IOI_WEIGHT * i_lp
                if ns not in candidates or candidates[ns][0] < total:
                    candidates[ns] = [total, state]

        if not candidates:
            return self._build_report()

        sorted_c = sorted(candidates.items(), key=lambda x: -x[1][0])
        best_raw = sorted_c[0][1][0]

        # Shift so best beam = 0 to prevent float underflow
        self._beams = [[s, v[0] - best_raw, v[1]] for s, v in sorted_c[:self._k]]

        # Low-confidence recovery: blend in globally-scored hypotheses
        if best_raw < LOG_CONF_THRESH:
            self._inject_global(midi_est)

        # Tempo update from top beam (only on clean forward steps; skip if locked)
        if not self._tempo_locked and USE_IOI and obs_ioi > 0 and not soft_reset:
            bs, _, bps = self._beams[0]
            if bps != _NO_PREV:
                d = bs - bps
                if 1 <= d <= 4:
                    dur_q = float(np.sum(_DUR[bps: bps + d]))
                    self._tempo.update(obs_ioi, dur_q)

        return self._build_report()

    # N-gram context emission

    def _context_lp(self, ns: int) -> float:
        """
        Retroactive context score for candidate state ns.

        For each past observation in _obs_buf (oldest first, newest = current),
        score it against the expected note k steps before ns, weighted by
        CONTEXT_DECAY^k.  The current observation itself is NOT included here —
        it is already counted as p_lp in the caller.

        Example (CONTEXT_SIZE=3, DECAY=0.5):
            score += 0.50 * pitch_lp(obs[-2], ns-1)
            score += 0.25 * pitch_lp(obs[-3], ns-2)
        """
        if CONTEXT_SIZE <= 1 or len(self._obs_buf) < 2:
            return 0.0
        total  = 0.0
        weight = CONTEXT_DECAY
        obs    = list(self._obs_buf)   # oldest → newest; obs[-1] = current note
        for k in range(1, len(obs)):   # k steps back
            past_s = ns - k
            if past_s < 0:
                break
            total  += weight * _pitch_lp_state(obs[-(k + 1)], past_s)
            weight *= CONTEXT_DECAY
        return total

    # IOI emission

    def _ioi_lp(self, obs_ioi: float, prev_s: int, delta: int) -> float:
        """Log-probability of the observed IOI for the transition prev_s → prev_s+delta."""
        if delta <= 0 or prev_s < 0:
            return 0.0
        end     = min(prev_s + delta, self._n)
        dur_q   = float(np.sum(_DUR[prev_s:end]))
        exp_ioi = dur_q / self._tempo.qps
        if exp_ioi <= 0.0:
            return 0.0
        sigma = max(SIGMA_IOI_FLOOR, SIGMA_IOI_REL * exp_ioi)
        diff  = obs_ioi - exp_ioi
        return -(diff * diff) / (2.0 * sigma * sigma)

    # ── Global injection ───────────────────────────────────────────────────────

    def _inject_global(self, midi_est: float) -> None:
        """Blend top-K globally-scored states into beams."""
        log_emit = _pitch_lp_all(midi_est)
        n_inject = self._k // 2
        top_idx  = np.argsort(log_emit)[-n_inject:][::-1]
        PENALTY  = -5.0
        merged   = {s: (lp, ps) for s, lp, ps in self._beams}
        for i in top_idx:
            s       = int(i)
            cand_lp = float(log_emit[i]) + PENALTY
            if s not in merged or merged[s][0] < cand_lp:
                merged[s] = (cand_lp, _NO_PREV)
        sorted_m = sorted(merged.items(), key=lambda x: -x[1][0])
        best_lp  = sorted_m[0][1][0]
        self._beams = [[s, lp - best_lp, ps] for s, (lp, ps) in sorted_m[:self._k]]


    def _build_report(self) -> dict:
        if not self._beams:
            ev = self._score[0]
            return {"idx": 0, "measure": ev["measure"], "beat": ev["beat"],
                    "confidence": 0.0, "locked": False, "bpm": self._tempo.bpm}

        top_s = self._beams[0][0]
        lps   = np.array([b[1] for b in self._beams], dtype=float)
        prbs  = np.exp(lps - np.max(lps))
        conf  = float(prbs[0] / prbs.sum())

        # Update locked display position only when confidence is high enough
        locked = conf >= DISPLAY_MIN_CONF
        if locked:
            self._display_idx = top_s

        # If we have a locked position, show it; otherwise fall back to best guess
        show_s = self._display_idx if self._display_idx != _NO_PREV else top_s
        ev     = self._score[show_s]

        return {
            "idx":        show_s,
            "measure":    ev["measure"],
            "beat":       ev["beat"],
            "confidence": conf,
            "locked":     locked,
            "bpm":        self._tempo.bpm,
        }

    def _reset_state(self) -> None:
        self._beams       = []
        self._ready       = False
        self._last_ts     = None
        self._display_idx = _NO_PREV
        self._obs_buf.clear()
        init_bpm = FIXED_BPM if FIXED_BPM is not None else INIT_BPM
        self._tempo       = _Tempo(init_bpm)


follower = ScoreFollower(expected)
