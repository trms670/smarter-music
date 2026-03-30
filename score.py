from music21 import converter, note, chord

# Load the score; repeats are ignored so each measure appears exactly once.
score = converter.parse("Bach_Minuet.mxl")

violin_part = score.parts[0]

def extract_raw_events(part, part_label: str, include_rests=True):
    """Iterate over Measure objects directly so measure numbers are correct
    after expandRepeats(), and use the global offset_q for ordering."""
    events = []
    from music21 import stream as m21stream
    for measure in part.getElementsByClass(m21stream.Measure):
        m = int(measure.number)
        for el in measure.notesAndRests:
            global_offset = float(measure.offset) + float(el.offset)
            if isinstance(el, note.Note):
                events.append({
                    "part": part_label,
                    "kind": "note",
                    "measure": m,
                    "beat": float(el.beat),
                    "offset_q": global_offset,
                    "duration_q": float(el.duration.quarterLength),
                    "pitches_midi": (int(el.pitch.midi),),
                    "pitches_name": (el.pitch.nameWithOctave,),
                })
            elif isinstance(el, chord.Chord):
                events.append({
                    "part": part_label,
                    "kind": "chord",
                    "measure": m,
                    "beat": float(el.beat),
                    "offset_q": global_offset,
                    "duration_q": float(el.duration.quarterLength),
                    "pitches_midi": tuple(int(p.midi) for p in el.pitches),
                    "pitches_name": tuple(p.nameWithOctave for p in el.pitches),
                })
            elif include_rests and isinstance(el, note.Rest):
                events.append({
                    "part": part_label,
                    "kind": "rest",
                    "measure": m,
                    "beat": float(el.beat),
                    "offset_q": global_offset,
                    "duration_q": float(el.duration.quarterLength),
                    "pitches_midi": tuple(),
                    "pitches_name": tuple(),
                })
    return events

violin_label = "Solo"
violin_raw = extract_raw_events(violin_part, violin_label, include_rests=True)

def sort_by_musical_time(events):
    # Sort by global offset only — unique across both repeat iterations
    return sorted(events, key=lambda e: e["offset_q"])

violin_sorted = sort_by_musical_time(violin_raw)

# Include notes and chords; flatten chords to their lowest pitch so the
# rest of the pipeline always sees single-pitch events.
def normalize_event(e):
    if e["kind"] == "chord":
        highest_midi = max(e["pitches_midi"])
        highest_name = max(zip(e["pitches_midi"], e["pitches_name"]))[1]
        return {**e, "kind": "note", "pitches_midi": (highest_midi,), "pitches_name": (highest_name,)}
    return e

expected = [normalize_event(e) for e in violin_sorted if e["kind"] in ("note", "chord")]

def _mk(measure, beat, name, midi, m_start, dur):
    return {"part": violin_label, "kind": "note", "measure": measure,
            "beat": float(beat), "offset_q": m_start + float(beat) - 1.0,
            "duration_q": dur, "pitches_midi": (midi,), "pitches_name": (name,)}

def _insert_after(lst, after_measure, measure_num, notes_spec, m_start):
    """Insert manually specified notes right after all events with measure <= after_measure."""
    idx = next((i for i, e in enumerate(lst) if e["measure"] > after_measure), len(lst))
    notes = [_mk(measure_num, beat, name, midi, m_start, dur)
             for beat, name, midi, dur in notes_spec]
    return lst[:idx] + notes + lst[idx:], idx

# --- Gap 1: measures 25 and 26 missing from MusicXML ---
_idx_25 = next((i for i, e in enumerate(expected) if e["measure"] > 24), len(expected))
_m25_start = expected[_idx_25 - 1]["offset_q"] + expected[_idx_25 - 1]["duration_q"]

# --- Gap 2: measure 29 missing from MusicXML ---
_idx_29 = next((i for i, e in enumerate(expected) if e["measure"] > 28), len(expected))
_m29_start = expected[_idx_29 - 1]["offset_q"] + expected[_idx_29 - 1]["duration_q"]

# Insert higher index first so lower index stays valid
expected = (expected[:_idx_29] +
            [_mk(29, 1.0, "D4",  62, _m29_start, 0.5),
             _mk(29, 1.5, "E4",  64, _m29_start, 0.5),
             _mk(29, 2.0, "F#4", 66, _m29_start, 0.5),
             _mk(29, 2.5, "G4",  67, _m29_start, 0.5),
             _mk(29, 3.0, "A4",  69, _m29_start, 0.5),
             _mk(29, 3.5, "B4",  71, _m29_start, 0.5)] +
            expected[_idx_29:])

expected = (expected[:_idx_25] +
            [_mk(25, 1.0, "D5",  74, _m25_start, 1.0),
             _mk(25, 2.0, "G4",  67, _m25_start, 0.5),
             _mk(25, 2.5, "F#4", 66, _m25_start, 0.5),
             _mk(25, 3.0, "G4",  67, _m25_start, 1.0),
             _mk(26, 1.0, "E5",  76, _m25_start + 3.0, 1.0),
             _mk(26, 2.0, "G4",  67, _m25_start + 3.0, 0.5),
             _mk(26, 2.5, "F#4", 66, _m25_start + 3.0, 0.5),
             _mk(26, 3.0, "G4",  67, _m25_start + 3.0, 1.0)] +
            expected[_idx_25:])

# Renumber measures sequentially so there are no gaps in the display
_measure_order = sorted({e["measure"] for e in expected})
_measure_remap = {orig: new for new, orig in enumerate(_measure_order, start=1)}
for e in expected:
    e["measure"] = _measure_remap[e["measure"]]

