from music21 import converter, note, chord

# Load the score and expand all repeat barlines so the note list reflects
# actual playback order (both iterations of repeated sections).
score = converter.parse("Bach_Minuet.mxl")
score_expanded = score.expandRepeats()

violin_part = score_expanded.parts[0]

def extract_raw_events(part, part_label: str, include_rests=True):
    events = []
    flat = part.flatten()

    for el in flat.recurse():
        m = int(el.measureNumber) if getattr(el, "measureNumber", None) else 0

        if isinstance(el, note.Note):
            events.append({
                "part": part_label,
                "kind": "note",
                "measure": m,
                "beat": float(el.beat),
                "offset_q": float(el.offset),
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
                "offset_q": float(el.offset),
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
                "offset_q": float(el.offset),
                "duration_q": float(el.duration.quarterLength),
                "pitches_midi": tuple(),
                "pitches_name": tuple(),
            })

    return events

violin_label = "Solo"
violin_raw = extract_raw_events(violin_part, violin_label, include_rests=True)

def sort_by_musical_time(events):
    return sorted(events, key=lambda e: (e["measure"], e["beat"], e["offset_q"]))

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
