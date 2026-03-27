from music21 import converter, note, chord

# Load the score
score = converter.parse("Bach_Minuet.mxl")

# print("Number of parts:", len(score.parts))
# for i, p in enumerate(score.parts):
#     # partName and id are often set; instrument is sometimes set
#     inst = p.getInstrument(returnDefault=False)
#     inst_name = inst.instrumentName if inst and inst.instrumentName else None
#     print(f"[{i}] partName={p.partName!r}, id={p.id!r}, instrument={inst_name!r}")

violin_part = score.parts[0]

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

# print("Violin raw events:", len(violin_raw))
# print("First 10 violin events:")
# for e in violin_raw[:10]:
#     print(e)

def sort_by_musical_time(events):
    return sorted(events, key=lambda e: (e["measure"], e["beat"], e["offset_q"]))

violin_sorted = sort_by_musical_time(violin_raw)

print("First 10 violin sorted (measure, beat, offset_q):")
for e in violin_sorted[:10]:
    print(e["measure"], e["beat"], e["offset_q"], e["kind"], e["pitches_name"])

# Keep only note events for now (you can add rests/chords later)
expected = [e for e in violin_sorted if e["kind"] == "note"]

# Give each expected note an index (useful for debugging)
for i, e in enumerate(expected[:10]):
    print(i, e["measure"], e["beat"], e["pitches_name"][0])