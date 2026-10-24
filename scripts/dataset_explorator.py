#!/usr/bin/env python3
"""Audit of the Luhya-ASR-Data-subset-642H dataset.

Features:
- dialect normalization (Abanyala/banyala/... -> Nyala, Abawanga -> Wanga)
- detection of rows with shifted metadata (numeric `domain`)
- per-speaker statistics (speakers per group, top contributors, inconsistencies)
- durations in hours if a duration column exists (or with --audio-duration)
- transcript length, empty transcripts and duplicates
- reads only the useful columns (no audio download by default)
- report saved to a .txt file, with an optional JSON export

Usage:
    python audit_luhya.py
    python audit_luhya.py --max-examples 5000           # quick test
    python audit_luhya.py --txt my_report.txt           # custom text report path
    python audit_luhya.py --json audit.json             # JSON export of the results
    python audit_luhya.py --audio-duration              # duration read from audio (slow)
"""
import argparse
import hashlib
import io
import json
import statistics
from collections import Counter, defaultdict

from datasets import Audio, load_dataset

DATASET_NAME = "Digital-Divide-Data/Luhya-ASR-Data-subset-642H"
TEXT_COLUMNS = ["transcription", "text", "sentence", "transcript", "normalized_text"]
DURATION_COLUMNS = ["duration", "audio_duration", "duration_s", "length"]
AGE_BUCKETS = [(18, 25), (26, 35), (36, 50), (51, 120)]


def normalize_dialect(value):
    """Merge spelling variants of the same dialect."""
    s = str(value).strip().lower()
    if "nyala" in s:
        return "Nyala"
    if "wanga" in s:
        return "Wanga"
    return s.title()


def age_bucket(value):
    """Map an age to its age group."""
    try:
        age = int(float(str(value).strip()))
    except ValueError:
        return "invalid"
    for low, high in AGE_BUCKETS:
        if low <= age <= high:
            return f"{low}-{high if high < 120 else '+'}"
    return "out of range"


def is_shifted(row):
    """A purely numeric `domain` signals a column shift in the source files."""
    return str(row.get("domain", "")).strip().isdigit()


def clean(value):
    """Return the stripped string, or None if the value is null or empty."""
    return str(value).strip() if value is not None and str(value).strip() else None


class Group:
    """Aggregate per category: examples, distinct speakers, seconds of audio."""

    def __init__(self):
        self.examples = 0
        self.seconds = 0.0
        self.speakers = set()

    def add(self, speaker, seconds):
        self.examples += 1
        self.seconds += seconds
        if speaker:
            self.speakers.add(speaker)

    def to_dict(self):
        return {"examples": self.examples, "speakers": len(self.speakers), "hours": self.seconds / 3600}


def audio_seconds(row):
    """Duration read from the audio file header (no decoding), used with --audio-duration."""
    import soundfile as sf

    audio = row.get("audio") or {}
    data = audio.get("bytes")
    if data:
        info = sf.info(io.BytesIO(data))
    elif audio.get("path"):
        info = sf.info(audio["path"])
    else:
        return 0.0
    return info.frames / info.samplerate


def percentile(sorted_values, q):
    if not sorted_values:
        return 0
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--audio-duration", action="store_true",
                        help="read durations from the audio files (downloads all audio, slow)")
    parser.add_argument("--txt", default="audit_report.txt", help="path of the text report")
    parser.add_argument("--json", default=None, help="path of an optional JSON export")
    args = parser.parse_args()

    # Every report line is printed and also kept to be written to the .txt file.
    lines = []

    def emit(text=""):
        print(text, flush=True)
        lines.append(text)

    print(f"Loading: {args.dataset} (split {args.split})", flush=True)
    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    features = list(ds.features) if ds.features else []
    text_col = next((c for c in TEXT_COLUMNS if c in features), None)
    dur_col = next((c for c in DURATION_COLUMNS if c in features), None)
    print(f"Columns: {features or 'unknown (features not exposed in streaming mode)'}")
    print(f"Text column: {text_col} | duration column: {dur_col}\n", flush=True)

    # Read only the useful columns, except audio when durations come from the files.
    if features:
        wanted = {"speaker_name", "dialect", "gender", "age", "domain", "sub_domain", text_col, dur_col}
        if args.audio_duration and "audio" in features:
            ds = ds.cast_column("audio", Audio(decode=False))
            wanted.add("audio")
        ds = ds.select_columns([c for c in features if c in wanted])

    use_duration = bool(dur_col) or args.audio_duration
    by_dialect, by_gender, by_age = defaultdict(Group), defaultdict(Group), defaultdict(Group)
    raw_dialects = Counter()
    speaker_examples = Counter()
    speaker_seconds = Counter()
    speaker_attrs = defaultdict(lambda: {"dialect": set(), "gender": set(), "age": set()})
    missing = Counter()
    shifted = 0
    text_lengths = []
    text_hashes = Counter()
    empty_texts = 0
    total = 0

    for row in ds:
        total += 1
        speaker = clean(row.get("speaker_name"))
        dialect_raw = clean(row.get("dialect"))
        gender = clean(row.get("gender"))
        age = clean(row.get("age"))

        # Count missing metadata fields.
        for name, value in [("speaker_name", speaker), ("dialect", dialect_raw), ("gender", gender),
                            ("age", age), ("domain", clean(row.get("domain"))),
                            ("sub_domain", clean(row.get("sub_domain")))]:
            if value is None:
                missing[name] += 1

        if is_shifted(row):
            shifted += 1

        # Duration of the example, from the duration column or from the audio header.
        seconds = 0.0
        if dur_col and row.get(dur_col) is not None:
            seconds = float(row[dur_col])
        elif args.audio_duration:
            seconds = audio_seconds(row)

        dialect = normalize_dialect(dialect_raw) if dialect_raw else "missing"
        if dialect_raw:
            raw_dialects[dialect_raw] += 1
        by_dialect[dialect].add(speaker, seconds)
        by_gender[gender or "missing"].add(speaker, seconds)
        by_age[age_bucket(age) if age else "missing"].add(speaker, seconds)

        # Per-speaker totals, and attributes used to detect inconsistent metadata.
        if speaker:
            speaker_examples[speaker] += 1
            speaker_seconds[speaker] += seconds
            for key, value in [("dialect", dialect), ("gender", gender), ("age", age)]:
                if value:
                    speaker_attrs[speaker][key].add(value)

        # Transcript statistics: length, empty transcripts and duplicates.
        if text_col:
            text = clean(row.get(text_col))
            if text is None:
                empty_texts += 1
            else:
                text_lengths.append(len(text.split()))
                text_hashes[hashlib.md5(text.lower().encode()).hexdigest()] += 1

        if total % 20000 == 0:
            print(f"  {total:,} examples analyzed...", flush=True)
        if args.max_examples and total >= args.max_examples:
            break

    total_seconds = sum(speaker_seconds.values())

    def table(title, groups):
        emit(f"\n--- {title} ---")
        header = f"{'Category':<22}{'Examples':>10}{'Share':>9}{'Speakers':>10}"
        header += f"{'Hours':>9}" if use_duration else ""
        emit(header)
        for name, g in sorted(groups.items(), key=lambda kv: -kv[1].examples):
            line = f"{name:<22}{g.examples:>10,}{100 * g.examples / total:>8.2f}%{len(g.speakers):>10,}"
            line += f"{g.seconds / 3600:>9.1f}" if use_duration else ""
            emit(line)

    emit("=" * 66)
    emit("LUHYA DATASET AUDIT REPORT")
    emit("=" * 66)
    emit(f"Dataset: {args.dataset} | split: {args.split}")
    emit(f"\nExamples: {total:,} | Speakers: {len(speaker_examples):,}")
    if use_duration:
        emit(f"Total duration: {total_seconds / 3600:,.1f} h | average: {total_seconds / total:.1f} s/example")

    emit("\n--- Raw dialect labels -> normalized ---")
    for name, count in raw_dialects.most_common():
        emit(f"{name:<22}{count:>10,}  ->  {normalize_dialect(name)}")

    table("DIALECTS (normalized)", by_dialect)
    table("GENDER", by_gender)
    table("AGE GROUPS", by_age)

    emit("\n--- MISSING METADATA ---")
    emit(str(dict(missing)) if missing else "none")

    emit("\n--- ROWS WITH SHIFTED METADATA (numeric domain) ---")
    emit(f"{shifted:,} ({100 * shifted / total:.2f}%)")

    emit("\n--- SPEAKER CONTRIBUTION ---")
    counts = sorted(speaker_examples.values())
    emit(f"Examples per speaker: min {counts[0]:,} | median {int(statistics.median(counts)):,} | "
         f"max {counts[-1]:,}")
    for speaker, n in speaker_examples.most_common(5):
        extra = f" | {speaker_seconds[speaker] / 3600:.1f} h" if use_duration else ""
        emit(f"  {speaker}: {n:,} ({100 * n / total:.2f}%){extra}")
    top10 = sum(n for _, n in speaker_examples.most_common(10))
    emit(f"Share of the 10 largest contributors: {100 * top10 / total:.2f}%")

    # Speakers that appear with several dialects, genders or ages.
    inconsistent = {s: {k: sorted(v) for k, v in a.items() if len(v) > 1}
                    for s, a in speaker_attrs.items() if any(len(v) > 1 for v in a.values())}
    emit("\n--- SPEAKERS WITH INCONSISTENT METADATA ---")
    emit(f"{len(inconsistent)} speaker(s)")
    for speaker, issues in list(inconsistent.items())[:10]:
        emit(f"  {speaker}: {issues}")

    if text_col:
        lengths = sorted(text_lengths)
        duplicates = sum(c - 1 for c in text_hashes.values() if c > 1)
        emit(f"\n--- TRANSCRIPTS ({text_col}) ---")
        emit(f"Empty: {empty_texts:,} | duplicates: {duplicates:,} | unique: {len(text_hashes):,}")
        if lengths:
            emit(f"Words per transcript: mean {statistics.mean(lengths):.1f} | median "
                 f"{statistics.median(lengths):.0f} | p95 {percentile(lengths, 0.95)} | max {lengths[-1]}")
    else:
        emit("\nNo transcript column detected: adjust TEXT_COLUMNS.")

    emit("\nAudit finished.")

    # Save the report as a text file.
    with open(args.txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nText report saved to {args.txt}")

    # Optional JSON export, to reuse the results in other tools.
    if args.json:
        result = {
            "total_examples": total,
            "speakers": len(speaker_examples),
            "total_hours": total_seconds / 3600 if use_duration else None,
            "shifted_rows": shifted,
            "missing": dict(missing),
            "dialects": {k: g.to_dict() for k, g in by_dialect.items()},
            "gender": {k: g.to_dict() for k, g in by_gender.items()},
            "age_groups": {k: g.to_dict() for k, g in by_age.items()},
            "inconsistent_speakers": inconsistent,
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Results exported to {args.json}")


if __name__ == "__main__":
    main()
