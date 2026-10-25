#!/usr/bin/env python3
"""Luhya V3: speaker-disjoint, gender/S3QA-balanced splits.

Dominant dialects are coverage constraints only.
No dialect balancing is performed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DOMINANT = ("Kabarasi", "Wanga", "Bukusu", "Butsotso")
LEVELS = ("low", "medium", "high")


def normalize_dialect(value):
    s = str(value).strip().lower()
    if "kabarasi" in s:
        return "Kabarasi"
    if "wanga" in s:
        return "Wanga"
    if "bukusu" in s:
        return "Bukusu"
    if "butsotso" in s:
        return "Butsotso"
    if "samia" in s:
        return "Samia"
    if "banyala" in s:
        return "Banyala"
    return "Unknown"


def load_data(input_dir):
    paths = sorted(Path(input_dir).glob("part_*.csv"))
    if not paths:
        raise FileNotFoundError("No S3QA CSV files found")

    print(f"Loading {len(paths)} S3QA files...", flush=True)

    df = pd.concat(
        (pd.read_csv(p, low_memory=False) for p in paths),
        ignore_index=True,
    )

    required = [
        "speaker_name", "dialect", "gender",
        "duration_seconds", "s3qa_mean",
        "arrow_file", "row_index",
    ]

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if "status" in df.columns:
        df = df[
            df["status"].astype(str).str.lower().eq("ok")
        ].copy()

    df["dialect"] = df["dialect"].map(normalize_dialect)
    df["gender"] = (
        df["gender"].astype(str).str.strip().str.upper()
    )
    df["duration_seconds"] = pd.to_numeric(
        df["duration_seconds"], errors="coerce"
    )
    df["s3qa_mean"] = pd.to_numeric(
        df["s3qa_mean"], errors="coerce"
    )

    valid = (
        df["speaker_name"].notna()
        & df["dialect"].ne("Unknown")
        & df["gender"].isin(["F", "M"])
        & df["duration_seconds"].gt(0)
        & np.isfinite(df["duration_seconds"])
        & np.isfinite(df["s3qa_mean"])
        & df["arrow_file"].notna()
        & df["row_index"].notna()
    )

    if not valid.all():
        raise ValueError(
            f"{int((~valid).sum())} invalid rows. "
            "Audit the source data first."
        )

    if df.duplicated(["arrow_file", "row_index"]).any():
        raise ValueError("Duplicate recording IDs")

    # Canonical speaker IDs before splitting.
    df["speaker_name_original"] = df["speaker_name"]

    df["speaker_name"] = (
        df["speaker_name"]
        .astype(str)
        .str.replace(r"\s+\(\d+\)$", "", regex=True)
        .str.strip()
    )

    df = df.sort_values(
        ["speaker_name", "arrow_file", "row_index"],
        kind="stable",
    ).reset_index(drop=True)

    q25, q75 = df["s3qa_mean"].quantile([0.25, 0.75])

    df["s3qa_level"] = np.where(
        df["s3qa_mean"] <= q25,
        "low",
        np.where(
            df["s3qa_mean"] >= q75,
            "high",
            "medium",
        ),
    )

    return df, float(q25), float(q75)


def build_speaker_table(df):
    rows = []

    for speaker, group in df.groupby(
        "speaker_name", sort=True
    ):
        genders = group["gender"].unique()
        if len(genders) != 1:
            raise ValueError(
                f"Inconsistent gender for {speaker}"
            )

        row = {
            "speaker": speaker,
            "gender": genders[0],
            "seconds": group["duration_seconds"].sum(),
        }

        for d in DOMINANT:
            row[f"d_{d}"] = group.loc[
                group["dialect"].eq(d),
                "duration_seconds",
            ].sum()

        for q in LEVELS:
            row[f"q_{q}"] = group.loc[
                group["s3qa_level"].eq(q),
                "duration_seconds",
            ].sum()

        rows.append(row)

    return pd.DataFrame(rows).set_index("speaker")



def select_speakers(
    table, target_seconds, available,
    rng, min_speakers, min_gender_speakers,
):
    """Select speakers while minimizing excluded audio."""

    chosen = []
    remaining = set(available)

    def choose(candidates):
        candidates = sorted(candidates)

        if not candidates:
            raise RuntimeError("No eligible speakers")

        # Prefer speakers with less total audio.
        # Randomized among the 5 smallest for diversity.
        candidates = sorted(
            candidates,
            key=lambda s: (table.at[s, "seconds"], s),
        )[:5]

        return candidates[int(rng.integers(len(candidates)))]

    for gender in ("F", "M"):
        for _ in range(min_gender_speakers):
            candidates = [
                s for s in remaining
                if table.at[s, "gender"] == gender
            ]
            speaker = choose(candidates)
            chosen.append(speaker)
            remaining.remove(speaker)

    for dialect in DOMINANT:
        if any(
            table.at[s, f"d_{dialect}"] > 0
            for s in chosen
        ):
            continue

        candidates = [
            s for s in remaining
            if table.at[s, f"d_{dialect}"] > 0
        ]

        speaker = choose(candidates)
        chosen.append(speaker)
        remaining.remove(speaker)

    while len(chosen) < min_speakers:
        candidates = list(remaining)

        f_count = sum(
            table.at[s, "gender"] == "F"
            for s in chosen
        )

        preferred = (
            "F" if f_count < min_speakers * 0.64
            else "M"
        )

        preferred_candidates = [
            s for s in candidates
            if table.at[s, "gender"] == preferred
        ]

        if preferred_candidates:
            candidates = preferred_candidates

        speaker = choose(candidates)
        chosen.append(speaker)
        remaining.remove(speaker)

    return sorted(chosen)


def select_recordings(
    pool, target_seconds, female_ratio,
    rng, tolerance_seconds,
):
    """Allocate durations across gender x S3QA strata."""

    strata = {
        ("F", "low"): female_ratio * 0.25,
        ("F", "medium"): female_ratio * 0.50,
        ("F", "high"): female_ratio * 0.25,
        ("M", "low"): (1 - female_ratio) * 0.25,
        ("M", "medium"): (1 - female_ratio) * 0.50,
        ("M", "high"): (1 - female_ratio) * 0.25,
    }

    selected_indices = []
    used = set()

    # Seed each speaker with one recording.
    for speaker in sorted(pool["speaker_name"].unique()):
        subset = pool[
            pool["speaker_name"].eq(speaker)
        ]
        ix = subset["duration_seconds"].idxmin()
        selected_indices.append(ix)
        used.add(ix)

    def current_totals():
        if not selected_indices:
            return {}, 0.0

        part = pool.loc[selected_indices]

        values = part.groupby(
            ["gender", "s3qa_level"]
        )["duration_seconds"].sum().to_dict()

        return values, part["duration_seconds"].sum()

    totals, current = current_totals()

    if current > target_seconds + tolerance_seconds:
        raise RuntimeError(
            "Speaker seeding exceeds target duration"
        )

    # Deterministic randomized ordering within each stratum.
    queues = {}

    for key in sorted(strata):
        subset = pool[
            pool["gender"].eq(key[0])
            & pool["s3qa_level"].eq(key[1])
        ]

        indices = subset.index.to_numpy()
        rng.shuffle(indices)
        queues[key] = indices.tolist()

    positions = {key: 0 for key in strata}

    while current < target_seconds - tolerance_seconds:
        deficits = {
            key: target_seconds * proportion
            - totals.get(key, 0.0)
            for key, proportion in strata.items()
        }

        ordered = sorted(
            deficits,
            key=lambda key: (-deficits[key], key),
        )

        picked = None
        picked_key = None

        for key in ordered:
            queue = queues[key]
            position = positions[key]

            while position < len(queue):
                ix = queue[position]
                position += 1

                if ix in used:
                    continue

                duration = float(
                    pool.at[ix, "duration_seconds"]
                )

                if (
                    current + duration
                    <= target_seconds + tolerance_seconds
                ):
                    picked = ix
                    picked_key = key
                    break

            positions[key] = position

            if picked is not None:
                break

        if picked is None:
            break

        used.add(picked)
        selected_indices.append(picked)

        duration = float(
            pool.at[picked, "duration_seconds"]
        )

        current += duration
        totals[picked_key] = (
            totals.get(picked_key, 0.0) + duration
        )

    result = pool.loc[selected_indices].copy()

    return result


def metrics(df, female_ratio):
    seconds = df["duration_seconds"].sum()

    gender_seconds = df.groupby(
        "gender"
    )["duration_seconds"].sum()

    quality_seconds = df.groupby(
        "s3qa_level"
    )["duration_seconds"].sum()

    female = gender_seconds.get("F", 0) / seconds

    quality = {
        q: quality_seconds.get(q, 0) / seconds
        for q in LEVELS
    }

    gender_error = abs(female - female_ratio)

    quality_error = (
        abs(quality["low"] - 0.25)
        + abs(quality["medium"] - 0.50)
        + abs(quality["high"] - 0.25)
    )

    return gender_error, quality_error


def summary(df):
    seconds = df["duration_seconds"].sum()

    return {
        "recordings": int(len(df)),
        "speakers": int(df["speaker_name"].nunique()),
        "hours": round(seconds / 3600, 5),
        "female_share": round(
            df.loc[
                df["gender"].eq("F"),
                "duration_seconds",
            ].sum() / seconds,
            5,
        ),
        "s3qa_hours": {
            q: round(
                df.loc[
                    df["s3qa_level"].eq(q),
                    "duration_seconds",
                ].sum() / 3600,
                5,
            )
            for q in LEVELS
        },
        "dominant_dialect_hours": {
            d: round(
                df.loc[
                    df["dialect"].eq(d),
                    "duration_seconds",
                ].sum() / 3600,
                5,
            )
            for d in DOMINANT
        },
    }


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)

    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-hours", type=float, default=5)
    parser.add_argument("--test-hours", type=float, default=10)
    parser.add_argument("--attempts", type=int, default=200)
    parser.add_argument(
        "--min-speakers", type=int, default=12
    )
    parser.add_argument(
        "--min-gender-speakers", type=int, default=3
    )
    parser.add_argument(
        "--tolerance-seconds", type=float, default=30
    )

    args = parser.parse_args()

    df, q25, q75 = load_data(args.input_dir)

    table = build_speaker_table(df)

    female_ratio = (
        df.loc[
            df["gender"].eq("F"),
            "duration_seconds",
        ].sum() / df["duration_seconds"].sum()
    )

    print(
        f"Female target: {female_ratio:.4%}",
        flush=True,
    )

    rng = np.random.default_rng(args.seed)

    all_speakers = sorted(table.index.tolist())

    best = None

    for attempt in range(args.attempts):
        try:
            val_speakers = select_speakers(
                table,
                args.val_hours * 3600,
                all_speakers,
                rng,
                args.min_speakers,
                args.min_gender_speakers,
            )

            remaining = sorted(
                set(all_speakers) - set(val_speakers)
            )

            test_speakers = select_speakers(
                table,
                args.test_hours * 3600,
                remaining,
                rng,
                args.min_speakers,
                args.min_gender_speakers,
            )

            val_pool = df[
                df["speaker_name"].isin(val_speakers)
            ]

            test_pool = df[
                df["speaker_name"].isin(test_speakers)
            ]

            val = select_recordings(
                val_pool,
                args.val_hours * 3600,
                female_ratio,
                rng,
                args.tolerance_seconds,
            )

            test = select_recordings(
                test_pool,
                args.test_hours * 3600,
                female_ratio,
                rng,
                args.tolerance_seconds,
            )

            if abs(
                val["duration_seconds"].sum()
                - args.val_hours * 3600
            ) > args.tolerance_seconds:
                continue

            if abs(
                test["duration_seconds"].sum()
                - args.test_hours * 3600
            ) > args.tolerance_seconds:
                continue

            if any(
                not val["dialect"].eq(d).any()
                or not test["dialect"].eq(d).any()
                for d in DOMINANT
            ):
                continue

            heldout = (
                set(val_speakers) | set(test_speakers)
            )

            train = df[
                ~df["speaker_name"].isin(heldout)
            ]

            if any(
                not train["dialect"].eq(d).any()
                for d in DOMINANT
            ):
                continue

            vg, vq = metrics(val, female_ratio)
            tg, tq = metrics(test, female_ratio)

            unused = (
                df["duration_seconds"].sum()
                - train["duration_seconds"].sum()
                - val["duration_seconds"].sum()
                - test["duration_seconds"].sum()
            ) / 3600

            score = (
                4 * (vg + tg)
                + 2 * (vq + tq)
                + 0.02 * unused
            )

            if best is None or score < best[0]:
                best = (
                    score, train.copy(),
                    val.copy(), test.copy(),
                    unused, attempt + 1,
                )

                print(
                    f"Attempt {attempt+1}: "
                    f"score={score:.5f}, "
                    f"train={train.duration_seconds.sum()/3600:.2f}h, "
                    f"unused={unused:.2f}h",
                    flush=True,
                )

        except RuntimeError:
            continue

    if best is None:
        raise RuntimeError(
            "No feasible V3 solution found. "
            "No output files were created."
        )

    score, train, val, test, unused, attempt = best

    speaker_sets = [
        set(x["speaker_name"])
        for x in (train, val, test)
    ]

    assert all(
        speaker_sets[i].isdisjoint(speaker_sets[j])
        for i, j in ((0, 1), (0, 2), (1, 2))
    )

    all_selected = pd.concat(
        [train, val, test], ignore_index=True
    )

    assert not all_selected.duplicated(
        ["arrow_file", "row_index"]
    ).any()

    out = Path(args.output_dir)

    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {out}"
        )

    out.mkdir(parents=True, exist_ok=True)

    files = {
        "train_pool.csv": train,
        "validation_5h.csv": val,
        "test_10h.csv": test,
    }

    hashes = {}

    for name, part in files.items():
        path = out / name
        part.to_csv(path, index=False)
        hashes[name] = sha256_file(path)

    report = {
        "version": 5,
        "seed": args.seed,
        "attempts": args.attempts,
        "best_attempt": attempt,
        "score": score,
        "q25": q25,
        "q75": q75,
        "female_target": female_ratio,
        "dominant_dialects": list(DOMINANT),
        "speaker_disjoint": True,
        "train": summary(train),
        "validation": summary(val),
        "test": summary(test),
        "unused_held_out_hours": round(unused, 5),
        "sha256": hashes,
    }

    (out / "split_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
