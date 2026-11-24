#!/usr/bin/env python3
"""Reproducible recording-disjoint Luhya ASR splits."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from create_luhya_splits import (
    DOMINANT,
    LEVELS,
    load_data,
    summary,
    sha256_file,
)


def allocate_split(pool, hours, female_ratio, rng, tolerance):
    target = hours * 3600

    strata = {
        ("F", "low"): female_ratio * 0.25,
        ("F", "medium"): female_ratio * 0.50,
        ("F", "high"): female_ratio * 0.25,
        ("M", "low"): (1 - female_ratio) * 0.25,
        ("M", "medium"): (1 - female_ratio) * 0.50,
        ("M", "high"): (1 - female_ratio) * 0.25,
    }

    selected = []
    used = set()
    totals = {key: 0.0 for key in strata}
    current = 0.0

    # Ensure dominant dialect coverage.
    for dialect in DOMINANT:
        candidates = pool[pool["dialect"].eq(dialect)]
        if candidates.empty:
            raise RuntimeError(f"Missing dialect: {dialect}")

        indices = candidates.index.to_numpy().copy()
        rng.shuffle(indices)

        ix = int(indices[0])
        selected.append(ix)
        used.add(ix)

        row = pool.loc[ix]
        key = (row["gender"], row["s3qa_level"])
        duration = float(row["duration_seconds"])

        totals[key] += duration
        current += duration

    queues = {}

    for key in strata:
        subset = pool[
            pool["gender"].eq(key[0])
            & pool["s3qa_level"].eq(key[1])
        ]

        indices = subset.index.to_numpy().copy()
        rng.shuffle(indices)
        queues[key] = indices.tolist()

    positions = {key: 0 for key in strata}

    while current < target - tolerance:
        deficits = {
            key: target * proportion - totals[key]
            for key, proportion in strata.items()
        }

        picked = None
        picked_key = None

        for key in sorted(deficits, key=lambda k: -deficits[k]):
            queue = queues[key]

            while positions[key] < len(queue):
                ix = queue[positions[key]]
                positions[key] += 1

                if ix in used:
                    continue

                duration = float(pool.at[ix, "duration_seconds"])

                if current + duration <= target + tolerance:
                    picked = ix
                    picked_key = key
                    break

            if picked is not None:
                break

        if picked is None:
            break

        selected.append(picked)
        used.add(picked)

        duration = float(pool.at[picked, "duration_seconds"])
        current += duration
        totals[picked_key] += duration

    result = pool.loc[selected].copy()

    if abs(result["duration_seconds"].sum() - target) > tolerance:
        raise RuntimeError("Duration target not reached")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-hours", type=float, default=5)
    parser.add_argument("--test-hours", type=float, default=10)
    parser.add_argument("--tolerance-seconds", type=float, default=30)
    args = parser.parse_args()

    df, q25, q75 = load_data(args.input_dir)

    rng = np.random.default_rng(args.seed)

    female_ratio = (
        df.loc[df["gender"].eq("F"), "duration_seconds"].sum()
        / df["duration_seconds"].sum()
    )

    # Select evaluation recordings, not speakers.
    validation = allocate_split(
        df, args.val_hours, female_ratio,
        rng, args.tolerance_seconds,
    )

    remaining = df.drop(index=validation.index)

    test = allocate_split(
        remaining, args.test_hours, female_ratio,
        rng, args.tolerance_seconds,
    )

    train = remaining.drop(index=test.index).copy()

    parts = {
        "train_pool.csv": train,
        "validation_5h.csv": validation,
        "test_10h.csv": test,
    }

    combined = pd.concat(parts.values())

    assert not combined.duplicated(
        ["arrow_file", "row_index"]
    ).any()

    # Verify recording-disjoint splits.
    recording_sets = {
        name: set(zip(
            part["arrow_file"],
            part["row_index"],
        ))
        for name, part in parts.items()
    }

    names = list(recording_sets)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert recording_sets[names[i]].isdisjoint(
                recording_sets[names[j]]
            )

    out = Path(args.output_dir)

    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {out}"
        )

    out.mkdir(parents=True, exist_ok=True)

    hashes = {}

    for filename, part in parts.items():
        path = out / filename
        part.to_csv(path, index=False)
        hashes[filename] = sha256_file(path)

    report = {
        "version": 6,
        "protocol": "recording-disjoint",
        "speaker_disjoint": False,
        "seed": args.seed,
        "q25": q25,
        "q75": q75,
        "female_target": female_ratio,
        "train": summary(train),
        "validation": summary(validation),
        "test": summary(test),
        "sha256": hashes,
    }

    (out / "split_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
