#!/usr/bin/env python3
"""Representative, recording-disjoint Luhya ASR splits."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DOMINANT = ("Kabarasi", "Wanga", "Bukusu", "Butsotso")


def normalize_dialect(value):
    s = str(value).strip().lower()
    for d in (
        "Kabarasi", "Wanga", "Bukusu",
        "Butsotso", "Samia", "Banyala"
    ):
        if d.lower() in s:
            return d
    return "Unknown"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_data(input_dir):
    paths = sorted(Path(input_dir).glob("part_*.csv"))
    if not paths:
        raise FileNotFoundError("No part_*.csv files found")

    df = pd.concat(
        (pd.read_csv(p, low_memory=False) for p in paths),
        ignore_index=True,
    )

    required = {"arrow_file", "row_index", "duration_seconds"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["duration_seconds"] = pd.to_numeric(
        df["duration_seconds"], errors="coerce"
    )
    df["row_index"] = pd.to_numeric(
        df["row_index"], errors="coerce"
    )

    valid = (
        df["arrow_file"].notna()
        & df["row_index"].notna()
        & np.isfinite(df["row_index"])
        & df["row_index"].ge(0)
        & df["row_index"].mod(1).eq(0)
        & df["duration_seconds"].gt(0)
        & np.isfinite(df["duration_seconds"])
    )

    if not valid.all():
        raise ValueError(
            f"{int((~valid).sum())} invalid audio references "
            "or durations. No rows were silently removed."
        )

    df["row_index"] = df["row_index"].astype(int)

    if df.duplicated(["arrow_file", "row_index"]).any():
        raise ValueError("Duplicate recording identifiers")

    if "speaker_name" not in df:
        df["speaker_name"] = "Unknown"

    if "gender" not in df:
        df["gender"] = "Unknown"

    if "dialect" not in df:
        df["dialect"] = "Unknown"

    if "s3qa_mean" not in df:
        df["s3qa_mean"] = np.nan

    df["speaker_name"] = (
        df["speaker_name"].fillna("Unknown").astype(str)
    )
    df["gender"] = (
        df["gender"].fillna("Unknown")
        .astype(str).str.strip().str.upper()
    )
    df["gender"] = df["gender"].where(
        df["gender"].isin(["F", "M"]), "Unknown"
    )
    df["dialect"] = df["dialect"].map(normalize_dialect)

    df["s3qa_mean"] = pd.to_numeric(
        df["s3qa_mean"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

    valid_scores = df["s3qa_mean"].dropna()
    if len(valid_scores):
        q25, q75 = valid_scores.quantile([0.25, 0.75])
        df["s3qa_level"] = np.select(
            [
                df["s3qa_mean"].le(q25),
                df["s3qa_mean"].ge(q75),
            ],
            ["low", "high"],
            default="medium",
        )
        df.loc[df["s3qa_mean"].isna(), "s3qa_level"] = "Unknown"
    else:
        q25 = q75 = float("nan")
        df["s3qa_level"] = "Unknown"

    # Stable ordering before seeded sampling.
    df = df.sort_values(
        ["arrow_file", "row_index"], kind="stable"
    ).reset_index(drop=True)

    print(f"Loaded: {len(df):,} recordings")
    print(f"Total: {df.duration_seconds.sum()/3600:.3f} h")
    print("No S3QA/status/gender/dialect filtering applied")

    return df, q25, q75


def distributions(df, column):
    totals = df.groupby(column, dropna=False)[
        "duration_seconds"
    ].sum()
    return (totals / totals.sum()).to_dict()


def distance(sample, target, column):
    observed = distributions(sample, column)
    return sum(
        abs(observed.get(k, 0.0) - v)
        for k, v in target.items()
    )


def select_split(pool, hours, targets, rng, tolerance, trials):
    target_seconds = hours * 3600
    durations = pool["duration_seconds"].to_numpy()
    indices = np.arange(len(pool))

    best = None

    for _ in range(trials):
        order = rng.permutation(indices)
        chosen = []
        seconds = 0.0

        for i in order:
            duration = float(durations[i])
            if seconds + duration <= target_seconds + tolerance:
                chosen.append(int(i))
                seconds += duration

            if seconds >= target_seconds - tolerance:
                break

        if abs(seconds - target_seconds) > tolerance:
            continue

        sample = pool.iloc[chosen]

        score = sum(
            distance(sample, targets[col], col)
            for col in ("gender", "dialect", "s3qa_level")
        )

        # Prefer representation of all dominant dialects.
        missing_dialects = sum(
            d not in set(sample["dialect"])
            for d in DOMINANT
            if d in targets["dialect"]
        )

        score += 2.0 * missing_dialects

        if best is None or score < best[0]:
            best = (score, sample.copy())

    if best is None:
        raise RuntimeError(
            f"No split found near {hours} h. "
            "Increase trials or tolerance."
        )

    return best[1]


def summary(df):
    seconds = df["duration_seconds"].sum()
    return {
        "recordings": int(len(df)),
        "hours": round(seconds / 3600, 5),
        "speakers": int(df["speaker_name"].nunique()),
        "gender": distributions(df, "gender"),
        "dialect": distributions(df, "dialect"),
        "s3qa": distributions(df, "s3qa_level"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-hours", type=float, default=5)
    p.add_argument("--test-hours", type=float, default=10)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--tolerance-seconds", type=float, default=30)
    args = p.parse_args()

    df, q25, q75 = load_data(args.input_dir)
    rng = np.random.default_rng(args.seed)

    targets = {
        col: distributions(df, col)
        for col in ("gender", "dialect", "s3qa_level")
    }

    validation = select_split(
        df, args.val_hours, targets, rng,
        args.tolerance_seconds, args.trials,
    )

    remaining = df.drop(index=validation.index)

    test = select_split(
        remaining, args.test_hours, targets, rng,
        args.tolerance_seconds, args.trials,
    )

    train = remaining.drop(index=test.index)

    parts = {
        "train_pool.csv": train,
        "validation_5h.csv": validation,
        "test_10h.csv": test,
    }

    combined = pd.concat(parts.values())

    if len(combined) != len(df):
        raise RuntimeError("Some recordings were lost")

    if combined.duplicated(["arrow_file", "row_index"]).any():
        raise RuntimeError("Recording overlap detected")

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

    speaker_sets = {
        name: set(part["speaker_name"])
        for name, part in parts.items()
    }

    report = {
        "protocol": "representative-recording-disjoint",
        "seed": args.seed,
        "speaker_disjoint": False,
        "q25": None if np.isnan(q25) else float(q25),
        "q75": None if np.isnan(q75) else float(q75),
        "train": summary(train),
        "validation": summary(validation),
        "test": summary(test),
        "speaker_overlap_train_test": len(
            speaker_sets["train_pool.csv"]
            & speaker_sets["test_10h.csv"]
        ),
        "sha256": hashes,
    }

    (out / "split_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
