from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(
    "/network/scratch/a/ahomagnr/low-resource-asr-benchmark/"
    "outputs/luhya_fixed_splits_seed42_v5"
)

FILES = {
    "Train": "train_pool.csv",
    "Validation": "validation_5h.csv",
    "Test": "test_10h.csv",
}

Q25 = 0.027028566158630574
Q75 = 0.06776604171664935

frames = {}

for split, filename in FILES.items():
    df = pd.read_csv(BASE / filename)
    df["hours"] = df["duration_seconds"] / 3600

    df["s3qa_group"] = pd.cut(
        df["s3qa_mean"],
        bins=[-np.inf, Q25, Q75, np.inf],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )

    frames[split] = df

def distribution(df, column):
    tab = df.groupby(column, observed=True).agg(
        recordings=("duration_seconds", "size"),
        speakers=("speaker_name", "nunique"),
        hours=("hours", "sum"),
    )
    tab["hours_pct"] = 100 * tab["hours"] / df["hours"].sum()
    return tab.round(3)

print("\n" + "=" * 75)
print("1. GLOBAL SPLIT DISTRIBUTION")
print("=" * 75)

rows = []
for split, df in frames.items():
    rows.append({
        "split": split,
        "recordings": len(df),
        "speakers": df["speaker_name"].nunique(),
        "hours": round(df["hours"].sum(), 4),
        "mean_duration_s": round(df["duration_seconds"].mean(), 2),
        "median_duration_s": round(df["duration_seconds"].median(), 2),
    })

print(pd.DataFrame(rows).to_string(index=False))

for column, title in [
    ("gender", "2. GENDER DISTRIBUTION"),
    ("dialect", "3. DIALECT DISTRIBUTION"),
    ("s3qa_group", "4. S3QA DISTRIBUTION"),
]:
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)

    for split, df in frames.items():
        print(f"\n--- {split} ---")
        print(distribution(df, column).to_string())

print("\n" + "=" * 75)
print("5. DIALECT DISTRIBUTION BY SPLIT (% OF HOURS)")
print("=" * 75)

all_df = pd.concat(
    [df.assign(split=split) for split, df in frames.items()],
    ignore_index=True,
)

dialect_hours = all_df.pivot_table(
    index="dialect",
    columns="split",
    values="hours",
    aggfunc="sum",
    fill_value=0,
    observed=True,
)

dialect_pct = (
    dialect_hours
    .div(dialect_hours.sum(axis=0), axis=1)
    .mul(100)
)

print(dialect_pct.round(2).to_string())

print("\n" + "=" * 75)
print("6. GENDER x DIALECT (% OF SPLIT HOURS)")
print("=" * 75)

for split, df in frames.items():
    cross = pd.pivot_table(
        df,
        index="dialect",
        columns="gender",
        values="hours",
        aggfunc="sum",
        fill_value=0,
        observed=True,
    )
    cross = 100 * cross / df["hours"].sum()

    print(f"\n--- {split} ---")
    print(cross.round(2).to_string())

print("\n" + "=" * 75)
print("7. SPEAKER DURATION STATISTICS")
print("=" * 75)

for split, df in frames.items():
    speaker_hours = df.groupby("speaker_name")["hours"].sum()

    print(
        f"{split}: "
        f"min={speaker_hours.min():.3f}h, "
        f"median={speaker_hours.median():.3f}h, "
        f"max={speaker_hours.max():.3f}h"
    )

print("\n" + "=" * 75)
print("8. TARGET CHECKS")
print("=" * 75)

for split, df in frames.items():
    total = df["hours"].sum()
    female = df.loc[
        df["gender"].astype(str).str.lower().isin(["f", "female"]),
        "hours"
    ].sum() / total * 100

    s3qa = (
        df.groupby("s3qa_group", observed=True)["hours"]
        .sum()
        .div(total)
        .mul(100)
    )

    print(f"\n{split}")
    print(f"Female: {female:.2f}% (target 64.13%)")
    print(
        "S3QA: "
        + ", ".join(
            f"{k}={s3qa.get(k, 0):.2f}%"
            for k in ["Low", "Medium", "High"]
        )
    )

print("\nANALYSIS COMPLETE")
