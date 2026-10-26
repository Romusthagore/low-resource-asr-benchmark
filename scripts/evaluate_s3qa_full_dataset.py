import argparse
import ast
import csv
import io
import os
from pathlib import Path

import librosa
import numpy as np
import pyarrow as pa
import pyloudnorm as pyln
import soundfile as sf
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "external" / "S3QA"

DATASET_ROOT = (
    Path.home()
    / "scratch/huggingface/hub/"
    "datasets--Digital-Divide-Data--Luhya-ASR-Data-subset-642H"
)

CKPT = Path.home() / "scratch/huggingface/hub/S3QA/BEST.ckpt"

OUTPUT_DIR = (
    Path.home()
    / "scratch/low-resource-asr-benchmark/outputs/s3qa_luhya_full"
)

SR = 16000
WINDOW = 4 * SR
HOP = SR

FIELDS = [
    "arrow_file", "row_index", "no", "sentence_id",
    "speaker_name", "filename", "dialect", "gender", "age",
    "domain", "sub_domain", "theme", "subtheme",
    "date_of_recording", "transcripts",
    "duration_seconds", "original_sample_rate",
    "num_windows", "s3qa_mean", "s3qa_median",
    "s3qa_min", "s3qa_max", "status", "error"
]


def load_model():
    source = (ROOT / "run_model.py").read_text()
    tree = ast.parse(source)

    selected = [
        node for node in tree.body
        if isinstance(node, (
            ast.Import, ast.ImportFrom,
            ast.FunctionDef, ast.ClassDef
        ))
    ]

    module = ast.Module(body=selected, type_ignores=[])
    namespace = {"__name__": "s3qa_model"}

    exec(
        compile(module, str(ROOT / "run_model.py"), "exec"),
        namespace
    )

    Model = namespace["s3qa_Model"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = Model(
        nclasses=1,
        batch_size=1,
        lr=0.0005,
        dur_aug=False
    )

    checkpoint = torch.load(
        CKPT, map_location="cpu", weights_only=False
    )

    model.load_state_dict(
        checkpoint["state_dict"], strict=True
    )

    model = model.to(device).eval()

    print("Device:", device, flush=True)

    return model, device


def normalize_loudness(audio, meter):
    if len(audio) < int(0.5 * SR):
        return audio

    try:
        loudness = meter.integrated_loudness(audio)

        if not np.isfinite(loudness):
            return audio

        normalized = pyln.normalize.loudness(
            audio, loudness, -35.0
        )

        return np.asarray(normalized, dtype=np.float32)

    except (ValueError, OverflowError):
        return audio


def load_audio(audio_bytes):
    with io.BytesIO(audio_bytes) as stream:
        audio, original_sr = sf.read(
            stream, dtype="float32"
        )

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if original_sr != SR:
        audio = librosa.resample(
            audio,
            orig_sr=original_sr,
            target_sr=SR
        )

    return np.asarray(audio, dtype=np.float32), original_sr


def make_windows(audio):
    if len(audio) <= WINDOW:
        yield 0, np.pad(
            audio, (0, WINDOW - len(audio))
        )
        return

    last_start = None

    for start in range(
        0, len(audio) - WINDOW + 1, HOP
    ):
        yield start, audio[start:start + WINDOW]
        last_start = start

    final_start = len(audio) - WINDOW

    if last_start != final_start:
        yield final_start, audio[final_start:]


def evaluate_audio(audio, model, device, meter):
    audio = normalize_loudness(audio, meter)

    scores = []

    with torch.inference_mode():
        for start, window in make_windows(audio):
            window = normalize_loudness(window, meter)

            tensor = torch.from_numpy(
                np.ascontiguousarray(window)
            ).to(device).reshape(1, 1, -1)

            outputs = model(tensor)

            score = float(
                outputs[0].reshape(-1)[0].item()
            )

            scores.append(score)

    return scores


def process_arrow(
    path, file_index, model, device, meter,
    max_audios=None
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = OUTPUT_DIR / f"part_{file_index:04d}.csv"
    temp = OUTPUT_DIR / f"part_{file_index:04d}.tmp"

    if output.exists():
        print(
            f"[SKIP] {output.name} déjà terminé",
            flush=True
        )
        return

    count = 0
    errors = 0

    with temp.open(
        "w", newline="", encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f, fieldnames=FIELDS
        )
        writer.writeheader()

        with pa.memory_map(str(path), "r") as source:
            try:
                reader = pa.ipc.open_file(source)
                batches = (
                    reader.get_batch(i)
                    for i in range(reader.num_record_batches)
                )
            except pa.ArrowInvalid:
                source.seek(0)
                reader = pa.ipc.open_stream(source)
                batches = iter(reader)

            row_index = 0

            for batch in batches:
                for row in batch.to_pylist():
                    result = {
                        "arrow_file": str(
                            path.relative_to(DATASET_ROOT)
                        ),
                        "row_index": row_index,
                        "no": row.get("no"),
                        "sentence_id": row.get("sentence_id"),
                        "speaker_name": row.get("speaker_name"),
                        "filename": row.get("filename"),
                        "dialect": row.get("dialect"),
                        "gender": row.get("gender"),
                        "age": row.get("age"),
                        "domain": row.get("domain"),
                        "sub_domain": row.get("sub_domain"),
                        "theme": row.get("theme"),
                        "subtheme": row.get("subtheme"),
                        "date_of_recording": row.get(
                            "date_of_recording"
                        ),
                        "transcripts": row.get("transcripts"),
                        "status": "ok",
                        "error": ""
                    }

                    try:
                        audio_bytes = (
                            row.get("audio") or {}
                        ).get("bytes")

                        if not audio_bytes:
                            raise ValueError(
                                "audio.bytes manquant"
                            )

                        audio, original_sr = load_audio(
                            audio_bytes
                        )

                        if len(audio) == 0:
                            raise ValueError("Audio vide")

                        scores = evaluate_audio(
                            audio, model, device, meter
                        )

                        result.update({
                            "duration_seconds": round(
                                len(audio) / SR, 3
                            ),
                            "original_sample_rate": original_sr,
                            "num_windows": len(scores),
                            "s3qa_mean": float(np.mean(scores)),
                            "s3qa_median": float(np.median(scores)),
                            "s3qa_min": float(np.min(scores)),
                            "s3qa_max": float(np.max(scores))
                        })

                    except Exception as e:
                        result["status"] = "error"
                        result["error"] = str(e)[:500]
                        errors += 1

                    writer.writerow(result)

                    count += 1
                    row_index += 1

                    if count % 100 == 0:
                        f.flush()
                        print(
                            f"[{file_index}] "
                            f"{count} audios, "
                            f"{errors} erreurs",
                            flush=True
                        )

                    if (
                        max_audios is not None
                        and count >= max_audios
                    ):
                        break

                if (
                    max_audios is not None
                    and count >= max_audios
                ):
                    break

        f.flush()
        os.fsync(f.fileno())

    # Les tests limités ne sont jamais considérés
    # comme des fichiers Arrow complètement traités.
    if max_audios is None:
        os.replace(temp, output)
    else:
        test_output = OUTPUT_DIR / (
            f"test_part_{file_index:04d}.csv"
        )
        os.replace(temp, test_output)

    print(
        f"[DONE] fichier {file_index}: "
        f"{count} audios, {errors} erreurs",
        flush=True
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start-file", type=int, default=0
    )
    parser.add_argument(
        "--num-files", type=int, default=None
    )
    parser.add_argument(
        "--max-audios", type=int, default=None
    )

    args = parser.parse_args()

    files = sorted(
        DATASET_ROOT.glob(
            "snapshots/*/batch_*/*.arrow"
        )
    )

    print(
        f"Fichiers Arrow trouvés: {len(files)}",
        flush=True
    )

    if not files:
        raise RuntimeError(
            "Aucun fichier Arrow trouvé"
        )

    if args.start_file < 0:
        parser.error("--start-file doit être >= 0")

    if args.num_files is not None and args.num_files < 1:
        parser.error("--num-files doit être >= 1")

    if args.max_audios is not None and args.max_audios < 1:
        parser.error("--max-audios doit être >= 1")

    end = (
        len(files)
        if args.num_files is None
        else min(
            len(files),
            args.start_file + args.num_files
        )
    )

    selected_files = list(
        enumerate(files)
    )[args.start_file:end]

    if not selected_files:
        parser.error("Aucun fichier sélectionné")

    model, device = load_model()
    meter = pyln.Meter(SR)

    for index, path in selected_files:
        print(
            f"\n[{index + 1}/{len(files)}] "
            f"{path.name}",
            flush=True
        )

        process_arrow(
            path, index, model, device, meter,
            max_audios=args.max_audios
        )

    print("\nAnalyse terminée.", flush=True)


if __name__ == "__main__":
    main()
