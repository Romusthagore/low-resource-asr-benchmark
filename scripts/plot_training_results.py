#!/usr/bin/env python3
"""Plot existing African ASR training results using only Python's standard library."""

import html
import json
import math
from pathlib import Path

ROOT = Path.home() / "scratch/low-resource-asr-benchmark/outputs"
REPORTS = Path(__file__).resolve().parents[1] / "reports"

EXCLUDE = ("smoke", "pipeline-test")


def read_json(path):
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def model_label(name):
    return name.replace("-luhya", "").replace("-", " ")


def load_runs():
    runs = []
    if not ROOT.is_dir():
        raise SystemExit(f"Répertoire introuvable : {ROOT}")

    for directory in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        name = directory.name
        if any(word in name.lower() for word in EXCLUDE):
            continue

        history = []
        metrics = read_json(directory / "metrics.json")

        # AfriHuBERT : métriques enregistrées sous forme de proportions.
        if isinstance(metrics, list):
            for item in metrics:
                if not isinstance(item, dict):
                    continue
                epoch = item.get("epoch")
                wer, cer = item.get("wer"), item.get("cer")
                if epoch is not None and wer is not None and cer is not None:
                    history.append({
                        "epoch": float(epoch),
                        "wer": float(wer) * 100,
                        "cer": float(cer) * 100,
                    })

        # Whisper et Buzz : historique du Trainer Hugging Face.
        state_files = sorted(directory.glob("checkpoint-*/trainer_state.json"))
        if state_files:
            state = read_json(state_files[-1])
            if isinstance(state, dict):
                for item in state.get("log_history", []):
                    if not isinstance(item, dict):
                        continue
                    wer, cer = item.get("eval_wer"), item.get("eval_cer")
                    epoch = item.get("epoch")
                    if epoch is not None and wer is not None and cer is not None:
                        history.append({
                            "epoch": float(epoch),
                            "wer": float(wer),
                            "cer": float(cer),
                        })

        # Dédupliquer les évaluations éventuelles au même epoch.
        by_epoch = {}
        for point in history:
            by_epoch[point["epoch"]] = point
        history = [by_epoch[e] for e in sorted(by_epoch)]

        compute = read_json(directory / "compute.json")
        if history:
            runs.append({
                "name": name,
                "label": model_label(name),
                "history": history,
                "compute": compute if isinstance(compute, dict) else {},
            })

    return runs


def svg_start(title, width=1100, height=650):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="35" text-anchor="middle" '
        f'font-family="sans-serif" font-size="22" font-weight="bold">'
        f'{html.escape(title)}</text>',
    ]


def text(x, y, value, size=13, anchor="middle"):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="sans-serif" font-size="{size}">'
        f'{html.escape(str(value))}</text>'
    )


def save_svg(filename, elements):
    elements.append("</svg>")
    path = REPORTS / filename
    path.write_text("\n".join(elements), encoding="utf-8")
    print(f"Créé : {path}")


def comparison_chart(runs, metric, title, filename):
    values = []
    for run in runs:
        # Meilleur WER/CER observé, sans mélanger les époques.
        best = min(run["history"], key=lambda p: p[metric])
        values.append((run["label"], best[metric], best["epoch"]))

    if not values:
        return

    width, height = 1100, max(380, 100 + 55 * len(values))
    elements = svg_start(title, width, height)
    left, right, top = 280, 70, 70
    plot_width = width - left - right
    max_value = max(v for _, v, _ in values) * 1.15 or 1

    for tick in range(6):
        val = max_value * tick / 5
        x = left + plot_width * tick / 5
        elements.append(
            f'<line x1="{x}" y1="{top}" x2="{x}" y2="{height-35}" '
            'stroke="#dddddd"/>'
        )
        elements.append(text(x, height - 12, f"{val:.1f}%"))

    for i, (label, value, epoch) in enumerate(values):
        y = top + 20 + i * 48
        elements.append(text(left - 12, y + 5, label, 12, "end"))
        bar_width = plot_width * value / max_value
        elements.append(
            f'<rect x="{left}" y="{y-12}" width="{bar_width:.2f}" '
            'height="23" fill="#4472c4"/>'
        )
        elements.append(text(left + bar_width + 38, y + 5,
                             f"{value:.2f}% (ép. {epoch:g})", 12, "start"))

    save_svg(filename, elements)


def curve_chart(runs, metric, title, filename):
    width, height = 1100, 650
    elements = svg_start(title, width, height)
    left, right, top, bottom = 80, 270, 75, 65
    plot_w, plot_h = width - left - right, height - top - bottom
    max_epoch = max(p["epoch"] for r in runs for p in r["history"])
    max_value = max(p[metric] for r in runs for p in r["history"]) * 1.1
    max_value = max(max_value, 1)
    colors = ["#4472c4", "#ed7d31", "#70ad47", "#a64d79",
              "#5b9bd5", "#ffc000", "#8064a2", "#333333"]

    for tick in range(6):
        val = max_value * tick / 5
        y = top + plot_h * (1 - tick / 5)
        elements.append(
            f'<line x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}" '
            'stroke="#dddddd"/>'
        )
        elements.append(text(left - 12, y + 5, f"{val:.1f}%", 12, "end"))

    for tick in range(int(max_epoch) + 1):
        if tick == 0:
            continue
        x = left + plot_w * tick / max_epoch
        elements.append(text(x, height - 35, tick))

    for i, run in enumerate(runs):
        pts = []
        for p in run["history"]:
            x = left + plot_w * p["epoch"] / max_epoch
            y = top + plot_h * (1 - p[metric] / max_value)
            pts.append((x, y))
        if pts:
            color = colors[i % len(colors)]
            point_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            elements.append(
                f'<polyline points="{point_str}" fill="none" '
                f'stroke="{color}" stroke-width="2.5"/>'
            )
            legend_y = top + 18 * i
            elements.append(
                f'<line x1="{left+plot_w+20}" y1="{legend_y}" '
                f'x2="{left+plot_w+42}" y2="{legend_y}" '
                f'stroke="{color}" stroke-width="3"/>'
            )
            elements.append(text(left + plot_w + 48, legend_y + 4,
                                 run["label"], 11, "start"))

    elements.append(text(left + plot_w / 2, height - 8, "Époque"))
    save_svg(filename, elements)


def compute_chart(runs):
    available = []
    for run in runs:
        data = run["compute"]
        hours = data.get("training_time_hours")
        memory = data.get("peak_gpu_memory_gb")
        if hours is not None or memory is not None:
            available.append((run["label"], hours, memory))

    if not available:
        print("Aucune donnée de calcul disponible.")
        return

    elements = svg_start("Coût de calcul par modèle", 1100,
                         max(380, 100 + 45 * len(available)))
    for i, (label, hours, memory) in enumerate(available):
        y = 85 + i * 45
        elements.append(text(260, y + 4, label, 12, "end"))
        if hours is not None:
            elements.append(
                f'<rect x="280" y="{y-12}" width="{float(hours)*35:.1f}" '
                'height="14" fill="#4472c4"/>'
            )
            elements.append(text(290 + float(hours)*35, y,
                                 f"{float(hours):.2f} h", 11, "start"))
        if memory is not None:
            elements.append(text(800, y,
                                 f"Pic mémoire GPU : {float(memory):.2f} Go",
                                 11, "start"))
    save_svg("training_compute.svg", elements)


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    if not runs:
        raise SystemExit("Aucune métrique d'entraînement exploitable trouvée.")

    print(f"Modèles avec historique exploitable : {len(runs)}")
    for run in runs:
        final = run["history"][-1]
        print(f"  {run['name']}: {len(run['history'])} points, "
              f"dernier WER={final['wer']:.2f}%, CER={final['cer']:.2f}%")

    comparison_chart(runs, "wer", "Comparaison du meilleur WER par modèle",
                     "best_wer.svg")
    comparison_chart(runs, "cer", "Comparaison du meilleur CER par modèle",
                     "best_cer.svg")
    curve_chart(runs, "wer", "Évolution du WER par époque", "wer_by_epoch.svg")
    curve_chart(runs, "cer", "Évolution du CER par époque", "cer_by_epoch.svg")
    compute_chart(runs)


if __name__ == "__main__":
    main()
