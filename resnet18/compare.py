#!/usr/bin/env python3
"""Collect available results from the five independent model projects."""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = ('mobilenet_v3_small', 'mobilenet_v3_large', 'efficientnet_b0', 'resnet18', 'resnet50')
MODES = ('feature', 'partial', 'scratch')
COLUMNS = ('architecture', 'mode', 'best_val_accuracy', 'best_epoch', 'epochs_to_90_percent',
           'training_seconds', 'checkpoint', 'trainable_params', 'total_params', 'epochs_run')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT.parent, help='directory containing the model projects')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'comparison')
    parser.add_argument('--latency-device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args()
    directories = [args.root / name for name in NAMES]
    if ROOT not in directories:
        directories.append(ROOT)
    collected = {}
    for directory in directories:
        source = directory / 'results' / 'experiments.csv'
        if not source.is_file():
            continue
        with source.open(newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                key = (row['architecture'], row['mode'])
                if key[0] not in NAMES or key[1] not in MODES:
                    raise ValueError(f'Unsupported experiment in {source}: {key}')
                accuracy = float(row['best_val_accuracy'])
                if not 0 <= accuracy <= 1:
                    raise ValueError(f'Invalid accuracy in {source}')
                checkpoint = directory / row['checkpoint']
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                row['checkpoint'] = str(checkpoint.resolve())
                collected[key] = row
    rows = [collected[(name, mode)] for name in NAMES for mode in MODES if (name, mode) in collected]
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'experiments.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Perbandingan hasil yang tersedia', '',
             '| Model | Mode | Best val accuracy | Epoch ≥90% | Waktu training (s) |',
             '|---|---|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['architecture']} | {row['mode']} | {float(row['best_val_accuracy']):.4f} | "
                     f"{row.get('epochs_to_90_percent') or '—'} | {float(row['training_seconds']):.3f} |")
    lines += ['', 'Hanya konfigurasi dengan tabel dan checkpoint tersedia yang dicantumkan.',
              'Perbandingan memerlukan split, preprocessing, epoch, dan konfigurasi training yang sama.',
              'Hasil lama tidak otomatis dilatih ulang. Validation temporal bukan evaluasi lintas sesi.']
    (args.output / 'experiments.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 6))
        for index, mode in enumerate(MODES):
            values = [float(collected[(name, mode)]['best_val_accuracy'])
                      if (name, mode) in collected else float('nan') for name in NAMES]
            ax.bar([position + (index - 1) * .24 for position in range(len(NAMES))], values, .24, label=mode)
        ax.set_xticks(range(len(NAMES)), NAMES, rotation=15, ha='right')
        ax.set(ylabel='Best validation accuracy', ylim=(0, 1.05), title='Available results across five model projects')
        ax.legend()
        ax.grid(axis='y', alpha=.3)
        fig.tight_layout()
        fig.savefig(args.output / 'accuracy_comparison.png', dpi=160)
        plt.close(fig)
    print(f'{len(rows)} available configurations -> {args.output}')
    if any((directory / 'results/latency.csv').is_file() for directory in directories):
        from plot_latency import plot_projects
        plot_projects(directories, args.output, args.latency_device, comparison=True)


if __name__ == '__main__':
    main()
