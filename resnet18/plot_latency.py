#!/usr/bin/env python3
"""Plot measured latency without benchmarking again or mixing configurations."""
import argparse
import csv
from datetime import datetime
import hashlib
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = ('mobilenet_v3_small', 'mobilenet_v3_large', 'efficientnet_b0', 'resnet18', 'resnet50')
MODES = ('feature', 'partial', 'scratch')
MODEL_LABELS = {'mobilenet_v3_small': 'MobileNetV3-Small', 'mobilenet_v3_large': 'MobileNetV3-Large',
                'efficientnet_b0': 'EfficientNet-B0', 'resnet18': 'ResNet-18', 'resnet50': 'ResNet-50'}
MODE_LABELS = {'feature': 'Feature extraction', 'partial': 'Partial fine-tuning', 'scratch': 'From scratch'}
COLORS = {'feature': '#2478B4', 'partial': '#E78424', 'scratch': '#339456'}
CONFIG = ('device', 'device_name', 'cpu_threads', 'torch_version', 'input_size', 'batch_size',
          'warmup_iterations', 'measured_iterations', 'image_sha256')


def read_measurements(folder):
    source = folder / 'results/latency.csv'
    if not source.is_file():
        return []
    with source.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row['architecture'] != folder.name or row['mode'] not in MODES:
            raise ValueError(f'Unexpected architecture/mode in {source}')
        for field in ('mean_ms', 'median_ms', 'p95_ms'):
            value = float(row[field])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'Invalid {field} in {source}')
        for field in ('cpu_threads', 'input_size', 'batch_size', 'measured_iterations'):
            if int(row[field]) < 1:
                raise ValueError(f'Invalid {field} in {source}')
        if int(row['warmup_iterations']) < 0:
            raise ValueError(f'Invalid warmup in {source}')
        timestamp = datetime.fromisoformat(row['measured_at_utc'])
        if timestamp.tzinfo is None:
            raise ValueError(f'Measurement timestamp needs a timezone in {source}')
        # Prefer this project's dataset so moved projects retain working paths.
        image = Path(row['image'])
        local = folder / 'dataset_raw' / image.parent.name / image.name
        image = local if local.is_file() else image
        if not image.is_file():
            raise FileNotFoundError(f'Cannot verify benchmark image: {image}')
        row['image_sha256'] = hashlib.sha256(image.read_bytes()).hexdigest()
    return rows


def select_comparable(rows, device='auto'):
    """Choose the latest row per model/mode in the largest matching cohort."""
    cohorts = {}
    for row in rows:
        if device != 'auto' and row['device'].split(':')[0] != device:
            continue
        signature = tuple(row[field] for field in CONFIG)
        group = cohorts.setdefault(signature, {})
        key = (row['architecture'], row['mode'])
        previous = group.get(key)
        if previous is None or datetime.fromisoformat(row['measured_at_utc']) >= datetime.fromisoformat(previous['measured_at_utc']):
            group[key] = row
    if not cohorts:
        raise ValueError(f'No measured latency for device={device}')
    group = max(cohorts.values(), key=lambda g: (len(g), max(datetime.fromisoformat(r['measured_at_utc']) for r in g.values())))
    return [group[(name, mode)] for name in NAMES for mode in MODES if (name, mode) in group]


def export_summary(rows, output, stem):
    output.mkdir(parents=True, exist_ok=True)
    with (output / f'{stem}.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cfg = rows[0]
    lines = ['# Hasil pengukuran latency', '',
             f"Perangkat: {cfg['device_name']} ({cfg['device']}); PyTorch {cfg['torch_version']}.",
             f"Input {cfg['input_size']}×{cfg['input_size']}, batch {cfg['batch_size']}, "
             f"CPU threads {cfg['cpu_threads']}, warm-up {cfg['warmup_iterations']}, iterasi {cfg['measured_iterations']}.", '',
             '| Model | Mode | Median (ms) | Mean (ms) | P95 (ms) |',
             '|---|---|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {MODEL_LABELS[row['architecture']]} | {row['mode']} | "
                     f"{float(row['median_ms']):.3f} | {float(row['mean_ms']):.3f} | {float(row['p95_ms']):.3f} |")
    lines += ['', 'Batang menunjukkan median; penanda P95 menunjukkan persentil ke-95, bukan confidence interval.',
              'Pengukuran mencakup forward pass PyTorch; tidak termasuk kamera, preprocessing, postprocessing, atau ROS2.',
              'Acuan slide 16: anggaran inferensi 35 ms dalam total sekitar 67 ms/frame (15 FPS).',
              'Anggaran PPT adalah target, bukan hasil pengukuran pipeline robot.',
              'Baris terbaru per model/mode dipilih dalam konfigurasi pengukuran yang sama, termasuk hash gambar.',
              'Auto memilih kelompok dengan konfigurasi terbanyak; jika seri, memilih kelompok paling baru.',
              'Konfigurasi yang tidak ada dalam kelompok tersebut tidak ditampilkan.',
              'Hasil satu sesi pada GPU laptop tidak menjamin kinerja pada komputer robot.', '']
    (output / f'{stem}.md').write_text('\n'.join(lines), encoding='utf-8')


def draw_bars(rows, output, comparison=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MultipleLocator

    cfg = rows[0]
    names = [name for name in NAMES if any(r['architecture'] == name for r in rows)]
    lookup = {(r['architecture'], r['mode']): r for r in rows}
    fig, ax = plt.subplots(figsize=(12, 7.5 if comparison else 6.2))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFBFD')
    positions, labels = [], []
    max_value = max(float(r['p95_ms']) for r in rows)
    if comparison:
        positions = list(range(len(names)))
        labels = [MODEL_LABELS[name] for name in names]
        entries = [(lookup[(name, mode)], i + (j - 1) * .23, .19)
                   for i, name in enumerate(names) for j, mode in enumerate(MODES) if (name, mode) in lookup]
    else:
        entries = [(r, i, .48) for i, r in enumerate(rows)]
        positions = list(range(len(rows)))
        labels = [MODE_LABELS[r['mode']] for r in rows]
    for row, y, height in entries:
        median, p95 = float(row['median_ms']), float(row['p95_ms'])
        ax.barh(y, median, height, color=COLORS[row['mode']], edgecolor='white', linewidth=.8, zorder=3)
        ax.plot([median, p95], [y, y], color='#303A49', linewidth=1.1, zorder=4)
        ax.plot(p95, y, marker='|', color='#303A49', markersize=9, markeredgewidth=1.5, zorder=5)
        ax.text(max(median, p95) + max_value * .025, y, f'{median:.2f}', va='center', fontsize=10, color='#263244')
    ax.set_yticks(positions, labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, max_value * 1.25)
    ax.xaxis.set_major_locator(MultipleLocator(max_value / 6 if max_value >= 6 else .5))
    ax.set_xlabel('Latency median (ms) — lebih kecil lebih cepat', fontsize=11, labelpad=10)
    ax.grid(axis='x', color='#DAE0E8', alpha=.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', length=0, pad=9)
    handles = [Patch(facecolor=COLORS[mode], label=MODE_LABELS[mode]) for mode in MODES if any(r['mode'] == mode for r in rows)]
    handles.append(Line2D([], [], color='#303A49', marker='|', linestyle='-', label='P95'))
    ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0, 1.025), ncol=4, frameon=False, fontsize=10)
    title = 'Perbandingan latency lima model' if comparison else f"Latency {MODEL_LABELS[names[0]]}"
    fig.suptitle(title, x=.02, y=.98, ha='left', fontsize=19, fontweight='bold', color='#172B45')
    fig.text(.02, .91, f"{cfg['input_size']}×{cfg['input_size']} · batch {cfg['batch_size']} · "
             f"{cfg['warmup_iterations']} warm-up · {cfg['measured_iterations']} iterasi", fontsize=10.5, color='#506176')
    fig.subplots_adjust(left=.21 if comparison else .23, right=.97, top=.81 if comparison else .76, bottom=.12 if comparison else .14)
    output.mkdir(parents=True, exist_ok=True)
    stem = 'latency_comparison' if comparison else 'latency_bar'
    for extension in ('png', 'svg'):
        fig.savefig(output / f'{stem}.{extension}', dpi=200, facecolor='white')
    plt.close(fig)


def plot_projects(folders, output, device='auto', comparison=False):
    measurements = [row for folder in folders for row in read_measurements(folder)]
    rows = select_comparable(measurements, device)
    draw_bars(rows, output, comparison)
    export_summary(rows, output, 'latency' if comparison else 'latency_summary')
    print(f"{len(rows)}/{len({(r['architecture'], r['mode']) for r in measurements})} configurations; "
          f"{rows[0]['device_name']} ({rows[0]['device']}) -> {output}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', action='store_true', help='compare available sibling projects')
    parser.add_argument('--root', type=Path, default=ROOT.parent)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto', help='filter recorded measurements, no device access needed')
    args = parser.parse_args()
    folders = [args.root / name for name in NAMES] if args.comparison else [ROOT]
    output = args.output or ROOT / 'results' / ('comparison' if args.comparison else '')
    try:
        plot_projects(folders, output, args.device, args.comparison)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
