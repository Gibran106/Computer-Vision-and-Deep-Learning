#!/usr/bin/env python3
"""Run and audit the five standalone P2 model projects sequentially."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlparse

import torch
from torchvision import models

PROJECTS = Path(__file__).resolve().parent.parent
WEIGHTS = {
    'mobilenet_v3_small': models.MobileNet_V3_Small_Weights.DEFAULT,
    'mobilenet_v3_large': models.MobileNet_V3_Large_Weights.DEFAULT,
    'efficientnet_b0': models.EfficientNet_B0_Weights.DEFAULT,
    'resnet18': models.ResNet18_Weights.DEFAULT,
    'resnet50': models.ResNet50_Weights.DEFAULT,
}
MODES = ('feature', 'partial', 'scratch')


def prepare_weights(cache):
    original = Path(torch.hub.get_dir()) / 'checkpoints'
    target = cache / 'hub' / 'checkpoints'
    target.mkdir(parents=True, exist_ok=True)
    for name, weights in WEIGHTS.items():
        filename = Path(urlparse(weights.url).path).name
        destination = target / filename
        if not destination.exists():
            if (original / filename).is_file():
                shutil.copy2(original / filename, destination)
            else:
                print(f'Preparing ImageNet weights: {name}', flush=True)
                torch.hub.download_url_to_file(weights.url, str(destination),
                                              hash_prefix=filename.rsplit('-', 1)[1].split('.')[0], progress=True)


def audit_model(folder, epochs):
    with (folder / 'results/experiments.csv').open(newline='') as handle:
        records = list(csv.DictReader(handle))
    assert {row['mode'] for row in records} == set(MODES)
    assert len(records) == 3
    digest = hashlib.sha256((folder / 'metadata.csv').read_bytes()).hexdigest()
    for row in records:
        with (folder / 'results' / f"{row['mode']}_history.csv").open(newline='') as handle:
            history = list(csv.DictReader(handle))
        assert len(history) == epochs
        checkpoint = torch.load(folder / row['checkpoint'], map_location='cpu', weights_only=True)
        assert checkpoint['architecture'] == folder.name and checkpoint['mode'] == row['mode']
        assert checkpoint['metadata_sha256'] == digest
        best = max(float(item['val_accuracy']) for item in history)
        assert checkpoint['val_accuracy'] == best == float(row['best_val_accuracy'])
        assert checkpoint['epoch'] == int(row['best_epoch'])
        assert (folder / 'results' / f"{row['mode']}_curves.png").is_file()
    with (folder / 'results/latency.csv').open(newline='') as handle:
        measurements = list(csv.DictReader(handle))
    latest = {row['mode']: row for row in measurements}
    assert set(latest) == set(MODES)
    for row in latest.values():
        assert row['architecture'] == folder.name
        assert int(row['cpu_threads']) == 1
        assert float(row['median_ms']) > 0
    return records, latest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--train-threads', type=int, default=4)
    parser.add_argument('--logs-dir', type=Path, help='log directory; default is a temporary directory')
    parser.add_argument('--models', nargs='+', choices=tuple(WEIGHTS), default=list(WEIGHTS))
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.train_threads) < 1:
        parser.error('epochs, batch-size, and train-threads must be positive')
    cache = Path(tempfile.gettempdir()) / 'p2-training-torch'
    prepare_weights(cache)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    logs = args.logs_dir or Path(tempfile.gettempdir()) / f'p2-run-{stamp}'
    logs.mkdir(parents=True)
    backup = Path(tempfile.gettempdir()) / f'p2-results-before-{stamp}'
    backup.mkdir()
    environment = dict(os.environ, TORCH_HOME=str(cache), PYTHONDONTWRITEBYTECODE='1',
                       MPLCONFIGDIR=str(Path(tempfile.gettempdir()) / 'p2-matplotlib'),
                       OMP_NUM_THREADS=str(args.train_threads), MKL_NUM_THREADS=str(args.train_threads))
    status = {'started_at_utc': stamp, 'device': 'cuda' if torch.cuda.is_available() else 'cpu',
              'epochs': args.epochs, 'backup': str(backup), 'steps': []}

    def save_status():
        (logs / 'status.json').write_text(json.dumps(status, indent=2))

    def step(name, label, arguments):
        command = [sys.executable, '-u', *arguments]
        print(f'\n>>> {name}: {label}', flush=True)
        started = time.perf_counter()
        record = {'model': name, 'step': label, 'command': command, 'state': 'running'}
        status['steps'].append(record)
        save_status()
        with (logs / f'{name}_{label}.log').open('w') as log:
            process = subprocess.Popen(command, cwd=PROJECTS / name, env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            result = process.wait()
        record.update(state='passed' if result == 0 else 'failed', exit_code=result,
                      elapsed_seconds=time.perf_counter() - started)
        save_status()
        if result:
            raise RuntimeError(f'{name}/{label} failed. Inspect {logs}')

    save_status()
    summaries = []
    for name in args.models:
        folder = PROJECTS / name
        for output in ('models', 'results'):
            source = folder / output
            if source.exists():
                shutil.copytree(source, backup / name / output,
                                ignore=shutil.ignore_patterns('run_all', '__pycache__'))
        step(name, 'tests', ['-m', 'unittest', 'discover', '-s', 'tests', '-v'])
        step(name, 'split', ['split.py'])
        step(name, 'training', ['train.py', '--epochs', str(args.epochs), '--batch-size', str(args.batch_size)])
        step(name, 'latency', ['latency.py', '--mode', 'all', '--device', status['device'], '--threads', '1',
                               '--image', 'dataset_raw/box_merah/box_merah__frame_000000000.jpg'])
        records, latest = audit_model(folder, args.epochs)
        for row in records:
            summaries.append(dict(row, median_latency_ms=latest[row['mode']]['median_ms'],
                                  latency_device=latest[row['mode']]['device']))
        print(f'>>> {name}: checkpoint/history/table/latency audit PASSED', flush=True)
    for name in WEIGHTS:
        step(name, 'comparison', ['compare.py'])
    with (logs / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    status['state'] = 'complete'
    status['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    status['configurations_trained'] = len(summaries)
    save_status()
    print(f'\nCOMPLETE: {len(summaries)} trained configurations. Summary: {logs / "summary.csv"}', flush=True)


if __name__ == '__main__':
    main()
