#!/usr/bin/env python3
"""Measure batch-one forward latency from trained model checkpoints."""

import argparse
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics
import time

from PIL import Image
import torch
from torchvision import transforms
from train import ARCHITECTURES, MODES, make_model

ROOT = Path(__file__).resolve().parent


def benchmark_checkpoint(checkpoint_path, image_path, device, warmup, iterations,
                         expected_architecture=None, expected_mode=None):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    architecture = checkpoint.get('architecture', 'resnet18')
    mode = checkpoint.get('mode')
    if architecture not in ARCHITECTURES or mode not in MODES:
        raise ValueError('Checkpoint must identify a supported architecture and mode')
    if expected_architecture and architecture != expected_architecture:
        raise ValueError(f'Expected {expected_architecture}, received {architecture}')
    if expected_mode and mode != expected_mode:
        raise ValueError(f'Expected mode {expected_mode}, received {mode}')
    class_to_idx = checkpoint['class_to_idx']
    if sorted(class_to_idx.values()) != [0, 1]:
        raise ValueError('Checkpoint must contain two classes indexed 0 and 1')
    model = make_model(architecture, mode, 2, pretrained=False)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.to(device).eval()
    preprocessing = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(checkpoint.get('mean', (.485, .456, .406)),
                             checkpoint.get('std', (.229, .224, .225))),
    ])
    with Image.open(image_path) as image:
        tensor = preprocessing(image.convert('RGB')).unsqueeze(0).to(device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        samples = []
        for _ in range(iterations):
            started = time.perf_counter()
            model(tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - started) * 1000)
    return {
        'architecture': architecture, 'mode': mode,
        'checkpoint': str(checkpoint_path), 'device': str(device),
        'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        'cpu_threads': torch.get_num_threads(), 'torch_version': str(torch.__version__),
        'input_size': 224, 'batch_size': 1,
        'warmup_iterations': warmup, 'measured_iterations': iterations,
        'mean_ms': statistics.mean(samples), 'median_ms': statistics.median(samples),
        'p95_ms': sorted(samples)[math.ceil(.95 * len(samples)) - 1],
        'image': str(image_path), 'measured_at_utc': datetime.now(timezone.utc).isoformat(),
    }


def write_result(output, result):
    output.parent.mkdir(parents=True, exist_ok=True)
    has_header = output.exists() and output.stat().st_size > 0
    if has_header:
        with output.open(newline='', encoding='utf-8') as handle:
            if next(csv.reader(handle)) != list(result):
                raise ValueError(f'CSV schema differs at {output}; choose a new --output file')
    with output.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        if not has_header:
            writer.writeheader()
        writer.writerow(result)


def main(argv=None, architecture=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=architecture is None)
    parser.add_argument('--mode', choices=(*MODES, 'all'), default=None)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--threads', type=int, default=1, help='fixed CPU thread count for comparable runs')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.iterations < 1 or args.threads < 1:
        parser.error('warmup must be non-negative; iterations and threads must be positive')
    if args.mode == 'all' and (args.checkpoint or architecture is None):
        parser.error('--mode all requires a per-model launcher without --checkpoint')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu or auto')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    torch.set_num_threads(args.threads)
    image = args.image if args.image.is_absolute() else ROOT / args.image
    if args.checkpoint:
        path = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
        checkpoints = [(path, args.mode)]
    else:
        modes = MODES if args.mode == 'all' else (args.mode or 'feature',)
        checkpoints = [(ROOT / 'models' / f'{mode}.pth', mode) for mode in modes]
    for path in [image, *(path for path, _ in checkpoints)]:
        if not path.is_file():
            parser.error(f'File not found: {path}. Training must create checkpoints first.')
    for path, mode in checkpoints:
        result = benchmark_checkpoint(path, image, device, args.warmup, args.iterations,
                                      architecture, mode)
        output = args.output or ROOT / 'results' / 'latency.csv'
        if not output.is_absolute():
            output = ROOT / output
        write_result(output, result)
        print(f"{result['architecture']}/{result['mode']} ({device}): "
              f"median={result['median_ms']:.3f} ms, mean={result['mean_ms']:.3f} ms, "
              f"p95={result['p95_ms']:.3f} ms -> {output}")
    # Custom CSV output can be plotted separately with plot_latency.py.
    if output.resolve() == (ROOT / 'results/latency.csv').resolve():
        from plot_latency import plot_projects
        plot_projects([ROOT], ROOT / 'results', device.type)


if __name__ == '__main__':
    main(architecture=ARCHITECTURES[0])
