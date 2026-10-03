#!/usr/bin/env python3
"""Train resnet18 in feature, partial, and scratch modes."""

import argparse
import csv
import hashlib
import random
import time
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parent
ARCHITECTURE = "resnet18"
ARCHITECTURES = (ARCHITECTURE,)
MODES = ("feature", "partial", "scratch")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODEL_BUILDERS = {"resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT)}


class MetadataDataset(Dataset):
    def __init__(self, rows, class_to_idx, transform):
        self.rows = rows
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = ROOT / row["path"]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), self.class_to_idx[row["label"]]


def load_metadata(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {'path', 'label', 'frame_index', 'group', 'split'}.issubset(rows[0]):
        raise ValueError('Metadata requires path, label, frame_index, group, and split columns')
    if any(row['split'] not in ('train', 'validation', 'excluded') for row in rows):
        raise ValueError('Invalid split assignment; run split.py first')
    paths = [row['path'] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate paths in metadata can cause data leakage')
    train_rows = [row for row in rows if row.get("split") == "train"]
    val_rows = [row for row in rows if row.get("split") == "validation"]
    if not train_rows or not val_rows:
        raise ValueError("No train/validation split found; run split.py first")
    classes = sorted({row["label"] for row in rows})
    if len(classes) != 2:
        raise ValueError('This project requires exactly two classes')
    hashes = {}
    for split_name, split_rows in (("train", train_rows), ("validation", val_rows)):
        missing = set(classes) - {row["label"] for row in split_rows}
        if missing:
            raise ValueError(f"{split_name} split is missing classes: {sorted(missing)}")
        for row in split_rows:
            if not (ROOT / row["path"]).is_file():
                raise FileNotFoundError(ROOT / row["path"])
            digest = hashlib.sha256((ROOT / row['path']).read_bytes()).hexdigest()
            previous = hashes.get(digest)
            if previous and (previous[0] != split_name or previous[1] != row['label']):
                raise ValueError('Identical image bytes occur across splits or class labels')
            hashes[digest] = (split_name, row['label'])
    for group in {row['group'] for row in rows}:
        for label in classes:
            train_frames = [int(r['frame_index']) for r in train_rows if r['group'] == group and r['label'] == label]
            val_frames = [int(r['frame_index']) for r in val_rows if r['group'] == group and r['label'] == label]
            if train_frames and val_frames and max(train_frames) >= min(val_frames):
                raise ValueError(f'Temporal split overlaps for {label}/{group}')
    return train_rows, val_rows, {name: index for index, name in enumerate(classes)}


def get_head(model, architecture):
    if architecture.startswith("resnet"):
        return model.fc
    return model.classifier


def replace_classifier(model, architecture, num_classes):
    if architecture.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        # Keep the pretrained classifier head structure and replace its output layer.
        for index in range(len(model.classifier) - 1, -1, -1):
            layer = model.classifier[index]
            if isinstance(layer, nn.Linear):
                model.classifier[index] = nn.Linear(layer.in_features, num_classes)
                break
        else:
            raise ValueError(f"Could not locate classifier output for {architecture}")
    return model


def partial_backbone_modules(model, architecture):
    if architecture.startswith("resnet"):
        return [model.layer4]
    # Both MobileNetV3 and EfficientNet expose the final MBConv stage followed
    # by a final feature projection in their torchvision `features` sequence.
    return list(model.features[-2:])


def make_model(architecture: str, mode: str, num_classes: int, pretrained: bool = True):
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {architecture!r}; choose from {ARCHITECTURES}")
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; choose from {MODES}")
    builder, weights = MODEL_BUILDERS[architecture]
    model = builder(weights=weights if pretrained and mode != "scratch" else None)
    model = replace_classifier(model, architecture, num_classes)

    if mode == "scratch":
        for parameter in model.parameters():
            parameter.requires_grad = True
    else:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in get_head(model, architecture).parameters():
            parameter.requires_grad = True
        if mode == "partial":
            for module in partial_backbone_modules(model, architecture):
                for parameter in module.parameters():
                    parameter.requires_grad = True
    return model


def make_optimizer(model, architecture, mode):
    head = get_head(model, architecture)
    if mode == "partial":
        head_ids = {id(parameter) for parameter in head.parameters()}
        backbone_parameters = [
            parameter for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in head_ids
        ]
        head_parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
        return torch.optim.Adam([
            {"params": backbone_parameters, "lr": 1e-4},
            {"params": head_parameters, "lr": 1e-3},
        ])
    return torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = correct = count = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)
    return loss_sum / count, correct / count


def set_training_mode(model, architecture, mode):
    if mode == 'scratch':
        model.train()
    else:
        # Freeze running statistics, dropout, and stochastic depth in the
        # frozen backbone, while keeping the selected trainable modules active.
        model.eval()
        get_head(model, architecture).train()
        if mode == 'partial':
            for module in partial_backbone_modules(model, architecture):
                module.train()


def train_experiment(architecture, mode, train_loader, val_loader, class_to_idx,
                     epochs, device, seed, metadata_digest=None):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = make_model(architecture, mode, len(class_to_idx), pretrained=(mode != "scratch")).to(device)
    # Model initialization consumes random numbers differently across models.
    # Reset augmentation/shuffle randomness independently of initialization.
    random.seed(seed)
    torch.manual_seed(seed)
    if train_loader.generator is not None:
        train_loader.generator.manual_seed(seed)
    optimizer = make_optimizer(model, architecture, mode)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    history = []
    best_accuracy = -1.0
    best_epoch = 0
    model_dir = ROOT / "models"
    result_dir = ROOT / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{mode}.pth"
    log_path = result_dir / f"{mode}_history.csv"
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        head_lr = optimizer.param_groups[-1]['lr']
        backbone_lr = '' if mode == 'feature' else optimizer.param_groups[0]['lr']
        set_training_mode(model, architecture, mode)
        loss_sum = correct = count = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)
        train_loss, train_accuracy = loss_sum / count, correct / count
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_started
        scheduler.step()
        record = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
                  "val_loss": val_loss, "val_accuracy": val_accuracy,
                  "epoch_seconds": epoch_seconds, "head_lr": head_lr, "backbone_lr": backbone_lr}
        history.append(record)
        print(f"{architecture}/{mode} epoch {epoch:02d}/{epochs}: "
              f"train_acc={train_accuracy:.4f} val_acc={val_accuracy:.4f}")
        if val_accuracy > best_accuracy:
            best_accuracy, best_epoch = val_accuracy, epoch
            torch.save({"state_dict": model.state_dict(), "architecture": architecture,
                        "mode": mode, "class_to_idx": class_to_idx, "input_size": 224,
                        "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
                        "seed": seed, "epochs_requested": epochs,
                        "metadata_sha256": metadata_digest,
                        "train_count": len(train_loader.dataset), "validation_count": len(val_loader.dataset),
                        "trainable_params": trainable_params, "total_params": total_params,
                        "epoch": epoch, "val_accuracy": val_accuracy}, model_path)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return {"architecture": architecture, "mode": mode,
            "best_val_accuracy": best_accuracy, "best_epoch": best_epoch,
            "epochs_to_90_percent": next((r["epoch"] for r in history if r["val_accuracy"] >= .9), ""),
            "training_seconds": elapsed, "checkpoint": str(model_path.relative_to(ROOT)),
            "trainable_params": trainable_params, "total_params": total_params, "epochs_run": len(history)}


RESULT_COLUMNS = ["architecture", "mode", "best_val_accuracy", "best_epoch",
                  "epochs_to_90_percent", "training_seconds", "checkpoint",
                  "trainable_params", "total_params", "epochs_run"]


def update_results(results):
    path = ROOT / "results" / "experiments.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames and "architecture" in reader.fieldnames:
                existing = {(row["architecture"], row["mode"]): row for row in reader}
    for result in results:
        existing[(result["architecture"], result["mode"])] = result
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for architecture in ARCHITECTURES:
            for mode in MODES:
                row = existing.get((architecture, mode))
                if row:
                    writer.writerow(row)
    write_markdown_results()


def write_markdown_results():
    path = ROOT / 'results' / 'experiments.csv'
    if not path.exists():
        return
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    (ROOT / 'results' / 'experiments.md').write_text(format_results_table(rows), encoding='utf-8')


def format_results_table(rows):
    lines = ['# Hasil eksperimen aktual', '',
             '| Arsitektur | Mode | Best val accuracy | Best epoch | Epoch ≥90% | Waktu training (s) | Parameter dilatih / total | Epoch dijalankan |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        counts = f"{row['trainable_params']} / {row['total_params']}" if row.get('trainable_params') and row.get('total_params') else '—'
        lines.append(f"| {row['architecture']} | {row['mode']} | {float(row['best_val_accuracy']):.4f} | "
                     f"{row['best_epoch']} | {row.get('epochs_to_90_percent') or '—'} | "
                     f"{float(row['training_seconds']):.3f} | {counts} | {row.get('epochs_run') or '—'} |")
    lines += ['', 'Hanya konfigurasi yang memiliki hasil training dicantumkan. Tanda — berarti data tidak tersedia atau ambang belum tercapai.',
              'CSV lama tetap dapat dibaca; jumlah parameter dan epoch yang belum dicatat tidak diisi dengan perkiraan.',
              'Validation berasal dari holdout temporal satu video per kelas, sehingga bukan evaluasi lintas sesi.']
    return '\n'.join(lines) + '\n'


def plot_results():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    histories = {}
    for architecture in ARCHITECTURES:
        for mode in MODES:
            path = ROOT / "results" / f"{mode}_history.csv"
            if path.exists():
                with path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                if rows:
                    histories[(architecture, mode)] = rows

    # Inspired by the reference's train/validation learning curves. Optional
    # panels require actual logged values; legacy CSVs show loss and accuracy.
    for (architecture, mode), rows in histories.items():
        has_details = all(row.get('epoch_seconds') and row.get('head_lr') for row in rows)
        fig, axes = plt.subplots(2, 2, figsize=(11, 8)) if has_details else plt.subplots(1, 2, figsize=(11, 4))
        panels = list(axes.flat)
        epochs = [int(row['epoch']) for row in rows]
        for index, (metric, title) in enumerate((('loss', 'Loss'), ('accuracy', 'Accuracy'))):
            for split_name in ('train', 'val'):
                panels[index].plot(epochs, [float(row[f'{split_name}_{metric}']) for row in rows],
                                   marker='o', label=split_name)
            panels[index].set(title=title, xlabel='Epoch', ylabel=metric)
            panels[index].legend()
            panels[index].grid(alpha=.3)
        panels[1].set_ylim(0, 1.05)
        panels[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
        best_index = max(range(len(rows)), key=lambda index: float(rows[index]['val_accuracy']))
        panels[1].axvline(epochs[best_index], color='gray', linestyle=':', label='Best validation epoch')
        panels[1].legend()
        if has_details:
            panels[2].bar(epochs, [float(row['epoch_seconds']) for row in rows])
            panels[2].set(title='Epoch duration', xlabel='Epoch', ylabel='Seconds')
            panels[3].plot(epochs, [float(row['head_lr']) for row in rows], label='Classifier')
            if all(row.get('backbone_lr') for row in rows):
                panels[3].plot(epochs, [float(row['backbone_lr']) for row in rows], label='Backbone')
            panels[3].set(title='Applied learning rate', xlabel='Epoch', ylabel='LR', yscale='log')
            panels[3].legend()
        fig.suptitle(f'{architecture} / {mode}')
        fig.tight_layout()
        fig.savefig(ROOT / 'results' / f'{mode}_curves.png', dpi=160)
        plt.close(fig)

    for architecture in ARCHITECTURES:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False
        for mode in MODES:
            rows = histories.get((architecture, mode), [])
            if rows:
                styles = {
                    'feature': dict(color='tab:blue', linestyle='-', marker='o', markersize=5),
                    'partial': dict(color='tab:orange', linestyle='--', marker='s', markersize=8, markerfacecolor='none'),
                    'scratch': dict(color='tab:green', linestyle=':', marker='^', markersize=6, markerfacecolor='none'),
                }
                ax.plot([int(row["epoch"]) for row in rows],
                        [float(row["val_accuracy"]) for row in rows],
                        label=mode, **styles[mode])
                plotted = True
        if plotted:
            ax.set(xlabel="Epoch", ylabel="Validation accuracy",
                   title=f"{architecture}: validation accuracy per epoch")
            ax.set_ylim(0, 1.05)
            ax.set_yticks([0, .2, .4, .6, .8, 1])
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            ax.set_xticks(sorted({int(row['epoch']) for mode in MODES
                                   for row in histories.get((architecture, mode), [])}))
            ax.grid(True, alpha=.3)
            ax.legend(title="Training mode")
            fig.tight_layout()
            fig.savefig(ROOT / "results" / "accuracy_per_epoch.png", dpi=160)
        plt.close(fig)
    write_markdown_results()



def main(argv=None, architecture=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture",
                        choices=(architecture,) if architecture else (*ARCHITECTURES, "all"),
                        default=architecture or "all")
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all")
    parser.add_argument("--metadata", type=Path, default=ROOT / "metadata.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--plot-only', action='store_true', help='regenerate graphs from existing history CSVs without training')
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error('--epochs and --batch-size must be positive; --workers must be non-negative')
    if args.plot_only:
        plot_results()
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_rows, val_rows, class_to_idx = load_metadata(args.metadata)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.05),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_ds = MetadataDataset(train_rows, class_to_idx, train_transform)
    val_ds = MetadataDataset(val_rows, class_to_idx, val_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == "cuda",
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
    architectures = ARCHITECTURES if args.architecture == "all" else (args.architecture,)
    modes = MODES if args.mode == "all" else (args.mode,)
    results = []
    for architecture in architectures:
        for mode in modes:
            results.append(train_experiment(architecture, mode, train_loader, val_loader,
                                            class_to_idx, args.epochs, device, args.seed,
                                            hashlib.sha256(args.metadata.read_bytes()).hexdigest()))
            update_results(results)
            plot_results()


if __name__ == "__main__":
    main(architecture=ARCHITECTURE)
