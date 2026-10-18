"""Membagi dataset deteksi YOLO secara temporal per kelas.

Masukan : dataset_bbox/ (classes.txt, images/, labels/)
Keluaran: dataset_bbox_split/ (data.yaml, images/{train,val,test}, labels/{...})

Pemakaian: python 05_split_dataset_yolo.py [folder_masukan] [folder_keluaran]

Frame disortir per kelas, lalu dibagi train/val/test secara berurutan. Dua frame
di sekitar tiap batas split dikeluarkan untuk mengurangi kebocoran frame video
yang berdekatan. Folder keluaran yang sudah ada tidak ditimpa.
"""
import shutil
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "dataset_bbox")
DST = Path(sys.argv[2] if len(sys.argv) > 2 else "dataset_bbox_split")
GUARD = 2  # jumlah sampel yang dibuang di sekitar tiap batas

if DST.exists():
    raise SystemExit(f"Folder keluaran {DST} sudah ada; hapus/pindahkan secara manual sebelum membuat ulang.")
classes_file = SRC / "classes.txt"
if not classes_file.is_file():
    raise SystemExit(f"File kelas tidak ditemukan: {classes_file}")
classes = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
if not classes:
    raise SystemExit("classes.txt kosong.")

splits = {name: [] for name in ("train", "val", "test")}
excluded = 0
for class_id, class_name in enumerate(classes):
    images = sorted(
        p for p in (SRC / "images").iterdir()
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
        and p.name.startswith(f"{class_name}__")
    )
    if len(images) < 10:
        raise SystemExit(f"Kelas {class_name} hanya memiliki {len(images)} gambar.")
    labels = SRC / "labels"
    missing = [p.name for p in images if not (labels / f"{p.stem}.txt").is_file()]
    if missing:
        raise SystemExit(f"{class_name}: {len(missing)} gambar tidak punya label, contoh: {missing[:3]}")

    n = len(images)
    train_end = int(n * 0.70)
    val_end = int(n * 0.90)
    train = images[:train_end - GUARD]
    val = images[train_end + GUARD:val_end - GUARD]
    test = images[val_end + GUARD:]
    excluded += n - len(train) - len(val) - len(test)
    splits["train"].extend((class_id, p) for p in train)
    splits["val"].extend((class_id, p) for p in val)
    splits["test"].extend((class_id, p) for p in test)

if sum(map(len, splits.values())) == 0:
    raise SystemExit("Tidak ada gambar untuk dibagi.")

for split_name, entries in splits.items():
    (DST / "images" / split_name).mkdir(parents=True, exist_ok=True)
    (DST / "labels" / split_name).mkdir(parents=True, exist_ok=True)
    for _, image in entries:
        shutil.copy2(image, DST / "images" / split_name / image.name)
        shutil.copy2(SRC / "labels" / f"{image.stem}.txt", DST / "labels" / split_name / f"{image.stem}.txt")
    print(f"{split_name:5s}: {len(entries)} gambar")

yaml = [
    f"path: {DST.resolve().as_posix()}",
    "train: images/train",
    "val: images/val",
    "test: images/test",
    "names:",
] + [f"  {i}: {name}" for i, name in enumerate(classes)]
(DST / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
print(f"Dikeluarkan di sekitar batas split: {excluded} gambar")
print(f"data.yaml dibuat di {DST / 'data.yaml'}")
