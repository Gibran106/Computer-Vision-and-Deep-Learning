# Desain awal P2 — resnet18

Tujuan: klasifikasi box_merah dan box_cokelat serta membandingkan tiga mode
pelatihan pada resnet18. Dataset 200 citra seimbang, disertai metadata.csv.
Input 224×224; loss CrossEntropyLoss; output dua logits dengan label tercatat
pada checkpoint. Perangkat CUDA otomatis, CPU jika CUDA tidak tersedia.

| Keluaran slide 23 | Berkas/folder pada proyek ini |
|---|---|
| Dokumen desain awal | DESIGN.md |
| Dataset raw dan metadata, ≥50 citra/kelas | dataset_raw/, metadata.csv (100 citra/kelas) |
| Tabel tiga mode | results/experiments.csv dan experiments.md setelah training |
| Grafik accuracy per epoch | results/accuracy_per_epoch.png dan <mode>_curves.png |
| Latensi model pilihan | results/latency.csv setelah latency.py dijalankan |
| Analisis hasil model dan perbandingan antarmodel | README.md dan hasil aktual di results/ |

Gunakan split temporal yang sama untuk semua mode dan arsitektur. Risiko
utama: frame dari satu video per kelas serta resolusi/latar berbeda antar
kelas dapat memberi petunjuk selain objek. Guard band mengurangi kebocoran
frame berdekatan; penilaian lintas sesi memerlukan rekaman tambahan.
Tabel/grafik tidak diisi dengan hasil yang belum dilatih. Mode terbaik dipilih
dari best validation accuracy dengan mempertimbangkan waktu training dan
latensi aktual; validation dari satu sesi memiliki keterbatasan di atas.
