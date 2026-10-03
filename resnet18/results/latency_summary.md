# Hasil pengukuran latency

Perangkat: NVIDIA GeForce RTX 4050 Laptop GPU (cuda); PyTorch 2.14.0+cu130.
Input 224×224, batch 1, CPU threads 1, warm-up 20, iterasi 100.

| Model | Mode | Median (ms) | Mean (ms) | P95 (ms) |
|---|---|---:|---:|---:|
| ResNet-18 | feature | 1.617 | 1.618 | 1.627 |
| ResNet-18 | partial | 1.396 | 1.388 | 1.409 |
| ResNet-18 | scratch | 1.408 | 1.406 | 1.420 |

Batang menunjukkan median; penanda P95 menunjukkan persentil ke-95, bukan confidence interval.
Pengukuran mencakup forward pass PyTorch; tidak termasuk kamera, preprocessing, postprocessing, atau ROS2.
Acuan slide 16: anggaran inferensi 35 ms dalam total sekitar 67 ms/frame (15 FPS).
Anggaran PPT adalah target, bukan hasil pengukuran pipeline robot.
Baris terbaru per model/mode dipilih dalam konfigurasi pengukuran yang sama, termasuk hash gambar.
Auto memilih kelompok dengan konfigurasi terbanyak; jika seri, memilih kelompok paling baru.
Konfigurasi yang tidak ada dalam kelompok tersebut tidak ditampilkan.
Hasil satu sesi pada GPU laptop tidak menjamin kinerja pada komputer robot.
