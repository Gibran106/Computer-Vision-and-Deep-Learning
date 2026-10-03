# Hasil pengukuran latency

Perangkat: NVIDIA GeForce RTX 4050 Laptop GPU (cuda); PyTorch 2.14.0+cu130.
Input 224×224, batch 1, CPU threads 1, warm-up 20, iterasi 100.

| Model | Mode | Median (ms) | Mean (ms) | P95 (ms) |
|---|---|---:|---:|---:|
| MobileNetV3-Small | feature | 1.820 | 1.840 | 1.926 |
| MobileNetV3-Small | partial | 1.819 | 1.842 | 1.901 |
| MobileNetV3-Small | scratch | 1.834 | 1.850 | 1.929 |
| MobileNetV3-Large | feature | 2.118 | 2.175 | 2.471 |
| MobileNetV3-Large | partial | 2.290 | 2.285 | 2.456 |
| MobileNetV3-Large | scratch | 2.157 | 2.165 | 2.231 |
| EfficientNet-B0 | feature | 2.805 | 2.824 | 2.936 |
| EfficientNet-B0 | partial | 2.802 | 2.824 | 2.946 |
| EfficientNet-B0 | scratch | 2.807 | 2.822 | 2.886 |
| ResNet-18 | feature | 1.617 | 1.618 | 1.627 |
| ResNet-18 | partial | 1.396 | 1.388 | 1.409 |
| ResNet-18 | scratch | 1.408 | 1.406 | 1.420 |
| ResNet-50 | feature | 2.852 | 2.982 | 3.229 |
| ResNet-50 | partial | 2.857 | 2.856 | 2.880 |
| ResNet-50 | scratch | 2.886 | 2.888 | 2.909 |

Batang menunjukkan median; penanda P95 menunjukkan persentil ke-95, bukan confidence interval.
Pengukuran mencakup forward pass PyTorch; tidak termasuk kamera, preprocessing, postprocessing, atau ROS2.
Acuan slide 16: anggaran inferensi 35 ms dalam total sekitar 67 ms/frame (15 FPS).
Anggaran PPT adalah target, bukan hasil pengukuran pipeline robot.
Baris terbaru per model/mode dipilih dalam konfigurasi pengukuran yang sama, termasuk hash gambar.
Auto memilih kelompok dengan konfigurasi terbanyak; jika seri, memilih kelompok paling baru.
Konfigurasi yang tidak ada dalam kelompok tersebut tidak ditampilkan.
Hasil satu sesi pada GPU laptop tidak menjamin kinerja pada komputer robot.
