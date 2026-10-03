# Perbandingan hasil yang tersedia

| Model | Mode | Best val accuracy | Epoch ≥90% | Waktu training (s) |
|---|---|---:|---:|---:|
| mobilenet_v3_small | feature | 1.0000 | 1 | 16.595 |
| mobilenet_v3_small | partial | 1.0000 | 1 | 15.549 |
| mobilenet_v3_small | scratch | 0.5000 | — | 16.120 |
| mobilenet_v3_large | feature | 1.0000 | 1 | 15.882 |
| mobilenet_v3_large | partial | 1.0000 | 1 | 16.020 |
| mobilenet_v3_large | scratch | 0.5000 | — | 18.277 |
| efficientnet_b0 | feature | 1.0000 | 1 | 16.952 |
| efficientnet_b0 | partial | 1.0000 | 1 | 17.017 |
| efficientnet_b0 | scratch | 1.0000 | 7 | 21.437 |
| resnet18 | feature | 1.0000 | 1 | 16.406 |
| resnet18 | partial | 1.0000 | 1 | 16.739 |
| resnet18 | scratch | 1.0000 | 1 | 19.495 |
| resnet50 | feature | 1.0000 | 1 | 19.462 |
| resnet50 | partial | 1.0000 | 1 | 20.573 |
| resnet50 | scratch | 1.0000 | 2 | 27.293 |

Hanya konfigurasi dengan tabel dan checkpoint tersedia yang dicantumkan.
Perbandingan memerlukan split, preprocessing, epoch, dan konfigurasi training yang sama.
Hasil lama tidak otomatis dilatih ulang. Validation temporal bukan evaluasi lintas sesi.
