# Hasil eksperimen aktual

| Arsitektur | Mode | Best val accuracy | Best epoch | Epoch ≥90% | Waktu training (s) | Parameter dilatih / total | Epoch dijalankan |
|---|---|---:|---:|---:|---:|---:|---:|
| resnet18 | feature | 1.0000 | 2 | 1 | 16.406 | 1026 / 11177538 | 10 |
| resnet18 | partial | 1.0000 | 1 | 1 | 16.739 | 8394754 / 11177538 | 10 |
| resnet18 | scratch | 1.0000 | 1 | 1 | 19.495 | 11177538 / 11177538 | 10 |

Hanya konfigurasi yang memiliki hasil training dicantumkan. Tanda — berarti data tidak tersedia atau ambang belum tercapai.
CSV lama tetap dapat dibaca; jumlah parameter dan epoch yang belum dicatat tidak diisi dengan perkiraan.
Validation berasal dari holdout temporal satu video per kelas, sehingga bukan evaluasi lintas sesi.
