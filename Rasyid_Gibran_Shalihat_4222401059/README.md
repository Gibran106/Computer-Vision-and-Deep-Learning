# PJ1 — Object Detection dengan YOLO

**Nama:** Rasyid Gibran Shalihat  
**NIM:** 4222401059

Proyek ini menyiapkan anotasi bounding box, membagi data, lalu membandingkan tiga eksperimen YOLO pada kelas `red_earth_block`, `blue_earth_block`, dan `sky_block`.

## Eksperimen

- **E1 — pretrained:** baseline dengan bobot pretrained COCO.
- **E2 — scratch:** arsitektur yang sama dilatih dari bobot acak.
- **E3 — tuning:** baseline pretrained dengan satu perubahan: E3 mengubah ukuran gambar dari 416 menjadi 480.

E1 dan E2 memakai 30 epoch, ukuran gambar 416, dan batch 8. Semua eksperimen menggunakan split dataset yang sama. Hasil aktual tersimpan di laporan CSV/TXT serta grafik pada `hasil_praktik/` setelah training.

## Dataset

Dataset split berada di `../dataset_bbox_split/` dan dipakai bersama oleh salinan PJ1 di folder `PJ1_Teman`. Kalau folder dipindahkan ke komputer lain, sesuaikan `path` di `dataset_bbox_split/data.yaml`.

## Menjalankan training

Dari folder proyek ini, siapkan dependensi (`pip install ultralytics`) lalu jalankan:

```bash
python 06_latih_bandingkan.py --nim 4222401059 --nama "Rasyid Gibran Shalihat" --data ../dataset_bbox_split/data.yaml --model yolo26n --epochs 30 --imgsz 416 --batch 8 --device 0 --ubah imgsz=480
```

Laporan, grafik, dan checkpoint tersimpan di `hasil_praktik/`. Skrip `00`–`05` menjelaskan tahap pemeriksaan serta pemrosesan anotasi; presentasi tugas ada di `RET503_Data dan YOLO.pptx`.
