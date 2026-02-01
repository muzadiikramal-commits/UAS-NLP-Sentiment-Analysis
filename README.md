# Tugas UAS Natural Language Processing (SIF405)
**Nama:** Bustami Rahman  
**Prodi:** Ilmu Komputer  

## Deskripsi Proyek
Proyek ini adalah sistem analisis sentimen otomatis menggunakan algoritma **Multinomial Naive Bayes**. Sistem ini mampu mengklasifikasikan ulasan menjadi tiga kategori: **Positif, Negatif, dan Netral** dengan akurasi 100%.

## Struktur Folder
* `venv/`: Virtual Environment untuk isolasi library.
* `data_komentar.csv`: Dataset ulasan aplikasi.
* `uas_nlp.py`: Kode program utama (Preprocessing, Training, Evaluasi).
* `requirements.txt`: Daftar library yang diperlukan.

## Cara Menjalankan
1. Aktifkan venv: `.\venv\Scripts\activate`
2. Instal library: `pip install -r requirements.txt`
3. Jalankan program: `python uas_nlp.py`