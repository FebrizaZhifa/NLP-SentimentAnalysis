# Evaluasi Sentimen IMDB di Streamlit

Aplikasi ini membandingkan akurasi validasi model RNN, LSTM, dan Transformer
pada dataset ulasan film IMDB.

## Menjalankan secara lokal

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Dataset `IMDB_MOVIE_REVIEW.csv` harus berada di folder yang sama dengan
`app.py` dan ikut diunggah ke repository. Dashboard memuat dataset tersebut
secara otomatis; pengunjung tidak perlu mengunggah CSV. Pengguna mengatur
eksperimen dari panel kiri lalu menekan **Mulai evaluasi**. Training dapat
memerlukan waktu, terutama jika jumlah ulasan atau epoch diperbesar.

## Deploy ke Streamlit Community Cloud

1. Push `app.py` dan `IMDB_MOVIE_REVIEW.csv` ke repository GitHub, dalam folder
   yang sama. Letakkan `requirements.txt` di root repository. Pastikan
   repository dapat diakses akun yang digunakan untuk Streamlit.
2. Buka [share.streamlit.io](https://share.streamlit.io/) dan masuk dengan
   GitHub.
3. Pilih **Create app**, pilih repository dan branch, lalu isi **Main file path**
   dengan path `app.py` yang sesuai lokasi file (contoh `NLP/app.py` jika
   aplikasi berada di subfolder).
4. Pilih **Deploy**. Streamlit akan memasang dependensi dari `requirements.txt`.

Dataset bawaan berukuran sekitar 53 MB. Aplikasi menyediakan batas jumlah
ulasan dan epoch agar eksperimen dapat disesuaikan dengan kapasitas Cloud.
