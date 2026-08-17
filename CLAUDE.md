# Petunjuk untuk Claude Code

**Baca `.claude/context.md` lebih dulu sebelum mengerjakan apa pun di project
ini.** Berkas itu memuat konteks lengkapnya, yaitu apa project ini, cara kerja
simulasinya, keputusan rancangan yang sudah diambil, status pekerjaan sekarang,
dan jebakan teknis yang sudah pernah ditemukan.

Tiga hal yang berlaku sejak awal sesi, sebelum sempat membaca berkas itu.

1. **Data hasil simulasi tidak boleh diubah, dihapus, atau ditimpa.** Ini hard
   constraint dari peneliti. Data hasil adalah patokan kebenaran.
2. **Jangan menjalankan korpus penuh tanpa diperintahkan.** Skalanya sekitar
   1.120 run untuk v_2 saja, yaitu puluhan jam. Bahas cakupannya dulu.
3. **Gaya penulisan.** Bahasa Indonesia, tanpa em-dash, hindari titik koma dan
   titik dua di tengah kalimat, satu gambar satu panel.

Seluruh perintah dijalankan dari root `clean_code/`, bukan dari dalam subfolder.
Interpreter ada di `.venv/Scripts/python.exe` (Python 3.12.7).
