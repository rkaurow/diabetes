# 🚀 Panduan Belajar: Membedah Aplikasi Deteksi Dini Diabetes
*Spesial disusun untuk pemula & anak SMA agar mudah paham cara kerja IT dan Artificial Intelligence (AI)!*

Halo! Pernah penasaran nggak sih gimana caranya sebuah website bisa "meramal" apakah seseorang berisiko kena penyakit diabetes atau nggak? Di aplikasi ini, kita menggabungkan **Pemrograman Web** dan **Kecerdasan Buatan (Machine Learning)** untuk melakukan keajaiban itu. 

Yuk, kita bedah fitur-fiturnya satu per satu dari kacamata anak SMA!

---

## 1. Fitur Kalkulator BMI (Body Mass Index) ⚖️
BMI atau Indeks Massa Tubuh adalah angka yang ngasih tahu apakah berat badan kita ideal atau nggak dibandingkan dengan tinggi badan kita.

**🧠 Rumus Matematika yang Dipakai:**
```text
BMI = Berat Badan (kg) / (Tinggi Badan (m) × Tinggi Badan (m))
```
*Catatan: Tinggi badan di aplikasi diubah dulu dari centimeter (cm) ke meter (m) dengan cara dibagi 100.*

**💻 Logika Pemrograman (Kondisi If-Else):**
- Jika **BMI < 18.5** ➔ Status: *Underweight* (Kekurangan berat badan)
- Jika **BMI antara 18.5 - 24.9** ➔ Status: *Normal* (Ideal!)
- Jika **BMI antara 25 - 29.9** ➔ Status: *Overweight* (Kelebihan berat badan)
- Jika **BMI ≥ 30** ➔ Status: *Obesitas* (Sangat bahaya untuk diabetes)

---

## 2. Fitur Skrining Pasien (Pengumpulan Data) 📋
Aplikasi akan menanyakan berbagai hal tentang gaya hidupmu (seperti: apakah sering makan buah? apakah merokok? apakah punya darah tinggi?). Komputer nggak ngerti kata "Ya" atau "Tidak". Jadi, di balik layar, jawabanmu diubah jadi angka!
- **Ya** = `1`
- **Tidak** = `0`

Kumpulan angka-angka inilah (ada sekitar 21 pertanyaan) yang nantinya disetor ke si "Otak AI" untuk dihitung.

---

## 3. Otak Utamanya: 3 Model Machine Learning (AI) 🧠🤖
Nah, ini bagian paling keren! Aplikasi ini nggak cuma pakai 1 otak AI, tapi pakai **3 otak AI** sekaligus untuk berdiskusi. Bayangkan kamu lagi konsultasi ke 3 dokter spesialis yang berbeda.

### A. Gradient Boosting (GB) - "Si Perfeksionis" 🎯
Algoritma ini cara kerjanya mirip siswa yang rajin ikut tryout. Dia kerjain soal (menebak), kalau salah, dia catat kesalahannya. Di tryout berikutnya, dia akan fokus memperbaiki kelemahan yang tadi sampai akhirnya akurasinya jadi sangat tinggi. Di aplikasi kita, ini adalah **model utama** yang paling pintar!

### B. Random Forest (RF) - "Voting 100 Dokter" 🌳
Algoritma ini cara kerjanya mirip musyawarah. Bayangkan ada 100 "pohon keputusan" (decision tree). Setiap pohon akan menebak "Dia diabetes!" atau "Dia aman!". Setelah semua menebak, mereka akan melakukan *voting* (pemungutan suara). Suara terbanyak yang menang. Karena keputusannya diambil bareng-bareng, algoritma ini sangat stabil dan jarang ngaco.

### C. Logistic Regression (LR) - "Si Analis Statistik" 📈
Ini adalah algoritma klasik yang pakai rumus matematika regresi. Bayangkan dia menggambar sebuah garis batas di tengah lapangan. Kalau data kamu jatuh di sebelah kiri garis, kamu "Aman". Kalau jatuh di sebelah kanan, kamu "Bahaya".

### ⚖️ Bagaimana Hasil Akhirnya Ditentukan?
Biar adil, aplikasi ini menghitung **Nilai Tengah (Rata-rata)** dari prediksi ketiga "dokter" tadi!
**Rumus di Aplikasi:**
```javascript
Skor_Akhir = (Skor_GB + Skor_RF + Skor_LR) / 3
```
*Contoh: GB menebak risiko 40%, RF menebak 45%, LR menebak 35%. Maka skor akhirmu = (40 + 45 + 35) / 3 = 40%.*

---

## 4. Kategori Risiko & Penarikan Kesimpulan 🚦
Setelah dapat *Skor_Akhir* berupa persentase (0% sampai 100%), aplikasi pakai logika `If-Else` sederhana untuk ngasih kamu warna dan peringatan:

- 🟢 **Risiko Rendah (< 20%)**: Aman! Teruskan gaya hidup sehatmu.
- 🟡 **Risiko Sedang (20% - 49%)**: Waspada! Kamu punya beberapa kebiasaan buruk yang bisa jadi cikal bakal diabetes.
- 🔴 **Risiko Tinggi (≥ 50%)**: Bahaya! Sebaiknya segera cek gula darah ke puskesmas/rumah sakit.

---

## 5. Fitur HealthAssistant AI (Chatbot Pintar) 💬
Setelah tahu hasilnya, kamu mungkin bingung, "Terus aku harus makan apa? Olahraga apa?". Tenang, aplikasi ini punya fitur Chatbot!

**Bagaimana cara kerjanya?**
Aplikasi kita menggunakan sesuatu yang disebut **API (Application Programming Interface)**. Anggap API ini seperti pramusaji di restoran. 
1. Aplikasi web kita (si Pelanggan) menitipkan pesan: *"Tolong kasih tahu anak SMA yang punya BMI 26 dan jarang makan sayur ini, gimana cara diet yang seru?"*
2. Lewat internet, pesan itu dikirim ke server AI pintar (bernama Sumopod / GPT-4).
3. Server AI memikirkan jawabannya, lalu mengembalikannya ke aplikasi web kita.
4. Voila! Teks jawabannya muncul di layar hp kamu!

Semua proses ngobrol ini bahkan sudah disesuaikan secara otomatis dengan hasil tes kamu, jadi sarannya sangat personal!

---

## 💡 Kesimpulan Pembelajaran
Membuat aplikasi seperti ini ternyata seru kan? Kamu belajar bahwa:
1. **Frontend (Tampilan Visual):** Dibuat pakai HTML dan Tailwind CSS supaya terlihat cantik dan kekinian.
2. **Logika UI (Matematika Dasar):** Pakai JavaScript untuk menghitung BMI dan mencari rata-rata 3 model algoritma.
3. **Backend & AI (Otak Utama):** Pakai Python dan Machine Learning (Library Scikit-Learn/Joblib) untuk memproses data medis menjadi persentase risiko yang akurat!
