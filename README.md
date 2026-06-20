# 🔍 SentiScope — Bangla E-Commerce Sentiment Analyzer

Aplikasi analisis sentimen ulasan e-commerce berbahasa Bangla menggunakan 4 model machine learning: **Logistic Regression**, **Naive Bayes**, **XGBoost**, dan **BERT**.

---

## 📁 Struktur Project

```
project/
├── app.py                    # Streamlit app
├── requirements.txt          # Dependencies
├── README.md
│
├── tfidf_vectorizer.pkl      # TF-IDF vectorizer (fit dari training)
├── label_encoder.pkl         # Label encoder (Negative/Neutral/Positive)
├── lr_model.pkl              # Model Logistic Regression
├── nb_model.pkl              # Model Naive Bayes
├── xgb_model.json            # Model XGBoost
│
└── bert_best/                # Model BERT hasil fine-tuning
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json
```

---

## 🗂️ Dataset

**BanglaEcomReviewCorpus** — dataset ulasan produk e-commerce berbahasa Bangla dengan terjemahan Inggris.

| Kolom | Keterangan |
|---|---|
| `Sentence` | Teks ulasan dalam aksara Bangla |
| `English Translation` | Terjemahan ke bahasa Inggris |
| `Sentiment` | Label sentimen: `Positive`, `Neutral`, `Negative` |

**Statistik dataset:**
- Total data: 8.685 baris
- Setelah preprocessing (drop duplikat & konflik): ±8.587 baris
- Split: 80% train / 10% val / 10% test

---

## ⚙️ Preprocessing

Model klasikal (LR, NB, XGBoost) dan BERT menggunakan jalur preprocessing yang berbeda:

```
Input teks
  ├── clean_classical()  →  TF-IDF  →  LR / Naive Bayes / XGBoost
  │     • Hapus Zero Width Space (U+200B)
  │     • Lowercase, hapus URL, angka, tanda baca
  │     • Stopword removal (NLTK English)
  │     • Stemming (PorterStemmer)
  │
  └── clean_base()  →  BertTokenizer  →  BERT
        • Hapus Zero Width Space (U+200B)
        • Lowercase, hapus URL, angka, tanda baca
        • (tanpa stopword removal & stemming)
```

**TF-IDF config:** `max_features=10000`, `ngram_range=(1,2)`, `sublinear_tf=True`

**BERT config:** `bert-base-uncased`, `max_length=128`, fine-tuned 4 epoch dengan `AdamW` + linear warmup scheduler

---

## 🤖 Model

| Model | Representasi Fitur | Library |
|---|---|---|
| Logistic Regression | TF-IDF | scikit-learn |
| Naive Bayes (Multinomial) | TF-IDF | scikit-learn |
| XGBoost | TF-IDF | xgboost |
| BERT | Token ID + Attention Mask | HuggingFace Transformers |

---

## 🚀 Cara Menjalankan

### Lokal

**1. Clone repository**
```bash
git clone https://github.com/username/sentiscope.git
cd sentiscope
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Pastikan semua file model tersedia** (lihat struktur project di atas)

**4. Jalankan aplikasi**
```bash
streamlit run app.py
```

Buka browser di `http://localhost:8501`

---

### Deploy ke Streamlit Cloud

1. Push seluruh project ke GitHub (termasuk folder `bert_best/` dan semua file `.pkl`)
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Connect ke repository → pilih `app.py` sebagai main file
4. Klik **Deploy**

> ⚠️ **Catatan ukuran file:** `model.safetensors` di folder `bert_best/` berukuran sekitar 400MB. Streamlit Cloud gratis membatasi ukuran repo hingga 1GB, jadi masih aman namun mendekati batas. Alternatifnya, upload model ke [HuggingFace Hub](https://huggingface.co) dan load menggunakan `from_pretrained('username/model-name')`.

---

## 📓 Notebook

| File | Keterangan |
|---|---|
| `EDA_Lengkap.ipynb` | Exploratory Data Analysis lengkap |
| `preprocessing_pipeline.ipynb` | Pipeline preprocessing lengkap |
| `model_training_evaluation.ipynb` | Training, evaluasi, dan inference semua model |

---

## 📦 Dependencies

```
streamlit>=1.35.0
pandas
numpy
scikit-learn
xgboost
transformers
torch
nltk
matplotlib
```

Install sekaligus:
```bash
pip install -r requirements.txt
```

---

## 👥 Tim

> Ganti bagian ini dengan nama anggota tim kalian.

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan penelitian/akademik.