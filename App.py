import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertForSequenceClassification

st.set_page_config(
    page_title="CheckShop · English E-Commerce Review Analyzer",
    page_icon="🛍️",
    layout="wide",
)


st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Background */
  .stApp { background: #0f1117; color: #e8eaf0; }

  /* Hide default header */
  header[data-testid="stHeader"] { background: transparent; }

  /* Hero */
  .hero {
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid #1e2130;
    margin-bottom: 2rem;
  }
  .hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.3rem;
    letter-spacing: -0.5px;
  }
  .hero p { color: #7b8099; font-size: 1rem; margin: 0; }
  .hero span { color: #6c8eff; }

  /* Input area */
  .stTextArea textarea {
    background: #161925 !important;
    border: 1px solid #272c3f !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
    font-size: 1rem !important;
    padding: 14px !important;
  }
  .stTextArea textarea:focus {
    border-color: #6c8eff !important;
    box-shadow: 0 0 0 2px rgba(108,142,255,0.15) !important;
  }

  /* Buttons */
  .stButton > button {
    background: #6c8eff !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.8rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: background 0.2s !important;
  }
  .stButton > button:hover { background: #5570e0 !important; }

  /* Result cards */
  .result-card {
    background: #161925;
    border: 1px solid #1e2336;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
  }
  .result-card .model-name {
    font-size: 0.75rem;
    font-weight: 600;
    color: #7b8099;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
  }
  .result-card .verdict {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0;
  }
  .verdict-positive { color: #4cde80; }
  .verdict-neutral  { color: #f5a623; }
  .verdict-negative { color: #ff5f5f; }

  /* Confidence badge */
  .conf-badge {
    display: inline-block;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 20px;
    margin-top: 4px;
  }
  .conf-high   { background: #1a3d2b; color: #4cde80; }
  .conf-medium { background: #3d2e0f; color: #f5a623; }
  .conf-low    { background: #3d1515; color: #ff5f5f; }

  /* Section header */
  .section-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #7b8099;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2130;
  }

  /* Divider */
  hr { border-color: #1e2130 !important; }

  /* Matplotlib transparent */
  .stPlotlyChart, .stPyplot { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

load_nltk()
stemmer    = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_base(text):
    text = str(text).replace('\u200b', '').lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'_', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def clean_classical(text):
    tokens = word_tokenize(clean_base(text))
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words and len(t) > 1]
    return ' '.join(tokens)

@st.cache_resource
def load_models():
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('lr_model.pkl', 'rb') as f:
        lr = pickle.load(f)
    with open('nb_model.pkl', 'rb') as f:
        nb = pickle.load(f)

    import xgboost as xgb
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model('xgb_model.json')

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert_best')
    bert      = BertForSequenceClassification.from_pretrained('bert_best').to(device)
    bert.eval()

    return tfidf, le, lr, nb, xgb_model, bert, tokenizer, device


def predict(text, tfidf, le, lr, nb, xgb_model, bert, tokenizer, device):
    text_cl = clean_classical(text)
    text_be = clean_base(text)

    vec      = tfidf.transform([text_cl])
    classes  = le.classes_   # ['Negative', 'Neutral', 'Positive']

    results = {}
    for name, model in [('Logistic Regression', lr), ('Naive Bayes', nb), ('XGBoost', xgb_model)]:
        proba = dict(zip(classes, model.predict_proba(vec)[0]))
        pred  = classes[np.argmax(list(proba.values()))]
        results[name] = {'pred': pred, 'proba': proba}

    enc = tokenizer(
        [text_be], max_length=128,
        padding='max_length', truncation=True, return_tensors='pt'
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = bert(**enc).logits
    proba_bert = torch.softmax(logits, dim=1).cpu().numpy()[0]
    proba_dict = dict(zip(classes, proba_bert))
    pred_bert  = classes[proba_bert.argmax()]
    results['BERT'] = {'pred': pred_bert, 'proba': proba_dict}

    return results

COLORS = {
    'Positive': '#4cde80',
    'Neutral':  '#f5a623',
    'Negative': '#ff5f5f',
}
VERDICT_CLASS = {
    'Positive': 'verdict-positive',
    'Neutral':  'verdict-neutral',
    'Negative': 'verdict-negative',
}

def conf_badge(confidence):
    if confidence >= 0.75:
        return f'<span class="conf-badge conf-high">{confidence*100:.1f}% confident</span>'
    elif confidence >= 0.5:
        return f'<span class="conf-badge conf-medium">{confidence*100:.1f}% confident</span>'
    else:
        return f'<span class="conf-badge conf-low">{confidence*100:.1f}% confident</span>'

def plot_proba(results):
    classes = ['Negative', 'Neutral', 'Positive']
    models  = list(results.keys())
    bar_colors = [COLORS[c] for c in classes]

    fig, axes = plt.subplots(1, len(models), figsize=(14, 3.2), facecolor='#0f1117')
    for ax, model_name in zip(axes, models):
        vals = [results[model_name]['proba'].get(c, 0) for c in classes]
        bars = ax.bar(classes, vals, color=bar_colors, width=0.55, edgecolor='#0f1117', linewidth=1.5)
        ax.set_facecolor('#161925')
        ax.set_ylim(0, 1.15)
        ax.set_title(model_name, color='#c0c5d8', fontsize=9.5, fontweight='600', pad=8)
        ax.tick_params(colors='#7b8099', labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_color('#272c3f')
        ax.yaxis.set_tick_params(color='#272c3f')
        ax.xaxis.set_tick_params(color='#272c3f')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.03,
                    f'{val:.2f}', ha='center', color='#c0c5d8', fontsize=8.5, fontweight='600')

    plt.tight_layout(pad=1.2)
    return fig

st.markdown("""
<div class="hero">
  <h1>👾🛍️ CheckShop</h1>
  <p>English e-commerce review sentiment analyzer &mdash; powered by <span>4 models</span></p>
  <p>Naive Bayes, Logistic Regression, XGBOOST, BERT<p>
</div>
""", unsafe_allow_html=True)

with st.spinner('Loading models...'):
    try:
        tfidf, le, lr, nb, xgb_model, bert, tokenizer, device = load_models()
        models_ready = True
    except Exception as e:
        st.error(f"⚠️ Gagal load model: {e}\n\nPastikan file berikut ada di direktori yang sama:\n`tfidf_vectorizer.pkl`, `label_encoder.pkl`, `lr_model.pkl`, `nb_model.pkl`, `xgb_model.json`, folder `bert_best/`")
        models_ready = False

col_input, col_gap = st.columns([3, 1])
with col_input:
    st.markdown('<div class="section-label">Masukkan teks review</div>', unsafe_allow_html=True)
    user_text = st.text_area(
        label="review_input",
        label_visibility="collapsed",
        placeholder="Contoh: The product quality is very good and delivery was fast!",
        height=120,
    )
    analyze_btn = st.button("Analisis Sentimen →", disabled=not models_ready)

st.markdown("<hr>", unsafe_allow_html=True)

# Results
if analyze_btn and user_text.strip():
    with st.spinner('Menganalisis...'):
        results = predict(user_text, tfidf, le, lr, nb, xgb_model, bert, tokenizer, device)

    # Verdict cards
    st.markdown('<div class="section-label">Prediksi per Model</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    model_order = ['Logistic Regression', 'Naive Bayes', 'XGBoost', 'BERT']
    for col, name in zip(cols, model_order):
        r    = results[name]
        pred = r['pred']
        conf = max(r['proba'].values())
        with col:
            st.markdown(f"""
            <div class="result-card">
              <div class="model-name">{name}</div>
              <div class="verdict {VERDICT_CLASS[pred]}">{pred}</div>
              {conf_badge(conf)}
            </div>
            """, unsafe_allow_html=True)

    # Agreement badge
    preds = [results[m]['pred'] for m in model_order]
    if len(set(preds)) == 1:
        st.success(f"✅ Semua model sepakat: **{preds[0]}**")
    else:
        majority = max(set(preds), key=preds.count)
        st.warning(f"⚠️ Model tidak sepenuhnya sepakat. Mayoritas: **{majority}**")

    # Probability chart
    st.markdown('<br><div class="section-label">Distribusi Probabilitas per Kelas</div>', unsafe_allow_html=True)
    fig = plot_proba(results)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Detail table
    st.markdown('<br><div class="section-label">Tabel Probabilitas Lengkap</div>', unsafe_allow_html=True)
    rows = []
    for name in model_order:
        r = results[name]
        rows.append({
            'Model'    : name,
            'Prediksi' : r['pred'],
            'Negative' : f"{r['proba'].get('Negative', 0):.3f}",
            'Neutral'  : f"{r['proba'].get('Neutral', 0):.3f}",
            'Positive' : f"{r['proba'].get('Positive', 0):.3f}",
        })
    st.dataframe(pd.DataFrame(rows).set_index('Model'), use_container_width=True)

elif analyze_btn and not user_text.strip():
    st.warning("Teks tidak boleh kosong.")
