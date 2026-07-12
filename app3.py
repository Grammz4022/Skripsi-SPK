import re
import ast
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

from gensim import corpora
from gensim.models import LdaModel, CoherenceModel

from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

import multiprocessing

warnings.filterwarnings("ignore")

LABEL_TOPIK = {
    0: "Kuliner & Makanan",
    1: "Fashion & Aksesoris",
    2: "Konten Digital & Media",
    3: "Kerajinan & Kriya",
    4: "Jasa Kreatif & Desain",
    5: "Teknologi & Startup",
    6: "Pendidikan Kreatif",
    7: "Pariwisata & Event",
    8: "Umum / Campuran",
    9: "Ekonomi & UMKM",
}

MODEL_NAME = "mdhugol/indonesia-bert-sentiment-classification"
LABEL_MAP = {"LABEL_0": "Positif", "LABEL_1": "Netral", "LABEL_2": "Negatif"}


@st.cache_data(show_spinner="Mencari jumlah topik optimal...")
def cari_topik_optimal(tokens_tuple):
    texts = [list(t) for t in tokens_tuple]
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=2, no_above=0.9)
    corpus = [dictionary.doc2bow(t) for t in texts]

    topic_range = list(range(2, 11))
    coherence_scores = []
    for n_topics in topic_range:
        lda_tmp = LdaModel(
            corpus=corpus, id2word=dictionary, num_topics=n_topics,
            random_state=42, passes=10, alpha="auto", per_word_topics=True,
        )
        cm = CoherenceModel(model=lda_tmp, texts=texts, dictionary=dictionary, coherence="c_v", processes=1)
        coherence_scores.append(cm.get_coherence())

    best_n = topic_range[int(np.argmax(coherence_scores))]
    return dictionary, corpus, topic_range, coherence_scores, best_n


@st.cache_resource(show_spinner="Melatih model LDA final...")
def latih_lda_final(tokens_tuple, best_n):
    texts = [list(t) for t in tokens_tuple]
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=2, no_above=0.9)
    corpus = [dictionary.doc2bow(t) for t in texts]
    lda_model = LdaModel(
        corpus=corpus, id2word=dictionary, num_topics=best_n,
        random_state=42, passes=20, alpha="auto", eta="auto", per_word_topics=True,
    )
    return lda_model, dictionary, corpus


@st.cache_resource(show_spinner=False)
def muat_model_sentimen():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_sa = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return pipeline(
        "sentiment-analysis", model=model_sa, tokenizer=tokenizer,
        truncation=True, max_length=128,
    )


@st.cache_data(show_spinner="Menjalankan analisis sentimen (sekali saja, hasil akan disimpan)...")
def prediksi_sentimen(teks_tuple):
    sentiment_pipeline = muat_model_sentimen()
    texts = list(teks_tuple)
    BATCH_SIZE = 32
    all_labels, all_scores = [], []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch = [t if str(t).strip() != "" else "tidak ada informasi" for t in batch]
        hasil = sentiment_pipeline(batch)
        for h in hasil:
            all_labels.append(LABEL_MAP[h["label"]])
            all_scores.append(round(h["score"], 4))

    return all_labels, all_scores


def main():

    # ---------------------------------------------------------------------------
    # Konfigurasi halaman & tema visual
    # ---------------------------------------------------------------------------

    st.set_page_config(
        page_title="SPK Usaha Kreatif",
        page_icon="🧭",
        layout="wide",
    )

    INK = "#22252A"
    MUTED = "#6B7280"
    ACCENT = "#A6633D"
    BG = "#F7F5F0"
    SURFACE = "#FFFFFF"
    LINE = "#E4E0D8"

    PALETTE = ["#A6633D", "#4B6B58", "#5B7B93", "#C9A66B", "#8C8377", "#7A5C61"]
    SENTIMEN_WARNA = {"Positif": "#4B7B5B", "Netral": "#B9B2A5", "Negatif": "#B4552F"}

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.edgecolor": LINE,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": LINE,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background-color: {BG};
    }}

    section[data-testid="stSidebar"] {{
        background-color: {SURFACE};
        border-right: 1px solid {LINE};
    }}

    .hero-title {{
        font-family: 'Playfair Display', serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: {INK};
        margin-bottom: 0.2rem;
    }}

    .hero-subtitle {{
        font-size: 1rem;
        color: {MUTED};
        margin-bottom: 1.2rem;
    }}

    .section-label {{
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: {INK};
        margin-top: 0.4rem;
    }}

    hr {{
        border: none;
        border-top: 1px solid {LINE};
        margin: 1.4rem 0;
    }}

    div[data-testid="stMetric"] {{
        background-color: {SURFACE};
        border: 1px solid {LINE};
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }}

    div[data-testid="stMetricLabel"] {{
        color: {MUTED};
    }}

    .stButton > button, .stDownloadButton > button {{
        background-color: {ACCENT};
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1.2rem;
        font-weight: 500;
    }}

    .stButton > button:hover, .stDownloadButton > button:hover {{
        background-color: #8C4F2E;
        color: white;
    }}

    div[data-testid="stFileUploaderDropzone"] {{
        background-color: {SURFACE};
        border: 1px dashed {LINE};
        border-radius: 10px;
    }}
    </style>
    """, unsafe_allow_html=True)


    def header():
        st.markdown(
            """
            <div class="hero-title">Sistem Pendukung Keputusan Usaha Kreatif</div>
            <div class="hero-subtitle">Topic Modeling (LDA), Analisis Sentimen, dan AHP untuk rekomendasi sektor usaha kreatif</div>
            """,
            unsafe_allow_html=True,
        )


    def section(title):
        st.markdown(f'<div class="section-label">{title}</div>', unsafe_allow_html=True)


    # ---------------------------------------------------------------------------
    # Helper
    # ---------------------------------------------------------------------------

    def parse_tokens(x):
        if isinstance(x, list):
            return x
        try:
            return ast.literal_eval(x)
        except Exception:
            return str(x).split()


    def minmax_norm(series):
        mn, mx = series.min(), series.max()
        if mn == mx:
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - mn) / (mx - mn)


    BOBOT_POPULARITAS = 0.50
    BOBOT_POTENSI = 0.30
    BOBOT_RISIKO = 0.20

    # ---------------------------------------------------------------------------
    # Sidebar
    # ---------------------------------------------------------------------------

    header()

    with st.sidebar:
        st.markdown("**Dataset**")
        uploaded_file = st.file_uploader("Upload dataset (.csv)", type=["csv"], label_visibility="collapsed")
        st.markdown("")
        mulai = st.button("Mulai Analisis", use_container_width=True, disabled=uploaded_file is None)
        st.markdown("---")
        st.caption(
            "Pipeline: pembersihan teks → topic modeling (LDA) → analisis sentimen → "
            "perhitungan AHP → rekomendasi sektor usaha."
        )

    if uploaded_file is None:
        st.info("Upload dataset melalui panel di sebelah kiri untuk memulai.")
        st.stop()

    df = pd.read_csv(uploaded_file)

    with st.container(border=True):
        section("Ringkasan Dataset")
        c1, c2, c3 = st.columns(3)
        c1.metric("Jumlah Data", len(df))
        c2.metric("Jumlah Kolom", len(df.columns))
        c3.metric("Status", "Siap Diproses")
        st.dataframe(df.head(), use_container_width=True)

    if "sudah_analisis" not in st.session_state:
        st.session_state.sudah_analisis = False

    if mulai:
        st.session_state.sudah_analisis = True

    if not st.session_state.sudah_analisis:
        st.stop()

    progress = st.progress(0)

    # ---------------------------------------------------------------------------
    # 1. Topic Modeling (LDA)
    # ---------------------------------------------------------------------------

    if "tokens_stem" not in df.columns:
        st.error("Kolom tokens_stem tidak ditemukan.")
        st.stop()
    if "teks_final" not in df.columns:
        st.error("Kolom teks_final tidak ditemukan.")
        st.stop()

    df = df.dropna(subset=["teks_final"]).reset_index(drop=True)
    df["tokens_list"] = df["tokens_stem"].apply(parse_tokens)
    df = df[df["tokens_list"].apply(len) >= 2]

    tokens_tuple = tuple(tuple(t) for t in df["tokens_list"])

    progress.progress(15)

    dictionary, corpus, topic_range, coherence_scores, best_n = cari_topik_optimal(tokens_tuple)

    with st.container(border=True):
        section("Topic Modeling (LDA)")
        st.caption("Menentukan jumlah topik optimal berdasarkan coherence score")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(topic_range, coherence_scores, marker="o", linewidth=2, color=ACCENT)
        ax.axvline(x=best_n, color="#B4552F", linestyle="--", label=f"Optimal: {best_n}")
        ax.set_xlabel("Jumlah Topik")
        ax.set_ylabel("Coherence Score")
        ax.legend(frameon=False)
        st.pyplot(fig)
        st.success(f"Jumlah topik optimal: {best_n}")

    progress.progress(25)

    lda_model, dictionary, corpus = latih_lda_final(tokens_tuple, best_n)

    with st.container(border=True):
        section("Kata Kunci Tiap Topik")
        cols = st.columns(2)
        for i, topic in lda_model.print_topics(num_words=10):
            with cols[i % 2]:
                st.markdown(f"**{LABEL_TOPIK.get(i, f'Topik {i + 1}')}**")
                st.code(topic, language=None)

    progress.progress(35)

    label_topik = {k: v for k, v in LABEL_TOPIK.items() if k < best_n}


    def dominant_topic(bow):
        dist = lda_model.get_document_topics(bow)
        if not dist:
            return -1, 0
        return max(dist, key=lambda x: x[1])


    hasil = [dominant_topic(bow) for bow in corpus]
    df["topik_id"] = [h[0] for h in hasil]
    df["topik_prob"] = [h[1] for h in hasil]
    df["topik_label"] = df["topik_id"].map(label_topik).fillna("Tidak Teridentifikasi")

    progress.progress(45)

    # ---------------------------------------------------------------------------
    # 2. Distribusi topik
    # ---------------------------------------------------------------------------

    with st.container(border=True):
        section("Distribusi Kategori Usaha Kreatif")

        topik_count = df["topik_label"].value_counts()

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(topik_count))]
        bars = ax.barh(topik_count.index, topik_count.values, color=colors, edgecolor="white")
        for bar, val in zip(bars, topik_count.values):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, str(val), va="center", color=INK)
        ax.set_xlabel("Jumlah Tweet")
        st.pyplot(fig)

        st.dataframe(
            topik_count.reset_index().rename(columns={"index": "Kategori", "topik_label": "Jumlah Tweet"}),
            use_container_width=True,
        )

    progress.progress(50)

    # ---------------------------------------------------------------------------
    # 3. Word cloud
    # ---------------------------------------------------------------------------

    with st.container(border=True):
        section("Word Cloud Tiap Topik")

        rows = (best_n + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(16, 4 * rows))
        axes = axes.flatten()

        for i in range(best_n):
            weights = dict(lda_model.show_topic(i, topn=50))
            wc = WordCloud(width=600, height=300, background_color="white", colormap="copper").generate_from_frequencies(weights)
            axes[i].imshow(wc)
            axes[i].axis("off")
            axes[i].set_title(label_topik.get(i, ""), fontsize=12, fontweight="bold", color=INK)

        for j in range(best_n, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)

    progress.progress(60)

    # ---------------------------------------------------------------------------
    # 4. Visualisasi interaktif LDA
    # ---------------------------------------------------------------------------

    with st.container(border=True):
        section("Visualisasi Interaktif LDA")
        with st.spinner("Membuat visualisasi..."):
            vis = gensimvis.prepare(lda_model, corpus, dictionary)
            components.html(pyLDAvis.prepared_data_to_html(vis), height=800, scrolling=True)

    progress.progress(65)

    # ---------------------------------------------------------------------------
    # 5. Analisis sentimen (IndoBERT)
    # ---------------------------------------------------------------------------

    with st.container(border=True):
        section("Analisis Sentimen")

        teks_tuple = tuple(df["teks_bersih"].fillna(""))
        all_labels, all_scores = prediksi_sentimen(teks_tuple)

        df["sentimen"] = all_labels
        df["sentimen_score"] = all_scores

        st.success("Analisis sentimen selesai")
        st.write(df["sentimen"].value_counts())

        # Koreksi label: kalimat ajakan yang salah dilabeli negatif
        pola_koreksi = re.compile(r"(tidak|bukan)\s+\w*\s*(saat|waktu|moment)\w*.*?(pasif|diam|statis)", re.IGNORECASE)
        ajakan = re.compile(r"\b(yuk|ayo|mari)\b", re.IGNORECASE)

        def cek_koreksi(row):
            teks = str(row["full_text"])
            if row["sentimen"] == "Negatif" and pola_koreksi.search(teks) and ajakan.search(teks):
                return "Netral"
            return row["sentimen"]

        df["sentimen_asli"] = df["sentimen"]
        df["sentimen"] = df.apply(cek_koreksi, axis=1)
        jumlah_dikoreksi = (df["sentimen_asli"] != df["sentimen"]).sum()
        st.info(f"Label yang dikoreksi: {jumlah_dikoreksi}")

    progress.progress(75)

    with st.container(border=True):
        section("Visualisasi Sentimen")

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        sentimen_count = df["sentimen"].value_counts()
        axes[0].pie(
            sentimen_count.values, labels=sentimen_count.index, autopct="%1.1f%%",
            colors=[SENTIMEN_WARNA.get(x, MUTED) for x in sentimen_count.index], startangle=90,
        )
        axes[0].set_title("Distribusi Sentimen", color=INK)

        sentimen_topik = df.groupby(["topik_label", "sentimen"]).size().unstack(fill_value=0)
        sentimen_pct = sentimen_topik.div(sentimen_topik.sum(axis=1), axis=0) * 100
        kolom = [c for c in ["Positif", "Netral", "Negatif"] if c in sentimen_pct.columns]

        sentimen_pct[kolom].plot(
            kind="barh", stacked=True, color=[SENTIMEN_WARNA[x] for x in kolom], ax=axes[1],
        )
        axes[1].set_xlabel("Persentase (%)")
        axes[1].legend(frameon=False)

        plt.tight_layout()
        st.pyplot(fig)

    progress.progress(85)

    # ---------------------------------------------------------------------------
    # 6. Perhitungan AHP
    # ---------------------------------------------------------------------------

    with st.container(border=True):
        section("Perhitungan AHP")

        for col in ["favorite_count", "retweet_count", "reply_count"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        df["engagement"] = df["favorite_count"] + df["retweet_count"] + df["reply_count"]

        popularitas = (
            df.groupby("topik_label")
            .agg(jumlah_tweet=("teks_final", "count"), total_engagement=("engagement", "sum"), avg_engagement=("engagement", "mean"))
            .reset_index()
        )
        popularitas["skor_popularitas"] = 0.6 * popularitas["jumlah_tweet"] + 0.4 * popularitas["total_engagement"]

        sentimen_topik_pct = (
            df.groupby("topik_label")
            .apply(lambda x: pd.Series({
                "pct_positif": (x["sentimen"] == "Positif").mean() * 100,
                "pct_negatif": (x["sentimen"] == "Negatif").mean() * 100,
                "pct_netral": (x["sentimen"] == "Netral").mean() * 100,
            }))
            .reset_index()
        )

        ahp_data = popularitas.merge(sentimen_topik_pct, on="topik_label")
        ahp_data["skor_potensi"] = ahp_data["pct_positif"]
        ahp_data["skor_risiko"] = ahp_data["pct_negatif"]

        ahp_data["norm_popularitas"] = minmax_norm(ahp_data["skor_popularitas"])
        ahp_data["norm_potensi"] = minmax_norm(ahp_data["skor_potensi"])
        ahp_data["norm_risiko"] = 1 - minmax_norm(ahp_data["skor_risiko"])

        ahp_data["skor_akhir_ahp"] = (
            BOBOT_POPULARITAS * ahp_data["norm_popularitas"]
            + BOBOT_POTENSI * ahp_data["norm_potensi"]
            + BOBOT_RISIKO * ahp_data["norm_risiko"]
        )
        ahp_data["ranking"] = ahp_data["skor_akhir_ahp"].rank(ascending=False, method="min").astype(int)
        ahp_data = ahp_data.sort_values("skor_akhir_ahp", ascending=False)

        st.success("Perhitungan AHP selesai")
        st.dataframe(ahp_data, use_container_width=True)

    progress.progress(90)

    with st.container(border=True):
        section("Ranking Usaha Kreatif")

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        ranking = ahp_data.sort_values("skor_akhir_ahp")
        axes[0].barh(ranking["topik_label"], ranking["skor_akhir_ahp"], color=ACCENT)
        axes[0].set_title("Skor Akhir AHP", color=INK)

        plot = ahp_data[["topik_label", "norm_popularitas", "norm_potensi", "norm_risiko"]]
        x = np.arange(len(plot))
        w = 0.25
        axes[1].bar(x - w, plot["norm_popularitas"], w, label="Popularitas", color=PALETTE[0])
        axes[1].bar(x, plot["norm_potensi"], w, label="Potensi", color=PALETTE[1])
        axes[1].bar(x + w, plot["norm_risiko"], w, label="Risiko", color=PALETTE[2])
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(plot["topik_label"], rotation=30, ha="right")
        axes[1].legend(frameon=False)

        st.pyplot(fig)

    progress.progress(95)

    # ---------------------------------------------------------------------------
    # 7. Unduh hasil & ringkasan
    # ---------------------------------------------------------------------------

    with st.container(border=True):
        section("Unduh Hasil")

        c1, c2 = st.columns(2)
        csv = df.to_csv(index=False).encode("utf-8-sig")
        c1.download_button("Download Data Analisis", csv, "data_analisis.csv", "text/csv", use_container_width=True)

        csv2 = ahp_data.to_csv(index=False).encode("utf-8-sig")
        c2.download_button("Download Ranking AHP", csv2, "ranking_ahp.csv", "text/csv", use_container_width=True)

    progress.progress(100)
    st.success("Seluruh proses analisis selesai")

    with st.container(border=True):
        section("Ringkasan Hasil")

        c1, c2, c3 = st.columns(3)
        c1.metric("Jumlah Tweet", len(df))
        c2.metric("Jumlah Topik", best_n)
        c3.metric("Rekomendasi Terbaik", ahp_data.iloc[0]["topik_label"])

        st.markdown("**Top 3 Rekomendasi**")
        st.table(ahp_data[["ranking", "topik_label", "skor_akhir_ahp"]].head(3))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()