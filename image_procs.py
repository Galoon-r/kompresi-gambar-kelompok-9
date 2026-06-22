import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
import os
import zipfile
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ImageProcs · Kompresi & Pengenalan Wajah",
    page_icon="🧠",
    layout="wide",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #0a0a0e; color: #ddd8cc; }

h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f0f14 !important;
    border-right: 1px solid #1a1a24 !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-family: 'DM Sans', sans-serif;
    color: #888899;
    transition: color 0.15s;
    cursor: pointer;
}
[data-testid="stSidebar"] .stRadio label:hover { color: #c8c4f0; }

.block-container { padding-top: 1.8rem; padding-bottom: 2rem; }

/* File uploader */
div[data-testid="stFileUploader"] {
    border: 2px dashed #252530;
    border-radius: 12px;
    padding: 1.4rem;
    background: #12121a;
    transition: border-color 0.2s;
}
div[data-testid="stFileUploader"]:hover { border-color: #6872ff; }

/* Buttons */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #5c6aff, #7c44ff);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.58rem 2rem;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    width: 100%;
    letter-spacing: 0.3px;
    transition: opacity 0.15s, transform 0.1s;
}
div[data-testid="stButton"] > button:hover {
    opacity: 0.88;
    transform: translateY(-1px);
}

/* Stat cards */
.stat-box {
    background: #13131c;
    border: 1px solid #22222f;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.stat-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 1.6px;
    color: #44445a;
    margin-bottom: 5px;
}
.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: #e8e4dc;
}
.stat-value.green { color: #50e090; }
.stat-value.blue  { color: #60b0ff; }
.stat-value.amber { color: #f0b040; }
.stat-value.purple{ color: #b880ff; }

.divider { border-top: 1px solid #18181f; margin: 1.2rem 0; }

/* Info pills */
.pill {
    display: inline-block;
    background: #1c1c28;
    border: 1px solid #2a2a3a;
    border-radius: 20px;
    padding: 2px 11px;
    font-size: 0.76rem;
    color: #6666aa;
    margin: 2px;
}

/* Result card */
.result-card {
    background: #141420;
    border: 1px solid #2a2a42;
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    text-align: center;
}
.similarity-bar-track {
    background: #1e1e2e;
    border-radius: 20px;
    height: 10px;
    width: 100%;
    margin: 0.5rem 0;
}

/* Nav mode label */
.mode-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #44445a;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
IMG_SIZE = (100, 100)

def file_size_kb(img: Image.Image, fmt="JPEG") -> float:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.tell() / 1024

def img_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

def compress_with_pca(image: Image.Image, n_components: int) -> Image.Image:
    img_array = np.array(image, dtype=np.float32)
    reconstructed = np.zeros_like(img_array)
    for ch in range(3):
        channel = img_array[:, :, ch]
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(channel)
        reconstructed[:, :, ch] = pca.inverse_transform(transformed)
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    return Image.fromarray(reconstructed)

def pil_to_cv2_gray(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)

def load_face_vector(img_gray: np.ndarray) -> np.ndarray:
    resized = cv2.resize(img_gray, IMG_SIZE)
    normalized = resized / 255.0
    return normalized.flatten()

def detect_and_crop(img_gray: np.ndarray):
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None, None
    x, y, w, h = faces[0]
    return img_gray[y:y+h, x:x+w], (x, y, w, h)

def draw_box_on_image(pil_img: Image.Image, box):
    img_np = np.array(pil_img.convert("RGB"))
    x, y, w, h = box
    cv2.rectangle(img_np, (x, y), (x+w, y+h), (100, 150, 255), 2)
    return Image.fromarray(img_np)

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1.2rem 0 0.5rem;'>
        <span style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;
                     color:#e0ddf8;letter-spacing:-0.5px;'>🧠 ImageProcs</span>
        <div style='font-size:0.72rem;color:#3a3a5a;margin-top:3px;
                    letter-spacing:1.5px;text-transform:uppercase;'>
            PCA · Eigenfaces
        </div>
    </div>
    <hr style='border:none;border-top:1px solid #1a1a28;margin:1rem 0;'>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "Pilih Fitur",
        options=["🗜️  Kompresi PCA", "👁️  Perbandingan Wajah", "🔍  Pengenalan Wajah"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border:none;border-top:1px solid #1a1a28;margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.73rem;color:#2e2e48;line-height:1.7;padding:0 0.2rem;'>
        <div class='pill' style='margin-bottom:6px;display:block;'>sklearn · PCA</div>
        <div class='pill' style='margin-bottom:6px;display:block;'>OpenCV · Haar Cascade</div>
        <div class='pill' style='display:block;'>Cosine Similarity</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# MODE 1 — KOMPRESI PCA
# ═══════════════════════════════════════════════════════════════
if mode == "🗜️  Kompresi PCA":
    st.markdown("""
    <h2 style='font-weight:800;font-size:2rem;letter-spacing:-0.8px;
               color:#e8e4f8;margin-bottom:2px;'>🗜️ Kompresi PCA</h2>
    <p style='color:#44445a;font-size:0.88rem;margin-top:0;'>
        Kurangi dimensi gambar dengan PCA per channel warna (R, G, B)
    </p>
    <hr style='border:none;border-top:1px solid #18181f;margin:1rem 0 1.4rem;'>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        st.markdown("#### 📂 Upload Gambar")
        uploaded = st.file_uploader(
            "Pilih file gambar",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
            key="comp_upload"
        )
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("#### ⚙️ Parameter PCA")
        n_components = st.slider(
            "n_components",
            min_value=1, max_value=200, value=50, step=1,
            help="Semakin kecil → kompresi lebih agresif",
            key="comp_slider"
        )
        st.markdown(f"""
        <div style='background:#12121c;border:1px solid #22222e;border-radius:8px;
                    padding:0.75rem 1rem;font-size:0.82rem;color:#55557a;margin-top:0.4rem;'>
            💡 Setiap channel warna direpresentasikan dengan <b style="color:#8888bb;">{n_components}</b> komponen utama.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        run_btn = st.button("🚀 Jalankan Kompresi", key="run_comp")

    with col_right:
        if uploaded is None:
            st.markdown("""
            <div style='height:380px;display:flex;align-items:center;justify-content:center;
                        border:2px dashed #1a1a28;border-radius:14px;background:#0e0e16;'>
                <div style='text-align:center;color:#2a2a3c;'>
                    <div style='font-size:3.2rem;'>🖼️</div>
                    <div style='font-family:Syne,sans-serif;font-size:1rem;margin-top:8px;'>
                        Upload gambar untuk mulai
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            image_orig = Image.open(uploaded).convert("RGB")
            w, h = image_orig.size
            orig_kb = file_size_kb(image_orig)

            max_comp = min(h, w, n_components)
            if max_comp < n_components:
                st.warning(f"⚠️ n_components dibatasi ke {max_comp} (gambar {w}×{h}px)")
                n_components = max_comp

            tab1, tab2 = st.tabs(["📸 Sebelum & Sesudah", "📊 Statistik"])

            with tab1:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("<p style='color:#44445a;font-size:0.75rem;text-align:center;letter-spacing:1.5px;'>ORIGINAL</p>", unsafe_allow_html=True)
                    st.image(image_orig, use_container_width=True)
                    st.markdown(f"<p style='text-align:center;color:#33334a;font-size:0.75rem;'>{orig_kb:.1f} KB · {w}×{h}px</p>", unsafe_allow_html=True)

                with c2:
                    st.markdown("<p style='color:#44445a;font-size:0.75rem;text-align:center;letter-spacing:1.5px;'>HASIL KOMPRESI</p>", unsafe_allow_html=True)
                    if run_btn:
                        with st.spinner("Memproses PCA..."):
                            result_img = compress_with_pca(image_orig, n_components)
                            st.session_state["comp_result"] = result_img
                            st.session_state["comp_orig_kb"] = orig_kb

                    if "comp_result" in st.session_state:
                        result_img = st.session_state["comp_result"]
                        result_kb = file_size_kb(result_img)
                        st.image(result_img, use_container_width=True)
                        st.markdown(f"<p style='text-align:center;color:#33334a;font-size:0.75rem;'>{result_kb:.1f} KB · {w}×{h}px</p>", unsafe_allow_html=True)
                    else:
                        st.markdown("""<div style='height:180px;display:flex;align-items:center;
                            justify-content:center;border:1px dashed #22222e;border-radius:10px;
                            background:#0e0e14;'><span style='color:#2a2a3c;font-size:0.85rem;'>
                            Tekan "Jalankan Kompresi"</span></div>""", unsafe_allow_html=True)

            with tab2:
                if "comp_result" in st.session_state:
                    result_img = st.session_state["comp_result"]
                    result_kb  = file_size_kb(result_img)
                    orig_kb_s  = st.session_state["comp_orig_kb"]
                    ratio      = (1 - result_kb / orig_kb_s) * 100 if orig_kb_s > 0 else 0

                    s1, s2, s3, s4 = st.columns(4)
                    s1.markdown(f"""<div class='stat-box'><div class='stat-label'>Ukuran Asli</div>
                        <div class='stat-value'>{orig_kb_s:.0f}<small style='color:#33334a;'> KB</small></div></div>""", unsafe_allow_html=True)
                    s2.markdown(f"""<div class='stat-box'><div class='stat-label'>Ukuran Hasil</div>
                        <div class='stat-value blue'>{result_kb:.0f}<small style='color:#33334a;'> KB</small></div></div>""", unsafe_allow_html=True)
                    s3.markdown(f"""<div class='stat-box'><div class='stat-label'>Penghematan</div>
                        <div class='stat-value green'>{ratio:.1f}<small style='color:#33334a;'> %</small></div></div>""", unsafe_allow_html=True)
                    s4.markdown(f"""<div class='stat-box'><div class='stat-label'>Komponen PCA</div>
                        <div class='stat-value purple'>{n_components}</div></div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    dl_bytes = img_to_bytes(result_img, "PNG")
                    st.download_button(
                        label="⬇️ Download Hasil (PNG)",
                        data=dl_bytes,
                        file_name=f"compressed_pca{n_components}.png",
                        mime="image/png",
                        use_container_width=True,
                    )
                else:
                    st.info("Jalankan kompresi terlebih dahulu untuk melihat statistik.")


# ═══════════════════════════════════════════════════════════════
# MODE 2 — PERBANDINGAN WAJAH
# ═══════════════════════════════════════════════════════════════
elif mode == "👁️  Perbandingan Wajah":
    st.markdown("""
    <h2 style='font-weight:800;font-size:2rem;letter-spacing:-0.8px;
               color:#e8e4f8;margin-bottom:2px;'>👁️ Perbandingan Wajah</h2>
    <p style='color:#44445a;font-size:0.88rem;margin-top:0;'>
        Bandingkan dua wajah menggunakan Eigenfaces (PCA + Cosine Similarity)
    </p>
    <hr style='border:none;border-top:1px solid #18181f;margin:1rem 0 1.4rem;'>
    """, unsafe_allow_html=True)

    cl, cr = st.columns([1, 2], gap="large")

    with cl:
        st.markdown("#### 📂 Upload Dua Gambar Wajah")
        face1_file = st.file_uploader("Wajah 1", type=["jpg","jpeg","png"], key="face1")
        face2_file = st.file_uploader("Wajah 2", type=["jpg","jpeg","png"], key="face2")
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        st.markdown("#### ⚙️ Parameter")
        n_comp_face = st.slider("n_components Eigenfaces", 10, 100, 50, key="face_comp")
        threshold_face = st.slider("Threshold Similarity", 0.50, 1.00, 0.80, 0.01, key="face_thresh")

        st.markdown(f"""
        <div style='background:#12121c;border:1px solid #22222e;border-radius:8px;
                    padding:0.75rem 1rem;font-size:0.82rem;color:#55557a;margin-top:0.4rem;'>
            🎯 Similarity ≥ <b style="color:#8888bb;">{threshold_face:.2f}</b> → dianggap wajah yang sama
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        compare_btn = st.button("🔎 Bandingkan Wajah", key="run_compare")

    with cr:
        if face1_file and face2_file:
            img1 = Image.open(face1_file)
            img2 = Image.open(face2_file)

            p1, p2 = st.columns(2)
            with p1:
                st.markdown("<p style='color:#44445a;font-size:0.75rem;text-align:center;letter-spacing:1.5px;'>WAJAH 1</p>", unsafe_allow_html=True)
                gray1 = pil_to_cv2_gray(img1)
                crop1, box1 = detect_and_crop(gray1)
                if box1:
                    st.image(draw_box_on_image(img1, box1), use_container_width=True)
                    st.markdown("<p style='text-align:center;font-size:0.72rem;color:#50e090;'>✓ Wajah terdeteksi</p>", unsafe_allow_html=True)
                else:
                    st.image(img1, use_container_width=True)
                    st.markdown("<p style='text-align:center;font-size:0.72rem;color:#f0b040;'>⚠ Wajah tidak terdeteksi, pakai gambar penuh</p>", unsafe_allow_html=True)

            with p2:
                st.markdown("<p style='color:#44445a;font-size:0.75rem;text-align:center;letter-spacing:1.5px;'>WAJAH 2</p>", unsafe_allow_html=True)
                gray2 = pil_to_cv2_gray(img2)
                crop2, box2 = detect_and_crop(gray2)
                if box2:
                    st.image(draw_box_on_image(img2, box2), use_container_width=True)
                    st.markdown("<p style='text-align:center;font-size:0.72rem;color:#50e090;'>✓ Wajah terdeteksi</p>", unsafe_allow_html=True)
                else:
                    st.image(img2, use_container_width=True)
                    st.markdown("<p style='text-align:center;font-size:0.72rem;color:#f0b040;'>⚠ Wajah tidak terdeteksi, pakai gambar penuh</p>", unsafe_allow_html=True)

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

            if compare_btn:
                with st.spinner("Menghitung Eigenfaces..."):
                    # Gunakan crop wajah jika terdeteksi, fallback ke full grayscale
                    region1 = crop1 if crop1 is not None else gray1
                    region2 = crop2 if crop2 is not None else gray2

                    v1 = load_face_vector(region1)
                    v2 = load_face_vector(region2)
                    # Langsung cosine similarity di raw space (10000-dim)
                    similarity = cosine_similarity(v1.reshape(1,-1), v2.reshape(1,-1))[0][0]
                    is_same = similarity >= threshold_face

                    st.session_state["cmp_sim"] = float(similarity)
                    st.session_state["cmp_same"] = is_same
                    

            if "cmp_sim" in st.session_state:
                sim = st.session_state["cmp_sim"]
                is_same = st.session_state["cmp_same"]

                verdict_color = "#50e090" if is_same else "#ff6070"
                verdict_text  = "✅ MIRIP / SAMA" if is_same else "❌ BERBEDA"
                bar_w = int(sim * 100)

                st.markdown(f"""
                <div class='result-card'>
                    <div style='font-family:Syne,sans-serif;font-size:2.2rem;font-weight:800;
                                color:{verdict_color};letter-spacing:-1px;'>{verdict_text}</div>
                    <div style='margin:1rem 0 0.4rem;font-size:0.8rem;color:#44445a;letter-spacing:1.5px;'>
                        COSINE SIMILARITY
                    </div>
                    <div style='font-family:Syne,sans-serif;font-size:3rem;font-weight:800;
                                color:#c8c4f0;'>{sim:.4f}</div>
                    <div class='similarity-bar-track'>
                        <div style='width:{bar_w}%;height:100%;border-radius:20px;
                            background:linear-gradient(90deg,{verdict_color}88,{verdict_color});
                            transition:width 0.4s;'></div>
                    </div>
                    <div style='display:flex;justify-content:space-between;
                                font-size:0.7rem;color:#2a2a4a;'>
                        <span>0.0</span><span>0.5</span><span>1.0</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.markdown(f"""<div class='stat-box'><div class='stat-label'>Similarity</div>
                    <div class='stat-value {"green" if is_same else "amber"}'>{sim:.3f}</div></div>""", unsafe_allow_html=True)
                m2.markdown(f"""<div class='stat-box'><div class='stat-label'>Threshold</div>
                    <div class='stat-value'>{threshold_face:.2f}</div></div>""", unsafe_allow_html=True)
                m3.markdown(f"""<div class='stat-box'><div class='stat-label'>Dimensi Vektor</div>
                    <div class='stat-value blue'>10000</div></div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='height:380px;display:flex;align-items:center;justify-content:center;
                        border:2px dashed #1a1a28;border-radius:14px;background:#0e0e16;'>
                <div style='text-align:center;color:#2a2a3c;'>
                    <div style='font-size:3rem;'>👁️</div>
                    <div style='font-family:Syne,sans-serif;font-size:1rem;margin-top:8px;'>
                        Upload dua gambar wajah untuk membandingkan
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# MODE 3 — PENGENALAN WAJAH
# ═══════════════════════════════════════════════════════════════
elif mode == "🔍  Pengenalan Wajah":
    st.markdown("""
    <h2 style='font-weight:800;font-size:2rem;letter-spacing:-0.8px;
               color:#e8e4f8;margin-bottom:2px;'>🔍 Pengenalan Wajah (Eigenfaces)</h2>
    <p style='color:#44445a;font-size:0.88rem;margin-top:0;'>
        Upload dataset wajah → latih model PCA → kenali wajah baru
    </p>
    <hr style='border:none;border-top:1px solid #18181f;margin:1rem 0 1.4rem;'>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 2], gap="large")

    with col_a:
        st.markdown("#### 📦 Dataset Wajah")
        st.markdown("""
        <div style='background:#12121c;border:1px solid #22222e;border-radius:8px;
                    padding:0.8rem 1rem;font-size:0.8rem;color:#55557a;margin-bottom:0.8rem;'>
            📁 Upload file <b>.zip</b> berisi folder per orang:<br>
            <code style='color:#6666aa;'>dataset.zip/</code><br>
            <code style='color:#6666aa;'>  ├── alice/foto1.jpg</code><br>
            <code style='color:#6666aa;'>  └── bob/foto1.jpg</code>
        </div>
        """, unsafe_allow_html=True)

        dataset_zip = st.file_uploader("Upload dataset.zip", type=["zip"], key="ds_zip")
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        st.markdown("#### 🖼️ Gambar yang Akan Dikenali")
        query_img_file = st.file_uploader("Upload gambar wajah", type=["jpg","jpeg","png"], key="query_img")
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        st.markdown("#### ⚙️ Parameter")
        n_comp_rec = st.slider("n_components Eigenfaces", 10, 150, 50, key="rec_comp")
        thresh_rec = st.slider("Threshold", 0.50, 1.00, 0.80, 0.01, key="rec_thresh")

        train_btn   = st.button("⚡ Latih Model", key="run_train")
        recog_btn   = st.button("🔍 Kenali Wajah", key="run_recog")

    with col_b:
        # ── LOAD MODEL dari .pkl ──
        model_file = st.file_uploader("Atau load model .pkl (dari Colab)", type=["pkl"], key="model_pkl")
        if model_file:
            import joblib
            data = joblib.load(model_file)
            st.session_state["pca_model"]  = data["pca"]
            st.session_state["X_pca"]      = data["X_pca"]
            st.session_state["labels"]     = data["labels"]
            st.session_state["n_persons"]  = len(set(data["labels"]))
            st.session_state["n_images"]   = len(data["labels"])
            st.session_state["exp_var_rec"]= float(np.sum(data["pca"].explained_variance_ratio_))
            st.success("✅ Model berhasil dimuat dari file!")

        # ── LATIH MODEL dari .zip ──
        if dataset_zip and train_btn:
            with st.spinner("Mengekstrak dataset dan melatih model PCA..."):
                import tempfile, shutil
                tmpdir = tempfile.mkdtemp()
                try:
                    zip_path = os.path.join(tmpdir, "dataset.zip")
                    with open(zip_path, "wb") as f:
                        f.write(dataset_zip.read())
                    with zipfile.ZipFile(zip_path, "r") as z:
                        z.extractall(tmpdir)

                    X_list, labels_list = [], []
                    ds_path = tmpdir
                    for entry in os.listdir(ds_path):
                        person_folder = os.path.join(ds_path, entry)
                        if not os.path.isdir(person_folder):
                            continue
                        if entry == "__MACOSX" or entry.startswith("."):
                            continue
                        for fname in os.listdir(person_folder):
                            if fname.lower().endswith((".jpg",".jpeg",".png")):
                                img_path = os.path.join(person_folder, fname)
                                img_cv = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                                if img_cv is None:
                                    continue
                                vec = load_face_vector(img_cv)
                                X_list.append(vec)
                                labels_list.append(entry)

                    if len(X_list) == 0:
                        st.error("Tidak ada gambar yang berhasil dibaca dari dataset.")
                    else:
                        X_arr = np.array(X_list)
                        n_safe = min(n_comp_rec, X_arr.shape[0]-1, X_arr.shape[1])
                        pca_model = PCA(n_components=n_safe)
                        X_pca = pca_model.fit_transform(X_arr)

                        st.session_state["pca_model"]  = pca_model
                        st.session_state["X_pca"]      = X_pca
                        st.session_state["labels"]     = np.array(labels_list)
                        st.session_state["n_persons"]  = len(set(labels_list))
                        st.session_state["n_images"]   = len(X_list)
                        st.session_state["exp_var_rec"]= float(np.sum(pca_model.explained_variance_ratio_))

                        st.success(f"✅ Model dilatih dengan {len(X_list)} gambar dari {len(set(labels_list))} orang.")

                        # Tawarkan download model
                        import joblib, io
                        buf = io.BytesIO()
                        joblib.dump({"pca": pca_model, "X_pca": X_pca, "labels": np.array(labels_list)}, buf)
                        st.download_button("💾 Download Model (.pkl)", buf.getvalue(),
                                           "eigenfaces_model.pkl", use_container_width=True)
                finally:
                    shutil.rmtree(tmpdir, ignore_errors=True)

        # ── STATUS MODEL ──
        if "pca_model" in st.session_state:
            n_persons  = st.session_state["n_persons"]
            n_images   = st.session_state["n_images"]
            exp_var_r  = st.session_state["exp_var_rec"]

            t1, t2, t3 = st.columns(3)
            t1.markdown(f"""<div class='stat-box'><div class='stat-label'>Total Orang</div>
                <div class='stat-value purple'>{n_persons}</div></div>""", unsafe_allow_html=True)
            t2.markdown(f"""<div class='stat-box'><div class='stat-label'>Total Gambar</div>
                <div class='stat-value blue'>{n_images}</div></div>""", unsafe_allow_html=True)
            t3.markdown(f"""<div class='stat-box'><div class='stat-label'>Variance PCA</div>
                <div class='stat-value green'>{exp_var_r:.1%}</div></div>""", unsafe_allow_html=True)
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── KENALI WAJAH ──
        if query_img_file:
            img_q = Image.open(query_img_file)
            q1, q2 = st.columns([1, 2])
            with q1:
                gray_q = pil_to_cv2_gray(img_q)
                crop_q, box_q = detect_and_crop(gray_q)
                if box_q:
                    st.image(draw_box_on_image(img_q, box_q), use_container_width=True)
                    st.markdown("<p style='text-align:center;font-size:0.72rem;color:#50e090;'>✓ Wajah terdeteksi</p>", unsafe_allow_html=True)
                else:
                    st.image(img_q, use_container_width=True)
                    st.markdown("<p style='text-align:center;font-size:0.72rem;color:#f0b040;'>⚠ Wajah tidak terdeteksi</p>", unsafe_allow_html=True)

            with q2:
                if recog_btn:
                    if "pca_model" not in st.session_state:
                        st.error("⚠️ Latih model terlebih dahulu dengan upload dataset!")
                    else:
                        with st.spinner("Mengenali wajah..."):
                            region_q = crop_q if crop_q is not None else gray_q
                            vec_q = load_face_vector(region_q).reshape(1, -1)

                            pca_m = st.session_state["pca_model"]
                            X_p   = st.session_state["X_pca"]
                            labs  = st.session_state["labels"]

                            face_pca = pca_m.transform(vec_q)
                            sims = cosine_similarity(face_pca, X_p)[0]
                            best_idx  = np.argmax(sims)
                            best_sim  = float(sims[best_idx])
                            best_name = labs[best_idx]

                            if best_sim >= thresh_rec:
                                rec_name = best_name
                                rec_found = True
                            else:
                                rec_name = "Tidak Dikenali"
                                rec_found = False

                            st.session_state["rec_name"]  = rec_name
                            st.session_state["rec_sim"]   = best_sim
                            st.session_state["rec_found"] = rec_found
                            # Top-5
                            top5_idx = np.argsort(sims)[::-1][:5]
                            st.session_state["top5"] = [(labs[i], float(sims[i])) for i in top5_idx]

                if "rec_name" in st.session_state:
                    rec_name  = st.session_state["rec_name"]
                    rec_sim   = st.session_state["rec_sim"]
                    rec_found = st.session_state["rec_found"]
                    top5      = st.session_state["top5"]

                    vc = "#50e090" if rec_found else "#ff6070"
                    st.markdown(f"""
                    <div class='result-card' style='text-align:left;'>
                        <div style='font-size:0.7rem;letter-spacing:1.5px;color:#44445a;margin-bottom:6px;'>
                            HASIL IDENTIFIKASI
                        </div>
                        <div style='font-family:Syne,sans-serif;font-size:1.9rem;font-weight:800;
                                    color:{vc};letter-spacing:-0.5px;'>{rec_name}</div>
                        <div style='font-size:0.8rem;color:#55556a;margin-top:4px;'>
                            Similarity terbaik: <b style='color:#c8c4f0;'>{rec_sim:.4f}</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("<p style='font-size:0.78rem;color:#33334a;margin-bottom:6px;'>TOP-5 KANDIDAT</p>", unsafe_allow_html=True)
                    for nm, sc in top5:
                        bar_pct = int(sc * 100)
                        color = "#7c80ff" if nm == rec_name else "#33334a"
                        st.markdown(f"""
                        <div style='display:flex;align-items:center;gap:10px;margin:4px 0;'>
                            <div style='width:90px;font-size:0.78rem;color:#8888aa;'>{nm}</div>
                            <div style='flex:1;background:#1a1a28;border-radius:10px;height:7px;'>
                                <div style='width:{bar_pct}%;height:100%;border-radius:10px;
                                    background:{color};'></div>
                            </div>
                            <div style='width:46px;font-size:0.76rem;color:#55556a;text-align:right;'>{sc:.3f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style='padding:2rem;border:1px dashed #22222e;border-radius:10px;
                                text-align:center;color:#2a2a3c;font-size:0.88rem;'>
                        Tekan "Kenali Wajah" untuk melihat hasil
                    </div>
                    """, unsafe_allow_html=True)
        elif not dataset_zip:
            st.markdown("""
            <div style='height:360px;display:flex;align-items:center;justify-content:center;
                        border:2px dashed #1a1a28;border-radius:14px;background:#0e0e16;'>
                <div style='text-align:center;color:#2a2a3c;'>
                    <div style='font-size:3rem;'>🔍</div>
                    <div style='font-family:Syne,sans-serif;font-size:1rem;margin-top:8px;'>
                        Upload dataset.zip untuk melatih model
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<hr style='border:none;border-top:1px solid #14141e;margin-top:2.5rem;'>
<p style='text-align:center;color:#22223a;font-size:0.75rem;'>
    ImageProcs · PCA Compression · Eigenfaces · sklearn · OpenCV
</p>
""", unsafe_allow_html=True)
