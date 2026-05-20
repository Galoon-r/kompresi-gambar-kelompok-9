import streamlit as st
from PIL import Image
import numpy as np
import io
import os
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ImagePress · Kompresi PCA",
    page_icon="🗜️",
    layout="wide",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0d0d0f; color: #e8e6e0; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
div[data-testid="stFileUploader"] {
    border: 2px dashed #2e2e38;
    border-radius: 12px;
    padding: 1.5rem;
    background: #16161a;
    transition: border-color 0.2s;
}
div[data-testid="stFileUploader"]:hover { border-color: #5c6aff; }
.stSlider > div { padding: 0.5rem 0; }
.stat-box {
    background: #16161a;
    border: 1px solid #2a2a35;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.stat-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #5a5a6e;
    margin-bottom: 4px;
}
.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #f0ece4;
}
.stat-value.green { color: #4ade80; }
.stat-value.blue  { color: #60a5fa; }
div[data-testid="stButton"] > button {
    background: #5c6aff;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    width: 100%;
    transition: background 0.2s, transform 0.1s;
}
div[data-testid="stButton"] > button:hover {
    background: #7b86ff;
    transform: translateY(-1px);
}
.divider { border-top: 1px solid #1e1e28; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FUNGSI KOMPRESI PCA  ← FIX: tidak nested lagi
# ─────────────────────────────────────────────
def compress_with_pca(image: Image.Image, n_components: int) -> Image.Image:
    img_array = np.array(image, dtype=np.float32)
    reconstructed = np.zeros_like(img_array)

    for ch in range(3):  # R, G, B
        channel = img_array[:, :, ch]
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(channel)
        reconstructed[:, :, ch] = pca.inverse_transform(transformed)

    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    return Image.fromarray(reconstructed)

# ─────────────────────────────────────────────
# HELPER  ← FIX: format default PNG, bukan JPEG
# ─────────────────────────────────────────────
def file_size_kb(img: Image.Image, fmt="JPEG") -> float:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=95)
    else:
        img.save(buf, format=fmt)
    return buf.tell() / 1024

def img_to_bytes(img: Image.Image, fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<h1 style='font-family:Syne,sans-serif;font-weight:800;font-size:2.6rem;
           letter-spacing:-1px;color:#f0ece4;margin-bottom:4px;'>
    🗜️ ImagePress
</h1>
<p style='color:#5a5a6e;font-size:0.95rem;margin-top:0;'>
    Kompresi gambar berbasis <strong style="color:#8888cc;">PCA (Principal Component Analysis)</strong> · sklearn
</p>
<hr style='border:none;border-top:1px solid #1e1e28;margin:1.2rem 0;'>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.markdown("#### 📂 Upload Gambar")
    uploaded = st.file_uploader(
        "Pilih file gambar",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed"
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown("#### ⚙️ Parameter PCA")
    n_components = st.slider(
        "Jumlah Komponen (n_components)",
        min_value=1,
        max_value=200,
        value=50,
        step=1,
        help="Semakin sedikit komponen → kompresi lebih agresif, kualitas lebih rendah."
    )

    st.markdown(f"""
    <div style='background:#16161a;border:1px solid #2a2a35;border-radius:8px;
                padding:0.8rem 1rem;font-size:0.83rem;color:#6666aa;margin-top:0.5rem;'>
        💡 <b>n_components = {n_components}</b> artinya setiap channel warna
        direpresentasikan dengan <b>{n_components}</b> komponen utama dari total piksel baris.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    run_btn = st.button("🚀 Jalankan Kompresi")

with col_right:
    if uploaded is None:
        st.markdown("""
        <div style='height:400px;display:flex;align-items:center;justify-content:center;
                    border:2px dashed #1e1e28;border-radius:14px;background:#121216;'>
            <div style='text-align:center;color:#333342;'>
                <div style='font-size:3rem;'>🖼️</div>
                <div style='font-family:Syne,sans-serif;font-size:1.1rem;margin-top:8px;'>
                    Upload gambar untuk mulai
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        image_orig = Image.open(uploaded).convert("RGB")
        w, h = image_orig.size
        orig_kb = len(uploaded.getvalue()) / 1024  # ← ukuran file asli

        max_comp = min(h, w, n_components)
        if max_comp < n_components:
            st.warning(f"⚠️ n_components dibatasi ke {max_comp} (ukuran gambar: {w}×{h}px)")
            n_components = max_comp

        tab1, tab2 = st.tabs(["📸 Sebelum & Sesudah", "📊 Statistik"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<p style='color:#5a5a6e;font-size:0.8rem;text-align:center;'>ORIGINAL</p>",
                            unsafe_allow_html=True)
                st.image(image_orig, use_container_width=True)
                st.markdown(f"<p style='text-align:center;color:#444455;font-size:0.78rem;'>{orig_kb:.1f} KB · {w}×{h}px</p>",
                            unsafe_allow_html=True)

            with c2:
                st.markdown("<p style='color:#5a5a6e;font-size:0.8rem;text-align:center;'>HASIL KOMPRESI</p>",
                            unsafe_allow_html=True)

                if run_btn:
                    with st.spinner("Memproses PCA..."):
                        result_img = compress_with_pca(image_orig, n_components)
                        st.session_state["result_img"] = result_img
                        st.session_state["orig_kb"] = orig_kb

                if "result_img" in st.session_state:
                    result_img = st.session_state["result_img"]
                    result_kb = file_size_kb(result_img)
                    st.image(result_img, use_container_width=True)
                    st.markdown(f"<p style='text-align:center;color:#444455;font-size:0.78rem;'>{result_kb:.1f} KB · {w}×{h}px</p>",
                                unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style='height:200px;display:flex;align-items:center;justify-content:center;
                                border:1px dashed #2a2a35;border-radius:10px;background:#13131a;'>
                        <span style='color:#333342;font-size:0.9rem;'>Tekan "Jalankan Kompresi"</span>
                    </div>
                    """, unsafe_allow_html=True)

        with tab2:
            if "result_img" in st.session_state:
                result_img = st.session_state["result_img"]
                result_kb  = file_size_kb(result_img)
                orig_kb_s  = st.session_state["orig_kb"]
                ratio      = (1 - result_kb / orig_kb_s) * 100 if orig_kb_s > 0 else 0

                s1, s2, s3, s4 = st.columns(4)
                s1.markdown(f"""
                    <div class='stat-box'>
                        <div class='stat-label'>Ukuran Asli</div>
                        <div class='stat-value'>{orig_kb_s:.0f}<span style='font-size:0.9rem;color:#555566;'> KB</span></div>
                    </div>""", unsafe_allow_html=True)
                s2.markdown(f"""
                    <div class='stat-box'>
                        <div class='stat-label'>Ukuran Hasil</div>
                        <div class='stat-value blue'>{result_kb:.0f}<span style='font-size:0.9rem;color:#555566;'> KB</span></div>
                    </div>""", unsafe_allow_html=True)
                s3.markdown(f"""
                    <div class='stat-box'>
                        <div class='stat-label'>Penghematan</div>
                        <div class='stat-value green'>{ratio:.1f}<span style='font-size:0.9rem;color:#555566;'> %</span></div>
                    </div>""", unsafe_allow_html=True)
                s4.markdown(f"""
                    <div class='stat-box'>
                        <div class='stat-label'>Komponen PCA</div>
                        <div class='stat-value'>{n_components}</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                dl_bytes = img_to_bytes(result_img, "PNG")
                st.download_button(
                   label="⬇️ Download Hasil (JPEG)",
                   data=dl_bytes,
                   file_name=f"compressed_pca{n_components}.jpg",
                   mime="image/jpeg",
                   use_container_width=True,
            )
            else:
                st.info("Jalankan kompresi terlebih dahulu untuk melihat statistik.")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<hr style='border:none;border-top:1px solid #1a1a22;margin-top:2rem;'>
<p style='text-align:center;color:#2e2e3e;font-size:0.78rem;'>
    ImagePress · PCA Image Compression · sklearn
</p>
""", unsafe_allow_html=True)
