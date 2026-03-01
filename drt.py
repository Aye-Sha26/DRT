import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Dynamic Range Transformation",
    page_icon="🎨",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0e1117;
    color: #e0e6f0;
}
.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00d2ff, #7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #8892a4;
    margin-bottom: 0.3rem;
}
.info-box {
    background: #161b27;
    border: 1px solid #252d3d;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #a8d8ea;
    margin-bottom: 1rem;
}
.pixel-table {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    border-collapse: collapse;
    width: 100%;
}
.pixel-table td {
    border: 1px solid #2a3348;
    padding: 3px 4px;
    text-align: center;
    color: #e0e6f0;
    min-width: 28px;
}
.pixel-table th {
    background: #1e2640;
    border: 1px solid #2a3348;
    padding: 3px 4px;
    color: #7b8cde;
    font-size: 0.6rem;
    min-width: 28px;
}
.matrix-wrap {
    background: #161b27;
    border: 1px solid #252d3d;
    border-radius: 10px;
    padding: 0.7rem;
    margin-bottom: 0.5rem;
    overflow-x: auto;
    overflow-y: auto;
    max-height: 420px;
}
.matrix-wrap-sm {
    background: #161b27;
    border: 1px solid #252d3d;
    border-radius: 10px;
    padding: 0.5rem;
    margin-bottom: 0.5rem;
    overflow-x: auto;
    overflow-y: auto;
    max-height: 280px;
}
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    margin-bottom: 0.4rem;
}
.badge-original { background:#1e3a2f; color:#4ecca3; border:1px solid #4ecca3; }
.badge-log      { background:#1e2a3a; color:#4a9eff; border:1px solid #4a9eff; }
.badge-gamma    { background:#2a1e3a; color:#b47eff; border:1px solid #b47eff; }
.badge-linear   { background:#3a2a1e; color:#ffb347; border:1px solid #ffb347; }

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #00d2ff22, #7b2ff722);
    border: 1px solid #7b2ff7;
    color: #d0b4ff;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    border-radius: 8px;
    padding: 0.5rem 1.4rem;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #00d2ff44, #7b2ff744);
    color: #fff;
}
hr { border-color: #252d3d; }
section[data-testid="stFileUploadDropzone"] {
    background: #161b27 !important;
    border: 2px dashed #7b2ff7 !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────

def to_rgb_array(img):
    return np.array(img.convert("RGB"), dtype=np.uint8)

def rgb_matrix_html(arr, rows, cols):
    """RGB pixel matrix — shows R/G/B per cell with actual pixel colour as bg."""
    rows = min(rows, arr.shape[0])
    cols = min(cols, arr.shape[1])
    patch = arr[:rows, :cols]
    html = '<table class="pixel-table"><tr><th></th>'
    for c in range(cols): html += f"<th>{c}</th>"
    html += "</tr>"
    for r in range(rows):
        html += f"<tr><th>{r}</th>"
        for c in range(cols):
            R, G, B = int(patch[r, c, 0]), int(patch[r, c, 1]), int(patch[r, c, 2])
            brightness = 0.299*R + 0.587*G + 0.114*B
            text_color = "#000" if brightness > 128 else "#fff"
            html += (f'<td style="background:rgb({R},{G},{B});color:{text_color};'
                     f'font-size:0.55rem;line-height:1.3">{R}<br>{G}<br>{B}</td>')
        html += "</tr>"
    html += "</table>"
    return html


# ── Transformations (applied per RGB channel) ────────────────────────────────

def log_transform_rgb(arr):
    """Apply log transform to each R, G, B channel independently."""
    result = np.zeros_like(arr, dtype=np.uint8)
    for i in range(3):  # R=0, G=1, B=2
        ch = arr[:, :, i].astype(np.float64)
        c = 255 / np.log(1 + ch.max()) if ch.max() > 0 else 1
        result[:, :, i] = np.clip(c * np.log(1 + ch), 0, 255).astype(np.uint8)
    return result

def power_transform_rgb(arr, gamma):
    """Apply gamma/power transform to each R, G, B channel independently."""
    result = np.zeros_like(arr, dtype=np.uint8)
    for i in range(3):
        ch = arr[:, :, i].astype(np.float64) / 255.0
        result[:, :, i] = np.clip(np.power(ch, gamma) * 255.0, 0, 255).astype(np.uint8)
    return result

def linear_stretch_rgb(arr):
    """Apply linear stretch to each R, G, B channel independently."""
    result = np.zeros_like(arr, dtype=np.uint8)
    for i in range(3):
        ch = arr[:, :, i].astype(np.float64)
        mn, mx = ch.min(), ch.max()
        if mx == mn:
            result[:, :, i] = ch.astype(np.uint8)
        else:
            result[:, :, i] = np.clip((ch - mn) / (mx - mn) * 255, 0, 255).astype(np.uint8)
    return result

def apply_transform(arr, name, gamma=None):
    if name == "Log Transform":  return log_transform_rgb(arr)
    if name == "Linear Stretch": return linear_stretch_rgb(arr)
    if "γ<1" in name:            return power_transform_rgb(arr, gamma if gamma else 0.4)
    if "γ>1" in name:            return power_transform_rgb(arr, gamma if gamma else 2.5)

TRANSFORMS = ["Log Transform", "Power/Gamma (γ<1)", "Power/Gamma (γ>1)", "Linear Stretch"]
BADGE_MAP  = {
    "Log Transform":     "badge-log",
    "Power/Gamma (γ<1)": "badge-gamma",
    "Power/Gamma (γ>1)": "badge-gamma",
    "Linear Stretch":    "badge-linear",
}
FORMULA = {
    "Log Transform":     "s = c · log(1 + r)  — applied to each R, G, B channel",
    "Power/Gamma (γ<1)": "s = rᵞ, γ < 1  →  brightens each R, G, B channel",
    "Power/Gamma (γ>1)": "s = rᵞ, γ > 1  →  darkens each R, G, B channel",
    "Linear Stretch":    "s = (r − min) / (max − min) × 255  — per channel stretch",
}


# ══════════════════════════════════════════════
#  PAGE
# ══════════════════════════════════════════════
st.markdown('<div class="main-title">Dynamic Range Transformation</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("Drop your image here", type=["png", "jpg", "jpeg", "bmp", "tiff"],
                            label_visibility="collapsed")

if not uploaded:
    st.markdown('<div style="text-align:center;padding:3rem;color:#8892a4;font-family:Space Mono,monospace;">⬆ Drop or browse an image to get started</div>',
                unsafe_allow_html=True)
    st.stop()

img_pil = Image.open(uploaded)
rgb_arr = to_rgb_array(img_pil)

st.divider()

# ════════════════════════════════════════
#  SECTION 1 — Original Image
# ════════════════════════════════════════
st.markdown("#### Original Image")

max_r = min(img_pil.height, 50)
max_c = min(img_pil.width, 50)

ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 4])
with ctrl1:
    mat_rows = st.slider("Matrix rows", 4, max_r, min(10, max_r), key="mat_rows")
with ctrl2:
    mat_cols = st.slider("Matrix cols", 4, max_c, min(10, max_c), key="mat_cols")
with ctrl3:
    st.markdown(f'<div class="info-box" style="margin-top:0.3rem">Showing top-left <b>{mat_rows} × {mat_cols}</b> crop out of <b>{img_pil.height} × {img_pil.width}</b> total pixels. Each cell shows R / G / B values.</div>',
                unsafe_allow_html=True)

c_img, c_info, c_rgb = st.columns([1.2, 0.8, 2.5])

with c_img:
    st.markdown('<div class="badge badge-original">ORIGINAL</div>', unsafe_allow_html=True)
    st.image(img_pil, use_container_width=True)

with c_info:
    st.markdown('<div class="info-box">'
                f'<b>File:</b> {uploaded.name}<br><br>'
                f'<b>Size:</b> {img_pil.width} × {img_pil.height} px<br><br>'
                f'<b>Mode:</b> {img_pil.mode}<br><br>'
                f'<b>R range:</b> {rgb_arr[:,:,0].min()} – {rgb_arr[:,:,0].max()}<br>'
                f'<b>G range:</b> {rgb_arr[:,:,1].min()} – {rgb_arr[:,:,1].max()}<br>'
                f'<b>B range:</b> {rgb_arr[:,:,2].min()} – {rgb_arr[:,:,2].max()}'
                '</div>', unsafe_allow_html=True)

with c_rgb:
    st.markdown('<div class="section-label">🎨 ORIGINAL RGB PIXEL MATRIX</div>', unsafe_allow_html=True)
    st.markdown('<div class="matrix-wrap">' + rgb_matrix_html(rgb_arr, mat_rows, mat_cols) + "</div>",
                unsafe_allow_html=True)
    st.caption("Each cell = R / G / B values. Background = actual pixel colour.")

st.divider()

# ════════════════════════════════════════
#  SECTION 2 — Transform
# ════════════════════════════════════════
st.markdown("#### Apply Transformation")

tc1, tc2, tc3 = st.columns([1.2, 1, 2])

with tc1:
    chosen = st.selectbox("Transformation", TRANSFORMS, label_visibility="collapsed")

with tc2:
    gamma_val = None
    if "Gamma" in chosen:
        lo, hi, default = (0.1, 0.99, 0.4) if "γ<1" in chosen else (1.01, 5.0, 2.5)
        gamma_val = st.slider("γ", lo, hi, default, 0.01 if "γ<1" in chosen else 0.1)
    apply_btn = st.button("Apply Transform")

with tc3:
    st.markdown(f'<div class="info-box"><b>Formula:</b> {FORMULA[chosen]}</div>', unsafe_allow_html=True)

if apply_btn:
    result = apply_transform(rgb_arr, chosen, gamma_val)
    st.session_state["result"]      = result
    st.session_state["result_name"] = chosen
    st.session_state.setdefault("all_transforms", {})[chosen] = result

if "result" in st.session_state and st.session_state.get("result_name") == chosen:
    result = st.session_state["result"]
    badge  = BADGE_MAP[chosen]

    st.markdown(f"**Result: `{chosen}`**")
    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.markdown('<div class="badge badge-original">ORIGINAL</div>', unsafe_allow_html=True)
        st.image(img_pil, use_container_width=True)
    with r2:
        st.markdown(f'<div class="badge {badge}">TRANSFORMED</div>', unsafe_allow_html=True)
        st.image(result, clamp=True, use_container_width=True)
    with r3:
        st.markdown("**Original RGB Matrix**")
        st.markdown('<div class="matrix-wrap-sm">' + rgb_matrix_html(rgb_arr, mat_rows, mat_cols) + "</div>",
                    unsafe_allow_html=True)
    with r4:
        st.markdown("**Transformed RGB Matrix**")
        st.markdown('<div class="matrix-wrap-sm">' + rgb_matrix_html(result, mat_rows, mat_cols) + "</div>",
                    unsafe_allow_html=True)

st.divider()

# ════════════════════════════════════════
#  SECTION 3 — Comparison
# ════════════════════════════════════════
st.markdown("#### Comparison")

all_t = st.session_state.get("all_transforms", {})

if not all_t:
    st.markdown('<div style="text-align:center;padding:2rem;color:#8892a4;font-family:Space Mono,monospace;">Apply at least one transformation above to see the comparison.</div>',
                unsafe_allow_html=True)
else:
    # Store original as PIL for display, others as numpy RGB
    combined_imgs  = {"Original": img_pil, **{k: Image.fromarray(v) for k, v in all_t.items()}}
    combined_arrs  = {"Original": rgb_arr, **all_t}

    img_cols = st.columns(len(combined_imgs))
    for col, (name, img) in zip(img_cols, combined_imgs.items()):
        with col:
            badge = BADGE_MAP.get(name, "badge-original")
            st.markdown(f'<div class="badge {badge}">{name[:14].upper()}</div>', unsafe_allow_html=True)
            st.image(img, use_container_width=True)

    st.markdown("**RGB Pixel Matrices** *(scrollable)*")
    mat_cols_comp = st.columns(len(combined_arrs))
    for col, (name, arr) in zip(mat_cols_comp, combined_arrs.items()):
        with col:
            st.markdown(f"*{name}*")
            st.markdown('<div class="matrix-wrap-sm">' + rgb_matrix_html(arr, mat_rows, mat_cols) + "</div>",
                        unsafe_allow_html=True)