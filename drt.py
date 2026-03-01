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
/* Scrollable matrix container */
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

def to_gray(img):
    return np.array(img.convert("L"), dtype=np.uint8)

def to_rgb_array(img):
    return np.array(img.convert("RGB"), dtype=np.uint8)

def gray_matrix_html(arr, rows, cols):
    """Grayscale pixel matrix with heatmap background."""
    rows = min(rows, arr.shape[0])
    cols = min(cols, arr.shape[1])
    patch = arr[:rows, :cols]
    html = '<table class="pixel-table"><tr><th></th>'
    for c in range(cols): html += f"<th>{c}</th>"
    html += "</tr>"
    for r in range(rows):
        html += f"<tr><th>{r}</th>"
        for c in range(cols):
            val = int(patch[r, c])
            norm = val / 255.0
            bg = f"rgba({int(norm*60)},{int(norm*80)},{int(80+norm*120)},0.7)"
            html += f'<td style="background:{bg}">{val}</td>'
        html += "</tr>"
    html += "</table>"
    return html

def rgb_matrix_html(arr, size=6):
    """RGB pixel matrix — actual pixel color as background."""
    rows = min(size, arr.shape[0])
    cols = min(size, arr.shape[1])
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
                     f'font-size:0.55rem;line-height:1.2">{R}<br>{G}<br>{B}</td>')
        html += "</tr>"
    html += "</table>"
    return html


# ── Transformations ───────────────────────────

def log_transform(arr):
    arr_f = arr.astype(np.float64)
    c = 255 / np.log(1 + arr_f.max()) if arr_f.max() > 0 else 1
    return np.clip(c * np.log(1 + arr_f), 0, 255).astype(np.uint8)

def power_transform(arr, gamma):
    arr_f = arr.astype(np.float64) / 255.0
    return np.clip(np.power(arr_f, gamma) * 255.0, 0, 255).astype(np.uint8)

def linear_stretch(arr):
    mn, mx = arr.min(), arr.max()
    if mx == mn: return arr.copy()
    return ((arr.astype(np.float64) - mn) / (mx - mn) * 255).astype(np.uint8)

TRANSFORMS = ["Log Transform", "Power/Gamma (γ<1)", "Power/Gamma (γ>1)", "Linear Stretch"]
BADGE_MAP  = {
    "Log Transform":     "badge-log",
    "Power/Gamma (γ<1)": "badge-gamma",
    "Power/Gamma (γ>1)": "badge-gamma",
    "Linear Stretch":    "badge-linear",
}
FORMULA = {
    "Log Transform":     "s = c · log(1 + r)",
    "Power/Gamma (γ<1)": "s = rᵞ,  γ < 1  →  brightens image",
    "Power/Gamma (γ>1)": "s = rᵞ,  γ > 1  →  darkens image",
    "Linear Stretch":    "s = (r − min) / (max − min) × 255",
}

def apply_transform(arr, name, gamma=None):
    if name == "Log Transform":  return log_transform(arr)
    if name == "Linear Stretch": return linear_stretch(arr)
    if "γ<1" in name:            return power_transform(arr, gamma if gamma else 0.4)
    if "γ>1" in name:            return power_transform(arr, gamma if gamma else 2.5)


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
gray    = to_gray(img_pil)

st.divider()

# ════════════════════════════════════════
#  SECTION 1 — Original Image
# ════════════════════════════════════════
st.markdown("#### Original Image")

# Matrix size control
max_r = min(img_pil.height, 50)
max_c = min(img_pil.width, 50)

ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 4])
with ctrl1:
    mat_rows = st.slider("Matrix rows", 4, max_r, min(16, max_r), key="mat_rows")
with ctrl2:
    mat_cols = st.slider("Matrix cols", 4, max_c, min(16, max_c), key="mat_cols")
with ctrl3:
    st.markdown(f'<div class="info-box" style="margin-top:0.3rem">Showing top-left <b>{mat_rows} × {mat_cols}</b> pixel crop out of <b>{img_pil.height} × {img_pil.width}</b> total pixels.</div>',
                unsafe_allow_html=True)

c_img, c_info, c_rgb, c_gray = st.columns([1.2, 0.7, 1.8, 1.8])

with c_img:
    st.markdown('<div class="badge badge-original">ORIGINAL</div>', unsafe_allow_html=True)
    st.image(img_pil, use_container_width=True)

with c_info:
    st.markdown('<div class="info-box">'
                f'<b>File:</b> {uploaded.name}<br><br>'
                f'<b>Size:</b> {img_pil.width} × {img_pil.height} px<br><br>'
                f'<b>Mode:</b> {img_pil.mode}<br><br>'
                f'<b>Gray range:</b><br>{gray.min()} – {gray.max()}'
                '</div>', unsafe_allow_html=True)

with c_rgb:
    st.markdown('<div class="section-label"> ORIGINAL RGB PIXEL MATRIX</div>', unsafe_allow_html=True)
    st.markdown('<div class="matrix-wrap">' + rgb_matrix_html(rgb_arr, min(mat_rows, 12)) + "</div>",
                unsafe_allow_html=True)
    st.caption("Each cell = R / G / B. Background = actual pixel colour. (capped at 12 rows for readability)")

with c_gray:
    st.markdown('<div class="section-label">⬛ GRAYSCALE PIXEL MATRIX</div>', unsafe_allow_html=True)
    st.markdown('<div class="matrix-wrap">' + gray_matrix_html(gray, mat_rows, mat_cols) + "</div>",
                unsafe_allow_html=True)
    st.caption(f"Showing {mat_rows}×{mat_cols} crop. Scroll inside the box to see all values.")

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
    apply_btn = st.button(" Apply Transform")

with tc3:
    st.markdown(f'<div class="info-box"><b>Formula:</b> {FORMULA[chosen]}</div>', unsafe_allow_html=True)

if apply_btn:
    result = apply_transform(gray, chosen, gamma_val)
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
        st.image(gray, clamp=True, use_container_width=True)
    with r2:
        st.markdown(f'<div class="badge {badge}">TRANSFORMED</div>', unsafe_allow_html=True)
        st.image(result, clamp=True, use_container_width=True)
    with r3:
        st.markdown("**Original Grayscale Matrix**")
        st.markdown('<div class="matrix-wrap-sm">' + gray_matrix_html(gray, mat_rows, mat_cols) + "</div>",
                    unsafe_allow_html=True)
    with r4:
        st.markdown("**Transformed Matrix**")
        st.markdown('<div class="matrix-wrap-sm">' + gray_matrix_html(result, mat_rows, mat_cols) + "</div>",
                    unsafe_allow_html=True)

st.divider()

# ════════════════════════════════════════
#  SECTION 3 — Comparison
# ════════════════════════════════════════
st.markdown("####  Comparison")

all_t = st.session_state.get("all_transforms", {})

if not all_t:
    st.markdown('<div style="text-align:center;padding:2rem;color:#8892a4;font-family:Space Mono,monospace;">Apply at least one transformation above to see the comparison.</div>',
                unsafe_allow_html=True)
else:
    combined = {"Original": gray, **all_t}

    img_cols = st.columns(len(combined))
    for col, (name, arr) in zip(img_cols, combined.items()):
        with col:
            badge = BADGE_MAP.get(name, "badge-original")
            st.markdown(f'<div class="badge {badge}">{name[:14].upper()}</div>', unsafe_allow_html=True)
            st.image(arr, clamp=True, use_container_width=True)

    st.markdown("**Pixel Matrices** *(scrollable)*")
    mat_cols_comp = st.columns(len(combined))
    for col, (name, arr) in zip(mat_cols_comp, combined.items()):
        with col:
            st.markdown(f"*{name}*")
            st.markdown('<div class="matrix-wrap-sm">' + gray_matrix_html(arr, mat_rows, mat_cols) + "</div>",
                        unsafe_allow_html=True)