import io
import os
import zipfile
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

APP_NAME = "PrivaMed"
API_URL = os.getenv("API_URL", "http://api:8000")
API_DOCS_URL = os.getenv("API_DOCS_URL", "http://localhost:8010")

CLASS_NAMES = {0: "name", 1: "id", 2: "age", 3: "date", 4: "time"}
SENSITIVE_CLASSES = {0, 1, 2}

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #888;
        font-size: 1.1rem;
    }
    .stat-card {
        background: rgba(58, 123, 213, 0.08);
        border: 1px solid rgba(58, 123, 213, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .stat-card h3 { margin: 0; font-size: 2rem; color: #3a7bd5; }
    .stat-card p  { margin: 0; color: #aaa; font-size: 0.85rem; }
    .metric-card {
        background: rgba(0, 200, 83, 0.06);
        border: 1px solid rgba(0, 200, 83, 0.2);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .metric-card h3 { margin: 0; font-size: 1.8rem; color: #00c853; }
    .metric-card p  { margin: 0; color: #aaa; font-size: 0.8rem; }
    .status-trained {
        background: rgba(0, 200, 83, 0.1);
        border: 1px solid rgba(0, 200, 83, 0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
    }
    .status-untrained {
        background: rgba(255, 75, 75, 0.08);
        border: 1px solid rgba(255, 75, 75, 0.25);
        border-radius: 12px;
        padding: 1rem 1.5rem;
    }
    .status-error {
        background: rgba(255, 165, 0, 0.08);
        border: 1px solid rgba(255, 165, 0, 0.25);
        border-radius: 12px;
        padding: 1rem 1.5rem;
    }
    .redacted-tag {
        display: inline-block;
        background: #ff4b4b33;
        color: #ff4b4b;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .safe-tag {
        display: inline-block;
        background: #00c85333;
        color: #00c853;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ───────────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <div class="main-header">
        <h1>🛡️ {APP_NAME}</h1>
        <p>Anonimización inteligente de radiografías médicas</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuración de detección")

    confidence = st.slider("Umbral de confianza", 0.05, 0.95, 0.25, 0.05)
    margin = st.slider("Margen de redacción (px)", 0, 30, 5, 1)

    st.markdown("#### 🏷️ Clases a redactar")
    redact_classes: list[int] = []
    for cls_id, cls_name in CLASS_NAMES.items():
        if st.checkbox(cls_name.capitalize(), value=cls_id in SENSITIVE_CLASSES, key=f"cls_{cls_id}"):
            redact_classes.append(cls_id)

    st.divider()
    st.markdown(f"**API:** `{API_URL}`")
    st.markdown(f"[📖 Documentación API]({API_DOCS_URL}/docs)")
    st.markdown(f"[📘 ReDoc]({API_DOCS_URL}/redoc)")


# ── API helpers ──────────────────────────────────────────────────────────────

def api_get(path: str):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception:
        return None


def api_post_file(path: str, file_bytes: bytes, filename: str, params: dict):
    try:
        r = requests.post(
            f"{API_URL}{path}",
            files={"file": (filename, file_bytes, "image/png")},
            params=params,
            timeout=120,
        )
        r.raise_for_status()
        return r
    except Exception as e:
        st.error(f"Error en la API: {e}")
        return None


def api_post_files(path: str, file_list: list[tuple[str, bytes]], params: dict):
    try:
        files = [("files", (name, data, "image/png")) for name, data in file_list]
        r = requests.post(
            f"{API_URL}{path}",
            files=files,
            params=params,
            timeout=300,
        )
        r.raise_for_status()
        return r
    except Exception as e:
        st.error(f"Error en la API: {e}")
        return None


# ── Section 1: Model status ─────────────────────────────────────────────────

st.markdown("## 🧠 Estado del modelo")

model_info = api_get("/model/info")

if model_info is None:
    st.markdown(
        '<div class="status-error">🔌 <strong>No se pudo conectar con la API</strong> — '
        f"verifica que el servicio esté corriendo en <code>{API_URL}</code></div>",
        unsafe_allow_html=True,
    )
    st.stop()

if model_info.get("status") == "trained":
    from datetime import datetime
    trained_dt = datetime.fromisoformat(model_info["trained_at"])
    metrics = model_info["metrics"]

    st.markdown(
        '<div class="status-trained">✅ <strong>Modelo entrenado y listo</strong></div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(
            f'<div class="metric-card"><h3>{metrics["mAP50"]:.1%}</h3><p>mAP@50</p></div>',
            unsafe_allow_html=True,
        )
    with mc2:
        st.markdown(
            f'<div class="metric-card"><h3>{metrics["mAP50-95"]:.1%}</h3><p>mAP@50-95</p></div>',
            unsafe_allow_html=True,
        )
    with mc3:
        st.markdown(
            f'<div class="metric-card"><h3>{metrics["precision"]:.1%}</h3><p>Precisión</p></div>',
            unsafe_allow_html=True,
        )
    with mc4:
        st.markdown(
            f'<div class="metric-card"><h3>{metrics["recall"]:.1%}</h3><p>Recall</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        st.markdown(f"**Fecha:** {trained_dt.strftime('%d/%m/%Y %H:%M')}")
    with ic2:
        st.markdown(f"**Épocas:** {model_info['epochs']}  |  **Imagen:** {model_info['imgsz']}px")
    with ic3:
        st.markdown(f"**Batch:** {model_info['batch']}  |  **Base:** {model_info['base_model']}")
else:
    st.markdown(
        '<div class="status-untrained">⚠️ <strong>Modelo no entrenado</strong> — '
        "ejecuta el entrenamiento con: <code>docker compose --profile train run --rm train</code></div>",
        unsafe_allow_html=True,
    )
    st.stop()

st.divider()

# ── Section 2: Image processing ─────────────────────────────────────────────

st.markdown("## 🔒 Procesamiento de imágenes")

tab_single, tab_batch = st.tabs(["📄 Imagen individual", "📁 Procesamiento por lote"])

# ── Single image ─────────────────────────────────────────────────────────────

with tab_single:
    uploaded = st.file_uploader(
        "Sube una radiografía",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        key="single_upload",
    )

    if uploaded is not None:
        file_bytes = uploaded.read()
        params = {"confidence": confidence, "margin": margin}
        if redact_classes:
            params["classes"] = redact_classes

        detect_resp = api_post_file("/detect", file_bytes, uploaded.name, {"confidence": confidence})

        if detect_resp and detect_resp.status_code == 200:
            detect_data = detect_resp.json()
            detections = detect_data["detections"]
            total = detect_data["total"]
            redact_set = set(redact_classes)
            redacted_count = sum(1 for d in detections if d["class_id"] in redact_set)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f'<div class="stat-card"><h3>{total}</h3><p>Detecciones</p></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<div class="stat-card"><h3>{redacted_count}</h3><p>Redactadas</p></div>',
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f'<div class="stat-card"><h3>{total - redacted_count}</h3><p>Conservadas</p></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Detecciones**")
                preview_params = {"confidence": confidence}
                if redact_classes:
                    preview_params["classes"] = redact_classes
                preview_resp = api_post_file("/preview", file_bytes, uploaded.name, preview_params)
                if preview_resp:
                    st.image(preview_resp.content, use_container_width=True)

            with col2:
                st.markdown("**Resultado anonimizado**")
                anon_resp = api_post_file("/anonymize", file_bytes, uploaded.name, params)
                if anon_resp:
                    st.image(anon_resp.content, use_container_width=True)

                    st.download_button(
                        "⬇️ Descargar imagen anonimizada",
                        data=anon_resp.content,
                        file_name=f"{Path(uploaded.name).stem}_anon.png",
                        mime="image/png",
                        use_container_width=True,
                    )

            if detections:
                with st.expander("📋 Detalle de detecciones", expanded=False):
                    for i, det in enumerate(detections):
                        tag_class = "redacted-tag" if det["class_id"] in redact_set else "safe-tag"
                        tag_text = "REDACTADO" if det["class_id"] in redact_set else "CONSERVADO"
                        st.markdown(
                            f'`#{i+1}` **{det["class_name"]}** — '
                            f'confianza: {det["confidence"]:.1%} — '
                            f'<span class="{tag_class}">{tag_text}</span>',
                            unsafe_allow_html=True,
                        )

# ── Batch processing ─────────────────────────────────────────────────────────

with tab_batch:
    uploaded_files = st.file_uploader(
        "Sube una o más radiografías",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        accept_multiple_files=True,
        key="batch_upload",
    )

    if uploaded_files:
        st.info(f"{len(uploaded_files)} imagen(es) seleccionada(s)")

        if st.button("🔒 Procesar todas", use_container_width=True, type="primary"):
            with st.spinner("Procesando lote en la API..."):
                file_list = [(f.name, f.read()) for f in uploaded_files]
                params = {"confidence": confidence, "margin": margin}
                if redact_classes:
                    params["classes"] = redact_classes

                resp = api_post_files("/anonymize/batch", file_list, params)

            if resp:
                st.success(f"¡{len(uploaded_files)} imágenes procesadas!")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(
                        f'<div class="stat-card"><h3>{len(uploaded_files)}</h3>'
                        f"<p>Imágenes procesadas</p></div>",
                        unsafe_allow_html=True,
                    )

                st.download_button(
                    "⬇️ Descargar ZIP con todas las imágenes anonimizadas",
                    data=resp.content,
                    file_name="radshield_batch.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
