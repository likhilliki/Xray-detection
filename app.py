import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import plotly.express as px
import time

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI X-ray Foreign Body Detector",
    page_icon="🩻",
    layout="wide",
)

# ------------------ CUSTOM CSS ------------------
page_bg = """
<style>
/* Background Gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #ffffff;
    font-family: 'Poppins', sans-serif;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: rgba(20, 20, 20, 0.95);
    color: #ffffff;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Poppins', sans-serif;
    color: #00e6e6 !important;
    font-weight: 600;
}

/* Metrics */
.stMetric {
    background: rgba(255, 255, 255, 0.08);
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #00e6e6;
    box-shadow: 0 0 10px rgba(0, 230, 230, 0.5);
}

/* Buttons */
.stDownloadButton button, .stButton button {
    background: linear-gradient(90deg, #00e6e6, #7f00ff);
    color: white;
    border-radius: 12px;
    padding: 10px 20px;
    border: none;
    font-weight: bold;
    transition: 0.3s;
}
.stDownloadButton button:hover, .stButton button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 15px #00e6e6;
}

/* Expander */
.streamlit-expanderHeader {
    font-weight: bold;
    color: #00e6e6 !important;
}

/* Pie Chart Title */
.plotly .main-svg {
    background: transparent !important;
}
</style>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
model = YOLO("best.pt")  # make sure best.pt is in same folder

# ------------------ SIDEBAR ------------------
st.sidebar.title("Info")
st.sidebar.markdown("""
AI-powered X-ray Foreign Body Detector  

**Model:** YOLOv8 (trained on foreign body dataset)  

Upload an X-ray image to detect foreign bodies such as:
- Metal  
- Glass  
- Plastic  
- Wood  
""")

confidence_threshold = st.sidebar.slider("⚙️ Detection Confidence", 0.1, 1.0, 0.4, 0.05)

# ------------------ MAIN APP ------------------
st.title("🩻 AI X-ray Foreign Body Detector")
st.markdown("### Upload an X-ray below to analyze for foreign bodies.")

uploaded_file = st.file_uploader("📂 Upload an X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        pil_img = Image.open(uploaded_file).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        start = time.time()
        results = model.predict(img, conf=confidence_threshold, verbose=False)
        end = time.time()

        detected_classes, confidences = [], []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{model.names[cls]} ({conf:.2f})"
                detected_classes.append(model.names[cls])
                confidences.append(conf)

                # Confidence-based color (green = high, red = low)
                color = (0, int(conf * 255), int((1 - conf) * 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # ------------------ KPI METRICS ------------------
        st.markdown("## 📊 Dashboard Overview")
        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("Objects Detected", len(detected_classes))
        with colB:
            st.metric("Avg Confidence", f"{np.mean(confidences):.2f}" if confidences else "0")
        with colC:
            st.metric("Detection Time", f"{end - start:.2f} sec")

        # ------------------ DISPLAY RESULTS ------------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔍 Detection Result")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                     caption="Processed X-ray",
                     use_container_width=True)

            is_success, buffer = cv2.imencode(".jpg", img)
            if is_success:
                st.download_button(
                    label="📥 Download Processed Image",
                    data=buffer.tobytes(),
                    file_name="detection_result.jpg",
                    mime="image/jpeg"
                )

        with col2:
            st.subheader("📈 Detection Analysis")
            if detected_classes:
                class_counts = {cls: detected_classes.count(cls) for cls in set(detected_classes)}

                fig = px.pie(values=class_counts.values(), names=class_counts.keys(),
                             hole=0.3, title="Detected Objects Distribution",
                             color_discrete_sequence=["#00e6e6", "#7f00ff", "#ff0066", "#ffcc00"])
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("📋 Detection Details"):
                    for cls, count in class_counts.items():
                        st.write(f"- **{cls}**: {count} instance(s)")
            else:
                st.warning("⚠️ No foreign bodies detected.")

    except Exception as e:
        st.error(f"⚠️ Error reading the image: {e}")
