from dotenv import load_dotenv

load_dotenv()

import os  # ✅ NEW
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import textwrap
from streamlit.components.v1 import html as st_html

# ✅ NEW: decision trace
from agentic.trace import build_decision_trace

# ✅ Agentic imports
from agentic.decision_agent import DecisionAgent
from agentic.adapters import pick_primary_detection, detection_to_damage_signal
from agentic.explainer import (
    build_customer_explanation,
    format_kb_insights,
    build_expert_insight,
)

# ✅ 3.1) WOW layer
from agentic.strategies import build_repair_strategies, build_damage_story

# ✅ NEW: AI inpaint preview
from agentic.vision.after_inpaint import make_repaired_after_preview

# Import custom modules (optional; demo mode if missing)
try:
    from car_damage_detector import CarDamageDetector
    from utils import enhance_image, calculate_damage_stats
except ImportError:
    st.warning("Custom modules not found. Running in demo mode.")
    CarDamageDetector = None


# Page configuration
st.set_page_config(
    page_title="Car Damage Assessment AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root{
  --bg0:#060607;
  --bg1:#0a0b0d;
  --panel: rgba(255,255,255,0.035);
  --panel2: rgba(255,255,255,0.055);
  --stroke: rgba(255,255,255,0.10);
  --stroke2: rgba(255,255,255,0.14);
  --text: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.68);
  --muted2: rgba(255,255,255,0.52);
  --shadow: 0 28px 90px rgba(0,0,0,0.55);
  --shadow2: 0 16px 50px rgba(0,0,0,0.50);
  --radius: 22px;
  --radius2: 16px;
  --focus: rgba(255,255,255,0.20);
  --white: rgba(255,255,255,0.92);
}

html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp{
  background:
    radial-gradient(circle at 12% 10%, rgba(255,255,255,0.05), transparent 48%),
    radial-gradient(circle at 88% 18%, rgba(255,255,255,0.03), transparent 45%),
    linear-gradient(180deg, var(--bg0), var(--bg1));
  color: var(--text);
}

/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Page spacing */
.block-container{
  padding-top: 1.25rem;
  padding-bottom: 3rem;
  max-width: 1320px;
}

/* Sidebar */
section[data-testid="stSidebar"]{
  background: rgba(255,255,255,0.018);
  border-right: 1px solid rgba(255,255,255,0.07);
}
section[data-testid="stSidebar"] .block-container{
  padding-top: 1.2rem;
}

/* Typography */
h1,h2,h3{
  letter-spacing: -0.045em;
}
.stMarkdown p{
  color: var(--muted);
}

/* Alerts */
div[data-testid="stAlert"]{
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  box-shadow: 0 10px 28px rgba(0,0,0,0.40);
}
div[data-testid="stAlert"] *{
  color: rgba(255,255,255,0.86) !important;
}

/* Hero */
.hero-wrap{
  border: 1px solid rgba(255,255,255,0.10);
  background: linear-gradient(135deg, rgba(255,255,255,0.070), rgba(255,255,255,0.018));
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  overflow: hidden;
  position: relative;
  padding: 24px 24px;
  margin-bottom: 16px;
  animation: fadeUp .42s ease-out both;
}
@keyframes fadeUp{
  from{ opacity:0; transform: translateY(10px); }
  to{ opacity:1; transform: translateY(0px); }
}
.hero-title{
  font-size: clamp(2.2rem, 3.6vw, 3.35rem);
  font-weight: 900;
  letter-spacing: -0.06em;
  line-height: 1.05;
  margin: 0;
  background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(255,255,255,0.68));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.hero-subtitle{
  margin-top: 10px;
  font-size: 1.02rem;
  color: rgba(255,255,255,0.64);
  max-width: 78ch;
}
.tag-row{
  display:flex;
  gap:10px;
  flex-wrap:wrap;
  margin-top: 14px;
}
.tag{
  display:inline-flex;
  align-items:center;
  padding: 8px 12px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.02);
  color: rgba(255,255,255,0.78);
  font-size: 0.88rem;
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
}
.hero-right{
  border-radius: var(--radius2);
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.018);
  padding: 14px;
  box-shadow: var(--shadow2);
  animation: fadeUp .52s ease-out both;
}

/* Metrics */
[data-testid="stMetric"]{
  background: rgba(255,255,255,0.028);
  border: 1px solid rgba(255,255,255,0.10);
  padding: 14px;
  border-radius: 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.40);
}
[data-testid="stMetricLabel"] p{
  color: rgba(255,255,255,0.52) !important;
  font-weight: 650;
}
[data-testid="stMetricValue"]{
  font-weight: 850;
  letter-spacing: -0.02em;
}

/* Expanders */
div[data-testid="stExpander"]{
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.018);
  border-radius: 16px;
  overflow:hidden;
}
div[data-testid="stExpander"] details summary{
  padding: 10px 14px;
  font-weight: 800;
}

/* Inputs */
[data-testid="stFileUploader"]{
  border: 1px dashed rgba(255,255,255,0.16);
  padding: 14px;
  border-radius: 16px;
  background: rgba(255,255,255,0.018);
}

/* Buttons */
.stButton button{
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  background: linear-gradient(180deg, rgba(255,255,255,0.11), rgba(255,255,255,0.035)) !important;
  color: rgba(255,255,255,0.92) !important;
  font-weight: 850 !important;
  letter-spacing: -0.02em;
  padding: 0.82rem 1rem !important;
  box-shadow: 0 18px 55px rgba(0,0,0,0.55);
  transform: translateY(0);
  transition: transform .18s ease, box-shadow .18s ease, filter .18s ease, border-color .18s ease;
}
.stButton button:hover{
  transform: translateY(-2px);
  box-shadow: 0 26px 80px rgba(0,0,0,0.65);
  border-color: rgba(255,255,255,0.22) !important;
  filter: brightness(1.06);
}
.stButton button:active{
  transform: translateY(0px) scale(0.99);
}

/* Download button */
[data-testid="stDownloadButton"] button{
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  background: rgba(255,255,255,0.02) !important;
  color: rgba(255,255,255,0.90) !important;
  font-weight: 780 !important;
  transition: transform .18s ease, background .18s ease, border-color .18s ease;
}
[data-testid="stDownloadButton"] button:hover{
  transform: translateY(-2px);
  background: rgba(255,255,255,0.04) !important;
  border-color: rgba(255,255,255,0.20) !important;
}

/* Dataframe + Plotly */
[data-testid="stDataFrame"]{
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  overflow:hidden;
}
.js-plotly-plot, .plot-container{
  border-radius: 16px;
  overflow:hidden;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.018);
}

/* Dividers */
hr{
  border:none;
  height:1px;
  background: rgba(255,255,255,0.08);
  margin: 16px 0;
}

/* Better focus */
*:focus-visible{
  outline: 2px solid rgba(255,255,255,0.18) !important;
  outline-offset: 2px;
  border-radius: 10px;
}

/* Multiselect tags (remove red/pink) */
div[data-baseweb="tag"]{
  background-color: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
}
div[data-baseweb="tag"] span{
  color: rgba(255,255,255,0.86) !important;
  font-weight: 650 !important;
}
div[data-baseweb="tag"] svg{
  color: rgba(255,255,255,0.70) !important;
}

/* Slider accent (remove red) */
div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]{
  background-color: rgba(255,255,255,0.92) !important;
  box-shadow: 0 0 0 6px rgba(255,255,255,0.06);
}
div[data-testid="stSlider"] [data-baseweb="slider"] div[aria-valuemin]{
  background: rgba(255,255,255,0.20) !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] div[aria-valuenow]{
  background: rgba(255,255,255,0.55) !important;
}

/* Checkbox accents */
div[data-testid="stCheckbox"] svg{
  color: rgba(255,255,255,0.90) !important;
}

/* Headers inside main content */
h2{
  margin-bottom: 0.25rem;
}
</style>
""",
    unsafe_allow_html=True,
)


def hero_svg_car_damage() -> str:
    svg = r"""
<div style="border-radius:14px; border:1px solid rgba(255,255,255,0.10);
            background: radial-gradient(circle at 30% 35%, rgba(255,255,255,0.08), transparent 55%),
                        radial-gradient(circle at 75% 40%, rgba(255,255,255,0.05), transparent 55%),
                        linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.015));
            overflow:hidden;">
<svg viewBox="0 0 1200 700" width="100%" height="240" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="rgba(255,255,255,0.06)"/>
      <stop offset="1" stop-color="rgba(255,255,255,0.012)"/>
    </linearGradient>
    <radialGradient id="glow" cx="38%" cy="42%" r="70%">
      <stop offset="0" stop-color="rgba(255,255,255,0.10)"/>
      <stop offset="1" stop-color="rgba(255,255,255,0)"/>
    </radialGradient>
    <filter id="soft" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="10" result="b"/>
      <feMerge>
        <feMergeNode in="b"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>

  <rect x="0" y="0" width="1200" height="700" fill="url(#bg)"/>
  <rect x="0" y="0" width="1200" height="700" fill="url(#glow)"/>

  <g opacity="0.08">
    <path d="M0 120 H1200" stroke="white" stroke-width="1"/>
    <path d="M0 240 H1200" stroke="white" stroke-width="1"/>
    <path d="M0 360 H1200" stroke="white" stroke-width="1"/>
    <path d="M0 480 H1200" stroke="white" stroke-width="1"/>
    <path d="M0 600 H1200" stroke="white" stroke-width="1"/>
  </g>

  <g transform="translate(80,210)">
    <path d="M160 250
             C240 190, 350 145, 520 140
             C650 135, 770 150, 880 190
             C930 208, 960 240, 980 270
             L1020 270
             C1060 270, 1090 300, 1090 340
             L1090 390
             C1090 430, 1060 460, 1020 460
             L980 460
             C960 510, 915 545, 860 545
             C800 545, 755 510, 740 460
             L420 460
             C405 510, 360 545, 300 545
             C240 545, 195 510, 180 460
             L140 460
             C95 460, 60 425, 60 380
             L60 330
             C60 295, 85 268, 120 260
             Z"
          fill="rgba(255,255,255,0.05)"
          stroke="rgba(255,255,255,0.24)"
          stroke-width="2"/>

    <path d="M320 250
             C360 210, 420 185, 520 180
             C650 174, 740 195, 820 240
             L780 240
             C710 210, 640 198, 530 204
             C450 208, 400 228, 360 250
             Z"
          fill="rgba(255,255,255,0.035)"
          stroke="rgba(255,255,255,0.16)"
          stroke-width="2"/>

    <path d="M390 250
             C430 220, 480 205, 560 202
             C650 198, 715 210, 775 240
             L720 240
             C670 220, 620 212, 560 214
             C500 216, 450 228, 420 250
             Z"
          fill="rgba(0,0,0,0.22)"
          stroke="rgba(255,255,255,0.10)"
          stroke-width="2"/>

    <circle cx="300" cy="460" r="78" fill="rgba(0,0,0,0.35)" stroke="rgba(255,255,255,0.22)" stroke-width="2"/>
    <circle cx="300" cy="460" r="40" fill="rgba(255,255,255,0.04)" stroke="rgba(255,255,255,0.16)" stroke-width="2"/>
    <circle cx="860" cy="460" r="78" fill="rgba(0,0,0,0.35)" stroke="rgba(255,255,255,0.22)" stroke-width="2"/>
    <circle cx="860" cy="460" r="40" fill="rgba(255,255,255,0.04)" stroke="rgba(255,255,255,0.16)" stroke-width="2"/>

    <g filter="url(#soft)">
      <path d="M860 260 C920 260, 965 300, 965 360
               C965 420, 920 460, 860 460
               C810 460, 770 430, 760 390
               C820 370, 860 330, 860 260 Z"
            fill="rgba(255,255,255,0.09)"
            stroke="rgba(255,255,255,0.50)"
            stroke-width="2"/>
      <path d="M790 380 L940 320" stroke="rgba(255,255,255,0.72)" stroke-width="2" stroke-linecap="round" opacity="0.9"/>
      <path d="M805 400 L955 340" stroke="rgba(255,255,255,0.60)" stroke-width="2" stroke-linecap="round" opacity="0.85"/>
      <path d="M820 418 L930 372" stroke="rgba(255,255,255,0.50)" stroke-width="2" stroke-linecap="round" opacity="0.78"/>
    </g>

    <g opacity="0.92">
      <rect x="692" y="120" width="380" height="64" rx="14" fill="rgba(255,255,255,0.04)" stroke="rgba(255,255,255,0.12)"/>
      <text x="714" y="160" font-size="26" fill="rgba(255,255,255,0.84)" font-family="Inter, -apple-system, sans-serif">
        Damage area highlight
      </text>
      <path d="M860 184 L860 255" stroke="rgba(255,255,255,0.32)" stroke-width="2"/>
      <circle cx="860" cy="255" r="6" fill="rgba(255,255,255,0.50)"/>
    </g>
  </g>

  <rect x="0" y="0" width="1200" height="700" fill="rgba(0,0,0,0.22)"/>
</svg>
</div>
"""
    return textwrap.dedent(svg).strip()


def demo_damage_detection(image: Image.Image):
    """Demo function that simulates damage detection for demonstration purposes."""
    img_array = np.array(image)
    height, width = img_array.shape[:2]

    detections = [
        {
            "type": "Scratch",
            "severity": "Light",
            "confidence": 0.89,
            "bbox": [int(width * 0.2), int(height * 0.3), int(width * 0.4), int(height * 0.5)],
            "area_percentage": 2.5,
            "estimated_cost": 150,
        },
        {
            "type": "Dent",
            "severity": "Moderate",
            "confidence": 0.76,
            "bbox": [int(width * 0.6), int(height * 0.2), int(width * 0.8), int(height * 0.4)],
            "area_percentage": 8.3,
            "estimated_cost": 450,
        },
        {
            "type": "Paint Damage",
            "severity": "Light",
            "confidence": 0.82,
            "bbox": [int(width * 0.1), int(height * 0.6), int(width * 0.25), int(height * 0.8)],
            "area_percentage": 3.2,
            "estimated_cost": 200,
        },
    ]

    img_with_annotations = img_array.copy()

    colors = {
        "Scratch": (0, 255, 0),
        "Dent": (255, 165, 0),
        "Paint Damage": (255, 0, 255),
        "Broken Part": (255, 0, 0),
    }

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        color = colors.get(detection["type"], (255, 255, 0))

        cv2.rectangle(img_with_annotations, (x1, y1), (x2, y2), color, 3)

        label = f"{detection['type']} ({detection['confidence']:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

        cv2.rectangle(
            img_with_annotations,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            color,
            -1,
        )

        cv2.putText(
            img_with_annotations,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    return img_with_annotations, detections


def create_damage_distribution_chart(detections):
    damage_counts = {}
    for detection in detections:
        damage_type = detection["type"]
        damage_counts[damage_type] = damage_counts.get(damage_type, 0) + 1

    fig = px.pie(values=list(damage_counts.values()), names=list(damage_counts.keys()), title="Damage Type Distribution")
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.86)"),
        title_font=dict(size=18),
    )
    return fig


def create_severity_chart(detections):
    severity_counts = {}
    for detection in detections:
        severity = detection["severity"]
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    colors = {"Light": "#cfcfcf", "Moderate": "#9a9a9a", "Severe": "#f2f2f2"}

    fig = go.Figure(
        data=[
            go.Bar(
                x=list(severity_counts.keys()),
                y=list(severity_counts.values()),
                marker_color=[colors.get(k, "#8a8a8a") for k in severity_counts.keys()],
            )
        ]
    )

    fig.update_layout(
        title="Damage Severity Distribution",
        xaxis_title="Severity Level",
        yaxis_title="Number of Damages",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.86)"),
        title_font=dict(size=18),
    )

    return fig


def generate_assessment_report(detections, image_info):
    total_cost = sum([d.get("estimated_cost", 0) for d in detections])
    total_area = sum([d["area_percentage"] for d in detections])
    avg_confidence = np.mean([d["confidence"] for d in detections])

    severity_priority = {"Severe": 3, "Moderate": 2, "Light": 1}
    highest_severity = max([severity_priority.get(d["severity"], 0) for d in detections])
    severity_names = {3: "Severe", 2: "Moderate", 1: "Light"}

    report = {
        "timestamp": datetime.now(),
        "image_dimensions": image_info,
        "total_damages": len(detections),
        "total_affected_area": total_area,
        "estimated_repair_cost": total_cost,
        "average_confidence": avg_confidence,
        "highest_severity": severity_names.get(highest_severity, "None"),
        "damage_breakdown": detections,
    }
    return report


def render_decision_actions(decision, primary_detection):
    """
    Decision-driven UI: show different CTAs based on agent decision.
    This is a Spike UI layer (no external integrations yet).
    """
    st.markdown("---")
    st.subheader("Next Actions")

    if primary_detection:
        st.caption(
            f"Primary detection: {primary_detection.get('type')} / {primary_detection.get('severity')} "
            f"(conf={primary_detection.get('confidence', 0.0):.2f})"
        )

    if decision.action == "AUTO_APPROVE":
        st.success("This case is eligible for auto-approval. You can create a repair ticket.")
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("✅ Create Repair Ticket", use_container_width=True):
                st.session_state["ticket_created"] = True

        with col2:
            if st.button("📩 Notify Customer", use_container_width=True):
                st.session_state["customer_notified"] = True

        if st.session_state.get("ticket_created"):
            st.info("Ticket created (demo). Next: integrate with real ticketing system.")
        if st.session_state.get("customer_notified"):
            st.info("Customer notified (demo). Next: integrate with email/SMS.")

    elif decision.action == "HUMAN_REVIEW":
        st.warning("This case requires human review. Complete the checklist and submit.")

        st.write("**Operator checklist:**")
        c1 = st.checkbox("Verify vehicle ID / VIN from the photos")
        c2 = st.checkbox("Confirm severity and affected area")
        c3 = st.checkbox("Request additional angles if needed")
        c4 = st.checkbox("Check if replacement vs repair is required")

        ready = all([c1, c2, c3, c4])

        col1, col2 = st.columns([1, 1])
        with col1:
            notes = st.text_area("Operator notes (demo)", height=90, placeholder="Write short notes for the reviewer...")

        with col2:
            if st.button("⚠️ Submit for Human Review", use_container_width=True, disabled=not ready):
                st.session_state["submitted_for_review"] = True
                st.session_state["review_notes"] = notes

        if st.session_state.get("submitted_for_review"):
            st.info("Submitted for review (demo). Next: send to queue / DB.")
            if st.session_state.get("review_notes"):
                st.code(st.session_state["review_notes"])

    elif decision.action == "ESCALATE":
        st.error("This case must be escalated to a specialist assessor.")

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if st.button("🔺 Assign Senior Assessor", use_container_width=True):
                st.session_state["assigned_senior"] = True

        with col2:
            if st.button("📎 Request Additional Documents", use_container_width=True):
                st.session_state["requested_docs"] = True

        with col3:
            if st.button("🧾 Generate Escalation Packet", use_container_width=True):
                st.session_state["packet_generated"] = True

        if st.session_state.get("assigned_senior"):
            st.info("Assigned to senior assessor (demo). Next: integrate with workflow.")
        if st.session_state.get("requested_docs"):
            st.info("Requested additional documents (demo). Next: email/portal integration.")
        if st.session_state.get("packet_generated"):
            st.info("Escalation packet generated (demo). Next: PDF export.")

    else:
        st.info("No action UI for this decision type yet.")


def render_hero():
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown(
            """
<div class="hero-wrap">
  <h1 class="hero-title">Car Damage Assessment AI</h1>
  <div class="hero-subtitle">
    High-trust vehicle damage intelligence. Computer vision detection and agentic decisioning,
    presented in a premium monochrome interface.
  </div>
  <div class="tag-row">
    <span class="tag">Real-time inference</span>
    <span class="tag">Policy workflow</span>
    <span class="tag">Analytics and reporting</span>
    <span class="tag">Decision assist</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="hero-right">', unsafe_allow_html=True)
        st_html(hero_svg_car_damage(), height=255)
        st.markdown(
            "<div style='margin-top:10px; color:rgba(255,255,255,0.62); font-size:0.92rem;'>"
            "Tip: sharp, well-lit images improve confidence scores."
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ✅ Wrapper to match the requested preview API (dict with RGB arrays)
def make_repaired_preview(before_rgb: np.ndarray, primary: dict, intensity: float = 0.65):
    """
    Returns:
      {
        "after": after_rgb,
        "diff": diff_rgb,
        "mask": mask_rgb_or_gray
      }
    Internally uses agentic.vision.after_inpaint.make_repaired_after_preview (BGR-based).
    """
    if before_rgb is None or primary is None:
        return {"after": before_rgb, "diff": None, "mask": None}

    # input: RGB -> BGR
    before_bgr = before_rgb[:, :, ::-1].copy()

    # method is fixed here to "cv" as in your snippet label "CV inpaint preview"
    res = make_repaired_after_preview(before_bgr, primary, intensity=float(intensity), method="cv")

    after_bgr = res.after_bgr
    diff_bgr = res.diff_bgr

    # Best-effort mask extraction (if exists in result)
    mask = getattr(res, "mask", None)
    if mask is None:
        mask = getattr(res, "mask_bgr", None)
    if mask is None:
        mask = getattr(res, "mask_gray", None)

    # convert outputs to RGB for Streamlit
    after_rgb = after_bgr[:, :, ::-1] if after_bgr is not None else None
    diff_rgb = diff_bgr[:, :, ::-1] if diff_bgr is not None else None

    if mask is None:
        mask_vis = None
    else:
        if mask.ndim == 2:
            mask_vis = mask  # grayscale ok
        elif mask.ndim == 3 and mask.shape[2] == 3:
            # assume BGR -> RGB
            mask_vis = mask[:, :, ::-1]
        else:
            mask_vis = mask

    return {"after": after_rgb, "diff": diff_rgb, "mask": mask_vis}


def main():
    render_hero()

    with st.sidebar:
        st.markdown("## Configuration")
        st.markdown(
            "<div style='color:rgba(255,255,255,0.58); margin-top:-8px;'>Tune detection and reporting</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # ✅ dev mode toggle
        developer_mode = st.checkbox("Developer mode (show debug)", value=False)
        # ✅ IMPORTANT: make session_state key available everywhere
        st.session_state["dev_mode"] = developer_mode

        st.markdown("### Detection Parameters")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence score for damage detection",
        )

        damage_types = st.multiselect(
            "Damage Types to Detect",
            ["Scratches", "Dents", "Broken Parts", "Paint Damage"],
            default=["Scratches", "Dents", "Paint Damage"],
            help="Select which types of damage to analyze",
        )

        st.markdown("### Processing Options")
        enhance_image_option = st.checkbox("Enable Image Enhancement", value=True)
        show_confidence = st.checkbox("Display Confidence Scores", value=True)
        generate_report = st.checkbox("Generate Assessment Report", value=True)

        st.markdown("---")
        st.markdown(
            """
        #### System Information
        **Model**: YOLOv8 Custom Trained  
        **Accuracy**: 87% mAP@0.5  
        **Classes**: 4 damage types  
        **Processing**: Real-time inference  
        """
        )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("## Image Upload")
        st.markdown(
            "<div style='color:rgba(255,255,255,0.58); margin-top:-10px;'>Upload a vehicle image for analysis</div>",
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Select vehicle image for analysis",
            type=["png", "jpg", "jpeg"],
            help="Upload a clear, well-lit image of the vehicle showing potential damage areas",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_container_width=True)

            col1_info, col2_info = st.columns(2)
            with col1_info:
                st.metric("Width", f"{image.size[0]}px")
                st.metric("Height", f"{image.size[1]}px")

            with col2_info:
                file_size = len(uploaded_file.getvalue()) / 1024
                st.metric("File Size", f"{file_size:.1f} KB")
                st.metric("Format", image.format)

            if st.button("Analyze Damage", type="primary", use_container_width=True):
                with st.spinner("Processing image and detecting damage..."):
                    import time

                    time.sleep(2)

                    processed_image, detections = demo_damage_detection(image)

                    st.session_state.processed_image = processed_image
                    st.session_state.detections = detections

                    # ✅ IMPORTANT: store ORIGINAL RGB image as numpy array
                    st.session_state.original_image = np.array(image)  # RGB
                    st.session_state.image_info = image.size

                st.success(f"Analysis complete. Found {len(detections)} damage areas.")

    with col2:
        st.markdown("## Analysis Results")
        st.markdown(
            "<div style='color:rgba(255,255,255,0.58); margin-top:-10px;'>Detections, decision, summary and charts</div>",
            unsafe_allow_html=True,
        )

        if "detections" in st.session_state:
            st.image(
                st.session_state.processed_image,
                caption="Detected Damage Areas (Annotated)",
                use_container_width=True,
            )

            detections = st.session_state.detections

            # --- Agentic decision spike ---
            agent = DecisionAgent(policies_dir="policies")
            primary = pick_primary_detection(detections)

            st.markdown("### Agent Decision")

            if primary is not None:
                signal = detection_to_damage_signal(primary)
                decision = agent.decide(signal)

                # ✅ 2.2 Decision trace
                trace = build_decision_trace(primary_detection=primary, signal=signal, decision=decision)
                with st.expander("🧾 Decision Trace (chain of evidence)", expanded=True):
                    for i, step in enumerate(trace["steps"], 1):
                        st.markdown(f"**{i}. {step['title']}**")
                        for d in step["details"]:
                            st.write(f"- {d}")

                # Product-ready explanation (customer mode)
                sop_text = getattr(decision, "evidence", None)  # SOP section text
                expl = build_customer_explanation(
                    decision_action=decision.action,
                    decision_reason=decision.reason,
                    policy_refs=list(decision.policy_refs or []),
                    next_steps=list(decision.next_steps or []),
                    sop_text=sop_text,
                    signal=signal,
                )

                # Badge
                badge_emoji = {
                    "AUTO_APPROVE": "✅",
                    "HUMAN_REVIEW": "⚠️",
                    "ESCALATE": "🔺",
                }.get(decision.action, "ℹ️")

                st.markdown(f"#### {badge_emoji} {decision.action}")
                st.caption(expl["summary"])

                # WHY (customer-friendly)
                if expl["why_bullets"]:
                    st.write("**Why this decision:**")
                    for b in expl["why_bullets"]:
                        st.write(f"- {b}")

                # NEXT STEPS (customer-friendly)
                if expl["next_steps"]:
                    st.write("**Next steps:**")
                    for s in expl["next_steps"]:
                        st.write(f"- {s}")

                # Actions UI (your spike CTA layer)
                render_decision_actions(decision, primary_detection=primary)

                # ✅ 3) Human override loop (governance)
                st.markdown("---")
                st.subheader("Human Override (governance)")

                override_on = st.checkbox("Override agent decision", value=False)

                if override_on:
                    col_o1, col_o2 = st.columns([1, 1])

                    with col_o1:
                        override_action = st.selectbox(
                            "Override decision to",
                            ["AUTO_APPROVE", "HUMAN_REVIEW", "ESCALATE"],
                            index=["AUTO_APPROVE", "HUMAN_REVIEW", "ESCALATE"].index(decision.action)
                            if decision.action in ["AUTO_APPROVE", "HUMAN_REVIEW", "ESCALATE"]
                            else 1,
                        )
                        override_reason = st.selectbox(
                            "Reason",
                            [
                                "False severity",
                                "Poor image quality",
                                "Hidden damage suspected",
                                "Vehicle type / context mismatch",
                                "Other",
                            ],
                        )

                    with col_o2:
                        override_comment = st.text_area(
                            "Comment (optional)", height=90, placeholder="Short explanation..."
                        )

                    if st.button("✅ Submit override", use_container_width=True):
                        event = {
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "original_action": decision.action,
                            "original_reason": decision.reason,
                            "override_action": override_action,
                            "override_reason": override_reason,
                            "override_comment": override_comment,
                            "policy_refs": list(decision.policy_refs or []),
                            "primary": primary,
                        }
                        st.session_state.setdefault("override_events", [])
                        st.session_state["override_events"].append(event)
                        st.success("Override recorded (demo).")

                    if st.session_state.get("override_events"):
                        with st.expander("Override log (demo)"):
                            st.json(st.session_state["override_events"][-5:])

                # ==========================
                # Product "WOW" Layer (Demo)
                # ==========================
                st.markdown("---")
                st.markdown("## 🧠 Damage Story & Repair Strategy")

                story = build_damage_story(primary)
                strategies = build_repair_strategies(primary)

                col_s1, col_s2 = st.columns([1.15, 0.85])

                with col_s1:
                    st.subheader("Damage Story (What happens next)")
                    st.caption(
                        f"Severity: **{story['severity']}** · Type: **{story['damage_type']}** · "
                        f"Confidence: **{story['confidence']:.2f}**"
                    )

                    st.write("**Likely consequences (weeks → months):**")
                    for c in story["consequences"]:
                        st.write(f"- {c}")

                    st.info(f"**Safety note:** {story['safety_note']}")
                    st.write(f"**Resale impact:** {story['resale_impact'].upper()}")

                with col_s2:
                    st.subheader("Repair Strategy Simulator")
                    st.caption("Heuristic estimates (demo). No external pricing APIs used.")

                    for s in strategies:
                        with st.expander(f"{s.name} · {s.risk_level} risk"):
                            st.write(s.summary)
                            st.write(
                                f"**ETA:** {s.eta_days[0]}–{s.eta_days[1]} days"
                                if s.eta_days[1] < 999
                                else f"**ETA:** {s.eta_days[0]}+ days"
                            )
                            st.write(
                                f"**Estimated cost:** ${s.cost_usd[0]}–${s.cost_usd[1]}"
                                if s.cost_usd[1] > 0
                                else "**Estimated cost:** $0 now (risk of higher cost later)"
                            )
                            st.write("**Steps:**")
                            for step in s.steps:
                                st.write(f"- {step}")

                # ==========================
                # Before / After Vision (Preview)
                # ==========================
                st.markdown("---")
                st.markdown("## ✨ Before / After Vision (Preview)")

                preview_intensity = st.slider("Preview intensity", 0.0, 1.0, 0.65, 0.05)

                before_rgb = st.session_state.original_image  # ORIGINAL image RGB
                preview = make_repaired_preview(before_rgb, primary, intensity=preview_intensity)

                after_rgb = preview["after"]
                diff = preview["diff"]
                mask = preview["mask"]

                cA, cB = st.columns(2)
                with cA:
                    st.image(before_rgb, caption="Before (original)", use_container_width=True)
                with cB:
                    st.image(after_rgb, caption="After (CV inpaint preview)", use_container_width=True)

                # zoom to bbox
                bbox = primary.get("bbox") if primary else None
                if bbox:
                    x1, y1, x2, y2 = bbox
                    pad = 18
                    h, w = before_rgb.shape[:2]
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(w - 1, x2 + pad)
                    y2 = min(h - 1, y2 + pad)

                    z1, z2 = st.columns(2)
                    with z1:
                        st.image(before_rgb[y1:y2, x1:x2], caption="Zoom: Before", use_container_width=True)
                    with z2:
                        st.image(after_rgb[y1:y2, x1:x2], caption="Zoom: After", use_container_width=True)

                st.markdown("#### Difference (what changed in preview)")
                if diff is not None:
                    st.image(diff, caption="Diff map (higher = more changed)", use_container_width=True)
                else:
                    st.info("Diff map is unavailable for this preview method.")

                if mask is not None:
                    with st.expander("Preview mask (where inpainting was applied)"):
                        st.image(mask, caption="Mask", use_container_width=True)

                st.caption(
                    "Note: this is a UX simulation (CV inpainting), not real body repair. "
                    "It helps visualize expected improvement."
                )

                # Optional: keep your retriever KB insights
                q = f"{signal.get('damage_type', '')} {signal.get('severity', '')} repair guidance checklist risks"
                chunks = agent.retriever.retrieve(q, top_k=3)
                insights = format_kb_insights(chunks, max_items=4)

                if insights:
                    with st.expander("KB insight (retrieved guidance)"):
                        for it in insights:
                            st.write(f"- {it}")

                # Developer debug view
                if developer_mode:
                    with st.expander("DEBUG — policy refs"):
                        st.write(expl.get("policy_refs", []))

                    with st.expander("DEBUG — SOP section (raw)"):
                        st.code(sop_text or "")

                    with st.expander("DEBUG — Retrieved KB (raw)"):
                        if not chunks:
                            st.write("(no chunks)")
                        for ch in chunks:
                            st.write(f"- source: {ch.source} | score: {ch.score:.2f}")
                            st.code(ch.text)

            else:
                st.info("No detections available to drive an agent decision.")
            # --- end spike ---

            st.markdown("### Assessment Summary")

            col1_metrics, col2_metrics, col3_metrics, col4_metrics = st.columns(4)

            with col1_metrics:
                st.metric("Total Damages", len(detections))

            with col2_metrics:
                avg_confidence = np.mean([d["confidence"] for d in detections])
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")

            with col3_metrics:
                total_area = sum([d["area_percentage"] for d in detections])
                st.metric("Affected Area", f"{total_area:.1f}%")

            with col4_metrics:
                total_cost = sum([d.get("estimated_cost", 0) for d in detections])
                st.metric("Estimated Repair Cost", f"${total_cost}")

            st.markdown("### Damage Details")

            for detection in detections:
                severity = detection["severity"]
                with st.expander(f"{detection['type']} — {severity}"):
                    col1_detail, col2_detail = st.columns(2)

                    with col1_detail:
                        st.write(f"**Damage Type:** {detection['type']}")
                        st.write(f"**Severity Level:** {detection['severity']}")
                        st.write(f"**Confidence Score:** {detection['confidence']:.1%}")
                        st.write(f"**Area Affected:** {detection['area_percentage']:.1f}%")

                    with col2_detail:
                        bbox = detection["bbox"]
                        st.write(f"**Bounding Box:** ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})")
                        st.write(f"**Estimated Cost:** ${detection.get('estimated_cost', 'N/A')}")

                        if severity == "Severe":
                            st.error("Immediate repair recommended")
                        elif severity == "Moderate":
                            st.warning("Repair recommended within 30 days")
                        else:
                            st.info("Cosmetic repair — no urgency")

            st.markdown("### Damage Analysis Charts")

            col1_viz, col2_viz = st.columns(2)

            with col1_viz:
                damage_dist_chart = create_damage_distribution_chart(detections)
                st.plotly_chart(damage_dist_chart, use_container_width=True)

            with col2_viz:
                severity_chart = create_severity_chart(detections)
                st.plotly_chart(severity_chart, use_container_width=True)

            if generate_report:
                st.markdown("### Assessment Report")

                report = generate_assessment_report(detections, st.session_state.image_info)

                st.write("**Report Generated:**", report["timestamp"].strftime("%Y-%m-%d %H:%M:%S"))
                st.write("**Overall Assessment:**", f"{report['highest_severity']} damage level detected")
                st.write(
                    "**Recommended Action:**",
                    "Immediate attention required"
                    if report["highest_severity"] == "Severe"
                    else "Schedule repair within reasonable timeframe",
                )

                damage_df = pd.DataFrame(
                    [
                        {
                            "Type": d["type"],
                            "Severity": d["severity"],
                            "Confidence": f"{d['confidence']:.1%}",
                            "Area %": f"{d['area_percentage']:.1f}%",
                            "Estimated Cost": f"${d.get('estimated_cost', 0)}",
                        }
                        for d in detections
                    ]
                )

                st.dataframe(damage_df, use_container_width=True)

                st.download_button(
                    label="Download Assessment Report (JSON)",
                    data=str(report),
                    file_name=f"damage_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                )

        else:
            st.info("Upload an image and click Analyze Damage to see results here.")

            st.markdown("### Sample Analysis")
            st.write("The system can detect and analyze:")
            st.write("- Scratches: surface-level paint damage")
            st.write("- Dents: body deformation damage")
            st.write("- Paint Damage: coating and color issues")
            st.write("- Broken Parts: structural component damage")


if __name__ == "__main__":
    main()
