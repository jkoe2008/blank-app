import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import os
import tempfile
from io import BytesIO
from datetime import datetime
import mediapipe as mp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter
from dataclasses import dataclass, field, asdict
from typing import Optional

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
except Exception:
    LogisticRegression = None
    StandardScaler = None
    make_pipeline = None
    confusion_matrix = None
    roc_auc_score = None
    roc_curve = None

st.markdown("""
<style>
/* ─── Base ─────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: Inter, "IBM Plex Sans", "Segoe UI", system-ui, sans-serif !important;
}

.stApp {
    background: #070b16 !important;
    color: #e2e8f0 !important;
}

.block-container {
    max-width: 1180px !important;
    padding-top: 1.2rem !important;
    padding-bottom: 3.5rem !important;
}

h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    color: #f1f5f9 !important;
    font-weight: 700 !important;
}

p, li,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: #94a3b8 !important;
}

/* ─── Sidebar ───────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #0c1120 !important;
    border-right: 1px solid rgba(148,163,184,0.10) !important;
}
div[data-testid="stSidebarContent"] { padding: 1.2rem 0.9rem !important; }
div[data-testid="stSidebarContent"] [data-testid="stMarkdownContainer"] p,
div[data-testid="stSidebarContent"] label { color: #94a3b8 !important; }

/* ─── Inputs ────────────────────────────────────────────────────── */
input, textarea, [data-baseweb="select"] > div {
    background: #101827 !important;
    border-color: rgba(148,163,184,0.15) !important;
    color: #f1f5f9 !important;
    border-radius: 8px !important;
}
.stSelectbox label, .stTextInput label,
.stNumberInput label, .stSlider label, .stTextArea label {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
}

/* ─── Buttons ───────────────────────────────────────────────────── */
div.stButton > button,
div[data-testid="stDownloadButton"] button {
    background: #0d6b63 !important;
    color: #ffffff !important;
    border: 1px solid rgba(20,184,166,0.30) !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    box-shadow: none !important;
}
div.stButton > button:hover,
div[data-testid="stDownloadButton"] button:hover {
    background: #0a5750 !important;
}

/* ─── Tabs ──────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #101827 !important;
    border: 1px solid rgba(148,163,184,0.10) !important;
    border-radius: 10px !important;
    gap: 0.25rem !important;
    padding: 0.28rem !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: #64748b !important;
    font-weight: 600 !important;
    padding: 0.45rem 0.8rem !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(20,184,166,0.14) !important;
    color: #f1f5f9 !important;
    border: 1px solid rgba(20,184,166,0.22) !important;
}

/* ─── Metrics ───────────────────────────────────────────────────── */
.stMetric {
    background: #101827 !important;
    border: 1px solid rgba(148,163,184,0.10) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] {
    color: #2dd4bf !important;
    font-family: "IBM Plex Mono", monospace !important;
}

/* ─── File uploader ─────────────────────────────────────────────── */
div[data-testid="stFileUploader"] {
    background: #101827 !important;
    border: 1px dashed rgba(45,212,191,0.25) !important;
    border-radius: 10px !important;
}

/* ─── Dataframes / expanders ────────────────────────────────────── */
.stDataFrame, div[data-testid="stTable"] {
    border-radius: 10px !important;
    overflow: hidden !important;
}
.stExpander {
    border: 1px solid rgba(148,163,184,0.10) !important;
    border-radius: 10px !important;
    background: #101827 !important;
}
hr { border-color: rgba(148,163,184,0.10) !important; }

/* ─── App shell ─────────────────────────────────────────────────── */
.app-shell {
    background: linear-gradient(135deg, #0f1e38 0%, #101827 100%);
    border: 1px solid rgba(148,163,184,0.10);
    border-top: 2px solid #14b8a6;
    border-radius: 12px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
}
.app-kicker {
    color: #14b8a6;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 800;
    margin-bottom: 0.3rem;
}
.app-title {
    color: #f1f5f9;
    font-size: 1.95rem;
    font-weight: 800;
    line-height: 1.1;
    margin: 0 0 0.45rem 0;
}
.app-subtitle {
    color: #64748b;
    font-size: 0.92rem;
    max-width: 740px;
    line-height: 1.55;
    margin-bottom: 1.1rem;
}

/* ─── Workflow ──────────────────────────────────────────────────── */
.workflow {
    display: grid;
    grid-template-columns: repeat(4, minmax(0,1fr));
    gap: 0.55rem;
}
.workflow-step {
    background: rgba(16,24,39,0.80);
    border: 1px solid rgba(148,163,184,0.10);
    border-radius: 8px;
    padding: 0.7rem 0.8rem;
}
.workflow-num {
    color: #14b8a6;
    font-size: 0.68rem;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.18rem;
}
.workflow-label { color: #f1f5f9; font-size: 0.88rem; font-weight: 700; }
.workflow-note  { color: #475569; font-size: 0.76rem; margin-top: 0.12rem; line-height: 1.3; }

/* ─── Snapshot strip (top clinical summary) ─────────────────────── */
.snapshot-strip {
    display: grid;
    grid-template-columns: repeat(4, minmax(0,1fr));
    gap: 0.55rem;
    margin: 0.85rem 0 0.6rem 0;
}
.snapshot-card {
    background: #101827;
    border: 1px solid rgba(148,163,184,0.10);
    border-radius: 10px;
    padding: 0.85rem 0.95rem;
}
.snapshot-label {
    color: #475569;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-weight: 800;
    margin-bottom: 0.22rem;
}
.snapshot-value { color: #f1f5f9; font-size: 1.05rem; font-weight: 800; line-height: 1.2; }
.snapshot-note  { color: #475569; font-size: 0.72rem; margin-top: 0.18rem; line-height: 1.3; }
.snapshot-accent { color: #2dd4bf !important; }

/* ─── Result strip ──────────────────────────────────────────────── */
.result-strip {
    display: grid;
    grid-template-columns: 1.2fr 1fr 1fr;
    gap: 0.55rem;
    margin: 0.75rem 0;
}
.result-strip-card {
    background: #101827;
    border: 1px solid rgba(148,163,184,0.10);
    border-radius: 10px;
    padding: 0.85rem 0.95rem;
}
.result-strip-label {
    color: #475569;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-weight: 800;
    margin-bottom: 0.22rem;
}
.result-strip-value { color: #f1f5f9; font-size: 0.97rem; font-weight: 700; line-height: 1.3; }
.result-strip-note  { color: #475569; font-size: 0.72rem; margin-top: 0.18rem; line-height: 1.3; }

/* ─── Summary panels ────────────────────────────────────────────── */
.summary-panel {
    background: #101827;
    border: 1px solid rgba(148,163,184,0.10);
    border-radius: 10px;
    padding: 0.95rem 1.05rem;
    min-height: 105px;
}
.summary-label {
    color: #475569;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-weight: 800;
    margin-bottom: 0.22rem;
}
.summary-value { color: #f1f5f9; font-size: 1.1rem; font-weight: 800; line-height: 1.2; }
.summary-note  { color: #475569; font-size: 0.76rem; margin-top: 0.28rem; line-height: 1.35; }

/* ─── Clinical banner ───────────────────────────────────────────── */
.clinical-banner {
    background: #101827;
    border: 1px solid rgba(148,163,184,0.10);
    border-left: 3px solid #14b8a6;
    border-radius: 10px;
    padding: 0.95rem 1.1rem;
    margin: 0.75rem 0;
}
.clinical-banner h3 {
    margin: 0 0 0.22rem 0 !important;
    color: #f1f5f9 !important;
    font-size: 0.9rem !important;
    font-weight: 800 !important;
}
.clinical-banner p {
    margin: 0 !important;
    color: #94a3b8 !important;
    line-height: 1.5 !important;
    font-size: 0.9rem !important;
}

/* ─── Clinical grid ─────────────────────────────────────────────── */
.clinical-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0,1fr));
    gap: 0.55rem;
    margin: 0.55rem 0 0.85rem 0;
}
.clinical-card {
    background: #101827;
    border: 1px solid rgba(148,163,184,0.10);
    border-radius: 10px;
    padding: 0.8rem 0.9rem;
    min-height: 85px;
}
.clinical-card-label {
    color: #475569;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-weight: 800;
    margin-bottom: 0.22rem;
}
.clinical-card-value { color: #f1f5f9; font-size: 0.92rem; font-weight: 700; line-height: 1.25; }
.clinical-card-note  { color: #475569; font-size: 0.74rem; line-height: 1.35; margin-top: 0.28rem; }

/* ─── Clinical section ──────────────────────────────────────────── */
.clinical-section {
    background: #101827;
    border: 1px solid rgba(148,163,184,0.10);
    border-radius: 10px;
    padding: 0.95rem 1.05rem;
    margin: 0.65rem 0;
}
.clinical-section-title {
    color: #f1f5f9;
    font-size: 0.92rem;
    font-weight: 800;
    margin-bottom: 0.55rem;
}

/* ─── Decision pill ─────────────────────────────────────────────── */
.decision-pill {
    display: inline-block;
    border-radius: 999px;
    padding: 0.25rem 0.6rem;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.55rem;
}
.decision-green  { background: rgba(34,197,94,.12);  color: #86efac; border: 1px solid rgba(134,239,172,.18); }
.decision-yellow { background: rgba(245,158,11,.12); color: #fcd34d; border: 1px solid rgba(252,211,77,.18); }
.decision-red    { background: rgba(239,68,68,.12);  color: #fca5a5; border: 1px solid rgba(252,165,165,.18); }

/* ─── Pro flag cards ────────────────────────────────────────────── */
.pro-flag-card {
    display: flex;
    gap: 0.7rem;
    align-items: flex-start;
    background: #101827;
    border: 1px solid rgba(148,163,184,0.10);
    border-radius: 10px;
    padding: 0.75rem 0.85rem;
    margin: 0.4rem 0;
}
.pro-flag-icon {
    width: 28px; height: 28px;
    border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 900; font-size: 0.82rem;
    flex: 0 0 auto;
}
.pro-flag-content { flex: 1; min-width: 0; }
.pro-flag-title   { color: #f1f5f9; font-weight: 700; font-size: 0.9rem; margin-bottom: 0.12rem; line-height: 1.3; }
.pro-flag-body    { color: #64748b; font-size: 0.82rem; line-height: 1.4; }
.pro-flag-meta    { color: #334155; font-size: 0.67rem; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 800; margin-top: 0.3rem; }

.pro-flag-critical { border-left: 3px solid #ef4444; }
.pro-flag-critical .pro-flag-icon { background: rgba(239,68,68,.13); color: #fca5a5; }
.pro-flag-warning  { border-left: 3px solid #f59e0b; }
.pro-flag-warning  .pro-flag-icon { background: rgba(245,158,11,.13); color: #fcd34d; }
.pro-flag-info     { border-left: 3px solid #38bdf8; }
.pro-flag-info     .pro-flag-icon { background: rgba(56,189,248,.13); color: #7dd3fc; }
.pro-flag-success  { border-left: 3px solid #22c55e; }
.pro-flag-success  .pro-flag-icon { background: rgba(34,197,94,.13); color: #86efac; }

/* ─── Risk cards ────────────────────────────────────────────────── */
.risk-card { padding: 1.4rem; border-radius: 10px; margin: 0.45rem 0; border: 1px solid rgba(148,163,184,0.10); }
.risk-card h3 { margin: 0; font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.07em; opacity: 0.8; }
.risk-card h1 { margin: 0.35rem 0; font-size: 2.6rem; font-family: "IBM Plex Mono", monospace; }
.risk-card p  { margin: 0; font-weight: 800; letter-spacing: 0.07em; font-size: 0.82rem; }

.risk-low        { background: rgba(34,197,94,.09);  color: #86efac; }
.risk-moderate   { background: rgba(245,158,11,.09); color: #fcd34d; }
.risk-high       { background: rgba(249,115,22,.09); color: #fdba74; }
.risk-very-high  { background: rgba(239,68,68,.09);  color: #fca5a5; }

.badge-low       { background: rgba(34,197,94,.12);  color:#86efac; border:1px solid rgba(134,239,172,.18); }
.badge-moderate  { background: rgba(245,158,11,.12); color:#fcd34d; border:1px solid rgba(252,211,77,.18); }
.badge-high      { background: rgba(249,115,22,.12); color:#fdba74; border:1px solid rgba(253,186,116,.18); }
.badge-very-high { background: rgba(239,68,68,.12);  color:#fca5a5; border:1px solid rgba(252,165,165,.18); }

/* ─── Progression cards ─────────────────────────────────────────── */
.rx-card {
    background: #101827;
    border: 1px solid rgba(148,163,184,0.10);
    border-radius: 10px;
    padding: 0.95rem;
    margin: 0.65rem 0;
}
.rx-header { display: flex; justify-content: space-between; gap: 1rem; align-items: flex-start; margin-bottom: 0.7rem; }
.rx-title   { color: #f1f5f9; font-size: 0.97rem; font-weight: 800; margin-bottom: 0.18rem; }
.rx-trigger { color: #475569; font-size: 0.8rem; line-height: 1.35; }
.rx-chip {
    white-space: nowrap;
    background: rgba(20,184,166,.12);
    color: #2dd4bf;
    border: 1px solid rgba(45,212,191,.18);
    border-radius: 999px;
    padding: 0.22rem 0.55rem;
    font-size: 0.67rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.rx-level-grid { display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: 0.55rem; }
.rx-level {
    background: #0a1020;
    border: 1px solid rgba(148,163,184,0.08);
    border-radius: 8px;
    padding: 0.7rem;
}
.rx-level-name { color: #2dd4bf; font-size: 0.67rem; font-weight: 900; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 0.28rem; }
.rx-exercise   { color: #f1f5f9; font-weight: 700; line-height: 1.25; margin-bottom: 0.3rem; font-size: 0.88rem; }
.rx-dose       { color: #64748b; font-size: 0.8rem; margin-bottom: 0.3rem; }
.rx-advance    { color: #334155; font-size: 0.76rem; line-height: 1.35; }

/* ─── Rec panel ─────────────────────────────────────────────────── */
.rec-panel {
    background: #101827;
    border: 1px solid rgba(148,163,184,0.10);
    border-radius: 10px;
    padding: 0.85rem;
    margin-bottom: 0.8rem;
}
.rec-title { color: #f1f5f9; font-weight: 800; margin-bottom: 0.45rem; font-size: 0.92rem; }
.rec-item {
    color: #cbd5e1;
    background: #0a1020;
    border-radius: 8px;
    padding: 0.55rem 0.7rem;
    margin: 0.3rem 0;
    border-left: 3px solid #14b8a6;
    font-size: 0.87rem;
    line-height: 1.4;
}

/* ─── Download ──────────────────────────────────────────────────── */
.download-panel {
    background: #101827;
    border: 1px solid rgba(148,163,184,0.10);
    border-radius: 10px;
    padding: 0.85rem;
    margin-bottom: 0.65rem;
}
.download-title { color: #f1f5f9; font-weight: 800; margin-bottom: 0.18rem; font-size: 0.92rem; }
.download-copy  { color: #475569; font-size: 0.8rem; line-height: 1.35; }

/* ─── Empty state ───────────────────────────────────────────────── */
.empty-state {
    border: 1px dashed rgba(45,212,191,0.2);
    background: #101827;
    border-radius: 12px;
    padding: 2rem 1.5rem;
    margin: 0.85rem 0;
    text-align: center;
}
.empty-title { color: #f1f5f9; font-size: 1.1rem; font-weight: 800; margin-bottom: 0.35rem; }
.empty-copy  { color: #475569; line-height: 1.55; margin: 0; font-size: 0.88rem; }

/* ─── Section lead ──────────────────────────────────────────────── */
.section-lead {
    color: #475569;
    margin-top: -0.2rem;
    margin-bottom: 0.8rem;
    line-height: 1.5;
    font-size: 0.88rem;
}

/* ─── Flow shell ────────────────────────────────────────────────── */
.flow-shell {
    background: #101827;
    border: 1px solid rgba(148,163,184,0.10);
    border-radius: 10px;
    padding: 0.8rem 0.95rem;
    margin: 0.65rem 0;
}
.flow-title { color: #f1f5f9; font-size: 0.9rem; font-weight: 800; margin-bottom: 0.12rem; }
.flow-note  { color: #475569; font-size: 0.82rem; line-height: 1.4; }

/* ─── Side table ────────────────────────────────────────────────── */
.side-table { width: 100%; border-collapse: collapse; font-size: 0.86rem; border-radius: 10px; overflow: hidden; }
.side-table th {
    text-align: left; color: #94a3b8;
    background: #0a1020;
    padding: 0.6rem 0.7rem;
    font-size: 0.67rem; letter-spacing: 0.07em; text-transform: uppercase; font-weight: 800;
}
.side-table td {
    padding: 0.6rem 0.7rem; color: #cbd5e1;
    border-top: 1px solid rgba(148,163,184,0.07);
    background: #101827;
}

.clinical-list { margin: 0; padding-left: 1rem; color: #94a3b8; line-height: 1.65; font-size: 0.88rem; }

/* ─── Responsive ────────────────────────────────────────────────── */
@media (max-width: 900px) {
    .workflow, .clinical-grid, .result-strip, .snapshot-strip, .rx-level-grid {
        grid-template-columns: 1fr !important;
    }
    .app-title { font-size: 1.55rem !important; }
}
</style>
""", unsafe_allow_html=True)

mp_pose = mp.solutions.pose
LM = mp_pose.PoseLandmark

POSE_CFG = dict(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

THRESHOLDS = {
    "min_safe_knee_flexion_IC": 30.0,
    "min_safe_knee_flexion_peak": 50.0,   # lowered from 60 - single-camera MediaPipe underestimates
    "max_safe_valgus_deg": 7.0,
    "max_safe_asymmetry_pct": 15.0,
    "max_safe_trunk_lateral_deg": 10.0,
    "max_safe_pelvis_drop_deg": 8.0,
}

VISIBILITY_THRESHOLD = 0.6
VIEW_METRIC_POLICY = {
    "frontal": {
        "reportable": [
            "peak_left_valgus",
            "peak_right_valgus",
            "peak_pelvis_drop",
            "max_lateral_trunk_lean",
        ],
        "not_reportable": [
            "left_knee_flexion_at_IC",
            "right_knee_flexion_at_IC",
            "left_knee_flexion_peak",
            "right_knee_flexion_peak",
            "left_hip_flexion_at_IC",
            "right_hip_flexion_at_IC",
            "max_anterior_trunk_lean",
            "landing_stiffness_index",
            "knee_flexion_asymmetry_pct",
        ],
    },
    "side": {
        "reportable": [
            "left_knee_flexion_at_IC",
            "right_knee_flexion_at_IC",
            "left_knee_flexion_peak",
            "right_knee_flexion_peak",
            "left_hip_flexion_at_IC",
            "right_hip_flexion_at_IC",
            "max_anterior_trunk_lean",
            "landing_stiffness_index",
            "knee_flexion_asymmetry_pct",
        ],
        "not_reportable": [
            "peak_left_valgus",
            "peak_right_valgus",
            "peak_pelvis_drop",
            "max_lateral_trunk_lean",
        ],
    },
}
REQUIRED_LANDMARKS = [
    LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE,
    LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE,
    LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
]

@dataclass
class FrameData:
    frame: int
    timestamp_s: float
    left_knee_flexion: Optional[float] = None
    right_knee_flexion: Optional[float] = None
    left_hip_flexion: Optional[float] = None
    right_hip_flexion: Optional[float] = None
    left_ankle_df: Optional[float] = None
    right_ankle_df: Optional[float] = None
    left_knee_valgus_2d: Optional[float] = None
    right_knee_valgus_2d: Optional[float] = None
    left_knee_rotation: Optional[float] = None
    right_knee_rotation: Optional[float] = None
    pelvis_drop: Optional[float] = None
    lateral_trunk_lean: Optional[float] = None
    anterior_trunk_lean: Optional[float] = None
    body_scale: Optional[float] = None
    shoulder_width: Optional[float] = None
    pelvis_width: Optional[float] = None
    com_y: Optional[float] = None
    left_knee_x_norm: Optional[float] = None
    right_knee_x_norm: Optional[float] = None
    com_y_norm: Optional[float] = None
    pose_detected: bool = False
    mean_landmark_visibility: Optional[float] = None
    left_ankle_y: Optional[float] = None
    right_ankle_y: Optional[float] = None

@dataclass
class RiskReport:
    ic_frame: Optional[int] = None
    ic_time_s: Optional[float] = None
    left_knee_flexion_at_IC: Optional[float] = None
    right_knee_flexion_at_IC: Optional[float] = None
    left_knee_flexion_peak: Optional[float] = None
    right_knee_flexion_peak: Optional[float] = None
    left_hip_flexion_at_IC: Optional[float] = None
    right_hip_flexion_at_IC: Optional[float] = None
    peak_left_valgus: Optional[float] = None
    peak_right_valgus: Optional[float] = None
    peak_pelvis_drop: Optional[float] = None
    max_lateral_trunk_lean: Optional[float] = None
    max_anterior_trunk_lean: Optional[float] = None
    knee_flexion_asymmetry_pct: Optional[float] = None
    landing_stiffness_index: Optional[float] = None
    camera_angle: str = "frontal"
    camera_confidence: float = 1.0
    pose_detection_rate: float = 0.0
    mean_visibility: float = 0.0
    ic_detection_method: str = "unknown"
    ic_vote_details: dict = field(default_factory=dict)
    phase_windows: dict = field(default_factory=dict)
    temporal_features: dict = field(default_factory=dict)
    normalization_summary: dict = field(default_factory=dict)
    failure_flags: list = field(default_factory=list)
    hybrid_score_details: dict = field(default_factory=dict)
    baseline_percentiles: dict = field(default_factory=dict)
    validation_results: dict = field(default_factory=dict)
    clinical_intake: dict = field(default_factory=dict)
    clinical_impression: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)
    acl_risk_score: float = 0.0
    general_injury_risk_score: float = 0.0
    acl_risk_level: str = "Unknown"
    general_risk_level: str = "Unknown"
    movement_profile: str = "Unclassified"
    metric_uncertainty: dict = field(default_factory=dict)
    measurement_validity: dict = field(default_factory=dict)
    recommendations: list = field(default_factory=list)
    progressions: list = field(default_factory=list)

def pt2(lm, lm_enum):
    l = lm[lm_enum.value]
    return np.array([l.x, l.y], dtype=float)

def angle_3pt_2d(a, b, c):
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1, 1))))

def valgus_2d_frontal(hip, knee, ankle):
    hip, knee, ankle = hip[:2], knee[:2], ankle[:2]
    t = (knee[1] - hip[1]) / (ankle[1] - hip[1] + 1e-9)
    t = np.clip(t, 0, 1)
    expected_x = hip[0] + t * (ankle[0] - hip[0])
    deviation_x = knee[0] - expected_x
    leg_length = np.linalg.norm(ankle - hip) + 1e-9
    angle = np.degrees(np.arctan2(abs(deviation_x), leg_length))
    return float(np.sign(deviation_x) * angle)

def pelvis_drop_2d(l_hip, r_hip):
    pelvis_vec = r_hip[:2] - l_hip[:2]
    horizontal = np.array([1.0, 0.0])
    cos_val = np.dot(pelvis_vec, horizontal) / (np.linalg.norm(pelvis_vec) + 1e-9)
    angle = np.degrees(np.arccos(np.clip(abs(cos_val), 0, 1)))
    return float((1.0 if l_hip[1] > r_hip[1] else -1.0) * angle)

def lateral_trunk_lean_2d(l_shoulder, r_shoulder, l_hip, r_hip):
    mid_shoulder = (l_shoulder[:2] + r_shoulder[:2]) / 2
    mid_hip = (l_hip[:2] + r_hip[:2]) / 2
    trunk_vec = mid_shoulder - mid_hip
    vertical = np.array([0.0, -1.0])
    cos_val = np.dot(trunk_vec, vertical) / (np.linalg.norm(trunk_vec) + 1e-9)
    angle = np.degrees(np.arccos(np.clip(cos_val, -1, 1)))
    return float((1.0 if l_shoulder[1] < r_shoulder[1] else -1.0) * angle)

def anterior_trunk_lean_2d(shoulder, hip):
    trunk_vec = shoulder[:2] - hip[:2]
    vertical = np.array([0.0, -1.0])
    cos_val = np.dot(trunk_vec, vertical) / (np.linalg.norm(trunk_vec) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1, 1))))

def landmarks_visible(lm, required, threshold):
    failed = [x.name for x in required if lm[x.value].visibility < threshold]
    return len(failed) == 0, failed

def safe_series(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)

def fill_smooth(values):
    y = np.array(values, dtype=float)
    if len(y) == 0 or np.isnan(y).all():
        return y
    idx = np.arange(len(y))
    nans = np.isnan(y)
    y = np.interp(idx, idx[~nans], y[~nans])
    window = min(11, len(y) - (1 - len(y) % 2))
    if window >= 5:
        y = savgol_filter(y, window_length=window, polyorder=3)
    return y


def smooth_metric_column(df, col):
    if col not in df.columns:
        return pd.Series(dtype=float)
    smooth_col = f"{col}_smooth"
    if smooth_col not in df.columns:
        df[smooth_col] = fill_smooth(safe_series(df, col).to_numpy(dtype=float))
    return safe_series(df, smooth_col)


def robust_window_value(df, col, start=0, n=90, percentile=90, absolute=False, positive_only=False):
    series = smooth_metric_column(df, col)
    if series.empty:
        return None

    window = pd.to_numeric(series.iloc[start:start + n], errors="coerce").dropna()
    if absolute:
        window = window.abs()
    if positive_only:
        positive = window[window > 0]
        if positive.empty:
            return 0.0
        window = positive

    if window.empty:
        return None
    return float(np.nanpercentile(window, percentile))


def detect_initial_contact_voting(df, fps):
    min_frames = max(8, int(0.25 * fps))
    if len(df) < min_frames:
        return None, {"reason": "insufficient frames"}

    from scipy.signal import find_peaks
    votes = {}

    ankle_y = pd.concat(
        [safe_series(df, "left_ankle_y"), safe_series(df, "right_ankle_y")],
        axis=1,
    ).mean(axis=1).to_numpy(dtype=float)
    ankle_y = fill_smooth(ankle_y)
    if not np.isnan(ankle_y).all():
        vel = fill_smooth(np.gradient(ankle_y))
        accel = np.gradient(vel)
        decel = -accel
        prom_floor = 0.03 * max(np.nanmax(decel) - np.nanmin(decel), 1e-9)
        peaks, _ = find_peaks(
            decel,
            prominence=prom_floor,
            distance=max(3, int(0.05 * fps)),
        )
        for p in peaks:
            pre = vel[max(0, p - int(0.15 * fps)):p]
            if len(pre) > 0 and np.nanmean(pre) > 1e-3:
                votes["ankle_plant"] = int(p)
                break

    lk = 180 - safe_series(df, "left_knee_flexion")
    rk = 180 - safe_series(df, "right_knee_flexion")
    knee_flex = fill_smooth(
        pd.concat([lk, rk], axis=1).mean(axis=1).to_numpy(dtype=float)
    )
    if not np.isnan(knee_flex).all():
        flex_rate = fill_smooth(np.gradient(knee_flex))
        max_rate = np.nanmax(flex_rate)
        if max_rate > 0:
            onset_thresh = 0.15 * max_rate
            above = flex_rate > onset_thresh
            for i in range(len(above) - 2):
                if above[i] and above[i + 1]:
                    votes["knee_flexion_onset"] = int(i)
                    break

    com_y = fill_smooth(safe_series(df, "com_y").to_numpy(dtype=float))
    if not np.isnan(com_y).all():
        vel = fill_smooth(np.gradient(com_y))
        accel = np.gradient(vel)
        decel = -accel
        prom_floor = 0.02 * max(np.nanmax(np.abs(decel)), 1e-9)
        peaks, _ = find_peaks(
            decel,
            prominence=prom_floor,
            distance=max(3, int(0.05 * fps)),
        )
        for p in peaks:
            pre = vel[max(0, p - int(0.10 * fps)):p]
            if len(pre) > 0 and np.nanmean(pre) > 1e-3:
                votes["com_deceleration"] = int(p)
                break

    if not votes:
        return None, {
            "reason": "no usable IC deceleration signals",
            "ic_confidence": 0.0,
            "vote_count": 0,
        }

    vals = list(votes.values())
    median_ic = int(round(float(np.median(vals))))
    spread = int(max(vals) - min(vals)) if len(vals) > 1 else 0
    spread_limit = max(3, int(0.08 * fps))
    vote_score = min(len(vals) / 3.0, 1.0)
    spread_score = max(0.0, 1.0 - (spread / max(spread_limit, 1)))
    ic_confidence = round(float(0.55 * vote_score + 0.45 * spread_score), 3)
    return median_ic, {
        "votes": votes,
        "vote_count": len(vals),
        "vote_spread_frames": spread,
        "vote_spread_limit_frames": spread_limit,
        "ic_confidence": ic_confidence,
        "method": "deceleration-event voting v2",
    }


def consecutive_abnormal(series, threshold, direction="above", min_frames=4):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return False
    abnormal = s > threshold if direction == "above" else s < threshold
    run = 0
    for val in abnormal:
        run = run + 1 if val else 0
        if run >= min_frames:
            return True
    return False

def get_phase_windows(ic, fps, n_frames):
    if ic is None:
        return {}
    return {
        "pre_contact": [max(0, ic - int(0.10 * fps)), ic],
        "initial_contact_0_100ms": [ic, min(n_frames, ic + max(1, int(0.10 * fps)))],
        "loading_0_200ms": [ic, min(n_frames, ic + max(1, int(0.20 * fps)))],
    }

def phase_slice(df, phase_windows, name):
    if name not in phase_windows:
        return df.iloc[0:0]
    a, b = phase_windows[name]
    return df.iloc[a:b]

def compute_temporal_features(df, ic, fps):
    if ic is None:
        return {}
    w = df.iloc[ic:min(len(df), ic + max(2, int(0.20 * fps)))].copy()
    if len(w) < 2:
        return {}

    knee_flex = pd.concat([180 - safe_series(w, "left_knee_flexion"), 180 - safe_series(w, "right_knee_flexion")], axis=1).mean(axis=1)
    valgus = pd.concat([safe_series(w, "left_knee_valgus_2d").abs(), safe_series(w, "right_knee_valgus_2d").abs()], axis=1).mean(axis=1)
    trunk = safe_series(w, "lateral_trunk_lean").abs()
    dt = max(1 / fps, 1e-6)
    out = {}

    if knee_flex.dropna().shape[0] >= 2:
        k = knee_flex.dropna()
        out["knee_flexion_loading_rate_deg_s"] = round(float((k.iloc[-1] - k.iloc[0]) / (dt * (len(k) - 1))), 2)
        out["time_to_peak_flexion_ms"] = round(float((knee_flex.idxmax() - ic) / fps * 1000), 1)

    if valgus.dropna().shape[0] >= 2:
        v = valgus.dropna()
        out["valgus_rate_deg_s"] = round(float((v.iloc[-1] - v.iloc[0]) / (dt * (len(v) - 1))), 2)

    if not valgus.dropna().empty and not trunk.dropna().empty:
        out["valgus_trunk_coupling"] = round(float(valgus.mean() * trunk.mean()), 2)

    return out

def detect_failures(report, df, fps):
    failures = []
    if df.empty:
        return ["no analyzable video frames"]

    if len(df) < max(15, int(0.5 * fps)):
        failures.append("insufficient frames for stable landing analysis")

    if report.ic_frame is not None:
        q_start = max(0, report.ic_frame - int(0.15 * fps))
        q_end = min(len(df), report.ic_frame + max(2, int(0.35 * fps)))
        quality_df = df.iloc[q_start:q_end]
    else:
        quality_df = df

    landing_detection_rate = float(quality_df["pose_detected"].mean()) if "pose_detected" in quality_df and not quality_df.empty else 0.0
    landing_visibility_series = safe_series(quality_df, "mean_landmark_visibility").dropna()
    landing_visibility = float(landing_visibility_series.mean()) if not landing_visibility_series.empty else 0.0
    usable_landing_frames = int(
        (
            (quality_df["pose_detected"] == True)
            & (safe_series(quality_df, "mean_landmark_visibility") >= VISIBILITY_THRESHOLD)
        ).sum()
    ) if "pose_detected" in quality_df and not quality_df.empty else 0

    report.ic_vote_details["landing_quality_window"] = {
        "start_frame": int(quality_df.index.min()) if not quality_df.empty else None,
        "end_frame": int(quality_df.index.max()) if not quality_df.empty else None,
        "pose_detection_rate": round(landing_detection_rate, 3),
        "mean_landmark_visibility": round(landing_visibility, 3),
        "usable_frames": usable_landing_frames,
    }

    min_usable_landing_frames = max(6, int(0.18 * fps))
    if landing_detection_rate < 0.85:
        failures.append("landing-window pose tracking unreliable")
    elif report.pose_detection_rate < 0.70:
        failures.append("overall pose tracking is low; trim video or use a cleaner capture")

    if landing_visibility < 0.75:
        failures.append("landing-window landmark visibility is too low")
    elif report.mean_visibility < 0.65:
        failures.append("overall landmark visibility is low; review capture quality")

    if usable_landing_frames < min_usable_landing_frames:
        failures.append("too few usable frames around landing for stable interpretation")

    if report.camera_angle not in ["frontal", "side"]:
        failures.append("camera angle invalid or unsupported")
    if report.ic_frame is None:
        failures.append("initial contact could not be detected reliably")
    if report.ic_vote_details.get("ic_confidence", 0.0) < 0.45:
        failures.append("initial contact confidence is low")
    if report.ic_vote_details.get("vote_spread_frames", 0) > report.ic_vote_details.get("vote_spread_limit_frames", max(3, int(0.08 * fps))):
        failures.append("initial contact signals disagree; IC timing has low confidence")
    return failures

@st.cache_data(show_spinner=False)
def analyze_video(video_path, view="frontal"):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    records = []
    frame_idx = 0

    progress_bar = st.progress(0)
    status_text = st.empty()

    with mp_pose.Pose(**POSE_CFG) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % 10 == 0:
                progress_bar.progress(min(frame_idx / total_frames, 1.0))
                status_text.text(f"Processing frame {frame_idx}/{total_frames}...")

            fd = FrameData(frame=frame_idx, timestamp_s=frame_idx / fps)
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                visible, _ = landmarks_visible(lm, REQUIRED_LANDMARKS, VISIBILITY_THRESHOLD)
                if not visible:
                    records.append(fd)
                    frame_idx += 1
                    continue

                fd.pose_detected = True
                fd.mean_landmark_visibility = float(np.mean([lm[x.value].visibility for x in REQUIRED_LANDMARKS]))

                L_HIP, L_KNEE, L_ANKLE = pt2(lm, LM.LEFT_HIP), pt2(lm, LM.LEFT_KNEE), pt2(lm, LM.LEFT_ANKLE)
                R_HIP, R_KNEE, R_ANKLE = pt2(lm, LM.RIGHT_HIP), pt2(lm, LM.RIGHT_KNEE), pt2(lm, LM.RIGHT_ANKLE)
                L_FOOT, R_FOOT = pt2(lm, LM.LEFT_FOOT_INDEX), pt2(lm, LM.RIGHT_FOOT_INDEX)
                L_SHLDR, R_SHLDR = pt2(lm, LM.LEFT_SHOULDER), pt2(lm, LM.RIGHT_SHOULDER)

                fd.left_knee_flexion = angle_3pt_2d(L_HIP, L_KNEE, L_ANKLE)
                fd.right_knee_flexion = angle_3pt_2d(R_HIP, R_KNEE, R_ANKLE)
                fd.left_hip_flexion = angle_3pt_2d(L_SHLDR, L_HIP, L_KNEE)
                fd.right_hip_flexion = angle_3pt_2d(R_SHLDR, R_HIP, R_KNEE)
                fd.left_ankle_df = angle_3pt_2d(L_KNEE, L_ANKLE, L_FOOT)
                fd.right_ankle_df = angle_3pt_2d(R_KNEE, R_ANKLE, R_FOOT)

                fd.left_knee_valgus_2d = valgus_2d_frontal(L_HIP, L_KNEE, L_ANKLE)
                fd.right_knee_valgus_2d = valgus_2d_frontal(R_HIP, R_KNEE, R_ANKLE)
                fd.left_knee_rotation = 0.0
                fd.right_knee_rotation = 0.0
                fd.pelvis_drop = pelvis_drop_2d(L_HIP, R_HIP)

                mid_shldr = (L_SHLDR + R_SHLDR) / 2
                mid_hip = (L_HIP + R_HIP) / 2
                mid_ankle = (L_ANKLE + R_ANKLE) / 2

                fd.shoulder_width = float(np.linalg.norm(R_SHLDR - L_SHLDR))
                fd.pelvis_width = float(np.linalg.norm(R_HIP - L_HIP))
                fd.body_scale = float(np.mean([
                    np.linalg.norm(L_HIP - L_ANKLE),
                    np.linalg.norm(R_HIP - R_ANKLE),
                    fd.shoulder_width,
                    fd.pelvis_width,
                ]))
                fd.com_y = float(np.mean([mid_shldr[1], mid_hip[1]]))
                fd.left_knee_x_norm = float((L_KNEE[0] - L_HIP[0]) / (fd.body_scale + 1e-9))
                fd.right_knee_x_norm = float((R_KNEE[0] - R_HIP[0]) / (fd.body_scale + 1e-9))
                fd.com_y_norm = float((fd.com_y - mid_ankle[1]) / (fd.body_scale + 1e-9))
                fd.lateral_trunk_lean = lateral_trunk_lean_2d(L_SHLDR, R_SHLDR, L_HIP, R_HIP)
                fd.anterior_trunk_lean = anterior_trunk_lean_2d(mid_shldr, mid_hip)
                fd.left_ankle_y = lm[LM.LEFT_ANKLE.value].y
                fd.right_ankle_y = lm[LM.RIGHT_ANKLE.value].y

            records.append(fd)
            frame_idx += 1

    cap.release()
    progress_bar.progress(1.0)
    status_text.text("Analysis complete!")
    pose_detection_rate = float(np.mean([1.0 if r.pose_detected else 0.0 for r in records])) if records else 0.0
    vis_vals = [r.mean_landmark_visibility for r in records if r.mean_landmark_visibility is not None]
    mean_visibility = float(np.mean(vis_vals)) if vis_vals else 0.0
    camera_confidence = float(np.clip(0.15 + 0.5 * pose_detection_rate + 0.35 * mean_visibility, 0.15, 0.90))
    return records, fps, view, camera_confidence

def confidence_label(score):
    if score >= 0.75:
        return "High"
    if score >= 0.50:
        return "Moderate"
    return "Low"

def score_level(score):
    if score < 20:
        return "LOW"
    if score < 45:
        return "MODERATE"
    if score < 70:
        return "HIGH"
    return "VERY HIGH"

def apply_view_metric_policy(report, view):
    view = (view or "frontal").lower()
    policy = VIEW_METRIC_POLICY.get(view, {})

    report.measurement_validity = {
        "view": view,
        "reportable": policy.get("reportable", []),
        "not_reportable": policy.get("not_reportable", []),
    }

    for metric in policy.get("not_reportable", []):
        if hasattr(report, metric):
            setattr(report, metric, None)

    return report


def add_uncertainty(report):

    base = max(0, min(1, report.mean_visibility)) * max(0, min(1, report.pose_detection_rate))
    frontal_conf = base * report.camera_confidence
    sagittal_conf = base * 0.75
    general_conf = base * 0.85

    def pack(value, conf, high, low):
        if value is None or pd.isna(value):
            return {"value": None, "confidence": 0.0, "label": "N/A", "plus_minus": None}
        pm = low + (1 - conf) * (high - low)
        return {"value": round(float(value), 2), "confidence": round(float(conf), 2), "label": confidence_label(conf), "plus_minus": round(float(pm), 2)}

    report.metric_uncertainty = {
        "left_knee_flexion_at_IC": pack(180 - report.left_knee_flexion_at_IC if report.left_knee_flexion_at_IC is not None else None, sagittal_conf, 10, 3),
        "right_knee_flexion_at_IC": pack(180 - report.right_knee_flexion_at_IC if report.right_knee_flexion_at_IC is not None else None, sagittal_conf, 10, 3),
        "peak_left_valgus": pack(report.peak_left_valgus, frontal_conf, 8, 3),
        "peak_right_valgus": pack(report.peak_right_valgus, frontal_conf, 8, 3),
        "pelvis_drop": pack(report.peak_pelvis_drop, frontal_conf, 7, 2.5),
        "lateral_trunk_lean": pack(report.max_lateral_trunk_lean, general_conf, 6, 2),
        "asymmetry_pct": pack(report.knee_flexion_asymmetry_pct, general_conf, 12, 4),
    }
    return report

def classify_movement_profile(report):
    vals = [180 - v for v in [report.left_knee_flexion_at_IC, report.right_knee_flexion_at_IC] if v is not None]
    mean_ic = np.nanmean(vals) if vals else None
    valgus_vals = [v for v in [report.peak_left_valgus, report.peak_right_valgus] if v is not None]
    max_valgus = np.nanmax(valgus_vals) if valgus_vals else 0
    trunk = report.max_lateral_trunk_lean or 0
    pelvis = report.peak_pelvis_drop or 0

    if mean_ic is not None and mean_ic < 25 and max_valgus > THRESHOLDS["max_safe_valgus_deg"] * 1.5:
        return "Landing type A: stiff valgus-dominant"
    if max_valgus > THRESHOLDS["max_safe_valgus_deg"] * 1.5 and pelvis > THRESHOLDS["max_safe_pelvis_drop_deg"]:
        return "Landing type B: hip-control / valgus compensation"
    if trunk > 12:
        return "Landing type C: trunk-dominant compensation"
    if mean_ic is not None and mean_ic >= 30 and max_valgus <= THRESHOLDS["max_safe_valgus_deg"] and pelvis <= THRESHOLDS["max_safe_pelvis_drop_deg"]:
        return "Landing type D: balanced landing pattern"
    return "Landing type E: mixed pattern"

def build_progressions(report):
    plans = []
    left_ic = 180 - report.left_knee_flexion_at_IC if report.left_knee_flexion_at_IC is not None else None
    right_ic = 180 - report.right_knee_flexion_at_IC if report.right_knee_flexion_at_IC is not None else None

    stiff = any(v is not None and v < THRESHOLDS["min_safe_knee_flexion_IC"] for v in [left_ic, right_ic])
    valgus = any(v is not None and v > THRESHOLDS["max_safe_valgus_deg"] for v in [report.peak_left_valgus, report.peak_right_valgus])
    pelvis = report.peak_pelvis_drop is not None and report.peak_pelvis_drop > THRESHOLDS["max_safe_pelvis_drop_deg"]
    trunk = report.max_lateral_trunk_lean is not None and report.max_lateral_trunk_lean > THRESHOLDS["max_safe_trunk_lateral_deg"]
    asym = report.knee_flexion_asymmetry_pct is not None and report.knee_flexion_asymmetry_pct > THRESHOLDS["max_safe_asymmetry_pct"]

    def add(focus, trigger, levels):
        plans.append({"focus": focus, "trigger": trigger, "levels": levels})

    if stiff:
        add("Soft Landing Progression", "Low knee flexion at initial contact", [
            {"level": "Level 1", "exercise": "Snap-down to athletic stance", "dosage": "3 sets x 6 reps, 3 days/week", "advance": "Quiet landing with knees flexed >30°."},
            {"level": "Level 2", "exercise": "Double-leg drop landing", "dosage": "4 sets x 5 reps, 3 days/week", "advance": "No stiff landing flag across 2 sessions."},
            {"level": "Level 3", "exercise": "Countermovement jump to controlled landing", "dosage": "4 sets x 4 reps, 2-3 days/week", "advance": "Soft contact under fatigue."},
        ])
    if valgus:
        add("Valgus Control Progression", "Excessive 2D knee valgus", [
            {"level": "Level 1", "exercise": "Band-resisted squat with knee tracking", "dosage": "3 sets x 8 reps, 3 days/week", "advance": "Knee tracks over 2nd toe."},
            {"level": "Level 2", "exercise": "Single-leg step-down", "dosage": "3 sets x 6 each side, 3 days/week", "advance": "No medial collapse on 80% reps."},
            {"level": "Level 3", "exercise": "Single-leg landing and lateral bound stick", "dosage": "4 sets x 4 each side, 2 days/week", "advance": "Valgus below threshold on re-test."},
        ])
    if pelvis:
        add("Hip Abductor / Pelvic Control Progression", "Pelvis drop above threshold", [
            {"level": "Level 1", "exercise": "Side plank with top-leg abduction", "dosage": "3 sets x 20-30 sec, 3 days/week", "advance": "Hold pelvis level."},
            {"level": "Level 2", "exercise": "Lateral band walk + single-leg RDL reach", "dosage": "3 sets x 8-10, 3 days/week", "advance": "Pelvis level during unilateral task."},
            {"level": "Level 3", "exercise": "Single-leg hop stick", "dosage": "4 sets x 4 each side, 2 days/week", "advance": "Pelvis drop below threshold."},
        ])
    if trunk:
        add("Lateral Trunk Control Progression", "Excessive lateral trunk lean", [
            {"level": "Level 1", "exercise": "Pallof press hold", "dosage": "3 sets x 20 sec each side, 3 days/week", "advance": "No trunk shift."},
            {"level": "Level 2", "exercise": "Split-squat anti-rotation press", "dosage": "3 sets x 8 each side, 3 days/week", "advance": "Maintain ribcage/pelvis stack."},
            {"level": "Level 3", "exercise": "Lateral shuffle to stick", "dosage": "4 sets x 5 each side, 2 days/week", "advance": "Trunk lean below threshold."},
        ])
    if asym:
        add("Asymmetry Reduction Progression", "Side-to-side knee flexion asymmetry", [
            {"level": "Level 1", "exercise": "Tempo split squat", "dosage": "3 sets x 8 each side, 3 days/week", "advance": "Pain-free symmetrical control."},
            {"level": "Level 2", "exercise": "Rear-foot elevated split squat", "dosage": "3 sets x 6 each side, 2-3 days/week", "advance": "Strength difference subjectively <10%."},
            {"level": "Level 3", "exercise": "Alternating single-leg landing", "dosage": "4 sets x 4 each side, 2 days/week", "advance": "Asymmetry below 15%."},
        ])

    if not plans:
        add("Maintenance Landing Mechanics Plan", "No major biomechanical risk flag detected", [
            {"level": "Level 1", "exercise": "Landing mechanics warm-up", "dosage": "2 sets x 6 reps, 2 days/week", "advance": "Consistent quiet landings."},
            {"level": "Level 2", "exercise": "Multidirectional hop and stick", "dosage": "3 sets x 4 each direction, 2 days/week", "advance": "Stable trunk, pelvis, and knee."},
            {"level": "Level 3", "exercise": "Reactive deceleration drill", "dosage": "4 sets x 10-15 sec, 1-2 days/week", "advance": "Maintain mechanics at sport speed."},
        ])
    return plans

def learn_threshold_suggestions(history_df):
    if "injury_label" not in history_df.columns:
        return None, "Upload a CSV with an injury_label column."
    suggestions = {}
    for col in ["knee_flexion_ic", "peak_valgus", "pelvis_drop", "lateral_trunk_lean", "asymmetry_pct"]:
        if col not in history_df.columns:
            continue
        clean = history_df[[col, "injury_label"]].dropna()
        if clean.empty or clean["injury_label"].nunique() < 2:
            continue
        injured = clean[clean["injury_label"].astype(int) == 1][col]
        uninjured = clean[clean["injury_label"].astype(int) == 0][col]
        if injured.empty or uninjured.empty:
            continue
        suggestions[col] = {
            "uninjured_median": round(float(uninjured.median()), 2),
            "injured_median": round(float(injured.median()), 2),
            "suggested_cutoff": round(float((uninjured.quantile(0.75) + injured.quantile(0.25)) / 2), 2),
        }
    return (suggestions, None) if suggestions else (None, "No usable threshold suggestions found.")

def train_hybrid_model(history_df):
    if history_df is None or LogisticRegression is None:
        return None, "sklearn unavailable or no dataset uploaded"
    features = ["knee_flexion_ic", "peak_valgus", "pelvis_drop", "lateral_trunk_lean", "asymmetry_pct"]
    usable = [c for c in features if c in history_df.columns]
    if "injury_label" not in history_df.columns or len(usable) < 2:
        return None, "need injury_label plus at least 2 numeric feature columns"
    clean = history_df[usable + ["injury_label"]].dropna()
    if clean.empty or clean["injury_label"].nunique() < 2:
        return None, "need both positive and negative examples"
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(clean[usable], clean["injury_label"].astype(int))
    return {"model": model, "features": usable}, None

def current_feature_row(report, features):
    ic_flex_values = [180 - v for v in [report.left_knee_flexion_at_IC, report.right_knee_flexion_at_IC] if v is not None]
    vals = {
        "knee_flexion_ic": float(np.mean(ic_flex_values)) if ic_flex_values else np.nan,
        "peak_valgus": np.nanmax([v for v in [report.peak_left_valgus, report.peak_right_valgus] if v is not None]) if report.peak_left_valgus is not None or report.peak_right_valgus is not None else np.nan,
        "pelvis_drop": report.peak_pelvis_drop,
        "lateral_trunk_lean": report.max_lateral_trunk_lean,
        "asymmetry_pct": report.knee_flexion_asymmetry_pct,
    }
    return pd.DataFrame([{f: vals.get(f, np.nan) for f in features}])

def apply_hybrid_score(report, model_pack):
    if not model_pack:
        return report
    row = current_feature_row(report, model_pack["features"])
    if row.isna().any(axis=None):
        report.hybrid_score_details = {"used": False, "reason": "missing current-video features"}
        return report
    ml_score = float(model_pack["model"].predict_proba(row)[0, 1]) * 100
    rule_score = report.acl_risk_score
    blended = 0.65 * rule_score + 0.35 * ml_score
    report.hybrid_score_details = {"used": True, "rule_score": round(rule_score, 1), "ml_score": round(ml_score, 1), "blended_acl_score": round(blended, 1)}
    report.acl_risk_score = round(blended, 1)
    return report

def compute_baseline_percentiles(report, baseline_df):
    if baseline_df is None:
        return {}
    current = {
        "acl_risk_score": report.acl_risk_score,
        "general_injury_risk_score": report.general_injury_risk_score,
        "peak_valgus": np.nanmax([v for v in [report.peak_left_valgus, report.peak_right_valgus] if v is not None]) if report.peak_left_valgus is not None or report.peak_right_valgus is not None else None,
        "pelvis_drop": report.peak_pelvis_drop,
        "lateral_trunk_lean": report.max_lateral_trunk_lean,
        "asymmetry_pct": report.knee_flexion_asymmetry_pct,
    }
    out = {}
    for col, val in current.items():
        if val is None or col not in baseline_df.columns:
            continue
        ref = pd.to_numeric(baseline_df[col], errors="coerce").dropna()
        if not ref.empty:
            out[col] = round(float((ref <= val).mean() * 100), 1)
    return out

def validate_dataset(history_df):
    if history_df is None or LogisticRegression is None or confusion_matrix is None:
        return None, None
    features = ["knee_flexion_ic", "peak_valgus", "pelvis_drop", "lateral_trunk_lean", "asymmetry_pct"]
    usable = [c for c in features if c in history_df.columns]
    if "injury_label" not in history_df.columns or len(usable) < 2:
        return None, None
    clean = history_df[usable + ["injury_label"]].dropna()
    if len(clean) < 10 or clean["injury_label"].nunique() < 2:
        return None, None
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(clean[usable], clean["injury_label"].astype(int))
    probs = model.predict_proba(clean[usable])[:, 1]
    preds = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(clean["injury_label"].astype(int), preds).ravel()
    auc = roc_auc_score(clean["injury_label"].astype(int), probs) if roc_auc_score else None
    fpr, tpr, _ = roc_curve(clean["injury_label"].astype(int), probs) if roc_curve else ([], [], [])
    results = {
        "n": int(len(clean)),
        "features_used": usable,
        "sensitivity": round(float(tp / (tp + fn + 1e-9)), 3),
        "specificity": round(float(tn / (tn + fp + 1e-9)), 3),
        "auc": round(float(auc), 3) if auc is not None else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "note": "In-sample validation only. Use held-out data for real validation.",
    }
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash")))
    fig.update_layout(template="plotly_dark", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=350)
    return results, fig

def score_risk(records, fps, cam_angle="frontal", cam_conf=1.0, hybrid_model=None, baseline_df=None):
    report = RiskReport(camera_angle=cam_angle, camera_confidence=cam_conf)
    df = pd.DataFrame([asdict(r) for r in records])
    T = THRESHOLDS
    view = (cam_angle or "frontal").lower()
    score_frontal = view == "frontal"
    score_sagittal = view == "side"

    pose_detected = safe_series(df, "pose_detected")
    visibility = safe_series(df, "mean_landmark_visibility")
    report.pose_detection_rate = float(pose_detected.mean()) if not pose_detected.empty else 0.0
    report.mean_visibility = float(visibility.dropna().mean()) if not visibility.dropna().empty else 0.0

    ic, vote_details = detect_initial_contact_voting(df, fps)
    report.ic_frame = ic
    report.ic_time_s = ic / fps if ic is not None else None
    report.ic_detection_method = vote_details.get("method", "deceleration-event voting v2")
    report.ic_vote_details = vote_details
    report.phase_windows = get_phase_windows(ic, fps, len(df))

    def at_ic(col):
        if ic is None or col not in df.columns:
            return None
        w = df[col].iloc[max(0, ic - 2): min(len(df), ic + 3)].dropna()
        if w.empty:
            return None
        return float(w.median())

    def peak_min(col, n=90):
        if col not in df.columns:
            return None
        start = ic if ic is not None else 0
        w = df[col].iloc[start:start + n]
        return w.dropna().min() if not w.dropna().empty else None

    measurement_quality_flags = []
    suppress_ic_knee_scoring = False

    report.left_knee_flexion_at_IC = at_ic("left_knee_flexion")
    report.right_knee_flexion_at_IC = at_ic("right_knee_flexion")

    left_ic_raw = report.left_knee_flexion_at_IC
    right_ic_raw = report.right_knee_flexion_at_IC
    left_ic_flex = 180 - left_ic_raw if left_ic_raw is not None else None
    right_ic_flex = 180 - right_ic_raw if right_ic_raw is not None else None

    if left_ic_flex is not None and right_ic_flex is not None:
        side_diff = abs(left_ic_flex - right_ic_flex)

        if left_ic_raw > 170 and side_diff > 15:
            report.left_knee_flexion_at_IC = None
            measurement_quality_flags.append(
                "ℹ️ Left knee flexion at IC not reportable due to likely tracking failure; repeat capture recommended."
            )

        if right_ic_raw > 170 and side_diff > 15:
            report.right_knee_flexion_at_IC = None
            measurement_quality_flags.append(
                "ℹ️ Right knee flexion at IC not reportable due to likely tracking failure; repeat capture recommended."
            )

        if left_ic_flex < 10 and right_ic_flex < 10:
            suppress_ic_knee_scoring = True
            measurement_quality_flags.append(
                "ℹ️ Bilateral knee flexion at IC is near-locked (<10° both sides). IC knee-flexion scoring suppressed; repeat capture recommended before interpreting contact stiffness."
            )

    report.left_knee_flexion_peak = peak_min("left_knee_flexion")
    report.right_knee_flexion_peak = peak_min("right_knee_flexion")
    report.left_hip_flexion_at_IC = at_ic("left_hip_flexion")
    report.right_hip_flexion_at_IC = at_ic("right_hip_flexion")
    frontal_start = ic if ic is not None else 0
    frontal_window = max(12, int(0.75 * fps)) if fps else 90
    report.peak_left_valgus = robust_window_value(
        df,
        "left_knee_valgus_2d",
        start=frontal_start,
        n=frontal_window,
        percentile=90,
        absolute=True,
    )
    report.peak_right_valgus = robust_window_value(
        df,
        "right_knee_valgus_2d",
        start=frontal_start,
        n=frontal_window,
        percentile=90,
        absolute=True,
    )

    if "pelvis_drop" in df.columns:
        report.peak_pelvis_drop = robust_window_value(
            df,
            "pelvis_drop",
            start=frontal_start,
            n=frontal_window,
            percentile=90,
            absolute=True,
        )
    else:
        report.peak_pelvis_drop = None

    report.max_lateral_trunk_lean = robust_window_value(
        df,
        "lateral_trunk_lean",
        start=frontal_start,
        n=frontal_window,
        percentile=90,
        absolute=True,
    )
    post_ic_df = df.iloc[frontal_start:]
    anterior_series = safe_series(post_ic_df, "anterior_trunk_lean").dropna()
    report.max_anterior_trunk_lean = anterior_series.max() if not anterior_series.empty else None

    if report.left_knee_flexion_at_IC is not None and report.right_knee_flexion_at_IC is not None:
        left_flex = 180 - report.left_knee_flexion_at_IC
        right_flex = 180 - report.right_knee_flexion_at_IC
        denom = (left_flex + right_flex) / 2
        if denom > 0:
            report.knee_flexion_asymmetry_pct = abs(left_flex - right_flex) / denom * 100

    if ic is not None:
        window = safe_series(df, "left_knee_flexion").iloc[max(0, ic - 3):ic + 8].dropna()
        if len(window) >= 2:
            report.landing_stiffness_index = abs(
                (window.iloc[-1] - window.iloc[0]) / ((len(window) - 1) / fps)
            )

    report.temporal_features = compute_temporal_features(df, ic, fps)
    report.normalization_summary = {
        "mean_body_scale": round(float(df["body_scale"].dropna().mean()), 4) if "body_scale" in df and not df["body_scale"].dropna().empty else None,
        "mean_shoulder_width": round(float(df["shoulder_width"].dropna().mean()), 4) if "shoulder_width" in df and not df["shoulder_width"].dropna().empty else None,
        "mean_pelvis_width": round(float(df["pelvis_width"].dropna().mean()), 4) if "pelvis_width" in df and not df["pelvis_width"].dropna().empty else None,
        "units": "body-relative normalized image units",
    }

    confidence_ok = report.mean_visibility >= 0.65 and report.pose_detection_rate >= 0.70
    acl_score = 0.0
    gen_score = 0.0
    flags = []
    recs = []
    flags.extend(measurement_quality_flags)

    if score_sagittal:
        for side, val, col in [
            ("Left", report.left_knee_flexion_at_IC, "left_knee_flexion"),
            ("Right", report.right_knee_flexion_at_IC, "right_knee_flexion"),
        ]:
            if suppress_ic_knee_scoring:
                continue
            if val is None:
                continue

            flexion = 180 - val
            loading = phase_slice(df, report.phase_windows, "loading_0_200ms")
            persistent = consecutive_abnormal(
                180 - safe_series(loading, col),
                T["min_safe_knee_flexion_IC"],
                "below",
                2,
            )

            if flexion < T["min_safe_knee_flexion_IC"] and persistent and confidence_ok:
                sev = (T["min_safe_knee_flexion_IC"] - flexion) / T["min_safe_knee_flexion_IC"]
                acl_score += 15 * min(sev, 1.0)
                gen_score += 10 * min(sev, 1.0)
                flags.append(f"⚠️ {side} knee stiff landing - {flexion:.1f}° flexion at contact")
                recs.append(f"Practice soft landings with >{T['min_safe_knee_flexion_IC']}° knee flexion at contact ({side.lower()} side).")
            elif flexion < T["min_safe_knee_flexion_IC"]:
                flags.append(f"ℹ️ {side} stiff landing signal suppressed due to confidence/persistence gating.")

        for side, val in [
            ("Left", report.left_knee_flexion_peak),
            ("Right", report.right_knee_flexion_peak),
        ]:
            if val is None:
                continue

            peak_flex = 180 - val
            if peak_flex < T["min_safe_knee_flexion_peak"] and confidence_ok:
                sev = (T["min_safe_knee_flexion_peak"] - peak_flex) / T["min_safe_knee_flexion_peak"]
                acl_score += 7.5 * min(sev, 1.0)
                gen_score += 5.0 * min(sev, 1.0)
                flags.append(f"⚠️ {side} knee insufficient peak flexion - {peak_flex:.1f}°")
                recs.append(f"Improve {side.lower()} knee flexion depth at landing.")

    if score_frontal:
        for side, val, col in [
            ("Left", report.peak_left_valgus, "left_knee_valgus_2d"),
            ("Right", report.peak_right_valgus, "right_knee_valgus_2d"),
        ]:
            if val is None:
                continue

            peak_start = ic if ic is not None else 0
            valgus_series = smooth_metric_column(df, col).iloc[peak_start:peak_start + frontal_window].abs()
            persistent = consecutive_abnormal(
                valgus_series,
                T["max_safe_valgus_deg"],
                "above",
                4,
            )

            if val > T["max_safe_valgus_deg"] and persistent and confidence_ok:
                sev = (val - T["max_safe_valgus_deg"]) / 12.0
                acl_score += 25 * min(sev, 1.0)
                gen_score += 20 * min(sev, 1.0)
                flags.append(f"🚨 {side} 2D frontal knee displacement / valgus proxy - {val:.1f}°")
                recs.append(f"PRIORITY: {side} valgus control. Strengthen hip abductors. Consider PEP or FIFA 11+.")
            elif val > T["max_safe_valgus_deg"]:
                flags.append(f"ℹ️ {side} valgus signal suppressed due to confidence/persistence gating.")

        if report.peak_pelvis_drop is not None and report.peak_pelvis_drop > T["max_safe_pelvis_drop_deg"] and confidence_ok:
            sev = (report.peak_pelvis_drop - T["max_safe_pelvis_drop_deg"]) / 15.0
            acl_score += 8 * min(sev, 1.0)
            gen_score += 8 * min(sev, 1.0)
            flags.append(f"⚠️ Pelvis drop - {report.peak_pelvis_drop:.1f}°")
            recs.append("Pelvis drop indicates hip abductor weakness. Glute medius strengthening required.")

        if report.max_lateral_trunk_lean is not None and report.max_lateral_trunk_lean > T["max_safe_trunk_lateral_deg"] and confidence_ok:
            sev = (report.max_lateral_trunk_lean - T["max_safe_trunk_lateral_deg"]) / 20.0
            acl_score += 15 * min(sev, 1.0)
            gen_score += 12 * min(sev, 1.0)
            flags.append(f"⚠️ Lateral trunk lean - {report.max_lateral_trunk_lean:.1f}°")
            recs.append("Improve lateral core stability.")

    if score_sagittal and report.knee_flexion_asymmetry_pct is not None and report.knee_flexion_asymmetry_pct > T["max_safe_asymmetry_pct"] and confidence_ok:
        sev = (report.knee_flexion_asymmetry_pct - T["max_safe_asymmetry_pct"]) / 30.0
        gen_score += 10 * min(sev, 1.0)
        flags.append(f"⚠️ Bilateral asymmetry - {report.knee_flexion_asymmetry_pct:.1f}%")
        recs.append("Address asymmetry with unilateral training.")

    report.acl_risk_score = min(round(acl_score, 1), 100.0)
    report.general_injury_risk_score = min(round(gen_score, 1), 100.0)
    report.hybrid_score_details = {"used": False, "reason": "no labeled dataset model supplied"}
    report = apply_hybrid_score(report, hybrid_model)
    report.acl_risk_level = score_level(report.acl_risk_score)
    report.general_risk_level = score_level(report.general_injury_risk_score)

    if not any(f.startswith(("⚠️", "🚨")) for f in flags):
        flags.insert(0, "✅ No significant high-confidence biomechanical risk flags detected.")
        recs.append("Maintain current landing mechanics.")

    report.flags = flags
    report.recommendations = recs
    report.movement_profile = classify_movement_profile(report)

    report = apply_view_metric_policy(report, cam_angle)

    report.progressions = build_progressions(report)

    report = add_uncertainty(report)
    report.failure_flags = detect_failures(report, df, fps)
    report.baseline_percentiles = compute_baseline_percentiles(report, baseline_df)

    if report.failure_flags:
        report.flags.insert(0, "ℹ️ Analysis quality warning: review failure detection before interpreting risk.")

    return report, df

def build_clinical_intake(patient_age, sport, competition_level, dominant_limb, involved_limb, prior_acl, surgery_history, pain_score, swelling, instability, rts_phase, clinician_notes):
    return {
        "patient_age": patient_age,
        "sport": sport,
        "competition_level": competition_level,
        "dominant_limb": dominant_limb,
        "involved_limb": involved_limb,
        "prior_acl_injury": prior_acl,
        "surgery_history": surgery_history,
        "pain_during_test_0_10": pain_score,
        "swelling_present": swelling,
        "instability_or_giving_way": instability,
        "return_to_sport_phase": rts_phase,
        "clinician_notes": clinician_notes,
    }

def side_specific_interpretation(report, involved_limb, dominant_limb):
    left_flex = 180 - report.left_knee_flexion_at_IC if report.left_knee_flexion_at_IC is not None else None
    right_flex = 180 - report.right_knee_flexion_at_IC if report.right_knee_flexion_at_IC is not None else None
    side_map = {"Left": {"knee_flexion_ic": left_flex, "peak_valgus": report.peak_left_valgus}, "Right": {"knee_flexion_ic": right_flex, "peak_valgus": report.peak_right_valgus}}
    rows = []
    for side, vals in side_map.items():
        role = []
        if involved_limb == side:
            role.append("involved/surgical limb")
        if dominant_limb == side:
            role.append("dominant limb")
        if not role:
            role.append("comparison limb")
        concerns = []
        if vals["knee_flexion_ic"] is not None and vals["knee_flexion_ic"] < THRESHOLDS["min_safe_knee_flexion_IC"]:
            concerns.append("stiff initial contact")
        if vals["peak_valgus"] is not None and vals["peak_valgus"] > THRESHOLDS["max_safe_valgus_deg"]:
            concerns.append("valgus control deficit")
        if not concerns:
            concerns.append("no major side-specific flag")
        rows.append({
            "Side": side,
            "Clinical Role": ", ".join(role),
            "Knee Flexion @ IC": "N/A" if vals["knee_flexion_ic"] is None else f"{vals['knee_flexion_ic']:.1f}°",
            "Peak Valgus": "N/A" if vals["peak_valgus"] is None else f"{vals['peak_valgus']:.1f}°",
            "Interpretation": ", ".join(concerns),
        })
    return rows

def determine_rts_bucket(report, clinical_intake):
    pain = clinical_intake.get("pain_during_test_0_10", 0) or 0
    if report.failure_flags:
        return "Hold: video/pose quality is not reliable enough for progression decision."
    if pain > 3 or clinical_intake.get("swelling_present") or clinical_intake.get("instability_or_giving_way"):
        return "Do not progress: symptoms or instability require clinician review."
    if report.acl_risk_score >= 70 or report.general_injury_risk_score >= 70:
        return "Not ready for plyometrics: address primary impairments first."
    if report.acl_risk_score >= 45 or report.general_injury_risk_score >= 45:
        return "Ready for controlled double-leg landing drills only."
    if report.acl_risk_score >= 20 or report.general_injury_risk_score >= 20:
        return "Ready for controlled single-leg landing progression if symptoms remain quiet."
    return "Ready for reactive/change-of-direction screening progression if clinically appropriate."

def build_clinical_impression(report, clinical_intake):
    pain = clinical_intake.get("pain_during_test_0_10", 0) or 0
    risk_flags = [f for f in report.flags if f.startswith(("⚠️", "🚨"))]
    primary = risk_flags[0] if risk_flags else "No dominant impairment identified from high-confidence video metrics."
    secondary = risk_flags[1] if len(risk_flags) > 1 else "Continue monitoring mechanics under more sport-specific conditions."
    confidence = "High" if not report.failure_flags and report.mean_visibility >= 0.75 else "Moderate" if report.mean_visibility >= 0.60 else "Low"
    next_test = "Repeat drop landing from a clean frontal angle."
    if report.acl_risk_score < 45 and pain <= 3 and not report.failure_flags:
        next_test = "Add single-leg squat or single-leg landing test."
    if report.acl_risk_score < 20 and pain <= 2 and not report.failure_flags:
        next_test = "Consider lateral bound or deceleration/change-of-direction screen."
    review = "Yes" if pain > 3 or clinical_intake.get("swelling_present") or clinical_intake.get("instability_or_giving_way") or report.failure_flags else "Optional"
    return {
        "primary_impairment": primary,
        "secondary_impairment": secondary,
        "interpretation_confidence": confidence,
        "involved_limb_context": clinical_intake.get("involved_limb", "Not specified"),
        "recommended_next_test": next_test,
        "clinician_review_advised": review,
        "return_to_sport_bucket": determine_rts_bucket(report, clinical_intake),
    }


def display_executive_summary(report):
    impression = report.clinical_impression or {}
    rts_bucket = impression.get("return_to_sport_bucket", "Complete analysis to generate return-to-sport guidance.")
    confidence = impression.get("interpretation_confidence", "N/A")
    next_test = impression.get("recommended_next_test", "N/A")

    st.markdown(f"""
    <div class="clinical-banner">
        <h3>Clinical Summary</h3>
        <p>{rts_bucket}</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="summary-panel">
            <div class="summary-label">ACL Risk</div>
            <div class="summary-value">{report.acl_risk_score:.1f}/100</div>
            <div class="summary-note">{report.acl_risk_level}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="summary-panel">
            <div class="summary-label">Movement Profile</div>
            <div class="summary-value">{report.movement_profile}</div>
            <div class="summary-note">Pattern classification from landing features.</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="summary-panel">
            <div class="summary-label">Confidence</div>
            <div class="summary-value">{confidence}</div>
            <div class="summary-note">Pose visibility {report.mean_visibility:.0%}; detection {report.pose_detection_rate:.0%}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="summary-panel">
            <div class="summary-label">Recommended Next Test</div>
            <div class="summary-value">{next_test}</div>
            <div class="summary-note">Use with clinical judgment.</div>
        </div>
        """, unsafe_allow_html=True)


def _risk_badge_class(level):
    level = (level or "").lower().replace(" ", "-")
    if level == "low":
        return "badge-low"
    if level == "moderate":
        return "badge-moderate"
    if level == "high":
        return "badge-high"
    return "badge-very-high"


def parse_flag_for_display(flag):
    clean = flag.replace("🚨", "").replace("⚠️", "").replace("ℹ️", "").replace("✅", "").strip()

    if " - " in clean:
        title, body = clean.split(" - ", 1)
    elif ":" in clean:
        title, body = clean.split(":", 1)
    else:
        title, body = clean, ""

    return title.strip(), body.strip()

def display_professional_flags(flags, limit=None):
    shown = flags[:limit] if limit else flags

    if not shown:
        st.markdown("""
        <div class="pro-flag-card pro-flag-success">
            <div class="pro-flag-icon">✓</div>
            <div class="pro-flag-content">
                <div class="pro-flag-title">No clinical flags detected</div>
                <div class="pro-flag-body">No high-confidence movement concerns were identified in this screen.</div>
                <div class="pro-flag-meta">Screening result</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    for flag in shown:
        if flag.startswith("🚨"):
            cls = "pro-flag-critical"
            icon = "!"
            meta = "High-priority finding"
        elif flag.startswith("⚠️"):
            cls = "pro-flag-warning"
            icon = "!"
            meta = "Movement concern"
        elif flag.startswith("ℹ️"):
            cls = "pro-flag-info"
            icon = "i"
            meta = "Quality / interpretation note"
        else:
            cls = "pro-flag-success"
            icon = "✓"
            meta = "Screening result"

        title, body = parse_flag_for_display(flag)
        if not body:
            body = "Review this finding in context with the video, symptoms, and clinical exam."

        st.markdown(f"""
        <div class="pro-flag-card {cls}">
            <div class="pro-flag-icon">{icon}</div>
            <div class="pro-flag-content">
                <div class="pro-flag-title">{title}</div>
                <div class="pro-flag-body">{body}</div>
                <div class="pro-flag-meta">{meta}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_summary_dashboard(report):
    st.subheader("Risk Scores")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("ACL Injury Risk", f"{report.acl_risk_score:.1f} / 100", report.acl_risk_level)
        st.progress(min(max(report.acl_risk_score / 100, 0), 1))

    with c2:
        st.metric("General Lower Extremity Risk", f"{report.general_injury_risk_score:.1f} / 100", report.general_risk_level)
        st.progress(min(max(report.general_injury_risk_score / 100, 0), 1))

    st.divider()

    st.subheader("Key Findings")

    flex_l = f"{180 - report.left_knee_flexion_at_IC:.1f}°" if report.left_knee_flexion_at_IC is not None else "N/A"
    flex_r = f"{180 - report.right_knee_flexion_at_IC:.1f}°" if report.right_knee_flexion_at_IC is not None else "N/A"
    valg_l = f"{report.peak_left_valgus:.1f}°" if report.peak_left_valgus is not None else "N/A"
    valg_r = f"{report.peak_right_valgus:.1f}°" if report.peak_right_valgus is not None else "N/A"
    pelvis = f"{report.peak_pelvis_drop:.1f}°" if report.peak_pelvis_drop is not None else "N/A"
    asym = f"{report.knee_flexion_asymmetry_pct:.1f}%" if report.knee_flexion_asymmetry_pct is not None else "N/A"
    ic_time = f"{report.ic_time_s:.3f}s" if report.ic_time_s is not None else "N/A"

    findings = pd.DataFrame([
        {"Finding": "Movement profile", "Result": report.movement_profile},
        {"Finding": "Initial contact", "Result": f"Frame {report.ic_frame if report.ic_frame is not None else 'N/A'} • {ic_time}"},
        {"Finding": "Knee flexion @ IC", "Result": f"Left {flex_l} • Right {flex_r}"},
        {"Finding": "Peak valgus", "Result": f"Left {valg_l} • Right {valg_r}"},
        {"Finding": "Pelvis / asymmetry", "Result": f"{pelvis} • {asym}"},
    ])
    st.dataframe(findings, use_container_width=True, hide_index=True)

    st.subheader("Clinical Flags")
    display_professional_flags(report.flags, limit=6)

def display_recommendation_panel(report):
    recs = report.recommendations or ["Maintain current landing mechanics and continue routine monitoring."]
    items = ""
    for i, rec in enumerate(recs, 1):
        items += f'<div class="rec-item"><b>{i}.</b> {rec}</div>'
    st.markdown(f"""
    <div class="rec-panel">
        <div class="rec-title">Clinical Priorities</div>
        {items}
    </div>
    """, unsafe_allow_html=True)

def display_progression_cards(report):
    for i, plan in enumerate(report.progressions, 1):
        st.markdown(f"""
        <div class="rx-card">
            <div class="rx-header">
                <div>
                    <div class="rx-title">{plan.get("focus", "Progression Plan")}</div>
                    <div class="rx-trigger">Triggered by: {plan.get("trigger", "Clinical screen findings")}</div>
                </div>
                <div class="rx-chip">Priority {i}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(3)
        for col, item in zip(cols, plan.get("levels", [])):
            with col:
                st.markdown(f"""
                <div class="rx-level">
                    <div class="rx-level-name">{item.get("level", "")}</div>
                    <div class="rx-exercise">{item.get("exercise", "")}</div>
                    <div class="rx-dose">{item.get("dosage", "")}</div>
                    <div class="rx-advance"><b>Advance:</b> {item.get("advance", "")}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height: 0.85rem;'></div>", unsafe_allow_html=True)

def display_risk_cards(report):
    col1, col2 = st.columns(2)
    with col1:
        risk_class = f"risk-{report.acl_risk_level.lower().replace(' ', '-')}"
        st.markdown(f"""<div class="risk-card {risk_class}"><h3>ACL Injury Risk</h3><h1>{report.acl_risk_score:.1f} / 100</h1><p>{report.acl_risk_level}</p></div>""", unsafe_allow_html=True)
    with col2:
        risk_class = f"risk-{report.general_risk_level.lower().replace(' ', '-')}"
        st.markdown(f"""<div class="risk-card {risk_class}"><h3>General Lower Extremity Risk</h3><h1>{report.general_injury_risk_score:.1f} / 100</h1><p>{report.general_risk_level}</p></div>""", unsafe_allow_html=True)

def display_uncertainty(report):
    labels = {
        "left_knee_flexion_at_IC": "Left Knee Flexion @ IC",
        "right_knee_flexion_at_IC": "Right Knee Flexion @ IC",
        "peak_left_valgus": "Left Peak Valgus",
        "peak_right_valgus": "Right Peak Valgus",
        "pelvis_drop": "Pelvis Drop",
        "lateral_trunk_lean": "Lateral Trunk Lean",
        "asymmetry_pct": "Knee Flexion Asymmetry",
    }
    rows = []
    for key, info in report.metric_uncertainty.items():
        rows.append({
            "Metric": labels.get(key, key),
            "Estimate": "N/A" if info["value"] is None else f'{info["value"]} ± {info["plus_minus"]}',
            "Confidence": info["label"],
            "Confidence Score": info["confidence"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.caption(f"Pose detection rate: {report.pose_detection_rate:.0%} | Mean landmark visibility: {report.mean_visibility:.0%} | Camera confidence: {report.camera_confidence:.0%}")

def display_progression_programs(report):
    for plan in report.progressions:
        st.subheader(plan["focus"])
        st.caption(f"Trigger: {plan['trigger']}")
        st.dataframe(pd.DataFrame(plan["levels"]), use_container_width=True)

def display_failure_detection(report):
    if report.failure_flags:
        for item in report.failure_flags:
            st.warning(item)
    else:
        st.success("No major analysis-quality failures detected.")

def display_advanced_biomechanics(report):
    st.subheader("Initial Contact Voting")
    st.json(report.ic_vote_details)

    st.subheader("Movement Phases")
    st.json(report.phase_windows)

    st.subheader("Temporal Biomechanics")
    if report.temporal_features:
        st.dataframe(pd.DataFrame([report.temporal_features]), use_container_width=True)
    else:
        st.info("Temporal features unavailable.")

    st.subheader("Cross-Video Normalization")
    st.json(report.normalization_summary)

def display_baseline_percentiles(report):
    if not report.baseline_percentiles:
        st.info("Upload a baseline CSV with matching feature columns to show percentile rankings.")
        return
    st.dataframe(pd.DataFrame([{"Metric": k, "Percentile": f"{v}th"} for k, v in report.baseline_percentiles.items()]), use_container_width=True)

def display_hybrid_score(report):
    if not report.hybrid_score_details or not report.hybrid_score_details.get("used"):
        st.info(report.hybrid_score_details.get("reason", "Upload a labeled dataset to enable hybrid ML + rules scoring."))
    else:
        st.json(report.hybrid_score_details)

def display_clinical_decision_support(report, clinical_intake):
    impression = report.clinical_impression or build_clinical_impression(report, clinical_intake)
    bucket = impression.get("return_to_sport_bucket", "Decision unavailable.")
    confidence = impression.get("interpretation_confidence", "N/A")
    review = impression.get("clinician_review_advised", "N/A")

    if bucket.startswith(("Do not progress", "Hold", "Not ready")):
        pill_class = "decision-red"
        pill_text = "Do Not Progress"
    elif bucket.startswith("Ready for controlled"):
        pill_class = "decision-yellow"
        pill_text = "Controlled Progression"
    else:
        pill_class = "decision-green"
        pill_text = "Progression Candidate"

    st.markdown(f"""
    <div class="clinical-section">
        <span class="decision-pill {pill_class}">{pill_text}</span>
        <div class="clinical-section-title">Return-to-Sport Decision</div>
        <div class="clinical-card-value">{bucket}</div>
        <div class="clinical-card-note">This decision combines symptoms, analysis quality, movement risk, and side-specific findings. It is not a diagnosis.</div>
    </div>
    """, unsafe_allow_html=True)

    age = clinical_intake.get("patient_age", "N/A")
    sport = clinical_intake.get("sport") or "Not specified"
    level = clinical_intake.get("competition_level", "N/A")
    involved = clinical_intake.get("involved_limb", "Not specified")
    dominant = clinical_intake.get("dominant_limb", "Not specified")
    pain = clinical_intake.get("pain_during_test_0_10", 0)
    symptoms = []
    if clinical_intake.get("swelling_present"):
        symptoms.append("swelling")
    if clinical_intake.get("instability_or_giving_way"):
        symptoms.append("instability/giving-way")
    if clinical_intake.get("prior_acl_injury"):
        symptoms.append("prior ACL")
    symptom_text = ", ".join(symptoms) if symptoms else "No major symptom flags entered"

    st.markdown(f"""
    <div class="clinical-grid">
        <div class="clinical-card">
            <div class="clinical-card-label">Athlete Context</div>
            <div class="clinical-card-value">{age} yrs • {sport}</div>
            <div class="clinical-card-note">{level}</div>
        </div>
        <div class="clinical-card">
            <div class="clinical-card-label">Limb Context</div>
            <div class="clinical-card-value">Involved: {involved}</div>
            <div class="clinical-card-note">Dominant: {dominant}</div>
        </div>
        <div class="clinical-card">
            <div class="clinical-card-label">Symptoms</div>
            <div class="clinical-card-value">Pain {pain}/10</div>
            <div class="clinical-card-note">{symptom_text}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="clinical-grid">
        <div class="clinical-card">
            <div class="clinical-card-label">Primary Impairment</div>
            <div class="clinical-card-value">{impression.get("primary_impairment", "N/A")}</div>
        </div>
        <div class="clinical-card">
            <div class="clinical-card-label">Interpretation Confidence</div>
            <div class="clinical-card-value">{confidence}</div>
            <div class="clinical-card-note">Pose visibility {report.mean_visibility:.0%}; detection {report.pose_detection_rate:.0%}</div>
        </div>
        <div class="clinical-card">
            <div class="clinical-card-label">Clinician Review</div>
            <div class="clinical-card-value">{review}</div>
            <div class="clinical-card-note">{impression.get("recommended_next_test", "N/A")}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    side_rows = side_specific_interpretation(
        report,
        clinical_intake.get("involved_limb", "Not specified"),
        clinical_intake.get("dominant_limb", "Not specified"),
    )

    table_rows = ""
    for row in side_rows:
        table_rows += (
            f"<tr>"
            f"<td>{row['Side']}</td>"
            f"<td>{row['Clinical Role']}</td>"
            f"<td>{row['Knee Flexion @ IC']}</td>"
            f"<td>{row['Peak Valgus']}</td>"
            f"<td>{row['Interpretation']}</td>"
            f"</tr>"
        )

    html = f"""
    <div class="clinical-section">
        <div class="clinical-section-title">Side-Specific Interpretation</div>
        <table class="side-table">
            <thead>
                <tr>
                    <th>Side</th>
                    <th>Role</th>
                    <th>Knee Flexion @ IC</th>
                    <th>Peak Valgus</th>
                    <th>Interpretation</th>
                </tr>
            </thead>
            <tbody>{table_rows}</tbody>
        </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

    notes = clinical_intake.get("clinician_notes") or "No manual notes entered."
    surgery = clinical_intake.get("surgery_history") or "No surgery history entered."
    secondary = impression.get("secondary_impairment", "N/A")
    next_test = impression.get("recommended_next_test", "N/A")

    st.markdown(f"""
    <div class="clinical-section">
        <div class="clinical-section-title">Clinical Notes & Next Step</div>
        <ul class="clinical-list">
            <li><b>Secondary consideration:</b> {secondary}</li>
            <li><b>Recommended next test:</b> {next_test}</li>
            <li><b>Surgery history:</b> {surgery}</li>
            <li><b>Clinician note:</b> {notes}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def create_charts(df, report, fps):
    df = df.copy()
    frame_series = safe_series(df, "frame")
    if frame_series.empty:
        frame_series = pd.Series(range(len(df)), index=df.index, dtype=float)
    df["time_s"] = frame_series / max(float(fps or 1), 1e-9)
    df["left_knee_flexion_deg"] = 180 - safe_series(df, "left_knee_flexion")
    df["right_knee_flexion_deg"] = 180 - safe_series(df, "right_knee_flexion")

    fig = make_subplots(rows=2, cols=2, subplot_titles=("Knee Flexion (2D)", "2D Knee Valgus", "Pelvis Drop & Rotation", "Trunk Lean"))
    fig.add_trace(go.Scatter(x=df["time_s"], y=df["left_knee_flexion_deg"], name="Left Knee", line=dict(color="#4fc3f7")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time_s"], y=df["right_knee_flexion_deg"], name="Right Knee", line=dict(color="#f48fb1")), row=1, col=1)
    fig.add_hline(y=THRESHOLDS["min_safe_knee_flexion_IC"], line_dash="dot", line_color="#ffb74d", row=1, col=1)
    fig.add_hline(y=THRESHOLDS["min_safe_knee_flexion_peak"], line_dash="dot", line_color="#ef5350", row=1, col=1)

    fig.add_trace(go.Scatter(x=df["time_s"], y=safe_series(df, "left_knee_valgus_2d"), name="Left Valgus", line=dict(color="#4fc3f7"), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=df["time_s"], y=safe_series(df, "right_knee_valgus_2d").abs(), name="Right Valgus", line=dict(color="#f48fb1"), showlegend=False), row=1, col=2)
    fig.add_hline(y=THRESHOLDS["max_safe_valgus_deg"], line_dash="dot", line_color="#ef5350", row=1, col=2)
    fig.add_hline(y=0, line_width=0.5, line_color="#aab4c4", row=1, col=2)

    fig.add_trace(go.Scatter(x=df["time_s"], y=safe_series(df, "pelvis_drop").abs(), name="Pelvis Drop", line=dict(color="#ffcc02"), showlegend=False), row=2, col=1)
    fig.add_hline(y=THRESHOLDS["max_safe_pelvis_drop_deg"], line_dash="dot", line_color="#ef5350", row=2, col=1)

    fig.add_trace(go.Scatter(x=df["time_s"], y=safe_series(df, "lateral_trunk_lean").abs(), name="Lateral Lean", line=dict(color="#ce93d8"), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=df["time_s"], y=safe_series(df, "anterior_trunk_lean"), name="Anterior Lean", line=dict(color="#80cbc4"), showlegend=False), row=2, col=2)
    fig.add_hline(y=THRESHOLDS["max_safe_trunk_lateral_deg"], line_dash="dot", line_color="#ef5350", row=2, col=2)

    if report.ic_frame is not None:
        for r in [1, 2]:
            for c in [1, 2]:
                fig.add_vline(x=report.ic_time_s, line_dash="dash", line_color="#00e5ff", row=r, col=c)

    fig.update_layout(height=700, template="plotly_dark", showlegend=True)
    return fig

def build_pdf_report(report, patient_id, clinician_name, fig=None):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    except Exception as e:
        raise RuntimeError(f"ReportLab import failed: {e}")

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("ACL / Lower Extremity Landing Mechanics Screening Report", styles["Title"]))
    story.append(Paragraph(f"Patient ID: {patient_id or 'Not provided'}", styles["Normal"]))
    story.append(Paragraph(f"Clinician / Reviewer: {clinician_name or 'Not provided'}", styles["Normal"]))
    story.append(Paragraph(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    summary = [
        ["Measure", "Result"],
        ["ACL Risk", f"{report.acl_risk_score:.1f}/100 ({report.acl_risk_level})"],
        ["General Lower Extremity Risk", f"{report.general_injury_risk_score:.1f}/100 ({report.general_risk_level})"],
        ["Movement Profile", report.movement_profile],
        ["Initial Contact Frame", str(report.ic_frame if report.ic_frame is not None else "N/A")],
        ["IC Method", report.ic_detection_method],
        ["Pose Detection Rate", f"{report.pose_detection_rate:.0%}"],
    ]
    table = Table(summary, colWidths=[220, 280])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]))
    story.append(table)
    story.append(Spacer(1, 12))

    if report.clinical_intake:
        story.append(Paragraph("Clinical Intake", styles["Heading2"]))
        rows = [["Field", "Value"]] + [[str(k), str(v)] for k, v in report.clinical_intake.items()]
        t = Table(rows, colWidths=[220, 280])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
        story.append(t)
        story.append(Spacer(1, 12))

    if report.clinical_impression:
        story.append(Paragraph("Clinical Impression", styles["Heading2"]))
        for k, v in report.clinical_impression.items():
            story.append(Paragraph(f"<b>{k}</b>: {v}", styles["Normal"]))
        story.append(Spacer(1, 12))

    story.append(Paragraph("Risk Flags", styles["Heading2"]))
    for flag in report.flags:
        story.append(Paragraph(flag, styles["Normal"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Failure Detection", styles["Heading2"]))
    if report.failure_flags:
        for flag in report.failure_flags:
            story.append(Paragraph(flag, styles["Normal"]))
    else:
        story.append(Paragraph("No major analysis-quality failures detected.", styles["Normal"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Temporal Features", styles["Heading2"]))
    for k, v in report.temporal_features.items():
        story.append(Paragraph(f"{k}: {v}", styles["Normal"]))

    if fig is not None:
        try:
            img_bytes = fig.to_image(format="png", width=900, height=650, scale=1)
            story.append(Spacer(1, 12))
            story.append(Paragraph("Graphs Snapshot", styles["Heading2"]))
            story.append(Image(BytesIO(img_bytes), width=500, height=360))
        except Exception:
            story.append(Paragraph("Graphs Snapshot unavailable. Install kaleido to embed Plotly figures in PDF.", styles["Italic"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Exercise Progression Plan", styles["Heading2"]))
    for plan in report.progressions:
        story.append(Paragraph(f"<b>{plan['focus']}</b> - Trigger: {plan['trigger']}", styles["Normal"]))
        for item in plan["levels"]:
            story.append(Paragraph(f"{item['level']}: {item['exercise']} | {item['dosage']} | Advance when: {item['advance']}", styles["Normal"]))
        story.append(Spacer(1, 8))

    story.append(Spacer(1, 24))
    story.append(Paragraph("Clinician Signature: ________________________________", styles["Normal"]))
    story.append(Paragraph("Disclaimer: This is a screening report, not a medical diagnosis. Interpret with clinical examination.", styles["Italic"]))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def display_clinical_snapshot(report):
    """Compact top strip: score · profile · confidence · next test."""
    impression = report.clinical_impression or {}
    confidence = impression.get("interpretation_confidence", "N/A")
    next_test   = impression.get("recommended_next_test", "—")
    rts         = impression.get("return_to_sport_bucket", "—")

    acl_color = (
        "#86efac" if report.acl_risk_score < 20 else
        "#fcd34d" if report.acl_risk_score < 45 else
        "#fdba74" if report.acl_risk_score < 70 else
        "#fca5a5"
    )

    st.markdown(f"""
    <div class="snapshot-strip">
        <div class="snapshot-card">
            <div class="snapshot-label">ACL Risk Score</div>
            <div class="snapshot-value" style="color:{acl_color}">
                {report.acl_risk_score:.1f} <span style="font-size:0.7rem;color:#475569">/ 100</span>
            </div>
            <div class="snapshot-note">{report.acl_risk_level}</div>
        </div>
        <div class="snapshot-card">
            <div class="snapshot-label">Movement Profile</div>
            <div class="snapshot-value" style="font-size:0.82rem">{report.movement_profile}</div>
        </div>
        <div class="snapshot-card">
            <div class="snapshot-label">Data Confidence</div>
            <div class="snapshot-value">{confidence}</div>
            <div class="snapshot-note">
                Visibility {report.mean_visibility:.0%} &nbsp;·&nbsp; Detection {report.pose_detection_rate:.0%}
            </div>
        </div>
        <div class="snapshot-card">
            <div class="snapshot-label">Recommended Next Step</div>
            <div class="snapshot-value" style="font-size:0.82rem">{next_test}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_premium_header():
    st.markdown("""
    <div class="app-shell">
        <div class="app-kicker">ACL &amp; Lower Extremity · Clinical Screen</div>
        <h1 class="app-title">Landing Mechanics Assessment</h1>
        <div class="app-subtitle">
            Upload a frontal or side-view landing video. The system scores only the metrics
            appropriate for the selected view, flags measurement quality, and generates
            a clinic-ready report.
        </div>
        <div class="workflow">
            <div class="workflow-step">
                <div class="workflow-num">01 · Intake</div>
                <div class="workflow-label">Patient context</div>
                <div class="workflow-note">Age, sport, limb, symptoms, RTS phase.</div>
            </div>
            <div class="workflow-step">
                <div class="workflow-num">02 · Capture</div>
                <div class="workflow-label">Upload video</div>
                <div class="workflow-note">Frontal for valgus. Side for flexion.</div>
            </div>
            <div class="workflow-step">
                <div class="workflow-num">03 · Review</div>
                <div class="workflow-label">Clinical output</div>
                <div class="workflow-note">Risk, quality flags, RTS guidance.</div>
            </div>
            <div class="workflow-step">
                <div class="workflow-num">04 · Export</div>
                <div class="workflow-label">Download report</div>
                <div class="workflow-note">PDF, CSV, and JSON.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_upload_guidance():
    st.markdown("""
    <div class="empty-state">
        <div class="empty-title">Ready for video review</div>
        <p class="empty-copy">
            Complete the intake, then upload a frontal or side-view landing video. The report will only
            score metrics that are appropriate for the selected view.
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_result_strip(report):
    impression = report.clinical_impression or {}
    rts = impression.get("return_to_sport_bucket", "Decision pending")
    confidence = impression.get("interpretation_confidence", "N/A")
    next_test = impression.get("recommended_next_test", "N/A")

    st.markdown(f"""
    <div class="result-strip">
        <div class="result-strip-card">
            <div class="result-strip-label">Return-to-Sport Guidance</div>
            <div class="result-strip-value">{rts}</div>
            <div class="result-strip-note">Combines symptoms, risk score, tracking quality, and side-specific findings.</div>
        </div>
        <div class="result-strip-card">
            <div class="result-strip-label">Interpretation Confidence</div>
            <div class="result-strip-value">{confidence}</div>
            <div class="result-strip-note">Visibility {report.mean_visibility:.0%} • Detection {report.pose_detection_rate:.0%}</div>
        </div>
        <div class="result-strip-card">
            <div class="result-strip-label">Recommended Next Step</div>
            <div class="result-strip-value">{next_test}</div>
            <div class="result-strip-note">Use alongside exam findings and clinical reasoning.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.caption("Version: 2D-ONLY-v5 | IC Detection: deceleration-event voting | Valgus: 90-frame persistence window | Peak flex threshold: 50°")
    display_premium_header()

    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False
    if "last_upload_key" not in st.session_state:
        st.session_state.last_upload_key = None
    if "cached_report" not in st.session_state:
        st.session_state.cached_report = None
    if "cached_df" not in st.session_state:
        st.session_state.cached_df = None
    if "cached_fps" not in st.session_state:
        st.session_state.cached_fps = None
    if "cached_fig" not in st.session_state:
        st.session_state.cached_fig = None
    if "cached_validation_fig" not in st.session_state:
        st.session_state.cached_validation_fig = None

    hybrid_model = None
    baseline_df = None
    validation_results = None
    validation_fig = None

    with st.sidebar:
        st.header("Session Setup")
        st.write("""
        Complete the clinical context first, then upload the landing video.

        Best capture:
        - Frontal camera angle
        - Full body visible
        - 30-60 fps
        - Bright, uncluttered background
        """)
        st.divider()
        patient_id = st.text_input("Patient ID", value="")
        clinician_name = st.text_input("Clinician / Reviewer", value="")

        st.divider()
        st.subheader("Clinical Intake")
        patient_age = st.number_input("Age", min_value=5, max_value=100, value=16)
        sport = st.text_input("Sport / activity", value="")
        competition_level = st.selectbox("Competition level", ["Youth", "High school", "College", "Professional", "Recreational", "Other"])
        dominant_limb = st.selectbox("Dominant limb", ["Not specified", "Left", "Right"])
        involved_limb = st.selectbox("Involved / surgical limb", ["Not specified", "Left", "Right", "Bilateral"])
        prior_acl = st.checkbox("Prior ACL injury")
        surgery_history = st.text_input("Relevant surgery history", value="")
        pain_score = st.slider("Pain during test", min_value=0, max_value=10, value=0)
        swelling = st.checkbox("Swelling present")
        instability = st.checkbox("Instability / giving-way reported")
        rts_phase = st.selectbox("Return-to-sport phase", ["Not specified", "Early rehab", "Strength phase", "Controlled plyometrics", "Single-leg plyometrics", "Reactive / change-of-direction", "Full return"])
        clinician_notes = st.text_area("Clinician notes / manual override", value="")

        st.divider()
        st.subheader("Optional Learning Dataset")
        historical_csv = st.file_uploader("Upload labeled CSV for thresholds, hybrid scoring, validation", type=["csv"], key="history_csv")
        if historical_csv is not None:
            try:
                hist_df = pd.read_csv(historical_csv)
                learned_thresholds, msg = learn_threshold_suggestions(hist_df)
                hybrid_model, hybrid_msg = train_hybrid_model(hist_df)
                validation_results, validation_fig = validate_dataset(hist_df)
                st.json(learned_thresholds) if learned_thresholds else st.info(msg)
                st.success("Hybrid ML scoring enabled.") if hybrid_model else st.info(hybrid_msg)
                st.success("Validation mode enabled.") if validation_results else st.info("Validation requires at least 10 rows, injury_label, and at least 2 usable features.")
            except Exception as e:
                st.warning(f"Could not read learning dataset: {e}")

        baseline_csv = st.file_uploader("Upload population baseline CSV", type=["csv"], key="baseline_csv")
        if baseline_csv is not None:
            try:
                baseline_df = pd.read_csv(baseline_csv)
                st.success("Baseline CSV loaded.")
            except Exception as e:
                st.warning(f"Could not read baseline CSV: {e}")

        st.caption("Screening only. Not a diagnosis. Single-camera estimates are sensitive to camera angle, clothing, occlusion, lighting, and calibration.")

    st.markdown("""
    <div class="flow-shell">
        <div class="flow-title">Step 2 — Upload Landing Video</div>
        <div class="flow-note">Upload frontal and/or side view files. The analysis and scoring remain view-aware and only evaluate valid metrics for each selected perspective.</div>
    </div>
    """, unsafe_allow_html=True)

    upload_col_1, upload_col_2 = st.columns(2)

    with upload_col_1:
        frontal_file = st.file_uploader(
            "Frontal view video",
            type=["mp4", "mov", "avi", "webm"],
            help="Use for valgus, pelvis drop, lateral trunk lean, and frontal-plane control.",
            key="frontal_video",
        )

    with upload_col_2:
        side_file = st.file_uploader(
            "Side view video",
            type=["mp4", "mov", "avi", "webm"],
            help="Use for knee flexion, hip flexion, anterior trunk lean, and landing depth.",
            key="side_video",
        )

    uploaded_file = frontal_file or side_file
    selected_view = "frontal" if frontal_file is not None else "side"

    if uploaded_file is not None:
        upload_bytes = uploaded_file.getvalue()
        upload_key = f"{uploaded_file.name}-{len(upload_bytes)}"

        should_analyze = (
            not st.session_state.analysis_done
            or st.session_state.last_upload_key != upload_key
            or st.session_state.cached_report is None
            or st.session_state.cached_df is None
        )

        if should_analyze:
            suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tfile.write(upload_bytes)
            tfile.close()
            video_path = tfile.name

            cap_test = cv2.VideoCapture(video_path)
            ret, first_frame = cap_test.read()
            cap_test.release()
            if ret:
                st.image(first_frame, caption="First frame - does the person look normal?", width=400)

            with st.spinner("Analyzing video... This may take 30-60 seconds."):
                records, fps, cam_angle, cam_conf = analyze_video(video_path, selected_view)
                report, df = score_risk(records, fps, cam_angle, cam_conf, hybrid_model=hybrid_model, baseline_df=baseline_df)

                clinical_intake = build_clinical_intake(
                    patient_age, sport, competition_level, dominant_limb, involved_limb,
                    prior_acl, surgery_history, pain_score, swelling, instability,
                    rts_phase, clinician_notes
                )
                report.clinical_intake = clinical_intake
                report.clinical_impression = build_clinical_impression(report, clinical_intake)

                if validation_results:
                    report.validation_results = validation_results

                fig = create_charts(df, report, fps)

            try:
                os.unlink(video_path)
            except Exception:
                pass

            st.session_state.analysis_done = True
            st.session_state.last_upload_key = upload_key
            st.session_state.cached_report = report
            st.session_state.cached_df = df
            st.session_state.cached_fps = fps
            st.session_state.cached_fig = fig
            st.session_state.cached_validation_fig = validation_fig

            st.success(f"Analysis complete! Processed {len(records)} frames at {fps:.1f} fps.")
        else:
            report = st.session_state.cached_report
            df = st.session_state.cached_df
            fps = st.session_state.cached_fps
            fig = st.session_state.cached_fig
            validation_fig = st.session_state.cached_validation_fig
            clinical_intake = report.clinical_intake
            st.success(f"Loaded saved analysis. Processed {len(df)} frames at {fps:.1f} fps.")

        st.info(f"Camera view selected by clinician: {report.camera_angle} (quality-adjusted confidence: {report.camera_confidence:.0%})")
        st.info(f"Movement profile: {report.movement_profile}")

        st.markdown("""
        <div class="flow-shell">
            <div class="flow-title">Step 3 — Clinical Results Overview</div>
            <div class="flow-note">Review return-to-sport guidance, interpretation confidence, and recommended next step before moving into detailed tabs.</div>
        </div>
        """, unsafe_allow_html=True)
        display_clinical_snapshot(report)
        display_result_strip(report)
        display_executive_summary(report)
        st.divider()

        with st.expander("Diagnostic Data"):
            st.write(f"Left knee flexion at IC: {report.left_knee_flexion_at_IC}")
            st.write(f"Right knee flexion at IC: {report.right_knee_flexion_at_IC}")
            st.write(f"Left valgus peak: {report.peak_left_valgus}")
            st.write(f"Right valgus peak: {report.peak_right_valgus}")
            st.write(f"Pelvis peak reported: {report.peak_pelvis_drop}")

            if "pelvis_drop" in df.columns:
                pelvis_diag_start = report.ic_frame if report.ic_frame is not None else 0
                pelvis_diag = pd.to_numeric(
                    df["pelvis_drop"].iloc[pelvis_diag_start:pelvis_diag_start + 90],
                    errors="coerce"
                ).dropna().abs()

                if not pelvis_diag.empty:
                    st.write({
                        "pelvis_raw_max": float(pelvis_diag.max()),
                        "pelvis_raw_p95": float(np.nanpercentile(pelvis_diag, 95)),
                        "pelvis_raw_p90": float(np.nanpercentile(pelvis_diag, 90)),
                        "pelvis_raw_median": float(pelvis_diag.median()),
                    })

            if "pelvis_drop_smooth" in df.columns:
                pelvis_diag_start = report.ic_frame if report.ic_frame is not None else 0
                pelvis_smooth_diag = pd.to_numeric(
                    df["pelvis_drop_smooth"].iloc[pelvis_diag_start:pelvis_diag_start + 90],
                    errors="coerce"
                ).dropna().abs()

                if not pelvis_smooth_diag.empty:
                    st.write({
                        "pelvis_smooth_max": float(pelvis_smooth_diag.max()),
                        "pelvis_smooth_p95": float(np.nanpercentile(pelvis_smooth_diag, 95)),
                        "pelvis_smooth_p90": float(np.nanpercentile(pelvis_smooth_diag, 90)),
                        "pelvis_smooth_median": float(pelvis_smooth_diag.median()),
                    })

            st.write(f"IC frame detected: {report.ic_frame}")
            st.write(f"IC vote details: {report.ic_vote_details}")
            st.write(df.head(10))

        overview_tab, clinical_tab, biomechanics_tab, plan_tab, export_tab = st.tabs([
            "Summary",
            "Clinical",
            "Biomechanics",
            "Plan",
            "Export"
        ])

        with overview_tab:
            st.header("Clinical Summary")
            st.markdown('<div class="section-lead">A concise, clinician-facing summary of risk, landing pattern, key measurements, and high-priority flags.</div>', unsafe_allow_html=True)
            display_summary_dashboard(report)

        with clinical_tab:
            st.markdown('<div class="section-lead">Clinical decision support combines symptoms, limb context, video quality, and movement findings.</div>', unsafe_allow_html=True)
            display_clinical_decision_support(report, clinical_intake)
            st.divider()
            st.subheader("Analysis Quality")
            display_failure_detection(report)
            st.subheader("Measurement Uncertainty")
            display_uncertainty(report)

        with biomechanics_tab:
            st.markdown('<div class="section-lead">Time-series biomechanics and advanced technical outputs. Use these when you need to audit the screen.</div>', unsafe_allow_html=True)
            st.subheader("Biomechanical Time Series")
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Advanced Technical Details", expanded=False):
                display_advanced_biomechanics(report)
                st.subheader("Hybrid Scoring")
                display_hybrid_score(report)
                st.subheader("Population Baseline Percentiles")
                display_baseline_percentiles(report)

                if report.validation_results:
                    st.subheader("Validation Mode")
                    st.json(report.validation_results)
                    if validation_fig is not None:
                        st.plotly_chart(validation_fig, use_container_width=True)

        with plan_tab:
            st.markdown('<div class="section-lead">A structured corrective plan generated from the dominant movement deficits, symptom screen, and confidence-gated findings.</div>', unsafe_allow_html=True)
            display_recommendation_panel(report)
            display_progression_cards(report)

        with export_tab:
            st.markdown("""
            <div class="download-panel">
                <div class="download-title">Clinic-ready outputs</div>
                <div class="download-copy">Export the report for documentation, review the raw CSV for analysis, or save the JSON for future model development.</div>
            </div>
            """, unsafe_allow_html=True)
            st.subheader("Download Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "knee_angles_full.csv", "text/csv")
            with col2:
                st.download_button("Download JSON", json.dumps(asdict(report), indent=2).encode("utf-8"), "risk_report.json", "application/json")
            with col3:
                try:
                    pdf_data = build_pdf_report(report, patient_id, clinician_name, fig)
                    st.download_button("Download Clinic PDF", pdf_data, "clinic_report.pdf", "application/pdf")
                except Exception as e:
                    st.warning(f"PDF export unavailable: {e}")

    else:
        display_upload_guidance()

if __name__ == "__main__":
    main()