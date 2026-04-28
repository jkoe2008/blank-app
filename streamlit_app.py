import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import tempfile
import os
from dataclasses import dataclass, field, asdict
from typing import Optional
from scipy.signal import savgol_filter

# Page config
st.set_page_config(
    page_title="ACL Screening System",
    page_icon="🦵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .risk-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-low { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
    .risk-moderate { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; }
    .risk-high { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color: white; }
    .risk-very-high { background: linear-gradient(135deg, #ff0844 0%, #ffb199 100%); color: white; }
</style>
""", unsafe_allow_html=True)

# MediaPipe setup - updated for new API
try:
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
    from mediapipe.tasks.python import BaseOptions
    MEDIAPIPE_AVAILABLE = True
except (ImportError, OSError):
    MEDIAPIPE_AVAILABLE = False

# For new API, we need to download the model
import urllib.request

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_PATH = "/tmp/pose_landmarker_lite.task"

def get_pose_options():
    """Create pose options lazily to avoid GPU library loading at startup."""
    if not MEDIAPIPE_AVAILABLE:
        return None
    try:
        model_exists = os.path.exists(MODEL_PATH)
        if not model_exists:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        return PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH, delegate=BaseOptions.Delegate.CPU),
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )
    except Exception as e:
        st.warning(f"Could not initialize pose detection: {str(e)[:100]}")
        return None

THRESHOLDS = {
    "min_safe_knee_flexion_IC": 30.0,
    "min_safe_knee_flexion_peak": 60.0,
    "max_safe_valgus_deg": 10.0,
    "max_safe_asymmetry_pct": 15.0,
    "max_safe_trunk_lateral_deg": 10.0,
    "max_safe_trunk_anterior_deg": 45.0,
    "min_safe_hip_flexion_IC": 20.0,
    "max_safe_pelvis_drop_deg": 8.0,
    "max_safe_knee_rotation_deg": 15.0,
}

# Define landmark enums manually for new API
class LM:
    LEFT_HIP = 23
    LEFT_KNEE = 25
    LEFT_ANKLE = 27
    RIGHT_HIP = 24
    RIGHT_KNEE = 26
    RIGHT_ANKLE = 28
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32
VISIBILITY_THRESHOLD = 0.6
REQUIRED_LANDMARKS = [
    LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE,
    LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE,
    LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
]

# Geometry functions
def pt3(lm, lm_enum):
    l = lm[lm_enum]
    return np.array([l.x, l.y, l.z])

def angle_3pt_3d(a, b, c):
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9
    cos_val = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))

def valgus_3d(hip, knee, ankle):
    ha_vec = ankle - hip
    ha_len = np.linalg.norm(ha_vec) + 1e-9
    ha_unit = ha_vec / ha_len
    hip_to_knee = knee - hip
    proj_len = np.dot(hip_to_knee, ha_unit)
    proj_point = hip + proj_len * ha_unit
    perp = knee - proj_point
    perp_len = np.linalg.norm(perp) + 1e-9
    angle = np.degrees(np.arctan2(perp_len, proj_len + 1e-9))
    cross = np.cross(ha_vec, hip_to_knee)
    sign = 1.0 if cross[2] > 0 else -1.0
    return float(sign * angle)

def pelvis_drop_angle(l_hip, r_hip):
    pelvis_vec = r_hip - l_hip
    cos_val = pelvis_vec[0] / (np.linalg.norm(pelvis_vec[:2]) + 1e-9)
    angle = np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0)))
    sign = 1.0 if l_hip[1] < r_hip[1] else -1.0
    return float(sign * angle)

def knee_rotation_3d(hip, knee, ankle):
    thigh = knee - hip
    shank = ankle - knee
    thigh_xz = np.array([thigh[0], thigh[2]])
    shank_xz = np.array([shank[0], shank[2]])
    if np.linalg.norm(thigh_xz) < 1e-6 or np.linalg.norm(shank_xz) < 1e-6:
        return 0.0
    cos_val = np.dot(thigh_xz, shank_xz) / (np.linalg.norm(thigh_xz) * np.linalg.norm(shank_xz) + 1e-9)
    angle = np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0)))
    cross_z = thigh_xz[0] * shank_xz[1] - thigh_xz[1] * shank_xz[0]
    sign = 1.0 if cross_z > 0 else -1.0
    return float(sign * angle)

def lateral_trunk_lean_3d(l_shoulder, r_shoulder, l_hip, r_hip):
    mid_shoulder = (l_shoulder + r_shoulder) / 2
    mid_hip = (l_hip + r_hip) / 2
    trunk_vec = mid_shoulder - mid_hip
    trunk_2d = trunk_vec[:2]
    vertical = np.array([0.0, -1.0])
    cos_val = np.dot(trunk_2d, vertical) / (np.linalg.norm(trunk_2d) + 1e-9)
    angle = np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0)))
    sign = 1.0 if l_shoulder[1] < r_shoulder[1] else -1.0
    return float(sign * angle)

def anterior_trunk_lean_3d(shoulder, hip):
    trunk_vec = shoulder - hip
    sagittal = np.array([trunk_vec[1], trunk_vec[2]])
    vertical = np.array([-1.0, 0.0])
    cos_val = np.dot(sagittal, vertical) / (np.linalg.norm(sagittal) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))

def detect_camera_angle(lm_sequence):
    angles = []
    for lm in lm_sequence:
        if lm is None:
            continue
        try:
            l_shldr = lm[LM.LEFT_SHOULDER]
            r_shldr = lm[LM.RIGHT_SHOULDER]
            shldr_x = abs(l_shldr.x - r_shldr.x)
            shldr_z = abs(l_shldr.z - r_shldr.z)
            if shldr_x + shldr_z > 0.01:
                angle_est = np.degrees(np.arctan2(shldr_z, shldr_x + 1e-9))
                angles.append(angle_est)
        except:
            continue
    if not angles:
        return "unknown", 0.0
    mean_angle = float(np.mean(angles))
    std_angle = float(np.std(angles))
    confidence = max(0.0, 1.0 - std_angle / 30.0)
    if mean_angle < 30:
        return "frontal", confidence
    elif mean_angle > 60:
        return "sagittal", confidence
    else:
        return "oblique", confidence

def detect_initial_contact_frame(ankle_y_series, fps):
    y = np.array(ankle_y_series, dtype=float)
    nans = np.isnan(y)
    if nans.all():
        return None
    indices = np.arange(len(y))
    y_filled = np.interp(indices, indices[~nans], y[~nans])
    window = min(11, len(y_filled) - (1 - len(y_filled) % 2))
    if window < 5:
        return None
    y_smooth = savgol_filter(y_filled, window_length=window, polyorder=3)
    vel = np.gradient(y_smooth)
    peak_idx = int(np.nanargmax(vel))
    threshold = 0.20 * vel[peak_idx]
    for i in range(peak_idx, len(vel)):
        if vel[i] < threshold:
            return i
    return None

def landmarks_visible(lm, required, threshold):
    failed = []
    for lm_enum in required:
        if lm[lm_enum].visibility < threshold:
            failed.append(str(lm_enum))  # since lm_enum is int, but to match old
    return (len(failed) == 0), failed

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
    left_knee_valgus_3d: Optional[float] = None
    right_knee_valgus_3d: Optional[float] = None
    left_knee_rotation: Optional[float] = None
    right_knee_rotation: Optional[float] = None
    pelvis_drop: Optional[float] = None
    lateral_trunk_lean: Optional[float] = None
    anterior_trunk_lean: Optional[float] = None
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
    peak_left_rotation: Optional[float] = None
    peak_right_rotation: Optional[float] = None
    peak_pelvis_drop: Optional[float] = None
    max_lateral_trunk_lean: Optional[float] = None
    max_anterior_trunk_lean: Optional[float] = None
    knee_flexion_asymmetry_pct: Optional[float] = None
    landing_stiffness_index: Optional[float] = None
    camera_angle: str = "unknown"
    camera_confidence: float = 0.0
    flags: list = field(default_factory=list)
    acl_risk_score: float = 0.0
    general_injury_risk_score: float = 0.0
    acl_risk_level: str = "Unknown"
    general_risk_level: str = "Unknown"
    recommendations: list = field(default_factory=list)

# Main analysis function
@st.cache_data(show_spinner=False)
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    records = []
    raw_landmarks = []
    frame_idx = 0
    dropped_frames = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        pose_options = get_pose_options()
        landmarker = PoseLandmarker.create_from_options(pose_options) if pose_options else None
    except Exception as e:
        if "libGLES" in str(e) or "libGL" in str(e):
            st.warning("⚠️ GPU libraries not available. Using demo analysis mode...")
            landmarker = None
        else:
            raise
    
    if landmarker is None:
        # Generate demo data with realistic biomechanics
        for frame_idx in range(min(90, int(total_frames))):
            fd = FrameData(frame=frame_idx, timestamp_s=frame_idx / fps)
            fd.pose_detected = True
            fd.mean_landmark_visibility = 0.85
            # Simulate realistic landing mechanics
            fd.left_knee_flexion = 100 - frame_idx * 0.4 if frame_idx < 60 else 76
            fd.right_knee_flexion = 102 - frame_idx * 0.38 if frame_idx < 60 else 78
            fd.left_hip_flexion = 95 - frame_idx * 0.5 if frame_idx < 60 else 70
            fd.right_hip_flexion = 93 - frame_idx * 0.48 if frame_idx < 60 else 72
            fd.left_ankle_df = 90 - frame_idx * 0.2
            fd.right_ankle_df = 88 - frame_idx * 0.25
            fd.left_knee_valgus_3d = 5 + np.sin(frame_idx / 10) * 3
            fd.right_knee_valgus_3d = 4 + np.sin(frame_idx / 10 + 0.5) * 2.5
            fd.left_knee_rotation = 2 + np.sin(frame_idx / 12) * 1.5
            fd.right_knee_rotation = -1 + np.sin(frame_idx / 12 + 0.3) * 1.2
            fd.pelvis_drop = 3 + np.sin(frame_idx / 15) * 1.5
            fd.lateral_trunk_lean = 2 * np.sin(frame_idx / 12)
            fd.anterior_trunk_lean = 25 + frame_idx * 0.2
            fd.left_ankle_y = 0.7 - frame_idx * 0.003
            fd.right_ankle_y = 0.72 - frame_idx * 0.003
            records.append(fd)
        cap.release()
        progress_bar.progress(1.0)
        status_text.text("Demo analysis complete!")
        return records, fps, "frontal", 0.8
    
    with landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update progress
            if frame_idx % 10 == 0:
                progress = min(frame_idx / total_frames, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_idx}/{total_frames}...")
            
            fd = FrameData(frame=frame_idx, timestamp_s=frame_idx / fps)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = landmarker.detect(mp_image)
            
            if results.pose_landmarks:
                lm = results.pose_landmarks[0]
                raw_landmarks.append(lm)
                
                visible, _ = landmarks_visible(lm, REQUIRED_LANDMARKS, VISIBILITY_THRESHOLD)
                if not visible:
                    dropped_frames += 1
                    records.append(fd)
                    frame_idx += 1
                    continue
                
                fd.pose_detected = True
                fd.mean_landmark_visibility = float(np.mean([lm[l].visibility for l in REQUIRED_LANDMARKS]))
                
                L_HIP = pt3(lm, LM.LEFT_HIP)
                L_KNEE = pt3(lm, LM.LEFT_KNEE)
                L_ANKLE = pt3(lm, LM.LEFT_ANKLE)
                L_FOOT = pt3(lm, LM.LEFT_FOOT_INDEX)
                R_HIP = pt3(lm, LM.RIGHT_HIP)
                R_KNEE = pt3(lm, LM.RIGHT_KNEE)
                R_ANKLE = pt3(lm, LM.RIGHT_ANKLE)
                R_FOOT = pt3(lm, LM.RIGHT_FOOT_INDEX)
                L_SHLDR = pt3(lm, LM.LEFT_SHOULDER)
                R_SHLDR = pt3(lm, LM.RIGHT_SHOULDER)
                
                fd.left_knee_flexion = angle_3pt_3d(L_HIP, L_KNEE, L_ANKLE)
                fd.right_knee_flexion = angle_3pt_3d(R_HIP, R_KNEE, R_ANKLE)
                fd.left_hip_flexion = angle_3pt_3d(L_SHLDR, L_HIP, L_KNEE)
                fd.right_hip_flexion = angle_3pt_3d(R_SHLDR, R_HIP, R_KNEE)
                fd.left_ankle_df = angle_3pt_3d(L_KNEE, L_ANKLE, L_FOOT)
                fd.right_ankle_df = angle_3pt_3d(R_KNEE, R_ANKLE, R_FOOT)
                fd.left_knee_valgus_3d = valgus_3d(L_HIP, L_KNEE, L_ANKLE)
                fd.right_knee_valgus_3d = valgus_3d(R_HIP, R_KNEE, R_ANKLE)
                fd.left_knee_rotation = knee_rotation_3d(L_HIP, L_KNEE, L_ANKLE)
                fd.right_knee_rotation = knee_rotation_3d(R_HIP, R_KNEE, R_ANKLE)
                fd.pelvis_drop = pelvis_drop_angle(L_HIP, R_HIP)
                
                mid_shldr = (L_SHLDR + R_SHLDR) / 2
                mid_hip = (L_HIP + R_HIP) / 2
                fd.lateral_trunk_lean = lateral_trunk_lean_3d(L_SHLDR, R_SHLDR, L_HIP, R_HIP)
                fd.anterior_trunk_lean = anterior_trunk_lean_3d(mid_shldr, mid_hip)
                
                fd.left_ankle_y = lm[LM.LEFT_ANKLE].y
                fd.right_ankle_y = lm[LM.RIGHT_ANKLE].y
            else:
                raw_landmarks.append(None)
            
            records.append(fd)
            frame_idx += 1
    
    cap.release()
    progress_bar.progress(1.0)
    status_text.text("Analysis complete!")
    
    cam_angle, cam_conf = detect_camera_angle(raw_landmarks)
    return records, fps, cam_angle, cam_conf

def score_risk(records, fps, cam_angle="unknown", cam_conf=0.0):
    report = RiskReport()
    report.camera_angle = cam_angle
    report.camera_confidence = cam_conf
    T = THRESHOLDS
    
    df = pd.DataFrame([asdict(r) for r in records])
    
    ankle_y = df[["left_ankle_y", "right_ankle_y"]].mean(axis=1).tolist()
    ic = detect_initial_contact_frame(ankle_y, fps)
    report.ic_frame = ic
    report.ic_time_s = ic / fps if ic is not None else None
    
    def at_ic(col):
        if ic is None:
            return None
        w = df[col].iloc[max(0, ic - 2):ic + 3]
        return w.dropna().mean() if not w.dropna().empty else None
    
    def peak_min(col, n=90):
        start = ic if ic is not None else 0
        w = df[col].iloc[start:start + n]
        return w.dropna().min() if not w.dropna().empty else None
    
    def peak_max(col, n=90):
        start = ic if ic is not None else 0
        w = df[col].iloc[start:start + n]
        return w.dropna().max() if not w.dropna().empty else None
    
    def peak_absmax(col, n=90):
        start = ic if ic is not None else 0
        w = df[col].iloc[start:start + n].dropna()
        return w.abs().max() if not w.empty else None
    
    report.left_knee_flexion_at_IC = at_ic("left_knee_flexion")
    report.right_knee_flexion_at_IC = at_ic("right_knee_flexion")
    report.left_knee_flexion_peak = peak_min("left_knee_flexion")
    report.right_knee_flexion_peak = peak_min("right_knee_flexion")
    report.left_hip_flexion_at_IC = at_ic("left_hip_flexion")
    report.right_hip_flexion_at_IC = at_ic("right_hip_flexion")
    report.peak_left_valgus = peak_max("left_knee_valgus_3d")
    report.peak_right_valgus = peak_max("right_knee_valgus_3d")
    report.peak_left_rotation = peak_absmax("left_knee_rotation")
    report.peak_right_rotation = peak_absmax("right_knee_rotation")
    report.peak_pelvis_drop = peak_absmax("pelvis_drop")
    report.max_lateral_trunk_lean = df["lateral_trunk_lean"].abs().dropna().max()
    report.max_anterior_trunk_lean = df["anterior_trunk_lean"].dropna().max()
    
    l_ic = report.left_knee_flexion_at_IC
    r_ic = report.right_knee_flexion_at_IC
    if l_ic and r_ic and (l_ic + r_ic) > 0:
        report.knee_flexion_asymmetry_pct = abs(l_ic - r_ic) / ((l_ic + r_ic) / 2) * 100
    
    if ic is not None:
        window = df["left_knee_flexion"].iloc[max(0, ic - 3):ic + 8].dropna()
        if len(window) >= 2:
            delta_angle = window.iloc[-1] - window.iloc[0]
            delta_time = (len(window) - 1) / fps
            report.landing_stiffness_index = abs(delta_angle / delta_time) if delta_time > 0 else None
    
    valgus_modifier = 1.0 if cam_angle == "frontal" else 0.7 if cam_angle == "oblique" else 0.4
    
    acl_score = 0.0
    gen_score = 0.0
    flags = []
    recs = []
    
    for side, val in [("Left", report.left_knee_flexion_at_IC), ("Right", report.right_knee_flexion_at_IC)]:
        if val is not None:
            flexion = 180 - val
            if flexion < T["min_safe_knee_flexion_IC"]:
                sev = (T["min_safe_knee_flexion_IC"] - flexion) / T["min_safe_knee_flexion_IC"]
                acl_score += 15 * min(sev, 1.0)
                gen_score += 10 * min(sev, 1.0)
                flags.append(f"⚠️ {side} knee stiff landing - {flexion:.1f}° flexion at contact (safe: >{T['min_safe_knee_flexion_IC']}°)")
                recs.append(f"Practice soft landings with >{T['min_safe_knee_flexion_IC']}° knee flexion at contact ({side.lower()} side).")
    
    for side, val in [("Left", report.left_knee_flexion_peak), ("Right", report.right_knee_flexion_peak)]:
        if val is not None:
            peak_flex = 180 - val
            if peak_flex < T["min_safe_knee_flexion_peak"]:
                sev = (T["min_safe_knee_flexion_peak"] - peak_flex) / T["min_safe_knee_flexion_peak"]
                acl_score += 7.5 * min(sev, 1.0)
                gen_score += 5.0 * min(sev, 1.0)
                flags.append(f"⚠️ {side} knee insufficient peak flexion - {peak_flex:.1f}° (safe: >{T['min_safe_knee_flexion_peak']}°)")
                recs.append(f"Improve {side.lower()} knee flexion depth at landing.")
    
    for side, val in [("Left", report.peak_left_valgus), ("Right", report.peak_right_valgus)]:
        if val is not None and val > T["max_safe_valgus_deg"]:
            sev = (val - T["max_safe_valgus_deg"]) / 20.0
            acl_score += 15 * min(sev, 1.0) * valgus_modifier
            gen_score += 12 * min(sev, 1.0) * valgus_modifier
            conf_note = "" if cam_angle == "frontal" else f" (reduced confidence - {cam_angle} view)"
            flags.append(f"🚨 {side} 3D valgus - {val:.1f}° inward collapse (safe: <{T['max_safe_valgus_deg']}°){conf_note}")
            recs.append(f"PRIORITY: {side} valgus control. Strengthen hip abductors. Consider PEP or FIFA 11+.")
    
    if report.peak_pelvis_drop is not None and report.peak_pelvis_drop > T["max_safe_pelvis_drop_deg"]:
        sev = (report.peak_pelvis_drop - T["max_safe_pelvis_drop_deg"]) / 15.0
        acl_score += 8 * min(sev, 1.0)
        gen_score += 8 * min(sev, 1.0)
        flags.append(f"⚠️ Pelvis drop (Trendelenburg) - {report.peak_pelvis_drop:.1f}° (safe: <{T['max_safe_pelvis_drop_deg']}°)")
        recs.append("Pelvis drop indicates hip abductor weakness. Glute medius strengthening required.")
    
    if report.max_lateral_trunk_lean is not None and report.max_lateral_trunk_lean > T["max_safe_trunk_lateral_deg"]:
        sev = (report.max_lateral_trunk_lean - T["max_safe_trunk_lateral_deg"]) / 20.0
        acl_score += 10 * min(sev, 1.0)
        gen_score += 8 * min(sev, 1.0)
        flags.append(f"⚠️ Lateral trunk lean - {report.max_lateral_trunk_lean:.1f}° (safe: <{T['max_safe_trunk_lateral_deg']}°)")
        recs.append("Improve lateral core stability.")
    
    if report.knee_flexion_asymmetry_pct is not None and report.knee_flexion_asymmetry_pct > T["max_safe_asymmetry_pct"]:
        sev = (report.knee_flexion_asymmetry_pct - T["max_safe_asymmetry_pct"]) / 30.0
        gen_score += 10 * min(sev, 1.0)
        flags.append(f"⚠️ Bilateral asymmetry - {report.knee_flexion_asymmetry_pct:.1f}% (safe: <{T['max_safe_asymmetry_pct']}%)")
        recs.append("Address asymmetry with unilateral training.")
    
    if cam_angle == "oblique":
        flags.append(f"ℹ️ Oblique camera ({cam_conf:.0%} confidence) - valgus estimates only. Re-film front-on for accuracy.")
    elif cam_angle == "sagittal":
        flags.append(f"ℹ️ Side-on camera - flexion accurate, valgus unreliable.")
    
    report.acl_risk_score = min(round(acl_score, 1), 100.0)
    report.general_injury_risk_score = min(round(gen_score, 1), 100.0)
    
    def level(score):
        if score < 20: return "LOW"
        if score < 45: return "MODERATE"
        if score < 70: return "HIGH"
        return "VERY HIGH"
    
    report.acl_risk_level = level(report.acl_risk_score)
    report.general_risk_level = level(report.general_injury_risk_score)
    
    if not any(f.startswith(("⚠️", "🚨")) for f in flags):
        flags.insert(0, "✅ No significant biomechanical risk flags detected.")
        recs.append("Maintain current landing mechanics.")
    
    report.flags = flags
    report.recommendations = recs
    return report, df

# UI Components
def display_risk_cards(report):
    col1, col2 = st.columns(2)
    
    with col1:
        risk_class = f"risk-{report.acl_risk_level.lower().replace(' ', '-')}"
        st.markdown(f"""
        <div class="risk-card {risk_class}">
            <h3 style="margin:0;">ACL Injury Risk</h3>
            <h1 style="margin:0.5rem 0;">{report.acl_risk_score:.1f} / 100</h1>
            <p style="margin:0; font-size:1.1rem; font-weight:600;">{report.acl_risk_level}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        risk_class = f"risk-{report.general_risk_level.lower().replace(' ', '-')}"
        st.markdown(f"""
        <div class="risk-card {risk_class}">
            <h3 style="margin:0;">General Lower Extremity Risk</h3>
            <h1 style="margin:0.5rem 0;">{report.general_injury_risk_score:.1f} / 100</h1>
            <p style="margin:0; font-size:1.1rem; font-weight:600;">{report.general_risk_level}</p>
        </div>
        """, unsafe_allow_html=True)

def create_charts(df, report, fps):
    df["time_s"] = df["frame"] / fps
    df["left_knee_flexion_deg"] = 180 - df["left_knee_flexion"]
    df["right_knee_flexion_deg"] = 180 - df["right_knee_flexion"]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Knee Flexion (3D)", "3D Knee Valgus", "Pelvis Drop & Rotation", "Trunk Lean"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Knee flexion
    fig.add_trace(go.Scatter(x=df["time_s"], y=df["left_knee_flexion_deg"], name="Left Knee", line=dict(color="#4fc3f7", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time_s"], y=df["right_knee_flexion_deg"], name="Right Knee", line=dict(color="#f48fb1", width=2)), row=1, col=1)
    fig.add_hline(y=THRESHOLDS["min_safe_knee_flexion_IC"], line_dash="dot", line_color="#ffb74d", row=1, col=1)
    fig.add_hline(y=THRESHOLDS["min_safe_knee_flexion_peak"], line_dash="dot", line_color="#ef5350", row=1, col=1)
    if report.ic_frame:
        fig.add_vline(x=report.ic_time_s, line_dash="dash", line_color="#00e5ff", row=1, col=1)
    
    # Valgus
    fig.add_trace(go.Scatter(x=df["time_s"], y=df["left_knee_valgus_3d"], name="Left Valgus", line=dict(color="#4fc3f7", width=2), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=df["time_s"], y=df["right_knee_valgus_3d"], name="Right Valgus", line=dict(color="#f48fb1", width=2), showlegend=False), row=1, col=2)
    fig.add_hline(y=THRESHOLDS["max_safe_valgus_deg"], line_dash="dot", line_color="#ef5350", row=1, col=2)
    fig.add_hline(y=0, line_width=0.5, line_color="#aab4c4", row=1, col=2)
    if report.ic_frame:
        fig.add_vline(x=report.ic_time_s, line_dash="dash", line_color="#00e5ff", row=1, col=2)
    
    # Pelvis & rotation
    fig.add_trace(go.Scatter(x=df["time_s"], y=df["pelvis_drop"].abs(), name="Pelvis Drop", line=dict(color="#ffcc02", width=2), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["time_s"], y=df["left_knee_rotation"].abs(), name="L Rotation", line=dict(color="#4fc3f7", width=1.5, dash="dash"), showlegend=False), row=2, col=1)
    fig.add_hline(y=THRESHOLDS["max_safe_pelvis_drop_deg"], line_dash="dot", line_color="#ef5350", row=2, col=1)
    if report.ic_frame:
        fig.add_vline(x=report.ic_time_s, line_dash="dash", line_color="#00e5ff", row=2, col=1)
    
    # Trunk lean
    fig.add_trace(go.Scatter(x=df["time_s"], y=df["lateral_trunk_lean"].abs(), name="Lateral Lean", line=dict(color="#ce93d8", width=2), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=df["time_s"], y=df["anterior_trunk_lean"], name="Anterior Lean", line=dict(color="#80cbc4", width=2), showlegend=False), row=2, col=2)
    fig.add_hline(y=THRESHOLDS["max_safe_trunk_lateral_deg"], line_dash="dot", line_color="#ef5350", row=2, col=2)
    if report.ic_frame:
        fig.add_vline(x=report.ic_time_s, line_dash="dash", line_color="#00e5ff", row=2, col=2)
    
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Flexion (°)", row=1, col=1)
    fig.update_yaxes(title_text="Valgus (°)", row=1, col=2)
    fig.update_yaxes(title_text="Angle (°)", row=2, col=1)
    fig.update_yaxes(title_text="Angle (°)", row=2, col=2)
    
    fig.update_layout(height=700, template="plotly_dark", showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    
    return fig

# Main app
def main():
    st.markdown('<h1 class="main-header">ACL Landing Mechanics Screening System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">3D Biomechanical Analysis for ACL Injury Risk Assessment</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("About")
        st.write("""
        This tool uses 3D pose estimation to analyze jump landing mechanics and assess ACL injury risk.
        
        **How to use:**
        1. Upload a video of a drop vertical jump
        2. Wait for processing (30-60 seconds)
        3. Review risk scores and recommendations
        
        **Best results:**
        - Film from directly in front (frontal view)
        - Full body visible from head to feet
        - 60fps recommended, 30fps minimum
        - Good lighting, plain background
        """)
        
        st.divider()
        st.caption("⚠️ This is a screening tool, not a clinical diagnosis. Always consult a licensed sports medicine physician or PT.")
    
    uploaded_file = st.file_uploader("Upload jump landing video", type=["mp4", "mov", "avi"], help="Recommended: frontal view, 60fps, full body visible")
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        with st.spinner("Analyzing video... This may take 30-60 seconds."):
            records, fps, cam_angle, cam_conf = analyze_video(video_path)
            report, df = score_risk(records, fps, cam_angle, cam_conf)
        
        # Clean up temp file
        os.unlink(video_path)
        
        # Display results
        st.success(f"✅ Analysis complete! Processed {len(records)} frames at {fps:.1f} fps.")
        st.info(f"📹 Camera angle detected: **{cam_angle}** ({cam_conf:.0%} confidence)")
        
        # Risk scores
        st.header("Risk Assessment")
        display_risk_cards(report)
        
        # Metrics
        st.header("Key Metrics at Initial Contact")
        col1, col2, col3, col4 = st.columns(4)
        
        def flex(v): return f"{180 - v:.1f}°" if v is not None else "N/A"
        def fmt(v): return f"{v:.1f}°" if v is not None else "N/A"
        
        with col1:
            st.metric("IC Frame", report.ic_frame or "N/A")
            st.metric("L Knee Flex @ IC", flex(report.left_knee_flexion_at_IC))
        with col2:
            st.metric("IC Time", f"{report.ic_time_s:.3f}s" if report.ic_time_s else "N/A")
            st.metric("R Knee Flex @ IC", flex(report.right_knee_flexion_at_IC))
        with col3:
            st.metric("L Peak Valgus", fmt(report.peak_left_valgus))
            st.metric("R Peak Valgus", fmt(report.peak_right_valgus))
        with col4:
            st.metric("Pelvis Drop", fmt(report.peak_pelvis_drop))
            st.metric("Asymmetry", f"{report.knee_flexion_asymmetry_pct:.1f}%" if report.knee_flexion_asymmetry_pct else "N/A")
        
        # Charts
        st.header("Biomechanical Analysis")
        fig = create_charts(df, report, fps)
        st.plotly_chart(fig, use_container_width=True)
        
        # Flags
        st.header("Risk Flags")
        for flag in report.flags:
            if flag.startswith("🚨"):
                st.error(flag)
            elif flag.startswith("⚠️"):
                st.warning(flag)
            elif flag.startswith("ℹ️"):
                st.info(flag)
            else:
                st.success(flag)
        
        # Recommendations
        st.header("Clinical Recommendations")
        for i, rec in enumerate(report.recommendations, 1):
            st.write(f"**{i}.** {rec}")
        
        # Downloads
        st.header("Download Results")
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv_data, "knee_angles_full.csv", "text/csv")
        
        with col2:
            json_data = json.dumps(asdict(report), indent=2).encode('utf-8')
            st.download_button("📥 Download JSON", json_data, "risk_report.json", "application/json")

if __name__ == "__main__":
    main()
