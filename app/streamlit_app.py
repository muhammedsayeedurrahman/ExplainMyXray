import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io
import time
from datetime import datetime

# ---------------------------------------------------------------------------
import os
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="ExplainMyXray",
    page_icon="\u2695",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "analysis_image" not in st.session_state:
    st.session_state.analysis_image = None
if "raw_image" not in st.session_state:
    st.session_state.raw_image = None
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# ---------------------------------------------------------------------------
# Theme Palettes (5-tier elevation)
# ---------------------------------------------------------------------------
F_SANS = "'Inter',-apple-system,'Segoe UI','Helvetica Neue',Arial,sans-serif"
F_MONO = "'JetBrains Mono','Fira Code','SF Mono','Consolas',monospace"

DARK = {
    "e0": "#060A14", "e1": "#0A1020", "e2": "#0F1729", "e3": "#162034",
    "e4": "#1C2A42", "e5": "#243352",
    "bs": "#152030", "bn": "#1E3050", "bf": "#2A4068",
    "t1": "#F1F5F9", "t2": "#C8D2E0", "t3": "#7E90A8", "t4": "#4A5E78",
    "ac": "#3B82F6", "ac2": "#60A5FA", "ac3": "#2563EB",
    "ag": "rgba(59,130,246,0.10)", "ar": "rgba(59,130,246,0.22)",
    "sc": "#38BDF8",
    "gn": "#34D399", "gb": "rgba(52,211,153,0.07)", "gd": "rgba(52,211,153,0.20)",
    "rd": "#F87171", "rb": "rgba(248,113,113,0.05)",
    "am": "#FBBF24",
    "s1": "rgba(0,0,0,0.18)", "s2": "rgba(0,0,0,0.30)", "s3": "rgba(0,0,0,0.45)",
    "s4": "rgba(0,0,0,0.60)", "si": "rgba(0,0,0,0.55)",
    "he": "rgba(255,255,255,0.035)", "hg": "rgba(255,255,255,0.055)",
    "ov": "rgba(6,10,20,0.88)",
    "ib": "rgba(59,130,246,0.05)", "id": "rgba(59,130,246,0.28)",
    # cockpit edge highlight
    "ck": "rgba(56,189,248,0.08)",
}

LIGHT = {
    "e0": "#EBEEF3", "e1": "#D5DAE2", "e2": "#F2F4F7", "e3": "#FFFFFF",
    "e4": "#FFFFFF", "e5": "#F0F2F7",
    "bs": "#DEE2EA", "bn": "#C8CED8", "bf": "#A8B2C0",
    "t1": "#0F172A", "t2": "#334155", "t3": "#64748B", "t4": "#94A3B8",
    "ac": "#2563EB", "ac2": "#3B82F6", "ac3": "#1D4ED8",
    "ag": "rgba(37,99,235,0.06)", "ar": "rgba(37,99,235,0.15)",
    "sc": "#0284C7",
    "gn": "#059669", "gb": "rgba(5,150,105,0.06)", "gd": "rgba(5,150,105,0.16)",
    "rd": "#DC2626", "rb": "rgba(220,38,38,0.04)",
    "am": "#D97706",
    "s1": "rgba(0,0,0,0.04)", "s2": "rgba(0,0,0,0.07)", "s3": "rgba(0,0,0,0.11)",
    "s4": "rgba(0,0,0,0.16)", "si": "rgba(0,0,0,0.06)",
    "he": "rgba(255,255,255,0.85)", "hg": "rgba(255,255,255,0.65)",
    "ov": "rgba(255,255,255,0.92)",
    "ib": "rgba(37,99,235,0.04)", "id": "rgba(37,99,235,0.18)",
    "ck": "rgba(2,132,199,0.06)",
}

T = DARK if st.session_state.theme == "dark" else LIGHT

# ---------------------------------------------------------------------------
# Ultra-Premium CSS
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ==================================================================
   SMOOTH THEME TRANSITION + 3D ENTRANCE SYSTEM
================================================================== */
@keyframes themeIn {{
    0%   {{ opacity:0; filter:brightness(0.88) saturate(0.9); transform:translateY(6px); }}
    100% {{ opacity:1; filter:brightness(1) saturate(1); transform:translateY(0); }}
}}
@keyframes riseIn {{
    from {{ opacity:0; transform:translateY(14px) perspective(1000px) rotateX(1.2deg); }}
    to   {{ opacity:1; transform:translateY(0) perspective(1000px) rotateX(0deg); }}
}}
@keyframes breathe {{
    0%,100% {{ border-color:{T["bf"]}; box-shadow:inset 0 1px 6px {T["si"]},0 1px 3px {T["s1"]}; }}
    50%     {{ border-color:{T["ac"]}; box-shadow:inset 0 1px 6px {T["si"]},0 0 0 3px {T["ag"]},0 0 18px {T["ag"]}; }}
}}
@keyframes floatIn {{
    from {{ opacity:0; transform:translateX(12px); }}
    to   {{ opacity:1; transform:translateX(0); }}
}}
@keyframes pulse {{
    0%,100% {{ opacity:1; box-shadow:0 0 4px {T["gn"]}; }}
    50%     {{ opacity:0.5; box-shadow:0 0 10px {T["gn"]}; }}
}}
@keyframes spin {{ to {{ transform:rotate(360deg); }} }}

.stApp {{
    background:{T["e0"]};
    color:{T["t2"]};
    font-family:{F_SANS};
    animation:themeIn 0.3s cubic-bezier(0.22,1,0.36,1) forwards;
}}
*,*::before,*::after {{
    transition-property:background,background-color,border-color,color,box-shadow,opacity,filter,transform,backdrop-filter;
    transition-duration:0.25s;
    transition-timing-function:cubic-bezier(0.22,1,0.36,1);
}}
.spin-ring,.dot {{ transition:none !important; }}

/* ==================================================================
   HIDE STREAMLIT CHROME
================================================================== */
#MainMenu,header,footer,.stDeployButton {{ visibility:hidden; display:none; }}
.block-container {{
    padding-top:0 !important;
    padding-bottom:0.5rem !important;
    max-width:100% !important;
    perspective:2400px;
}}
::-webkit-scrollbar {{ width:5px; height:5px; }}
::-webkit-scrollbar-track {{ background:{T["e0"]}; }}
::-webkit-scrollbar-thumb {{ background:{T["bn"]}; border-radius:3px; }}

/* ==================================================================
   HEADER — floating cockpit bar
================================================================== */
.hdr {{
    background:linear-gradient(180deg,{T["e4"]} 0%,{T["e3"]} 100%);
    border-bottom:1px solid {T["bn"]};
    border-top:1px solid {T["ck"]};
    padding:12px 28px;
    display:flex; align-items:center; justify-content:space-between;
    margin:-1rem -1rem 0 -1rem;
    position:relative; z-index:100;
    box-shadow:
        inset 0 1px 0 {T["he"]},
        0 1px 2px {T["s2"]},
        0 4px 16px {T["s3"]},
        0 12px 48px {T["s4"]};
}}
.hdr-l {{ display:flex; align-items:center; gap:16px; }}
.hdr-logo {{ font-size:1.2rem; font-weight:800; color:{T["t1"]}; letter-spacing:-0.04em; }}
.hdr-logo b {{ color:{T["ac"]}; }}
.hdr-sep {{ width:1px; height:24px; background:linear-gradient(180deg,transparent,{T["bn"]},transparent); }}
.hdr-sub {{ font-size:0.78rem; color:{T["t3"]}; font-weight:400; letter-spacing:0.01em; }}
.hdr-r {{ display:flex; align-items:center; gap:10px; }}

/* ==================================================================
   GLASS BADGES — frosted capsule with backdrop-blur
================================================================== */
.bdg {{
    display:inline-flex; align-items:center; gap:5px;
    padding:4px 12px; border-radius:8px;
    font-size:0.64rem; font-weight:600; font-family:{F_MONO};
    border:1px solid {T["bn"]};
    background:linear-gradient(180deg,{T["hg"]} 0%,transparent 50%),{T["e3"]};
    backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px);
    box-shadow:inset 0 1px 0 {T["he"]},0 1px 4px {T["s1"]};
}}
.bdg-model {{ color:{T["t3"]}; }}
.bdg-ver {{ color:{T["t4"]}; }}
.bdg-status {{
    background:linear-gradient(180deg,{T["hg"]} 0%,transparent 50%),{T["gb"]};
    color:{T["gn"]}; border-color:{T["gd"]};
    box-shadow:inset 0 1px 0 {T["he"]},0 0 10px {T["gb"]};
}}
.dot {{
    width:6px; height:6px; border-radius:50%;
    background:{T["gn"]}; box-shadow:0 0 6px {T["gn"]};
    animation:pulse 2s ease-in-out infinite;
    display:inline-block;
}}

/* ==================================================================
   PANEL HEADER — bevelled instrument bar
================================================================== */
.ph {{
    background:linear-gradient(180deg,{T["e4"]} 0%,{T["e2"]} 100%);
    border-bottom:1px solid {T["bs"]};
    border-top:1px solid {T["ck"]};
    padding:10px 16px;
    display:flex; align-items:center; justify-content:space-between;
    margin:0 -1rem;
    box-shadow:inset 0 1px 0 {T["he"]},0 2px 8px {T["s2"]};
    animation:riseIn 0.35s cubic-bezier(0.22,1,0.36,1) backwards;
    animation-delay:0.05s;
}}
.ph-t {{ font-size:0.7rem; font-weight:700; color:{T["t3"]}; letter-spacing:0.09em; text-transform:uppercase; }}
.ph-tags {{ display:flex; gap:5px; }}
.ph-tag {{
    background:linear-gradient(180deg,{T["hg"]} 0%,transparent 50%),{T["e3"]};
    border:1px solid {T["bs"]};
    color:{T["t4"]}; font-size:0.57rem; padding:2px 8px; border-radius:4px;
    font-family:{F_MONO}; font-weight:500;
    box-shadow:inset 0 1px 0 {T["he"]};
    backdrop-filter:blur(6px); -webkit-backdrop-filter:blur(6px);
}}

/* ==================================================================
   IMAGE VIEWER — 3D medical display with bezel + toolbar + DICOM
================================================================== */
.vw {{
    background:{T["e1"]};
    border:1px solid {T["bn"]};
    border-top-color:{T["bf"]};
    border-bottom-color:{T["bs"]};
    border-radius:12px;
    padding:5px;
    position:relative;
    display:flex; align-items:center; justify-content:center;
    transform:perspective(1200px) rotateX(0.4deg);
    transform-origin:center bottom;
    transform-style:preserve-3d;
    box-shadow:
        inset 0 2px 12px {T["si"]},
        inset 0 0 1px {T["s2"]},
        inset 0 -1px 0 {T["ck"]},
        0 2px 4px {T["s1"]},
        0 6px 20px {T["s2"]},
        0 16px 48px {T["s3"]};
    overflow:hidden;
    animation:riseIn 0.4s cubic-bezier(0.22,1,0.36,1) backwards;
    animation-delay:0.1s;
}}
/* inner bezel ring */
.vw::before {{
    content:'';
    position:absolute; inset:4px;
    border:1px solid {T["bs"]};
    border-top-color:{T["bf"]};
    border-radius:9px;
    pointer-events:none; z-index:5;
    box-shadow:inset 0 0 20px {T["ag"]};
}}
/* floating measurement toolbar — appears on hover */
.vw::after {{
    content:'\\2316  \\2295  \\2220  \\21D5';
    position:absolute; right:10px; top:50%;
    transform:translateY(-50%) translateX(50px);
    opacity:0;
    display:flex; flex-direction:column; gap:8px;
    padding:10px 8px;
    font-size:0.8rem; letter-spacing:0.3em;
    color:{T["t4"]};
    background:linear-gradient(180deg,{T["hg"]},transparent 60%),{T["ov"]};
    border:1px solid {T["bs"]};
    border-radius:8px;
    backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px);
    box-shadow:0 4px 16px {T["s3"]};
    z-index:20;
    pointer-events:none;
    writing-mode:vertical-lr;
    text-orientation:upright;
    transition:all 0.3s cubic-bezier(0.22,1,0.36,1);
}}
.vw:hover {{
    transform:perspective(1200px) rotateX(-0.15deg) translateZ(4px);
    border-color:{T["bf"]};
    box-shadow:
        inset 0 2px 12px {T["si"]},
        inset 0 0 1px {T["s2"]},
        inset 0 -1px 0 {T["ck"]},
        0 4px 8px {T["s1"]},
        0 12px 32px {T["s2"]},
        0 24px 64px {T["s3"]},
        0 0 30px {T["ag"]};
}}
.vw:hover::after {{
    opacity:1;
    transform:translateY(-50%) translateX(0);
}}
/* viewport-fitted image */
.vw [data-testid="stImage"] {{
    max-height:50vh !important;
    display:flex; align-items:center; justify-content:center;
}}
.vw [data-testid="stImage"] img {{
    max-height:50vh !important;
    width:auto !important;
    object-fit:contain !important;
    border-radius:7px;
}}
/* corner tag */
.vw-tag {{
    position:absolute; top:10px; left:12px; z-index:10;
    font-size:0.58rem; font-family:{F_MONO}; font-weight:600;
    color:{T["t4"]};
    background:linear-gradient(180deg,{T["hg"]},transparent 40%),{T["ov"]};
    padding:3px 11px; border-radius:6px;
    letter-spacing:0.05em;
    border:1px solid {T["bs"]};
    box-shadow:0 2px 8px {T["s2"]};
    backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px);
}}
/* DICOM overlay — appears on hover */
.vw-dicom {{
    position:absolute; bottom:10px; right:12px; z-index:10;
    font-family:{F_MONO}; font-size:0.52rem; font-weight:500;
    color:{T["t4"]}; line-height:1.7;
    background:linear-gradient(180deg,{T["hg"]},transparent 40%),{T["ov"]};
    padding:6px 10px; border-radius:6px;
    border:1px solid {T["bs"]};
    backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px);
    box-shadow:0 2px 8px {T["s2"]};
    opacity:0;
    transform:translateY(4px);
    transition:all 0.25s cubic-bezier(0.22,1,0.36,1);
    pointer-events:none; white-space:pre;
    letter-spacing:0.03em;
}}
.vw:hover .vw-dicom {{ opacity:1; transform:translateY(0); }}

/* ==================================================================
   FILMSTRIP — simulated prior studies
================================================================== */
.film {{
    display:flex; gap:5px; padding:6px 0;
    overflow-x:auto; scrollbar-width:none;
    animation:riseIn 0.4s cubic-bezier(0.22,1,0.36,1) backwards;
    animation-delay:0.15s;
}}
.film::-webkit-scrollbar {{ display:none; }}
.film-t {{
    width:42px; height:42px; flex-shrink:0;
    background:{T["e1"]};
    border:1px solid {T["bs"]};
    border-radius:5px;
    box-shadow:inset 0 1px 4px {T["si"]};
    position:relative; overflow:hidden;
}}
.film-t::after {{
    content:'';
    position:absolute; inset:0;
    background:linear-gradient(135deg,transparent 40%,{T["hg"]} 50%,transparent 60%);
}}
.film-t.active {{
    border-color:{T["ac"]};
    box-shadow:inset 0 1px 4px {T["si"]},0 0 8px {T["ag"]};
}}
.film-l {{
    font-size:0.52rem; color:{T["t4"]}; font-family:{F_MONO};
    padding:2px 0; letter-spacing:0.04em;
}}

/* ==================================================================
   EMPTY STATE
================================================================== */
.vw-empty {{
    display:flex; flex-direction:column;
    align-items:center; justify-content:center;
    min-height:38vh; gap:14px; text-align:center;
    animation:riseIn 0.4s cubic-bezier(0.22,1,0.36,1) backwards;
    animation-delay:0.08s;
}}
.vw-empty-ico {{
    width:56px; height:56px;
    border:2px dashed {T["bf"]};
    border-radius:14px;
    display:flex; align-items:center; justify-content:center;
    color:{T["t4"]}; font-size:1.2rem; opacity:0.3;
    box-shadow:inset 0 2px 6px {T["si"]};
}}
.vw-empty-t {{ font-size:0.86rem; color:{T["t3"]}; font-weight:600; }}
.vw-empty-s {{ font-size:0.72rem; color:{T["t4"]}; line-height:1.6; }}

/* ==================================================================
   UPLOAD ZONE — breathing idle pulse
================================================================== */
[data-testid="stFileUploader"] {{ background:transparent !important; }}
[data-testid="stFileUploader"] section {{
    background:linear-gradient(180deg,{T["hg"]} 0%,transparent 30%),{T["e3"]} !important;
    border:2px dashed {T["bf"]} !important;
    border-radius:14px !important;
    padding:28px 16px !important;
    box-shadow:inset 0 1px 6px {T["si"]},0 1px 3px {T["s1"]} !important;
    animation:breathe 4s ease-in-out infinite !important;
}}
[data-testid="stFileUploader"] section:hover {{
    border-color:{T["ac"]} !important;
    animation:none !important;
    box-shadow:inset 0 1px 6px {T["si"]},0 0 0 3px {T["ar"]},0 0 24px {T["ag"]} !important;
    transform:translateY(-2px);
}}
[data-testid="stFileUploader"] button {{
    background:linear-gradient(180deg,{T["ac2"]} 0%,{T["ac3"]} 100%) !important;
    color:#fff !important; border:none !important;
    border-radius:8px !important;
    font-size:0.76rem !important; padding:9px 22px !important;
    font-weight:600 !important;
    box-shadow:inset 0 1px 0 rgba(255,255,255,0.18),0 2px 8px {T["s2"]} !important;
}}
[data-testid="stFileUploader"] small {{ color:{T["t4"]} !important; font-size:0.66rem !important; }}

/* ==================================================================
   METADATA ROW
================================================================== */
.mr {{ display:flex; gap:14px; padding:7px 2px; flex-wrap:wrap; }}
.mi {{ font-size:0.63rem; color:{T["t4"]}; font-family:{F_MONO}; font-weight:500; }}
.mi span {{ color:{T["t3"]}; font-weight:600; }}

/* ==================================================================
   PRIMARY BUTTON — 3D raised with hover lift
================================================================== */
.stButton > button {{
    background:linear-gradient(180deg,{T["ac2"]} 0%,{T["ac3"]} 100%) !important;
    color:#fff !important; border:none !important;
    border-radius:10px !important;
    font-weight:700 !important; font-size:0.84rem !important;
    padding:12px 24px !important; font-family:{F_SANS} !important;
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.18),
        0 1px 2px {T["s2"]},
        0 4px 14px {T["s3"]} !important;
    transition:all 0.18s cubic-bezier(0.22,1,0.36,1) !important;
}}
.stButton > button:hover {{
    transform:translateY(-2px) scale(1.01) !important;
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.22),
        0 4px 8px {T["s2"]},
        0 12px 32px {T["s3"]},
        0 0 28px {T["ag"]} !important;
}}
.stButton > button:active {{
    transform:translateY(0) scale(0.98) !important;
    box-shadow:inset 0 2px 4px {T["s3"]},0 1px 2px {T["s1"]} !important;
}}
.clear-btn .stButton > button {{
    background:linear-gradient(180deg,{T["hg"]},transparent 40%),{T["e3"]} !important;
    color:{T["t3"]} !important;
    border:1px solid {T["bn"]} !important;
    font-weight:500 !important; font-size:0.72rem !important; padding:7px 14px !important;
    box-shadow:inset 0 1px 0 {T["he"]},0 1px 3px {T["s1"]} !important;
}}
.clear-btn .stButton > button:hover {{
    border-color:{T["rd"]} !important; color:{T["rd"]} !important;
    background:{T["rb"]} !important;
    box-shadow:0 0 14px {T["rb"]} !important;
    transform:translateY(-1px) !important;
}}
[data-testid="stCheckbox"] label span {{ color:{T["t3"]} !important; font-size:0.76rem !important; }}

/* ==================================================================
   FINDINGS CARD — 3D floating card with bevel
================================================================== */
.fc {{
    background:linear-gradient(180deg,{T["hg"]} 0%,transparent 18%),{T["e3"]};
    border:1px solid {T["bs"]};
    border-top-color:{T["bn"]};
    border-bottom-color:{T["bs"]};
    border-radius:14px;
    padding:18px 22px;
    margin-top:10px;
    transform:perspective(1200px) rotateX(0.25deg);
    transform-origin:center bottom;
    box-shadow:
        inset 0 1px 0 {T["he"]},
        0 1px 3px {T["s1"]},
        0 4px 16px {T["s2"]},
        0 12px 40px {T["s2"]};
    animation:riseIn 0.4s cubic-bezier(0.22,1,0.36,1) backwards;
}}
.fc:hover {{
    transform:perspective(1200px) rotateX(0deg) translateY(-2px);
    box-shadow:
        inset 0 1px 0 {T["he"]},
        0 2px 6px {T["s1"]},
        0 8px 28px {T["s2"]},
        0 20px 56px {T["s3"]};
}}
.fc-title {{
    font-size:0.65rem; font-weight:700; color:{T["sc"]};
    text-transform:uppercase; letter-spacing:0.09em;
    margin-bottom:10px; padding-bottom:8px;
    border-bottom:1px solid {T["bs"]};
}}
.fc-body {{ font-size:0.87rem; line-height:1.78; color:{T["t2"]}; }}

/* ==================================================================
   IMPRESSION CARD — accent-tinted 3D with left edge
================================================================== */
.imp-card {{
    background:linear-gradient(90deg,{T["ib"]} 0%,transparent 55%),
        linear-gradient(180deg,{T["hg"]} 0%,transparent 18%),{T["e3"]};
    border:1px solid {T["id"]};
    border-left:4px solid {T["ac"]};
    border-radius:14px;
    padding:18px 22px; margin-top:10px;
    transform:perspective(1200px) rotateX(0.25deg);
    transform-origin:center bottom;
    box-shadow:
        inset 0 1px 0 {T["he"]},
        0 2px 8px {T["s2"]},
        0 0 28px {T["ag"]};
    animation:riseIn 0.4s cubic-bezier(0.22,1,0.36,1) backwards;
    animation-delay:0.06s;
}}
.imp-card:hover {{
    transform:perspective(1200px) rotateX(0deg) translateY(-2px);
    box-shadow:
        inset 0 1px 0 {T["he"]},
        0 4px 16px {T["s2"]},
        0 0 40px {T["ag"]};
}}
.imp-title {{
    font-size:0.65rem; font-weight:700; color:{T["ac"]};
    text-transform:uppercase; letter-spacing:0.09em;
    margin-bottom:10px; padding-bottom:8px;
    border-bottom:1px solid {T["id"]};
}}
.imp-body {{ font-size:0.9rem; line-height:1.78; color:{T["t1"]}; font-weight:600; }}

/* ==================================================================
   REGION TABLE
================================================================== */
.rt {{ width:100%; border-collapse:collapse; margin-top:8px; }}
.rt th {{
    text-align:left; color:{T["t4"]}; font-weight:600;
    font-size:0.6rem; text-transform:uppercase; letter-spacing:0.06em;
    padding:8px 10px; border-bottom:1px solid {T["bn"]};
}}
.rt td {{
    padding:8px 10px; color:{T["t3"]};
    font-family:{F_MONO}; font-size:0.69rem;
    border-bottom:1px solid {T["bs"]};
    transition:background 0.15s;
}}
.rt tr:hover td {{ background:{T["ag"]}; }}
.rdot {{
    display:inline-block; width:8px; height:8px;
    border-radius:2px; background:{T["rd"]};
    margin-right:8px; vertical-align:middle;
    box-shadow:0 0 6px {T["rb"]};
}}

/* ==================================================================
   RAW TOKENS / PERF BADGES / BANNERS / SPINNER / FOOTER
================================================================== */
.rtk {{
    background:{T["e1"]}; border:1px solid {T["bs"]};
    border-radius:7px; padding:10px 14px;
    font-family:{F_MONO}; font-size:0.65rem;
    color:{T["t4"]}; word-break:break-all; margin-top:8px;
    box-shadow:inset 0 2px 8px {T["si"]};
}}
.pb {{ display:flex; gap:7px; flex-wrap:wrap; margin-top:10px; }}
.pb-i {{
    display:inline-flex; align-items:center; gap:4px;
    padding:4px 10px;
    background:linear-gradient(180deg,{T["hg"]},transparent 50%),{T["e2"]};
    border:1px solid {T["bs"]};
    border-radius:7px;
    font-size:0.59rem; font-family:{F_MONO};
    color:{T["t4"]}; font-weight:500;
    box-shadow:inset 0 1px 0 {T["he"]},0 1px 3px {T["s1"]};
    backdrop-filter:blur(6px); -webkit-backdrop-filter:blur(6px);
}}
.pb-i:hover {{ border-color:{T["bn"]}; box-shadow:inset 0 1px 0 {T["he"]},0 2px 8px {T["s2"]}; }}
.pb-i span {{ color:{T["t3"]}; font-weight:600; }}

.ib {{
    background:linear-gradient(90deg,{T["ag"]},transparent 55%),{T["e3"]};
    border:1px solid {T["ar"]}; border-left:3px solid {T["ac"]};
    border-radius:12px; padding:14px 18px;
    color:{T["t3"]}; font-size:0.82rem; line-height:1.65;
    box-shadow:inset 0 1px 0 {T["he"]},0 2px 6px {T["s1"]};
    animation:riseIn 0.35s cubic-bezier(0.22,1,0.36,1) backwards;
    animation-delay:0.05s;
}}
.eb {{
    background:{T["rb"]}; border:1px solid rgba(248,113,113,0.18);
    border-left:3px solid {T["rd"]};
    border-radius:12px; padding:12px 16px; color:{T["rd"]}; font-size:0.8rem;
}}
.pi {{
    display:flex; align-items:center; gap:12px;
    padding:14px 18px;
    background:linear-gradient(90deg,{T["ag"]},transparent 50%),{T["e3"]};
    border:1px solid {T["ar"]};
    border-radius:12px; color:{T["ac"]}; font-size:0.82rem;
    box-shadow:0 0 24px {T["ag"]};
}}
.spin-ring {{
    width:16px; height:16px;
    border:2px solid {T["bn"]}; border-top-color:{T["ac"]};
    border-radius:50%; animation:spin 0.8s linear infinite;
    flex-shrink:0;
}}
.ft {{
    background:linear-gradient(180deg,{T["e3"]},{T["e2"]});
    border-top:1px solid {T["bs"]};
    padding:10px 28px;
    display:flex; align-items:center; justify-content:space-between;
    margin:14px -1rem -1rem -1rem;
    font-size:0.59rem; color:{T["t4"]}; font-family:{F_MONO};
    box-shadow:inset 0 1px 0 {T["he"]},0 -2px 10px {T["s2"]};
}}

/* ==================================================================
   DIVIDER / DOWNLOAD / MISC
================================================================== */
hr {{
    border:none !important; height:1px !important;
    background:linear-gradient(90deg,transparent,{T["bn"]},transparent) !important;
    margin:12px 0 !important;
}}
.stDownloadButton > button {{
    background:linear-gradient(180deg,{T["hg"]},transparent 40%),{T["e3"]} !important;
    color:{T["t3"]} !important; border:1px solid {T["bn"]} !important;
    font-size:0.72rem !important; padding:8px 16px !important;
    font-weight:600 !important; border-radius:8px !important;
    box-shadow:inset 0 1px 0 {T["he"]},0 1px 4px {T["s1"]} !important;
}}
.stDownloadButton > button:hover {{
    border-color:{T["ac"]} !important; color:{T["ac"]} !important;
    transform:translateY(-1px) !important;
    box-shadow:inset 0 1px 0 {T["he"]},0 2px 8px {T["s2"]},0 0 16px {T["ag"]} !important;
}}
[data-testid="stImage"] > div > div > p {{ display:none !important; }}
button[title="View fullscreen"] {{ display:none !important; }}
[data-testid="stHorizontalBlock"] {{ gap:8px !important; }}
.upl-help {{ font-size:0.73rem; color:{T["t4"]}; text-align:center; padding:6px 0 2px 0; font-weight:500; }}
.sp {{ height:8px; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helpers (unchanged)
# ---------------------------------------------------------------------------
def check_api_status():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def draw_clinical_boxes(image: Image.Image, bboxes: list, show_labels: bool = True) -> Image.Image:
    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    fill_color = (220, 70, 70, 30)
    outline_color = (220, 70, 70, 190)
    for i, box in enumerate(bboxes):
        xmin, ymin = box["xmin"], box["ymin"]
        xmax, ymax = box["xmax"], box["ymax"]
        draw.rectangle([xmin, ymin, xmax, ymax], fill=fill_color, outline=outline_color, width=2)
        cl = outline_color
        clen = min(14, (xmax - xmin) * 0.15, (ymax - ymin) * 0.15)
        for cx, cy, dx, dy in [
            (xmin, ymin, 1, 1), (xmax, ymin, -1, 1),
            (xmin, ymax, 1, -1), (xmax, ymax, -1, -1),
        ]:
            draw.line([(cx, cy), (cx + dx * clen, cy)], fill=cl, width=2)
            draw.line([(cx, cy), (cx, cy + dy * clen)], fill=cl, width=2)
        if show_labels:
            label = box.get("label", f"R{i + 1}")
            try:
                font = ImageFont.truetype("arial.ttf", 11)
            except Exception:
                font = ImageFont.load_default()
            bb = draw.textbbox((0, 0), label, font=font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            ly = max(ymin - th - 7, 0)
            draw.rectangle([xmin, ly, xmin + tw + 10, ly + th + 5], fill=(220, 70, 70, 210))
            draw.text((xmin + 5, ly + 2), label, fill=(255, 255, 255, 255), font=font)
    return Image.alpha_composite(img, overlay).convert("RGB")


def image_to_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def filmstrip_html(active_idx: int = 0) -> str:
    """Mini filmstrip simulating prior studies."""
    cells = ""
    for i in range(8):
        cls = "film-t active" if i == active_idx else "film-t"
        cells += f'<div class="{cls}"></div>'
    return f"""
    <div class="film-l">PRIOR STUDIES</div>
    <div class="film">{cells}</div>
    """


def dicom_overlay_html(w: int, h: int) -> str:
    """DICOM metadata overlay shown on viewer hover."""
    return f"""<div class="vw-dicom">KVP  120\nmAs  2.5\nSID  180cm\nBIT  14\nWL   40 / WW 400\nRES  {w}x{h}</div>"""


# ---------------------------------------------------------------------------
# API Health
# ---------------------------------------------------------------------------
health = check_api_status()
model_version = "MedGemma-4B"
device_info = "GPU"
sys_version = "3.1.0"
api_online = False

if health:
    api_online = True
    model_version = health.get("model_version", model_version)
    device_info = health.get("device", device_info)
    sys_version = health.get("system_version", sys_version)
    if device_info in ("N/A", "CPU"):
        device_info = "CPU"

# ---------------------------------------------------------------------------
# HEADER BAR
# ---------------------------------------------------------------------------
theme_label = "Light" if st.session_state.theme == "dark" else "Dark"

st.markdown(f"""
<div class="hdr">
    <div class="hdr-l">
        <div class="hdr-logo">Explain<b>My</b>Xray</div>
        <div class="hdr-sep"></div>
        <div class="hdr-sub">AI-Assisted Chest X-ray Interpretation</div>
    </div>
    <div class="hdr-r">
        <div class="bdg bdg-model">MedGemma-4B</div>
        <div class="bdg bdg-ver">v{sys_version}</div>
        <div class="bdg bdg-status"><span class="dot"></span>{"Ready" if api_online else "Connecting"}</div>
    </div>
</div>
""", unsafe_allow_html=True)

_, _, _, _, t_col = st.columns([1, 1, 1, 1, 0.4])
with t_col:
    if st.button(f"{theme_label} Mode", key="theme_toggle",
                 help="Switch between dark and light theme"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

# ---------------------------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1, 1], gap="small")

# =============================== LEFT PANEL ===============================
with col_left:
    st.markdown("""
    <div class="ph">
        <div class="ph-t">Patient Study / Input Radiograph</div>
        <div class="ph-tags"><span class="ph-tag">CR / DX</span><span class="ph-tag">PA View</span></div>
    </div><div class="sp"></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="upl-help">Upload a chest radiograph for AI analysis</div>',
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Chest Radiograph", type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
        help="Accepted: PNG, JPEG. Recommended resolution 512x512 or higher.",
    )

    if uploaded_file is not None:
        file_size = fmt_size(uploaded_file.size)
        raw_image = Image.open(uploaded_file)
        st.session_state.raw_image = raw_image
        img_w, img_h = raw_image.size

        if img_w >= 512 and img_h >= 512:
            q_label, q_color = "Adequate", T["gn"]
        elif img_w >= 256 and img_h >= 256:
            q_label, q_color = "Acceptable", T["am"]
        else:
            q_label, q_color = "Low", T["rd"]

        st.markdown(f"""
        <div class="mr">
            <div class="mi">FILE <span>{uploaded_file.name}</span></div>
            <div class="mi">SIZE <span>{file_size}</span></div>
            <div class="mi">RES <span>{img_w} x {img_h}</span></div>
            <div class="mi">QUALITY <span style="color:{q_color}">{q_label}</span></div>
        </div>
        """, unsafe_allow_html=True)

        # Viewer with DICOM overlay
        st.markdown(
            f'<div class="vw">'
            f'<div class="vw-tag">ORIGINAL &mdash; {img_w} x {img_h}</div>'
            f'{dicom_overlay_html(img_w, img_h)}',
            unsafe_allow_html=True,
        )
        st.image(raw_image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Filmstrip
        st.markdown(filmstrip_html(0), unsafe_allow_html=True)

        now = datetime.now()
        st.markdown(f"""
        <div class="mr" style="margin-top:2px;">
            <div class="mi">DATE <span>{now.strftime("%Y-%m-%d")}</span></div>
            <div class="mi">TIME <span>{now.strftime("%H:%M:%S")}</span></div>
            <div class="mi">MODALITY <span>CR / DX</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
        if st.button("Remove Image", key="clear_study"):
            st.session_state.analysis_result = None
            st.session_state.analysis_image = None
            st.session_state.raw_image = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="vw-empty">
            <div class="vw-empty-ico">&#x2B1C;</div>
            <div class="vw-empty-t">No radiograph loaded</div>
            <div class="vw-empty-s">
                Drag and drop a chest X-ray above, or click Browse<br>
                Accepted formats: PNG, JPEG
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================== RIGHT PANEL ===============================
with col_right:
    st.markdown("""
    <div class="ph">
        <div class="ph-t">AI Analysis / Annotated View</div>
        <div class="ph-tags"><span class="ph-tag">Spatial</span><span class="ph-tag">Findings</span></div>
    </div><div class="sp"></div>
    """, unsafe_allow_html=True)

    if uploaded_file is None:
        st.markdown("""
        <div class="ib">
            Upload a chest radiograph to begin. The AI will generate diagnostic findings
            with spatial localization of any detected abnormalities.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="vw-empty">
            <div class="vw-empty-ico">&#x2B1C;</div>
            <div class="vw-empty-t">Awaiting radiograph</div>
            <div class="vw-empty-s">Analysis results and annotated overlay will appear here</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        c1, c2 = st.columns([3, 1])
        with c1:
            run_btn = st.button("Analyze Radiograph", type="primary",
                                use_container_width=True, disabled=(not api_online))
        with c2:
            show_overlay = st.checkbox("Overlay", value=True, key="ov",
                                       help="Show or hide bounding box annotations")

        if not api_online:
            st.markdown("""
            <div class="eb">Unable to reach the analysis server. Please ensure the backend is running.</div>
            """, unsafe_allow_html=True)

        if run_btn:
            with st.spinner(""):
                st.markdown("""
                <div class="pi"><div class="spin-ring"></div>
                    Analyzing radiograph &mdash; generating findings and spatial localization...
                </div>
                """, unsafe_allow_html=True)
                try:
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    t0 = time.time()
                    resp = requests.post(f"{API_URL}/explain", files=files, timeout=120)
                    elapsed_local = (time.time() - t0) * 1000
                    if resp.status_code == 200:
                        result = resp.json()
                        result["elapsed_local_ms"] = round(elapsed_local, 1)
                        st.session_state.analysis_result = result
                        bboxes = result.get("bboxes", [])
                        if bboxes:
                            st.session_state.analysis_image = draw_clinical_boxes(
                                st.session_state.raw_image, bboxes, show_labels=True)
                        else:
                            st.session_state.analysis_image = None
                        st.rerun()
                    else:
                        st.markdown(f'<div class="eb">Server returned status {resp.status_code}.</div>',
                                    unsafe_allow_html=True)
                except requests.exceptions.Timeout:
                    st.markdown('<div class="eb">Request timed out.</div>', unsafe_allow_html=True)
                except requests.exceptions.ConnectionError:
                    st.markdown('<div class="eb">Connection refused. Backend may not be running.</div>',
                                unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="eb">Error: {e}</div>', unsafe_allow_html=True)

        # --- Results ---
        result = st.session_state.analysis_result
        if result:
            explanation = result.get("explanation", "")
            bboxes = result.get("bboxes", [])
            raw_tokens = result.get("raw_tokens", "")
            proc_time = result.get("processing_time_ms", 0)
            elapsed_local = result.get("elapsed_local_ms", 0)
            res_w = result.get("image_width", 0)
            res_h = result.get("image_height", 0)
            r_device = result.get("device", device_info)
            r_model = result.get("model_version", model_version)

            # Annotated viewer with DICOM overlay
            dicom = dicom_overlay_html(res_w, res_h)
            st.markdown(f'<div class="vw">{dicom}', unsafe_allow_html=True)
            if st.session_state.analysis_image is not None and show_overlay:
                st.markdown(f'<div class="vw-tag">ANNOTATED &mdash; {len(bboxes)} region(s)</div>',
                            unsafe_allow_html=True)
                st.image(st.session_state.analysis_image, use_container_width=True)
            elif st.session_state.raw_image is not None:
                st.markdown('<div class="vw-tag">ORIGINAL &mdash; overlay off</div>',
                            unsafe_allow_html=True)
                st.image(st.session_state.raw_image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Filmstrip
            st.markdown(filmstrip_html(0), unsafe_allow_html=True)

            # Performance badges
            st.markdown(f"""
            <div class="pb">
                <div class="pb-i">MODEL <span>{r_model}</span></div>
                <div class="pb-i">DEVICE <span>{r_device}</span></div>
                <div class="pb-i">INFERENCE <span>{proc_time:.0f} ms</span></div>
                <div class="pb-i">ROUND-TRIP <span>{elapsed_local:.0f} ms</span></div>
                <div class="pb-i">INPUT <span>{res_w}x{res_h}</span></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="fc">
                <div class="fc-title">Findings</div>
                <div class="fc-body">{explanation}</div>
            </div>
            """, unsafe_allow_html=True)

            impression = explanation.split(".")[0] + "." if explanation else "No significant findings."
            st.markdown(f"""
            <div class="imp-card">
                <div class="imp-title">Impression</div>
                <div class="imp-body">{impression}</div>
            </div>
            """, unsafe_allow_html=True)

            if bboxes:
                rows = ""
                for i, box in enumerate(bboxes):
                    label = box.get("label", f"Region {i+1}")
                    x1, y1 = int(box["xmin"]), int(box["ymin"])
                    x2, y2 = int(box["xmax"]), int(box["ymax"])
                    rows += f'<tr><td><span class="rdot"></span>{label}</td>'
                    rows += f'<td>({x1}, {y1}) &rarr; ({x2}, {y2})</td>'
                    rows += f'<td>{x2-x1} x {y2-y1} px</td></tr>'
                st.markdown(f"""
                <div class="fc">
                    <div class="fc-title">Regions Detected ({len(bboxes)})</div>
                    <table class="rt">
                        <thead><tr><th>Region</th><th>Coordinates</th><th>Size</th></tr></thead>
                        <tbody>{rows}</tbody>
                    </table>
                </div>
                """, unsafe_allow_html=True)

            if raw_tokens:
                st.markdown(f"""
                <div class="fc">
                    <div class="fc-title">Spatial Geometry</div>
                    <div class="rtk">{raw_tokens}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="sp"></div>', unsafe_allow_html=True)
            dl1, dl2 = st.columns(2)
            with dl1:
                if st.session_state.analysis_image is not None:
                    st.download_button(
                        "Download Annotated Image",
                        data=image_to_bytes(st.session_state.analysis_image),
                        file_name="explainmyxray_annotated.png", mime="image/png",
                    )
            with dl2:
                report = f"ExplainMyXray Analysis Report\n{'=' * 44}\n"
                report += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                report += f"Model: {r_model}\nDevice: {r_device}\n"
                report += f"Inference: {proc_time:.0f} ms\nResolution: {res_w}x{res_h}\n\n"
                report += f"FINDINGS\n{'-' * 44}\n{explanation}\n\n"
                report += f"IMPRESSION\n{'-' * 44}\n{impression}\n\n"
                report += f"REGIONS: {len(bboxes)}\n{'-' * 44}\n"
                for i, b in enumerate(bboxes):
                    report += f"  {b.get('label', f'Region {i+1}')}: ({int(b['xmin'])},{int(b['ymin'])}) -> ({int(b['xmax'])},{int(b['ymax'])})\n"
                report += f"\nSPATIAL TOKENS\n{'-' * 44}\n{raw_tokens}\n"
                report += f"\n{'=' * 44}\n"
                report += "Generated by ExplainMyXray. For research and educational use.\n"
                st.download_button(
                    "Download Report", data=report,
                    file_name="explainmyxray_report.txt", mime="text/plain",
                )

        elif uploaded_file is not None:
            st.markdown("""
            <div class="vw-empty">
                <div class="vw-empty-ico">&#x2B1C;</div>
                <div class="vw-empty-t">Ready for analysis</div>
                <div class="vw-empty-s">
                    Click "Analyze Radiograph" to generate diagnostic findings<br>
                    with spatial localization of abnormalities
                </div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="ft">
    <div>ExplainMyXray &middot; {model_version} &middot; Kaggle Medical AI Challenge</div>
    <div>For research and educational use &middot; Not a substitute for professional radiological interpretation</div>
</div>
""", unsafe_allow_html=True)
