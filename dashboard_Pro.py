import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import cmath
import os
import joblib

# AI / ML Imports
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Try importing Advanced AI Libraries; handle gracefully
try:
    from prophet import Prophet
    from prophet.serialize import model_to_json, model_from_json
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

# ----------------------------------------------------------
# 1. CONFIGURATION & CYBERPUNK STYLING
# ----------------------------------------------------------
st.set_page_config(
    page_title="NEON GRID CONTROL v30.0 (Physics Upgrade)", 
    layout="wide", 
    page_icon="☀️",
    initial_sidebar_state="expanded"
)

# CYBERPUNK CSS INJECTION (GLOBAL)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto+Mono:wght@400;700&display=swap');

    /* MAIN BACKGROUND & FONT */
    .stApp {
        background-color: #050505;
        background-image: radial-gradient(#1a1a1a 1px, transparent 1px);
        background-size: 40px 40px;
        font-family: 'Roboto Mono', monospace;
        color: #e0e0e0;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Orbitron', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #00f3ff !important;
        text-shadow: 0 0 10px rgba(0, 243, 255, 0.5);
    }
    
    /* CARD STYLING FOR METRICS */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(10,10,10,0.9) 0%, rgba(20,20,20,0.9) 100%);
        border: 1px solid #00f3ff;
        border-left: 5px solid #00f3ff;
        padding: 15px;
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.2);
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 25px rgba(0, 243, 255, 0.4);
    }
    
    /* CUSTOM METRIC LABELS & VALUES */
    [data-testid="stMetricLabel"] {
        color: #ff00ff !important;
        font-size: 12px;
        font-family: 'Orbitron', sans-serif;
    }
    [data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff !important;
        text-shadow: 0 0 10px #ffffff;
    }

    /* BUTTONS */
    div[data-testid="stButton"] button {
        border-radius: 0px;
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
        border: 1px solid #00f3ff;
        background-color: transparent;
        color: #00f3ff;
        transition: all 0.3s ease;
    }
    div[data-testid="stButton"] button:hover {
        background-color: #00f3ff;
        color: #000;
        box-shadow: 0 0 20px #00f3ff;
    }
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(90deg, #ff0055, #ff00aa);
        border: none;
        color: white;
        box-shadow: 0 0 15px rgba(255, 0, 85, 0.5);
    }
    
    /* INPUT FIELDS */
    .stTextInput input, .stSelectbox div[data-baseweb="select"], .stSlider div {
        color: #00f3ff !important;
        font-family: 'Roboto Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# 2. PHYSICS CONSTANTS & CONFIG
# ----------------------------------------------------------
# CSV FILES CONFIG
CSV_PATH = "Historical_Data\Total_P&Q.csv"
FEEDER_A_P = "Historical_Data\FeederA_P.csv"
FEEDER_A_Q = "Historical_Data\FeederA_Q.csv"
SOLAR_DATA_FILE = "Historical_Data\Solardata.csv"

# AI MODEL PATHS
PROPHET_MODEL_FILE = "prophet_model.json"
LSTM_MODEL_FILE = "lstm_model.h5"
SCALER_FILE = "scaler.pkl"

# ELECTRICAL PHYSICS CONSTANTS
OMEGA = 2 * np.pi * 50
A_OPERATOR = cmath.rect(1.0, np.deg2rad(120))
A2_OPERATOR = A_OPERATOR * A_OPERATOR
WAVE_TIME = np.linspace(0, 0.02, 100)

# SYSTEM PARAMETERS
NOMINAL_FREQ = 50.0
TRANSFORMER_RATING_KVA = 10000.0 
TRANSFORMER_TAU = 20.0  
SOURCE_IMPEDANCE = 0.01 + 0.1j  
LINE_IMPEDANCE_PER_UNIT_DIST = 0.005 + 0.002j 

# SWING EQUATION CONSTANTS
INERTIA_H = 5.0 
DAMPING_D = 1.0 
SYSTEM_BASE_MVA = 10.0 

# FAULT IMPEDANCE LIBRARY
FAULT_LIBRARY = {
    "L-G (Line-to-Ground)":   {"Zf": 0.0, "type": "LG", "color": "#ff9100", "icon": "⚡"},
    "L-L (Line-to-Line)":     {"Zf": 0.01, "type": "LL", "color": "#ff5500", "icon": "🔥"},
    "L-L-G (2-Line-Ground)":  {"Zf": 0.0, "type": "LLG", "color": "#ff0000", "icon": "💥"},
    "L-L-L (3-Phase Bolted)": {"Zf": 0.0, "type": "LLL", "color": "#ff0055", "icon": "☠️"}
}

# --- SOLAR PV CONFIGURATION ---
SOLAR_BUS_CONFIG = {
    "bus1005": 100.0, "bus1010": 150.0, "bus2008": 80.0, "bus2015": 120.0,
    "bus2025": 200.0, "bus3006": 50.0, "bus3020": 100.0, "bus3035": 150.0,
    "bus3050": 90.0, "bus3100": 75.0
}

# --- BESS (BATTERY) CONFIGURATION ---
BESS_CAPACITY_KWH = 500.0
BESS_MAX_POWER = 100.0 

# ----------------------------------------------------------
# 3. TOPOLOGY & DATA PARSING
# ----------------------------------------------------------
def parse_bus_coords(dss_content):
    coords = {}
    lines = dss_content.split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('//') or line.startswith('['): continue
        parts = line.replace(',', ' ').split()
        if len(parts) >= 3:
            try:
                b_name = parts[0].lower()
                x = float(parts[1])
                y = float(parts[2])
                coords[b_name] = [x, y]
            except: pass
    return coords

BUS_COORDS_RAW = """
bus1, 10.52, 13.6
bus1001, 10.19, 13.04
bus1002, 8.9, 14.11
bus1003, 8.9, 13.36
bus1004, 7.77, 14.11
bus1005, 6.4, 14.11
bus1006, 5.15, 14.11
bus1007, 5.15, 12.98
bus1008, 3.71, 14.11
bus1009, 2.4, 14.11
bus1010, 1.4, 14.11
bus1011, 2.4, 12.98
bus1012, 2.4, 12.11
bus1013, 2.4, 11.23
bus1014, 3.85, 11.23
bus1015, 4.85, 11.23
bus1016, 2.4, 10.48
bus1017, 2.4, 9.73
bus2001, 10.52, 12.23
bus2002, 9.56, 7.58
bus2003, 8.3, 7.58
bus2004, 7.07, 7.58
bus2005, 5.77, 7.58
bus2006, 4.54, 7.58
bus2007, 4.55, 8.25
bus2008, 3.66, 8.25
bus2009, 4.55, 8.93
bus2010, 3.68, 7.58
bus2011, 2.71, 7.58
bus2012, 2.32, 7.1
bus2013, 2.31, 6.44
bus2014, 2.31, 5.68
bus2015, 1.41, 5.68
bus2016, 2.31, 4.97
bus2017, 2.31, 4.24
bus2018, 2.31, 3.51
bus2019, 3.23, 6.45
bus2020, 3.23, 5.85
bus2021, 4.04, 6.44
bus2022, 4.04, 5.72
bus2023, 4.04, 4.99
bus2024, 4.04, 4.26
bus2025, 4.04, 3.53
bus2026, 4.8, 6.45
bus2027, 5.59, 6.45
bus2028, 5.59, 5.88
bus2029, 5.58, 5.31
bus2030, 5.58, 4.73
bus2031, 5.58, 4.16
bus2032, 6.61, 6.45
bus2033, 7.6, 6.45
bus2034, 7.6, 5.79
bus2035, 8.46, 6.44
bus2036, 9.3, 6.45
bus2037, 10.21, 6.45
bus2038, 10.21, 5.84
bus2039, 10.21, 4.98
bus2040, 11.07, 4.98
bus2041, 11.88, 4.98
bus2042, 10.21, 4.39
bus2043, 10.21, 3.63
bus2044, 8.76, 3.63
bus2045, 8.76, 4.29
bus2046, 8.76, 5
bus2047, 7.84, 4.29
bus2048, 6.92, 4.29
bus2049, 7.87, 3.63
bus2050, 7.01, 3.63
bus2051, 6.16, 3.62
bus2052, 6.16, 2.81
bus2053, 8.76, 2.1
bus2054, 7.78, 2.1
bus2055, 6.84, 2.1
bus2056, 5.86, 2.1
bus2057, 8.76, 1.4
bus2058, 9.9, 1.4
bus2059, 7.72, 1.4
bus2060, 6.76, 1.4
bus3001, 10.83, 13.02
bus3002, 12.5, 12.16
bus3003, 12.5, 12.85
bus3004, 12.5, 13.43
bus3005, 13.36, 12.85
bus3006, 13.36, 13.91
bus3007, 13.36, 14.71
bus3008, 14.62, 11.41
bus3009, 14.61, 12.04
bus3010, 14.62, 12.7
bus3011, 14.61, 13.36
bus3012, 14.62, 14.02
bus3013, 14.61, 10.77
bus3014, 14.62, 10.06
bus3015, 15.79, 11.41
bus3016, 15.8, 10.75
bus3017, 15.8, 10.1
bus3018, 15.8, 12.07
bus3019, 15.8, 12.72
bus3020, 15.8, 13.38
bus3021, 15.8, 14.03
bus3022, 16.99, 11.41
bus3023, 16.99, 10.76
bus3024, 16.99, 10.12
bus3025, 16.99, 9.47
bus3026, 16.99, 12.06
bus3027, 16.99, 12.72
bus3028, 16.99, 13.38
bus3029, 16.99, 14.03
bus3030, 18.26, 11.41
bus3031, 18.26, 10.91
bus3032, 18.26, 10.4
bus3033, 18.26, 9.9
bus3034, 18.26, 9.39
bus3035, 18.25, 12.05
bus3036, 18.27, 12.68
bus3037, 18.25, 13.32
bus3038, 18.25, 13.95
bus3039, 18.26, 14.59
bus3040, 19.51, 11.41
bus3041, 19.51, 10.87
bus3042, 19.51, 10.33
bus3043, 19.51, 9.79
bus3044, 19.51, 12.03
bus3045, 19.51, 12.63
bus3046, 20.88, 11.41
bus3047, 20.88, 12.16
bus3048, 21.17, 10.92
bus3049, 22, 10.92
bus3050, 22.82, 10.92
bus3051, 23.64, 10.92
bus3052, 24.46, 10.91
bus3053, 19.51, 14.79
bus3054, 19.51, 14.19
bus3055, 21.01, 14.79
bus3056, 21.5, 15.2
bus3057, 22.22, 15.2
bus3058, 22.93, 15.2
bus3059, 23.65, 15.2
bus3060, 24.37, 15.2
bus3061, 25.09, 15.2
bus3062, 21.53, 14.13
bus3063, 22.24, 14.13
bus3064, 22.95, 14.14
bus3065, 23.66, 14.13
bus3066, 24.38, 14.13
bus3067, 25.09, 14.13
bus3068, 18.26, 8.82
bus3069, 16.65, 8.82
bus3070, 16.65, 8.27
bus3071, 16.65, 7.71
bus3072, 16.65, 7.16
bus3073, 15.76, 8.82
bus3074, 14.98, 8.82
bus3075, 19.05, 8.82
bus3076, 19.97, 8.82
bus3077, 20.73, 8.82
bus3078, 21.49, 8.82
bus3079, 19.97, 8.07
bus3080, 19.97, 7.42
bus3081, 19.21, 7.42
bus3082, 19.97, 6.79
bus3083, 19.1, 6.79
bus3084, 18.23, 6.79
bus3085, 17.36, 6.79
bus3086, 16.49, 6.79
bus3087, 15.62, 6.79
bus3088, 14.75, 6.79
bus3089, 13.88, 6.79
bus3090, 13.01, 6.79
bus3091, 12.14, 6.79
bus3092, 22.05, 6.79
bus3093, 22.07, 7.3
bus3094, 22.04, 7.81
bus3095, 22.05, 8.32
bus3096, 22.05, 8.83
bus3097, 22.05, 9.34
bus3098, 22.05, 6.13
bus3099, 22.05, 5.56
bus3100, 23.09, 6.8
bus3101, 23.86, 6.8
bus3102, 24.53, 6.8
bus3103, 25.12, 6.8
bus3104, 23.09, 7.52
bus3105, 23.08, 8.08
bus3106, 23.09, 8.67
bus3107, 19.98, 5.54
bus3108, 18.97, 5.54
bus3109, 18.07, 5.54
bus3110, 16.97, 5.54
bus3111, 15.97, 5.54
bus3112, 15.07, 5.54
bus3113, 18.63, 4.76
bus3114, 18.63, 4.15
bus3115, 18.63, 3.53
bus3116, 17.55, 4.75
bus3117, 16.55, 4.76
bus3118, 21.36, 4.03
bus3119, 22.77, 4.03
bus3120, 22.77, 4.69
bus3121, 22.78, 5.34
bus3122, 22.78, 5.96
bus3123, 23.64, 4.04
bus3124, 24.42, 4.04
bus3125, 25.17, 4.04
bus3126, 25.97, 4.04
bus3127, 25.97, 4.58
bus3128, 25.97, 5.12
bus3129, 25.97, 5.66
bus3130, 25.97, 6.2
bus3131, 25.97, 6.75
bus3132, 21.35, 3.22
bus3133, 21.36, 2.41
bus3134, 22.53, 2.41
bus3135, 23.57, 2.41
bus3136, 24.53, 2.41
bus3137, 21.37, 1.79
bus3138, 21.37, 1.16
bus3139, 21.36, 0.53
bus3140, 17.61, 2.5
bus3141, 18.42, 2.48
bus3142, 19.23, 2.48
bus3143, 20.03, 2.49
bus3144, 18.31, 1.11
bus3145, 18.99, 1.11
bus3146, 19.67, 1.11
bus3147, 20.35, 1.11
bus3148, 16.87, 2.48
bus3149, 16.13, 2.48
bus3150, 15.4, 2.48
bus3151, 14.66, 2.48
bus3152, 13.92, 2.48
bus3153, 13.18, 2.48
bus3154, 12.44, 2.49
bus3155, 11.7, 2.49
bus3156, 17.64, 1.11
bus3157, 16.9, 1.11
bus3158, 16.17, 1.11
bus3159, 15.43, 1.11
bus3160, 14.7, 1.11
bus3161, 13.97, 1.11
bus3162, 13.23, 1.11
"""

EDGE_LIST_RAW = [
    ("bus1", "bus1001"), ("bus1001", "bus1002"), ("bus1002", "bus1003"), ("bus1002", "bus1004"),
    ("bus1004", "bus1005"), ("bus1005", "bus1006"), ("bus1006", "bus1007"), ("bus1006", "bus1008"),
    ("bus1008", "bus1009"), ("bus1009", "bus1010"), ("bus1009", "bus1011"), ("bus1011", "bus1012"),
    ("bus1012", "bus1013"), ("bus1013", "bus1014"), ("bus1014", "bus1015"), ("bus1013", "bus1016"),
    ("bus1016", "bus1017"), 
    ("bus1", "bus2001"), ("bus2001", "bus2002"), ("bus2002", "bus2003"), ("bus2003", "bus2004"),
    ("bus2004", "bus2005"), ("bus2005", "bus2006"), ("bus2006", "bus2007"), ("bus2007", "bus2008"),
    ("bus2007", "bus2009"), ("bus2006", "bus2010"), ("bus2010", "bus2011"), ("bus2011", "bus2012"),
    ("bus2013", "bus2014"), ("bus2014", "bus2015"), ("bus2014", "bus2016"), ("bus2016", "bus2017"),
    ("bus2017", "bus2018"), ("bus2013", "bus2019"), ("bus2019", "bus2020"), ("bus2019", "bus2021"),
    ("bus2021", "bus2022"), ("bus2022", "bus2023"), ("bus2023", "bus2024"), ("bus2024", "bus2025"),
    ("bus2026", "bus2027"), ("bus2027", "bus2028"), ("bus2028", "bus2029"), ("bus2029", "bus2030"),
    ("bus2030", "bus2031"), ("bus2027", "bus2032"), ("bus2032", "bus2033"), ("bus2033", "bus2034"),
    ("bus2033", "bus2035"), ("bus2035", "bus2036"), ("bus2036", "bus2037"), ("bus2037", "bus2038"),
    ("bus2038", "bus2039"), ("bus2039", "bus2040"), ("bus2040", "bus2041"), ("bus2039", "bus2042"),
    ("bus2042", "bus2043"), ("bus2043", "bus2044"), ("bus2044", "bus2045"), ("bus2045", "bus2046"),
    ("bus2045", "bus2047"), ("bus2047", "bus2048"), ("bus2044", "bus2049"), ("bus2049", "bus2050"),
    ("bus2050", "bus2051"), ("bus2051", "bus2052"), ("bus2044", "bus2053"), ("bus2053", "bus2054"),
    ("bus2054", "bus2055"), ("bus2055", "bus2056"), ("bus2053", "bus2057"), ("bus2057", "bus2058"),
    ("bus2057", "bus2059"), ("bus2059", "bus2060"),
    ("bus1", "bus3001"), ("bus3001", "bus3003"), ("bus3003", "bus3002"), ("bus3003", "bus3004"),
    ("bus3003", "bus3005"), ("bus3005", "bus3006"), ("bus3006", "bus3007"), ("bus3005", "bus3008"),
    ("bus3008", "bus3009"), ("bus3009", "bus3010"), ("bus3010", "bus3011"), ("bus3011", "bus3012"),
    ("bus3008", "bus3013"), ("bus3013", "bus3014"), ("bus3008", "bus3015"), ("bus3015", "bus3016"),
    ("bus3016", "bus3017"), ("bus3015", "bus3018"), ("bus3018", "bus3019"), ("bus3019", "bus3020"),
    ("bus3020", "bus3021"), ("bus3015", "bus3022"), ("bus3022", "bus3023"), ("bus3023", "bus3024"),
    ("bus3024", "bus3025"), ("bus3022", "bus3026"), ("bus3026", "bus3027"), ("bus3027", "bus3028"),
    ("bus3028", "bus3029"), ("bus3022", "bus3030"), ("bus3030", "bus3035"), ("bus3035", "bus3036"),
    ("bus3036", "bus3037"), ("bus3037", "bus3038"), ("bus3038", "bus3039"), ("bus3039", "bus3053"),
    ("bus3053", "bus3054"), ("bus3053", "bus3055"), ("bus3055", "bus3056"), ("bus3056", "bus3057"),
    ("bus3057", "bus3058"), ("bus3058", "bus3059"), ("bus3059", "bus3060"), ("bus3060", "bus3061"),
    ("bus3055", "bus3062"), ("bus3062", "bus3063"), ("bus3063", "bus3064"), ("bus3064", "bus3065"),
    ("bus3065", "bus3066"), ("bus3066", "bus3067"), ("bus3030", "bus3040"), ("bus3040", "bus3044"),
    ("bus3044", "bus3045"), ("bus3040", "bus3041"), ("bus3041", "bus3042"), ("bus3042", "bus3043"),
    ("bus3040", "bus3046"), ("bus3046", "bus3047"), ("bus3046", "bus3048"), ("bus3048", "bus3049"),
    ("bus3049", "bus3050"), ("bus3050", "bus3051"), ("bus3051", "bus3052"), ("bus3030", "bus3031"),
    ("bus3031", "bus3032"), ("bus3032", "bus3033"), ("bus3033", "bus3034"), ("bus3034", "bus3068"),
    ("bus3068", "bus3069"), ("bus3069", "bus3070"), ("bus3070", "bus3071"), ("bus3071", "bus3072"),
    ("bus3069", "bus3073"), ("bus3073", "bus3074"), ("bus3068", "bus3075"), ("bus3076", "bus3077"),
    ("bus3077", "bus3078"), ("bus3076", "bus3079"), ("bus3079", "bus3080"), ("bus3080", "bus3081"),
    ("bus3080", "bus3082"), ("bus3082", "bus3083"), ("bus3083", "bus3084"), ("bus3084", "bus3085"),
    ("bus3085", "bus3086"), ("bus3086", "bus3087"), ("bus3087", "bus3088"), ("bus3088", "bus3089"),
    ("bus3089", "bus3090"), ("bus3090", "bus3091"), ("bus3082", "bus3092"), ("bus3092", "bus3093"),
    ("bus3093", "bus3094"), ("bus3094", "bus3095"), ("bus3095", "bus3096"), ("bus3096", "bus3097"),
    ("bus3092", "bus3098"), ("bus3098", "bus3099"), ("bus3092", "bus3100"), ("bus3100", "bus3101"),
    ("bus3101", "bus3102"), ("bus3102", "bus3103"), ("bus3100", "bus3104"), ("bus3104", "bus3105"),
    ("bus3105", "bus3106"), ("bus3082", "bus3107"), ("bus3107", "bus3108"), ("bus3108", "bus3109"),
    ("bus3109", "bus3110"), ("bus3110", "bus3111"), ("bus3111", "bus3112"), ("bus3107", "bus3113"),
    ("bus3113", "bus3114"), ("bus3114", "bus3115"), ("bus3113", "bus3116"), ("bus3116", "bus3117"),
    ("bus3107", "bus3118"), ("bus3118", "bus3119"), ("bus3119", "bus3120"), ("bus3120", "bus3121"),
    ("bus3121", "bus3122"), ("bus3119", "bus3123"), ("bus3123", "bus3124"), ("bus3124", "bus3125"),
    ("bus3125", "bus3126"), ("bus3126", "bus3127"), ("bus3127", "bus3128"), ("bus3128", "bus3129"),
    ("bus3129", "bus3130"), ("bus3130", "bus3131"), ("bus3118", "bus3132"), ("bus3132", "bus3133"),
    ("bus3133", "bus3134"), ("bus3134", "bus3135"), ("bus3135", "bus3136"), ("bus3133", "bus3137"),
    ("bus3137", "bus3138"), ("bus3138", "bus3139"), ("bus3107", "bus3140"), ("bus3140", "bus3141"),
    ("bus3141", "bus3142"), ("bus3142", "bus3143"), ("bus3140", "bus3148"), ("bus3148", "bus3149"),
    ("bus3149", "bus3150"), ("bus3150", "bus3151"), ("bus3151", "bus3152"), ("bus3152", "bus3153"),
    ("bus3153", "bus3154"), ("bus3154", "bus3155"), ("bus3140", "bus3156"), ("bus3156", "bus3144"),
    ("bus3144", "bus3145"), ("bus3145", "bus3146"), ("bus3146", "bus3147"), ("bus3156", "bus3157"),
    ("bus3157", "bus3158"), ("bus3158", "bus3159"), ("bus3159", "bus3160"), ("bus3160", "bus3161"),
    ("bus3161", "bus3162")
]

TRANSFORMER_NODES = ["bus1003", "bus1004", "bus1005", "bus1006", "bus1007", "bus1008", 
                     "bus1009", "bus1010", "bus1011", "bus1012", "bus1013", "bus1014",
                     "bus2002", "bus2003", "bus2005", "bus2008", "bus2009", "bus2010"]

bus_dict = parse_bus_coords(BUS_COORDS_RAW)
bus_list = list(bus_dict.keys())

def get_distance_map():
    d_map = {}
    source_x, source_y = bus_dict['bus1']
    for b_name, (bx, by) in bus_dict.items():
        dist = np.sqrt((bx - source_x)**2 + (by - source_y)**2)
        d_map[b_name] = dist
    return d_map

dist_map = get_distance_map()

# ----------------------------------------------------------
# 4. SESSION STATE & PHYSICS VARS
# ----------------------------------------------------------
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "idx" not in st.session_state: st.session_state.idx = 0
if "run_simulation" not in st.session_state: st.session_state.run_simulation = False
if "speed" not in st.session_state: st.session_state.speed = 1.0

# SWING EQUATION STATE VECTORS
if "grid_freq" not in st.session_state: st.session_state.grid_freq = 50.0
if "rotor_angle" not in st.session_state: st.session_state.rotor_angle = 0.0 
if "mech_power" not in st.session_state: st.session_state.mech_power = 5000.0 

if "transformer_thermal" not in st.session_state: st.session_state.transformer_thermal = 40.0 

if "fault_active" not in st.session_state: st.session_state.fault_active = False
if "fault_bus" not in st.session_state: st.session_state.fault_bus = ""
if "fault_type" not in st.session_state: st.session_state.fault_type = "L-G (Line-to-Ground)"

if "recloser_state" not in st.session_state: st.session_state.recloser_state = "CLOSED"
if "recloser_timer" not in st.session_state: st.session_state.recloser_timer = 0.0

if "relay_trip" not in st.session_state: st.session_state.relay_trip = False
if "relay_accumulator" not in st.session_state: st.session_state.relay_accumulator = 0.0

if "thd_mode" not in st.session_state: st.session_state.thd_mode = False 
if "filter_mode" not in st.session_state: st.session_state.filter_mode = False

if "capacitor_bank_kvAr" not in st.session_state: st.session_state.capacitor_bank_kvAr = 0.0
if "apfc_auto_mode" not in st.session_state: st.session_state.apfc_auto_mode = False

if "tap_position" not in st.session_state: st.session_state.tap_position = 1.0 
if "auto_tap_mode" not in st.session_state: st.session_state.auto_tap_mode = False

# NEW: Cyber Attack
if "fdi_attack" not in st.session_state: st.session_state.fdi_attack = False

# NEW: Cloud Shading & BESS
if "cloud_shading" not in st.session_state: st.session_state.cloud_shading = False
if "bess_soc" not in st.session_state: st.session_state.bess_soc = 50.0 

if "room_temp" not in st.session_state: st.session_state.room_temp = 28.0
if "hvac_on" not in st.session_state: st.session_state.hvac_on = False
if "hvac_setpoint" not in st.session_state: st.session_state.hvac_setpoint = 24.0
if "hvac_load_kw" not in st.session_state: st.session_state.hvac_load_kw = 0.0

if "prev_p" not in st.session_state: st.session_state.prev_p = 0.0
if "prev_q" not in st.session_state: st.session_state.prev_q = 0.0
if "prev_feeder_p" not in st.session_state: st.session_state.prev_feeder_p = 0.0

# --- HISTORICAL BUFFERS FOR PLOTS ---
if "history_tap" not in st.session_state: st.session_state.history_tap = [1.0] * 50
if "history_cap" not in st.session_state: st.session_state.history_cap = [0.0] * 24

# --- SE HISTORY BUFFERS ---
if "history_se_meas" not in st.session_state: st.session_state.history_se_meas = [1.0] * 50
if "history_se_est" not in st.session_state: st.session_state.history_se_est = [1.0] * 50
if "history_se_j" not in st.session_state: st.session_state.history_se_j = [0.0] * 50

# --- NEW: SOLAR PLOTTING BUFFERS ---
if "solar_p_history" not in st.session_state: st.session_state.solar_p_history = [0.0] * 50
if "solar_q_history" not in st.session_state: st.session_state.solar_q_history = [0.0] * 50
if "solar_v_history" not in st.session_state: st.session_state.solar_v_history = [1.0] * 50
if "solar_irr_history" not in st.session_state: st.session_state.solar_irr_history = [0.0] * 50
if "solar_temp_history" not in st.session_state: st.session_state.solar_temp_history = [25.0] * 50

if "audit_log" not in st.session_state: 
    st.session_state.audit_log = pd.DataFrame(columns=["Timestamp", "Event", "Type", "Details"])

def log_event(event, e_type, details):
    new_entry = pd.DataFrame([{
        "Timestamp": datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3],
        "Event": event,
        "Type": e_type,
        "Details": details
    }])
    st.session_state.audit_log = pd.concat([new_entry, st.session_state.audit_log], ignore_index=True)

# ----------------------------------------------------------
# 5. DATA LOADING & PHYSICS HELPERS
# ----------------------------------------------------------
@st.cache_data
def load_data(path):
    try:
        return pd.read_csv(path)
    except:
        dates = pd.date_range('2025-01-01', periods=8760, freq='H')
        df = pd.DataFrame({'Total_Active_Power': np.random.normal(5000, 500, 8760),
                           'Total_Reac_Power': np.random.normal(2000, 200, 8760)})
        for b in bus_dict.keys():
            df[b] = np.random.normal(50, 10, 8760)
        return df

@st.cache_data
def load_solar_profile():
    try:
        df = pd.read_csv(SOLAR_DATA_FILE)
        # Normalize the first column to use as a 0-1 generation profile
        vals = df.iloc[:, 0].values
        max_val = np.max(vals)
        if max_val > 0:
            return vals / max_val
        return vals 
    except:
        return None

df_raw = load_data(CSV_PATH)
df_fa_p = load_data(FEEDER_A_P)
df_fa_q = load_data(FEEDER_A_Q)
solar_profile = load_solar_profile()

if not st.session_state.fault_bus and len(bus_list) > 0: 
    st.session_state.fault_bus = bus_list[0]

def get_operating_state(voltage_pu, current_pu, fault_active, relay_trip):
    if relay_trip or st.session_state.recloser_state in ["TRIPPED", "WAITING", "LOCKOUT"]: return "BLACKOUT", "#ff0055", "BREAKER OPEN - NO VOLTAGE"
    if fault_active or voltage_pu < 0.90 or voltage_pu > 1.10: return "EMERGENCY", "#ff9100", "LIMITS EXCEEDED - HAZARD"
    if voltage_pu < 0.96 or current_pu > 1.2: return "WARNING", "#ffee00", "INSTABILITY DETECTED"
    return "NOMINAL", "#00f3ff", "SYSTEM OPTIMAL"

# ==========================================================
#  AI ENGINE
# ==========================================================
def prepare_lstm_data(series, lookback=24):
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i:i+lookback])
        y.append(series[i+lookback])
    return np.array(X), np.array(y)

@st.cache_resource(show_spinner=False)
def load_or_train_models(df_values):
    split_idx = int(len(df_values) * 0.8)
    train_data = df_values[:split_idx]
    test_data = df_values[split_idx:]
    metrics = {"prophet_rmse": 0, "prophet_mae": 0, "lstm_rmse": 0, "lstm_mae": 0}
    
    prophet_model = None
    forecast_full = None
    if PROPHET_AVAILABLE:
        if os.path.exists(PROPHET_MODEL_FILE):
            try:
                with open(PROPHET_MODEL_FILE, 'r') as fin:
                    prophet_model = model_from_json(fin.read())
            except: pass
        if prophet_model is None:
            base_time = datetime.datetime.now()
            time_list = [base_time + datetime.timedelta(hours=x) for x in range(len(df_values))]
            df_prophet = pd.DataFrame({'ds': time_list, 'y': df_values})
            prophet_model = Prophet(daily_seasonality=True, yearly_seasonality=False)
            prophet_model.fit(df_prophet)
            with open(PROPHET_MODEL_FILE, 'w') as fout:
                fout.write(model_to_json(prophet_model))
        base_time = datetime.datetime.now()
        time_list = [base_time + datetime.timedelta(hours=x) for x in range(len(df_values))]
        df_future = pd.DataFrame({'ds': time_list})
        forecast_full = prophet_model.predict(df_future)
        y_true = test_data
        y_pred = forecast_full['yhat'].values[split_idx:]
        min_len = min(len(y_true), len(y_pred))
        metrics['prophet_rmse'] = np.sqrt(mean_squared_error(y_true[:min_len], y_pred[:min_len]))
        metrics['prophet_mae'] = mean_absolute_error(y_true[:min_len], y_pred[:min_len])

    lstm_model = None
    lstm_predictions = None
    if LSTM_AVAILABLE:
        if os.path.exists(LSTM_MODEL_FILE) and os.path.exists(SCALER_FILE):
            try:
                lstm_model = load_model(LSTM_MODEL_FILE)
                scaler = joblib.load(SCALER_FILE)
            except: pass
        if lstm_model is None:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df_values.reshape(-1, 1))
            X, y = prepare_lstm_data(scaled_data, lookback=24)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            lstm_model = Sequential()
            lstm_model.add(LSTM(50, return_sequences=False, input_shape=(24, 1)))
            lstm_model.add(Dense(1))
            lstm_model.compile(optimizer='adam', loss='mean_squared_error')
            lstm_model.fit(X, y, epochs=5, batch_size=32, verbose=0)
            lstm_model.save(LSTM_MODEL_FILE)
            joblib.dump(scaler, SCALER_FILE)
        scaled_full = scaler.transform(df_values.reshape(-1, 1))
        X_full, _ = prepare_lstm_data(scaled_full, lookback=24)
        X_full = np.reshape(X_full, (X_full.shape[0], X_full.shape[1], 1))
        pred_scaled = lstm_model.predict(X_full)
        pred_actual = scaler.inverse_transform(pred_scaled).flatten()
        lstm_predictions = np.concatenate((np.zeros(24), pred_actual))
        y_true_lstm = df_values[split_idx:]
        y_pred_lstm = lstm_predictions[split_idx:]
        min_len = min(len(y_true_lstm), len(y_pred_lstm))
        metrics['lstm_rmse'] = np.sqrt(mean_squared_error(y_true_lstm[:min_len], y_pred_lstm[:min_len]))
        metrics['lstm_mae'] = mean_absolute_error(y_true_lstm[:min_len], y_pred_lstm[:min_len])

    return prophet_model, forecast_full, lstm_model, lstm_predictions, metrics

# ==========================================================
#  PHYSICS ENGINE: 2ND ORDER SWING EQUATION & SE
# ==========================================================

def calculate_iec_trip_time(current_pu, tms=0.5):
    if current_pu <= 1.0: return None
    k = 0.14
    alpha = 0.02
    denominator = (current_pu ** alpha) - 1
    if abs(denominator) < 1e-5: return 9999.0 
    return tms * (k / denominator)

def compute_symmetrical_components_physics(fault_type, V_prefault=1.0, Z1=0.2, Z2=0.2, Z0=0.6, Zf=0.0):
    if fault_type == "L-G (Line-to-Ground)":
        denom = Z1 + Z2 + Z0 + 3*Zf
        if denom == 0: denom = 0.001
        I_seq = V_prefault / denom
        return I_seq, I_seq, I_seq
    elif fault_type == "L-L (Line-to-Line)":
        denom = Z1 + Z2 + Zf
        if denom == 0: denom = 0.001
        I1 = V_prefault / denom
        return abs(I1), abs(I1), 0.0
    elif fault_type == "L-L-G (2-Line-Ground)":
        denom = Z1 + ((Z2 * Z0) / (Z2 + Z0))
        if denom == 0: denom = 0.001
        I1 = V_prefault / denom
        I2 = abs(I1 * (Z0 / (Z2 + Z0)))
        I0 = abs(I1 * (Z2 / (Z2 + Z0)))
        return abs(I1), I2, I0
    elif fault_type == "L-L-L (3-Phase Bolted)":
        denom = Z1 + Zf
        if denom == 0: denom = 0.001
        I1 = V_prefault / denom
        return I1, 0.0, 0.0
    return 0.0, 0.0, 0.0

def convert_seq_to_phase(I0, I1, I2):
    i0_c = complex(I0, 0)
    i1_c = complex(I1, 0)
    i2_c = complex(I2, 0)
    ia = i0_c + i1_c + i2_c
    ib = i0_c + (A2_OPERATOR * i1_c) + (A_OPERATOR * i2_c)
    ic = i0_c + (A_OPERATOR * i1_c) + (A2_OPERATOR * i2_c)
    return abs(ia), np.degrees(cmath.phase(ia)), abs(ib), np.degrees(cmath.phase(ib)), abs(ic), np.degrees(cmath.phase(ic))

def calculate_voltage_profile(bus_name, p_load_kw, q_load_kvar, tap_pos, p_gen_kw=0.0, q_gen_kvar=0.0):
    """
    Enhanced Physics: Calculates Voltage Drop considering Load and Generation.
    Now includes q_gen_kvar for Smart Inverter Volt-VAR logic.
    """
    dist = dist_map.get(bus_name, 1.0)
    if dist < 1e-6: dist = 1e-6 
    
    # Net Power at Bus (Load - Generation)
    p_net = p_load_kw - p_gen_kw
    # Net Reactive Power (Load - Inverter Injection)
    q_net = q_load_kvar - q_gen_kvar
    
    z_line = LINE_IMPEDANCE_PER_UNIT_DIST * dist
    v_source = 1.0 * tap_pos
    
    # Complex Power S = P + jQ
    s_apparent_net = complex(p_net, q_net) / 1000.0
    
    # I = (S/V)*
    i_approx = s_apparent_net.conjugate() / v_source
    
    v_drop = i_approx * z_line
    v_load = v_source - v_drop
    
    return abs(v_load)

def update_grid_physics(current_p, current_q):
    """2ND ORDER SWING EQUATION + GOVERNOR CONTROL"""
    H_const = INERTIA_H 
    f0 = NOMINAL_FREQ
    M = 2 * H_const / (2 * np.pi * f0) 
    D = DAMPING_D 
    curr_freq = st.session_state.grid_freq
    curr_delta = st.session_state.rotor_angle
    curr_Pm = st.session_state.mech_power 
    
    Pe_pu = current_p / (SYSTEM_BASE_MVA * 1000.0) 
    governor_response = (50.0 - curr_freq) * 0.5 
    curr_Pm += governor_response * 100.0 
    Pm_pu = curr_Pm / (SYSTEM_BASE_MVA * 1000.0)
    
    w_dev = 2 * np.pi * (curr_freq - 50.0)
    accel = (Pm_pu - Pe_pu - (D * w_dev)) / M
    dt = 0.05 
    
    df_dt = accel / (2 * np.pi)
    new_freq = curr_freq + (df_dt * dt)
    new_delta = curr_delta + (2 * np.pi * (new_freq - 50.0) * dt)
    
    st.session_state.grid_freq = new_freq
    st.session_state.rotor_angle = new_delta
    st.session_state.mech_power = curr_Pm

    # Thermal
    s_load = np.sqrt(current_p**2 + current_q**2)
    loading_pct = s_load / TRANSFORMER_RATING_KVA
    t_ambient = st.session_state.room_temp
    t_rise_max = 65.0
    t_ultimate = t_ambient + (t_rise_max * (loading_pct ** 2))
    tau = TRANSFORMER_TAU
    st.session_state.transformer_thermal += (1.0 / tau) * (t_ultimate - st.session_state.transformer_thermal)

    return st.session_state.grid_freq, st.session_state.transformer_thermal

def apply_scada_noise(val, sigma=0.015):
    return val + np.random.normal(0, sigma)

# --- REAL GAUSS-NEWTON WLS STATE ESTIMATOR ---
class StateEstimator:
    def __init__(self, R_line, X_line):
        self.R = R_line
        self.X = X_line
        
    def solve(self, z, V_source=1.0):
        # Initial State Guess (Flat Start)
        x = np.array([1.0, 0.0]) # [V, delta]
        
        # Weights (Inverse of Variance)
        # We trust Voltage more than Power
        W = np.diag([10000.0, 100.0, 100.0]) 
        
        max_iter = 10
        tol = 1e-4
        
        for i in range(max_iter):
            V, delta = x[0], x[1]
            
            # 1. Measurement Function h(x)
            V_c = cmath.rect(V, delta)
            V_s = complex(V_source, 0)
            Z = complex(self.R, self.X)
            I_c = (V_s - V_c) / Z
            S_c = V_c * I_c.conjugate()
            
            h_val = np.array([
                V,          # V_meas
                S_c.real,   # P_meas
                S_c.imag    # Q_meas
            ])
            
            # 2. Residual
            r = z - h_val
            
            # Check convergence
            if np.max(np.abs(r)) < tol:
                break
                
            # 3. Jacobian H = dh/dx
            epsilon = 1e-5
            
            # Perturb V
            V_p = V + epsilon
            V_c_p = cmath.rect(V_p, delta)
            I_c_p = (V_s - V_c_p) / Z
            S_c_p = V_c_p * I_c_p.conjugate()
            h_p_V = np.array([V_p, S_c_p.real, S_c_p.imag])
            col_V = (h_p_V - h_val) / epsilon
            
            # Perturb Delta
            d_p = delta + epsilon
            V_c_d = cmath.rect(V, d_p)
            I_c_d = (V_s - V_c_d) / Z
            S_c_d = V_c_d * I_c_d.conjugate()
            h_p_d = np.array([V, S_c_d.real, S_c_d.imag])
            col_d = (h_p_d - h_val) / epsilon
            
            H = np.column_stack((col_V, col_d))
            
            # 4. Gain Matrix G = H^T W H
            G = H.T @ W @ H
            
            # 5. Solve Step: dx = G^-1 H^T W r
            rhs = H.T @ W @ r
            try:
                dx = np.linalg.solve(G, rhs)
                x = x + dx
            except:
                break # Singular matrix protection
        
        # Chi-Square Calculation (Cost Function)
        J = np.dot(r.T, np.dot(W, r))
        return x[0], J

def run_wls_state_estimation(measured_v_pu, measured_p_kw, measured_q_kvar, bus_name):
    dist = dist_map.get(bus_name, 1.0)
    if dist < 1e-6: dist = 1e-6 
    R_total = LINE_IMPEDANCE_PER_UNIT_DIST.real * dist
    X_total = LINE_IMPEDANCE_PER_UNIT_DIST.imag * dist
    
    # --- CYBER ATTACK LOGIC (BAD DATA INJECTION) ---
    if st.session_state.fdi_attack:
        measured_v_pu += 0.15 # Inject false bias
    # -----------------------------------------------

    se = StateEstimator(R_total, X_total)
    z = np.array([measured_v_pu, measured_p_kw / 1000.0, measured_q_kvar / 1000.0])
    est_v, chi_sq = se.solve(z, V_source=st.session_state.tap_position)
    residual = abs(measured_v_pu - est_v)
    return est_v, residual, chi_sq

def recloser_logic():
    state = st.session_state.recloser_state
    if st.session_state.fault_active:
        if state == "CLOSED":
            st.session_state.recloser_state = "TRIPPED"
            st.session_state.relay_trip = True
            st.session_state.recloser_timer = 0
            log_event("Protection", "Trip", "Recloser: Instantaneous Trip")
        elif state == "TRIPPED":
            st.session_state.recloser_state = "WAITING"
        elif state == "WAITING":
            st.session_state.recloser_timer += 1
            if st.session_state.recloser_timer > 5: 
                st.session_state.recloser_state = "RECLOSE"
        elif state == "RECLOSE":
            st.session_state.relay_trip = False
            if st.session_state.fault_active:
                log_event("Protection", "Reclose", "Reclose Attempt Failed - Fault Persistent")
                st.session_state.recloser_state = "LOCKOUT"
                st.session_state.relay_trip = True
            else:
                log_event("Protection", "Reclose", "Reclose Successful")
                st.session_state.recloser_state = "CLOSED"
        elif state == "LOCKOUT":
            st.session_state.relay_trip = True

def generate_waveform(thd_active, filter_active):
    v_total = 230 * np.sin(OMEGA * WAVE_TIME)
    thd_pct = 0.5 + np.random.uniform(0, 0.2)
    if thd_active:
        attenuation = 0.05 if filter_active else 1.0
        v3 = (30 * attenuation) * np.sin(3 * OMEGA * WAVE_TIME)
        v5 = (15 * attenuation) * np.sin(5 * OMEGA * WAVE_TIME)
        v_total += v3 + v5
        v_residue = np.sqrt((30*attenuation)**2 + (15*attenuation)**2)
        thd_pct = (v_residue / 230) * 100
    return WAVE_TIME, v_total, thd_pct

# ----------------------------------------------------------
# SOLAR, BESS & SMART INVERTER LOGIC (UPGRADED)
# ----------------------------------------------------------
# --- UPGRADED MATHEMATICAL PV MODEL ---
NOCT = 45.0  # Nominal Operating Cell Temperature
TEMP_COEFF = -0.0041 # -0.41% / deg C
STC_TEMP = 25.0

def get_solar_contribution(idx):
    """
    Returns irradiance (0-1) from the uploaded solar data.
    """
    if solar_profile is not None and len(solar_profile) > 0:
        return solar_profile[idx % len(solar_profile)]
    else:
        # Fallback to bell curve if file read failed
        h = (idx % 24)
        if h < 6 or h >= 19: return 0.0
        elif 6 <= h < 10: return 0.25 * (h - 5)
        elif 10 <= h < 15: return 1.0
        elif 15 <= h < 19: return 1.0 - (0.25 * (h - 15))
        return 0.0

def calculate_pv_physics(bus_name, idx, ambient_temp):
    """
    Mathematically rigorous PV calculation.
    Returns: P_gen (kW), Cell_Temp (C), Irradiance (0-1)
    """
    if bus_name not in SOLAR_BUS_CONFIG: return 0.0, ambient_temp, 0.0
    
    capacity = SOLAR_BUS_CONFIG[bus_name]
    irradiance = get_solar_contribution(idx) # "Suns" (0-1, where 1=1000W/m2)
    
    # 1. Cloud Shading Logic (Global Override)
    if st.session_state.cloud_shading:
        irradiance *= 0.3 # 70% Drop
    
    # 2. Cell Temperature Calculation (Standard Model)
    # T_cell = T_amb + (NOCT - 20)/800 * G_watts
    # We assume irradiance 1.0 = 800 W/m2 NOCT standard, scaled
    t_cell = ambient_temp + ((NOCT - 20.0) / 0.8) * irradiance
    
    # 3. Active Power with Temperature Degradation
    # P = P_rated * Irradiance * (1 + gamma * (T_cell - 25))
    temp_loss_factor = 1.0 + (TEMP_COEFF * (t_cell - STC_TEMP))
    
    # Clamp factor to avoid physics glitch at extreme temps
    temp_loss_factor = max(0.5, min(1.2, temp_loss_factor))
    
    p_gen_ideal = capacity * irradiance
    p_gen_real = p_gen_ideal * temp_loss_factor
    
    # Add minor noise for realism
    noise = np.random.uniform(0.99, 1.01)
    
    return p_gen_real * noise, t_cell, irradiance

def smart_inverter_logic(v_pu, p_available_kw, capacity_kw):
    """
    IEEE 1547 compliant Smart Inverter Functions
    1. Volt-Watt Curtailment: Reduce P if V > 1.05
    2. Volt-VAR Control: Absorb Q if V > 1.02, Inject Q if V < 0.98
    """
    p_out = p_available_kw
    q_out = 0.0
    status = []
    
    # Volt-Watt
    if v_pu > 1.05:
        curtail_factor = max(0.0, 1.0 - (v_pu - 1.05) * 10) # Slope
        p_out *= curtail_factor
        status.append(f"VW-Curtail: {int((1-curtail_factor)*100)}%")
        
    # Volt-VAR
    if v_pu > 1.02:
        # Absorb Inductive (Negative Q) to lower voltage
        q_req = -1.0 * capacity_kw * (v_pu - 1.02) * 5 
        q_out = max(q_req, -0.44 * capacity_kw) # Cap at 0.44 pf
        status.append("VV-Absorbing")
    elif v_pu < 0.98:
        # Inject Capacitive (Positive Q) to raise voltage
        q_req = capacity_kw * (0.98 - v_pu) * 5
        q_out = min(q_req, 0.44 * capacity_kw)
        status.append("VV-Injecting")
        
    return p_out, q_out, ", ".join(status)

def bess_dispatch_logic(net_load_kw, current_hour):
    """
    Battery Energy Storage System (Peak Shaving)
    Charge: Early morning or when Solar > Load
    Discharge: Evening Peak (18:00 - 22:00)
    """
    h = current_hour % 24
    p_bess = 0.0 # +Discharge, -Charge
    mode = "IDLE"
    
    soc = st.session_state.bess_soc
    
    # Dispatch Strategy
    if 10 <= h <= 15: # Solar Peak -> Charge
        if soc < 95.0:
            p_bess = -0.8 * BESS_MAX_POWER
            mode = "CHARGING (Solar Soak)"
    elif 18 <= h <= 22: # Evening Peak -> Discharge
        if soc > 20.0:
            p_bess = 1.0 * BESS_MAX_POWER
            mode = "DISCHARGING (Peak Shave)"
            
    # Update SoC
    energy_delta = (p_bess * -1.0) / 60.0 # Assuming 1 min sim steps roughly for demo visuals
    if st.session_state.run_simulation:
         new_soc = soc + (energy_delta / BESS_CAPACITY_KWH * 100)
         st.session_state.bess_soc = max(0.0, min(100.0, new_soc))
         
    return p_bess, mode

# --- CYBERPUNK PLOTTING FUNCTIONS ---
def make_cyber_meter(value, delta_val, title, min_val, max_val, color_hex):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        delta = {
            'reference': value - delta_val, 
            'position': "top", 
            'relative': False, 
            'valueformat': '.1f',
            'font': {'size': 20, 'family': "Orbitron"}
        },
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 18, 'color': "#00f3ff", 'family': "Orbitron"}},
        number = {'font': {'size': 30, 'color': "white", 'family': "Orbitron"}, 'suffix': ""},
        gauge = {
            'axis': {'range': [min_val, max_val], 'tickwidth': 2, 'tickcolor': "#00f3ff", 'ticklen': 10},
            'bar': {'color': color_hex, 'thickness': 0.75}, 
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [
                {'range': [min_val, max_val], 'color': "rgba(20, 20, 20, 0.8)"} 
            ],
            'threshold': {
                'line': {'color': "#ff0055", 'width': 4},
                'thickness': 0.75,
                'value': max_val * 0.9
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)", 
        font = {'color': "white", 'family': "Orbitron"},
        margin=dict(l=30, r=30, t=70, b=10), 
        height=240
    )
    return fig

def make_cyber_plot(x, y, title, line_color, delta_val=None, height=200):
    title_text = title.upper()
    if delta_val is not None:
        symbol = "▲" if delta_val >= 0 else "▼"
        title_text += f"   {symbol} {abs(delta_val):.2f}"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines', 
        fill='tozeroy', 
        line=dict(color=line_color, width=3),
        name=title
    ))
    
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10), height=height,
        title=dict(text=title_text, font=dict(size=12, color=line_color, family="Orbitron")),
        xaxis=dict(showgrid=True, gridcolor='rgba(0, 243, 255, 0.1)', visible=True, color="#666"),
        yaxis=dict(showgrid=True, gridcolor='rgba(0, 243, 255, 0.1)', color="#666"),
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0.3)', 
        showlegend=False
    )
    return fig

def draw_phasor(Ia, Id, Ib, Ibd, Ic, Icd):
    max_curr = max(Ia, Ib, Ic, 0.1)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[0, 1.0], theta=[0, 90], mode='lines', name="Va", line=dict(color="#ff0055", width=2)))
    fig.add_trace(go.Scatterpolar(r=[0, 1.0], theta=[0, 330], mode='lines', name="Vb", line=dict(color="#ffae00", width=2)))
    fig.add_trace(go.Scatterpolar(r=[0, 1.0], theta=[0, 210], mode='lines', name="Vc", line=dict(color="#00f3ff", width=2)))
    
    scale = 0.8
    fig.add_trace(go.Scatterpolar(r=[0, (Ia/max_curr)*scale], theta=[0, Id], mode='lines+markers', name="Ia", line=dict(color="#ff0055", width=4, dash="dot")))
    fig.add_trace(go.Scatterpolar(r=[0, (Ib/max_curr)*scale], theta=[0, Ibd], mode='lines+markers', name="Ib", line=dict(color="#ffae00", width=4, dash="dot")))
    fig.add_trace(go.Scatterpolar(r=[0, (Ic/max_curr)*scale], theta=[0, Icd], mode='lines+markers', name="Ic", line=dict(color="#00f3ff", width=4, dash="dot")))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.2], gridcolor="#333", linecolor="#333"), 
            angularaxis=dict(gridcolor="#333", linecolor="#333"), 
            bgcolor="rgba(0,0,0,0.5)"
        ),
        showlegend=True, height=350, margin=dict(l=40, r=40, t=30, b=30), 
        title=dict(text="PHASOR ANALYZER", font=dict(color="#00f3ff", family="Orbitron")),
        paper_bgcolor='rgba(0,0,0,0)', legend=dict(font=dict(color="white"))
    )
    return fig

# ----------------------------------------------------------
# 6. SECURITY & LOGIN (CYBERPUNK THEME)
# ----------------------------------------------------------
if not st.session_state.logged_in:
    st.markdown("""<style>
        .stApp { 
            background-color: #000;
            background-image: 
                linear-gradient(rgba(0, 255, 0, 0.1) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 255, 0, 0.1) 1px, transparent 1px);
            background-size: 50px 50px;
        }
        .login-box {
            border: 2px solid #00f3ff;
            background: rgba(0, 10, 10, 0.9);
            box-shadow: 0 0 50px rgba(0, 243, 255, 0.2);
            padding: 40px;
            border-radius: 10px;
            text-align: center;
        }
        .login-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 40px;
            color: #00f3ff;
            text-shadow: 0 0 20px #00f3ff;
            margin-bottom: 10px;
        }
        .login-sub {
            font-family: 'Roboto Mono', monospace;
            color: #ff0055;
            letter-spacing: 3px;
            margin-bottom: 30px;
        }
    </style>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="login-box">
            <div class="login-title">AZU DIGITAL TWIN</div>
            <div class="login-sub">Final Year Project - II</div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.container(border=True):
            user = st.text_input("Username")
            pwd = st.text_input("ENCRYPTION KEY", type="password")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("LOGIN", type="primary", use_container_width=True):
                if user == "admin" and pwd == "admin":
                    st.session_state.logged_in = True
                    st.session_state.run_simulation = True
                    log_event("Session Start", "Security", "Admin Uplink Established")
                    st.rerun()
                else: st.error("ACCESS DENIED: INVALID CREDENTIALS")
    st.stop()

# ----------------------------------------------------------
# 7. SIDEBAR
# ----------------------------------------------------------
with st.sidebar:
    c1, c2 = st.columns([1, 25])
    with c1: st.write("")
    with c2: st.markdown("### AZU Digital Twin\n*Final Year Project - II*")
    
    curr_v = 0.0 if st.session_state.relay_trip else ((0.85 * st.session_state.tap_position) if st.session_state.fault_active else (0.99 * st.session_state.tap_position))
    curr_state, state_col, state_desc = get_operating_state(curr_v, 1.0, st.session_state.fault_active, st.session_state.relay_trip)
    
    st.markdown(f"""
    <div style="padding: 15px; border-left: 5px solid {state_col}; background: linear-gradient(90deg, rgba(0,0,0,0.8), transparent);">
        <h3 style="margin:0; color:{state_col}; font-size: 18px; font-family: 'Orbitron';">{curr_state}</h3>
        <p style="margin:0; font-size: 10px; color: #ccc; font-family: 'Roboto Mono';">{state_desc}</p>
        <p style="margin:0; font-size: 12px; color: #00f3ff; font-family: 'Roboto Mono';">79-RECLOSER: {st.session_state.recloser_state}</p>
    </div>
    """, unsafe_allow_html=True)
        
    st.markdown("---")
    nav = st.radio("NAVIGATION MODULE", ["Live Telemetry", "Grid Topology", "Feeder Analytics", "AI Forecasting"])
    st.markdown("---")
    
    with st.expander("Simulation Control", expanded=True):
        run_sim = st.toggle("▶ ACTIVATE STREAM", value=st.session_state.run_simulation)
        st.session_state.run_simulation = run_sim
        speed = st.select_slider("CLOCK SPEED", options=[0.5, 1.0, 2.0, 5.0], value=st.session_state.speed)
        st.session_state.speed = speed
        
        # --- CLOUD TRANSIENT TOGGLE ---
        cloud = st.toggle("☁️ CLOUD SHADING", value=st.session_state.cloud_shading)
        if cloud != st.session_state.cloud_shading:
             log_event("Environment", "Weather", "Cloud Front Detected" if cloud else "Clear Sky")
        st.session_state.cloud_shading = cloud
        
        if st.button("RESTART", use_container_width=True):
            st.session_state.idx = 0
            st.session_state.relay_trip = False
            st.session_state.fault_active = False
            st.session_state.recloser_state = "CLOSED"
            st.session_state.history_tap = [1.0] * 50
            st.session_state.history_cap = [0.0] * 24
            st.session_state.history_se_meas = [1.0] * 50
            st.session_state.history_se_est = [1.0] * 50
            st.session_state.history_se_j = [0.0] * 50
            st.session_state.transformer_thermal = 40.0
            st.session_state.grid_freq = 50.0
            st.session_state.rotor_angle = 0.0
            st.session_state.mech_power = 5000.0
            st.session_state.fdi_attack = False
            st.session_state.bess_soc = 50.0
            # Clear solar buffers
            st.session_state.solar_p_history = [0.0] * 50
            st.session_state.solar_q_history = [0.0] * 50
            st.session_state.solar_v_history = [1.0] * 50
            st.session_state.solar_irr_history = [0.0] * 50
            st.session_state.solar_temp_history = [25.0] * 50
            
            st.session_state.audit_log = pd.DataFrame(columns=["Timestamp", "Event", "Type", "Details"])
            log_event("System", "Reset", "Hard Reboot Initiated")
            st.rerun()
    
    st.markdown("---")
    st.caption("THREAT INJECTION")
    safety_lock = st.toggle("SAFETY OVERRIDE", value=False)
    
    with st.container(border=True):
        f_bus = st.selectbox("TARGET BUS", bus_list, index=0)
        f_type = st.selectbox("FAULT VECTOR", list(FAULT_LIBRARY.keys()))
        
        if st.session_state.recloser_state == "LOCKOUT":
            st.error("LOCKOUT - MANUAL RESET REQ")
            if st.button("RESET RECLOSER", type="secondary", use_container_width=True):
                st.session_state.relay_trip = False
                st.session_state.fault_active = False
                st.session_state.recloser_state = "CLOSED"
                log_event("Protection", "Reset", "Manual Recloser Reset")
                st.rerun()
        elif st.session_state.fault_active:
            st.warning("FAULT IN PROGRESS")
            st.caption(f"Recloser: {st.session_state.recloser_state}")
            if st.button("FORCE CLEAR", type="primary", use_container_width=True):
                st.session_state.fault_active = False
                st.session_state.relay_trip = False
                st.session_state.recloser_state = "CLOSED"
                log_event("Restoration", "Manual", "Fault Cleared by Operator")
                st.rerun()
        else:
            if st.button("EXECUTE FAULT", type="primary", use_container_width=True, disabled=not safety_lock):
                st.session_state.fault_active = True
                st.session_state.fault_bus = f_bus
                st.session_state.fault_type = f_type
                log_event("Contingency", "Fault", f"Injected: {f_type}")
                st.rerun()

    st.markdown("---")
    with st.expander("EVENT LOGS", expanded=False):
        st.dataframe(st.session_state.audit_log, hide_index=True, use_container_width=True)
        csv = st.session_state.audit_log.to_csv(index=False).encode('utf-8')
        st.download_button("EXPORT DATA", data=csv, file_name="log.csv", mime="text/csv", use_container_width=True)

    st.markdown("---")
    if st.button("LOGOUT", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

# ----------------------------------------------------------
# 8. DASHBOARD FRAGMENTS
# ----------------------------------------------------------

def render_hvac():
    """Enhanced HVAC rendering to show Load Impact."""
    ambient_temp = 35.0  
    insulation_factor = 0.05
    cooling_power_per_degree = 0.4       
    temp_diff = max(0, st.session_state.room_temp - st.session_state.hvac_setpoint)
    impact_kw = temp_diff * 1.5 if st.session_state.hvac_on else 0.0
    if st.session_state.hvac_on:
        st.session_state.room_temp -= cooling_power_per_degree
        st.session_state.hvac_load_kw = 15.0 + impact_kw
    else:
        if st.session_state.room_temp < ambient_temp:
            st.session_state.room_temp += insulation_factor
        st.session_state.hvac_load_kw = 0.0
    if st.session_state.room_temp > (st.session_state.hvac_setpoint + 1.0): st.session_state.hvac_on = True
    elif st.session_state.room_temp < (st.session_state.hvac_setpoint - 1.0): st.session_state.hvac_on = False

    st.markdown("### ❄️ HVAC System Status")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("ZONE TEMP", f"{st.session_state.room_temp:.1f} °C", delta="-COOLING" if st.session_state.hvac_on else "+HEATING", delta_color="inverse")
    with c2: st.metric("COMPRESSOR", "ONLINE" if st.session_state.hvac_on else "STANDBY")
    with c3: st.metric("LOAD DRIFT", f"{st.session_state.hvac_load_kw:.1f} kW", delta="Temp Impact", delta_color="off")
    with c4:
        st.write("**SETPOINT**")
        st.session_state.hvac_setpoint = st.slider("TARGET TEMP", 18.0, 30.0, st.session_state.hvac_setpoint, key="hvac_slider")

    fig_temp = go.Figure(go.Indicator(
        mode = "gauge+number", value = st.session_state.room_temp,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "TEMPERATURE METER", 'font': {'size': 15, 'color': "white", 'family': "Orbitron"}},
        gauge = {
            'axis': {'range': [15, 40], 'tickcolor': "white"},
            'bar': {'color': "#00f3ff" if st.session_state.hvac_on else "#ff0055"},
            'bgcolor': "rgba(0,0,0,0)",
            'steps': [{'range': [15, 24], 'color': 'rgba(0, 100, 255, 0.2)'}, {'range': [30, 40], 'color': 'rgba(255, 0, 0, 0.2)'}]
        }
    ))
    fig_temp.update_layout(height=180, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
    st.plotly_chart(fig_temp, use_container_width=True, key="hvac_gauge")

@st.fragment(run_every=speed if st.session_state.run_simulation else None)
def render_home():
    if st.session_state.run_simulation: st.session_state.idx = (st.session_state.idx + 1) % len(df_raw)
    idx = st.session_state.idx
    row = df_raw.iloc[idx]
    
    # ----------------------------------------
    # GRID LEVEL CALCULATIONS
    # ----------------------------------------
    
    # Base Loads
    p_load_total = row["Total_Active_Power"] + st.session_state.hvac_load_kw
    
    # Total Solar Generation (Driven by Physics)
    total_pv_gen = 0.0
    for bus_name in SOLAR_BUS_CONFIG.keys():
        p_val, _, _ = calculate_pv_physics(bus_name, idx, st.session_state.room_temp)
        total_pv_gen += p_val
        
    # BESS Dispatch
    p_bess, bess_mode = bess_dispatch_logic(p_load_total - total_pv_gen, idx)
    
    # Net Grid Load
    p_grid_net = p_load_total - total_pv_gen - p_bess
    
    q_val = row["Total_Reac_Power"] + (st.session_state.hvac_load_kw * 0.6)
    
    recloser_logic()
    freq, temp = update_grid_physics(p_grid_net, q_val)
    
    delta_p = p_grid_net - st.session_state.prev_p
    delta_q = q_val - st.session_state.prev_q
    st.session_state.prev_p = p_grid_net
    st.session_state.prev_q = q_val

    current_voltage_pu = 0.0 if st.session_state.relay_trip else ((0.85 * st.session_state.tap_position) if st.session_state.fault_active else (0.99 * st.session_state.tap_position))
    sys_state, sys_color, sys_desc = get_operating_state(current_voltage_pu, 1.0, st.session_state.fault_active, st.session_state.relay_trip)
    disp_freq = apply_scada_noise(freq, 0.02)
    
    # REVERSE POWER ALERT
    if p_grid_net < -50.0: # Significant backfeed
        sys_state = "REVERSE FLOW"
        sys_color = "#ffff00"
        sys_desc = "PROTECTION BLIND SPOT ACTIVE"

    with st.container(height=150):
        c1, c2, c3 = st.columns([2, 3, 1])
        c1.metric("GRID STATE", sys_state)
        c2.metric("INTEGRITY", sys_desc)
        c3.metric("FREQUENCY (INERTIA)", f"{disp_freq:.2f} Hz", delta=f"{disp_freq-50.0:.2f}")

    st.markdown(f"#### 📡 LIVE TELEMETRY (STATUS: {sys_state})")
    
    # --- METRICS ROW ---
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("TOTAL SOLAR PV", f"{total_pv_gen:.1f} kW", delta="Cloud Effect" if st.session_state.cloud_shading else "Optimal")
    col_s2.metric("BESS STATUS", f"{st.session_state.bess_soc:.1f}% SoC", delta=bess_mode)
    col_s3.metric("GRID NET LOAD", f"{p_grid_net:.1f} kW")
    # Penetration Metric - Requested by User
    penetration_pct = (total_pv_gen / p_load_total * 100) if p_load_total > 0 else 0
    col_s4.metric("PV PENETRATION", f"{penetration_pct:.1f} %", delta="Load Response")
    
    # --- PLOTS ROW ---
    hist_window = 60
    start_idx = max(0, idx - hist_window)
    hist_indices = list(range(start_idx, idx))
    
    h_solar, h_grid, h_pen = [], [], []
    for k in hist_indices:
        s_gen = 0.0
        for b in SOLAR_BUS_CONFIG:
            pv, _, _ = calculate_pv_physics(b, k, st.session_state.room_temp)
            s_gen += pv
        # Apply rough BESS calc for history consistency
        b_p, _ = bess_dispatch_logic(df_raw["Total_Active_Power"].iloc[k % len(df_raw)] - s_gen, k)
        
        base_load = df_raw["Total_Active_Power"].iloc[k % len(df_raw)]
        net_grid = base_load - s_gen - b_p
        pen = (s_gen / base_load * 100) if base_load > 0 else 0
        h_solar.append(s_gen)
        h_grid.append(net_grid)
        h_pen.append(pen)

    c_p1, c_p2, c_p3 = st.columns(3)
    with c_p1: st.plotly_chart(make_cyber_plot(hist_indices, h_solar, "SOLAR GENERATION (kW)", "#ffff00", height=150), use_container_width=True)
    with c_p2: st.plotly_chart(make_cyber_plot(hist_indices, h_grid, "NET GRID LOAD (kW)", "#00f3ff", height=150), use_container_width=True)
    with c_p3: st.plotly_chart(make_cyber_plot(hist_indices, h_pen, "PV PENETRATION (%)", "#00ff00", height=150), use_container_width=True)

    max_p = max(3000, p_load_total * 1.5) 
    max_q = max(2000, q_val * 1.5)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(make_cyber_meter(p_grid_net, delta_p, "NET ACTIVE POWER (kW)", 0, max_p, "#00f3ff"), use_container_width=True, key="gauge_p")
    with c2:
        st.plotly_chart(make_cyber_meter(q_val, delta_q, "REACTIVE POWER (kVAR)", 0, max_q, "#ff00ff"), use_container_width=True, key="gauge_q")

    st.markdown("---")
    render_hvac()

@st.fragment(run_every=speed if st.session_state.run_simulation else None)
def render_feeder(view_bus):
    if st.session_state.run_simulation: st.session_state.idx = (st.session_state.idx + 1) % len(df_fa_p)
    idx = st.session_state.idx
    
    try:
        raw_p = df_fa_p[view_bus].iloc[idx]
    except KeyError:
        raw_p = 45.0 + np.random.uniform(-5, 5)

    display_p = raw_p
    display_q = raw_p * 0.4
    
    # --- PV & SMART INVERTER CALCULATIONS ---
    # Use upgraded physics calc
    pv_output, cell_temp, irradiance = calculate_pv_physics(view_bus, idx, st.session_state.room_temp)
    has_pv = view_bus in SOLAR_BUS_CONFIG
    
    # Initial Voltage Calc for Smart Logic
    v_pre = calculate_voltage_profile(view_bus, display_p, display_q, st.session_state.tap_position, p_gen_kw=pv_output)
    
    smart_p, smart_q, smart_status = pv_output, 0.0, "Passive"
    if has_pv:
        smart_p, smart_q, smart_status = smart_inverter_logic(v_pre, pv_output, SOLAR_BUS_CONFIG[view_bus])
    
    # Final Voltage with Smart Inverter Actions
    voltage_pu_phys = calculate_voltage_profile(view_bus, display_p, display_q, st.session_state.tap_position, p_gen_kw=smart_p, q_gen_kvar=smart_q)
    
    # Hosting Capacity Check
    hc_status = "OK"
    hc_col = "normal"
    if voltage_pu_phys > 1.045: 
        hc_status = "CRITICAL (Limit Reached)"
        hc_col = "inverse"
    elif voltage_pu_phys > 1.03:
        hc_status = "MODERATE"
    
    # --- AUTO-TAP CHANGER LOGIC (AVR) ---
    if st.session_state.auto_tap_mode and not st.session_state.relay_trip:
        if voltage_pu_phys < 0.96:
            st.session_state.tap_position = min(1.10, st.session_state.tap_position + 0.005)
        elif voltage_pu_phys > 1.04:
            st.session_state.tap_position = max(0.90, st.session_state.tap_position - 0.005)
    # ------------------------------------

    # 2. SCADA Layer: Add Noise
    measured_v = apply_scada_noise(voltage_pu_phys, 0.01)
    measured_p = apply_scada_noise(display_p - smart_p, 1.0) # SCADA sees Net Load
    measured_q = apply_scada_noise(display_q - smart_q, 1.0)
    
    # 3. State Estimation (SE) Engine
    estimated_v, se_resid, se_chi = run_wls_state_estimation(measured_v, measured_p, measured_q, view_bus)

    is_local_fault = False
    relay_msg = "MONITORING"
    
    i0, i1, i2 = 0.0, 0.0, 0.0
    Ia, Ib, Ic = 0.8, 0.8, 0.8 # approx PU
    Id, Ibd, Icd = 0, -120, 120 # angles

    if st.session_state.fault_active and st.session_state.fault_bus == view_bus:
        if st.session_state.relay_trip:
            display_p, display_q = 0.0, 0.0
            relay_msg = "❌ TRIP"
            is_local_fault = True
            Ia, Ib, Ic = 0, 0, 0
            voltage_pu_phys = 0.0
            measured_v = 0.0
            estimated_v = 0.0
        else:
            f_data = FAULT_LIBRARY[st.session_state.fault_type]
            i1, i2, i0 = compute_symmetrical_components_physics(st.session_state.fault_type, V_prefault=1.0, Zf=f_data["Zf"])
            Ia_c, Id_c, Ib_c, Ibd_c, Ic_c, Icd_c = convert_seq_to_phase(i0, i1, i2)
            Ia, Ib, Ic = Ia_c, Ib_c, Ic_c
            Id, Ibd, Icd = Id_c, Ibd_c, Icd_c
            
            max_I = max(Ia, Ib, Ic)
            trip_time = calculate_iec_trip_time(max_I) 
            
            if trip_time:
                step_add = (st.session_state.speed / trip_time) * 100
                st.session_state.relay_accumulator += step_add
                pct = min(100, int(st.session_state.relay_accumulator))
                relay_msg = f"⚠️ TRIP CURVE: {pct}%"
            else: relay_msg = "FAULT DETECTED"
            
            voltage_pu_phys *= 0.3 
            measured_v = apply_scada_noise(voltage_pu_phys)
            estimated_v = run_wls_state_estimation(measured_v, measured_p, measured_q, view_bus)[0]
    else:
        st.session_state.relay_accumulator = max(0, st.session_state.relay_accumulator - 5.0)
        i1 = (display_p - smart_p) / 100.0 # Current reflects net power
        i2 = i1 * 0.05 
        i0 = i1 * 0.02
        Ia = Ib = Ic = abs(i1)

    delta_feeder = display_p - st.session_state.prev_feeder_p
    st.session_state.prev_feeder_p = display_p
    
    # -- UPDATE PLOT BUFFERS --
    st.session_state.history_tap.append(st.session_state.tap_position)
    if len(st.session_state.history_tap) > 50: st.session_state.history_tap.pop(0)

    st.session_state.history_cap.append(st.session_state.capacitor_bank_kvAr)
    if len(st.session_state.history_cap) > 24: st.session_state.history_cap.pop(0) 
    
    st.session_state.history_se_meas.append(measured_v)
    if len(st.session_state.history_se_meas) > 50: st.session_state.history_se_meas.pop(0)
    
    st.session_state.history_se_est.append(estimated_v)
    if len(st.session_state.history_se_est) > 50: st.session_state.history_se_est.pop(0)
    
    st.session_state.history_se_j.append(se_chi)
    if len(st.session_state.history_se_j) > 50: st.session_state.history_se_j.pop(0)

    # -- UPDATE SOLAR PHYSICS BUFFERS --
    st.session_state.solar_p_history.append(smart_p)
    st.session_state.solar_q_history.append(smart_q)
    st.session_state.solar_v_history.append(voltage_pu_phys)
    st.session_state.solar_irr_history.append(irradiance)
    st.session_state.solar_temp_history.append(cell_temp)
    
    for hist_list in [st.session_state.solar_p_history, st.session_state.solar_q_history, 
                      st.session_state.solar_v_history, st.session_state.solar_irr_history, 
                      st.session_state.solar_temp_history]:
        if len(hist_list) > 50: hist_list.pop(0)

    pf_denom = np.sqrt((display_p - smart_p)**2 + (display_q - st.session_state.capacitor_bank_kvAr)**2)
    pf_final = (display_p - smart_p) / pf_denom if pf_denom > 0 else 1.0
    
    if st.session_state.apfc_auto_mode and not is_local_fault and not st.session_state.relay_trip:
        if pf_final < 0.95: 
            st.session_state.capacitor_bank_kvAr += 25.0 
        elif pf_final > 0.99 and st.session_state.capacitor_bank_kvAr > 0: 
            st.session_state.capacitor_bank_kvAr -= 25.0 
    
    st.session_state.capacitor_bank_kvAr = max(0.0, min(200.0, st.session_state.capacitor_bank_kvAr))

    st.markdown(f"### 🔍 FEEDER: **{view_bus}**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("STATUS", "TRIPPED" if st.session_state.relay_trip else ("FAULT" if is_local_fault else "OK"))
    m2.metric("RAW LOAD", f"{display_p:.1f} kW", delta=f"{delta_feeder:.1f} kW")
    
    se_delta_msg = "State Est."
    if abs(measured_v - estimated_v) > 0.05: se_delta_msg = "⚠️ BAD DATA DETECTED"
    
    m3.metric("SE VOLTAGE", f"{estimated_v:.3f} pu", delta=se_delta_msg, delta_color="normal" if "BAD" not in se_delta_msg else "inverse")
    m4.metric("PROTECTION", relay_msg)

    # --- SOLAR & HOSTING CAPACITY VISUALIZATION ---
    if has_pv:
        st.markdown(f"""
        <div style="background: rgba(255, 255, 0, 0.1); border: 1px solid #ffff00; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between;">
                <div>
                    <h4 style="color: #ffff00; margin:0;">☀️ SMART PV SYSTEM (PHYSICS MODEL)</h4>
                    <span style="color: #ddd;">Generating: <b>{smart_p:.2f} kW</b> | Reactive Inj: <b>{smart_q:.1f} kVAR</b></span><br>
                    <span style="color: #bbb; font-size:12px;">Cell Temp: {cell_temp:.1f}°C (Eff Loss: {abs(TEMP_COEFF*(cell_temp-STC_TEMP)*100):.1f}%)</span>
                </div>
                <div style="text-align: right;">
                    <span style="color: #00f3ff; font-weight: bold;">IEEE 1547 MODE:</span><br>
                    <span style="color: #fff;">{smart_status}</span>
                </div>
            </div>
            <hr style="border-color: #333;">
            <div style="display: flex; justify-content: space-between; font-size: 12px;">
                <span style="color: #ddd;">HOSTING CAPACITY STATUS:</span>
                <span style="color: {'#ff0055' if 'CRITICAL' in hc_status else '#00ff00'}; font-weight: bold;">{hc_status}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # --- NEW PHYSICS PLOTS EXPANDER ---
        with st.expander("☀️ SOLAR PV PHYSICS WORKBENCH", expanded=True):
            p_hist = st.session_state.solar_p_history
            q_hist = st.session_state.solar_q_history
            v_hist = st.session_state.solar_v_history
            irr_hist = st.session_state.solar_irr_history
            t_hist = st.session_state.solar_temp_history
            x_ax = list(range(len(p_hist)))

            # PLOT 1: ACTIVE POWER vs THERMAL PHYSICS
            fig_p = make_subplots(specs=[[{"secondary_y": True}]])
            fig_p.add_trace(go.Scatter(x=x_ax, y=p_hist, name="Active Power (P)", line=dict(color="#ffff00", width=3), fill='tozeroy'), secondary_y=False)
            fig_p.add_trace(go.Scatter(x=x_ax, y=t_hist, name="Cell Temp (°C)", line=dict(color="#ff0055", width=1, dash='dot')), secondary_y=True)
            fig_p.update_layout(title="PV ACTIVE POWER & THERMAL DEGRADATION", height=200, margin=dict(l=10, r=10, t=30, b=10), 
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.3)', font=dict(color="#ccc", family="Orbitron"))
            fig_p.update_yaxes(title_text="Power (kW)", secondary_y=False, gridcolor='#333')
            fig_p.update_yaxes(title_text="Temp (°C)", secondary_y=True, showgrid=False)
            st.plotly_chart(fig_p, use_container_width=True)

            c_pv1, c_pv2 = st.columns(2)
            
            # PLOT 2: REACTIVE POWER (VOLT-VAR)
            with c_pv1:
                fig_q = go.Figure()
                fig_q.add_trace(go.Scatter(x=x_ax, y=q_hist, name="Reactive Q", line=dict(color="#00ff00", width=2)))
                fig_q.add_hline(y=0, line_dash="dash", line_color="white")
                fig_q.update_layout(title="INVERTER Q-INJECTION (VAR)", height=200, margin=dict(l=10, r=10, t=30, b=10), 
                                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.3)', font=dict(color="#ccc", family="Orbitron"),
                                    yaxis=dict(gridcolor='#333'))
                st.plotly_chart(fig_q, use_container_width=True)

            # PLOT 3: VOLTAGE PROFILE
            with c_pv2:
                fig_v = go.Figure()
                fig_v.add_trace(go.Scatter(x=x_ax, y=v_hist, name="PCC Voltage", line=dict(color="#00f3ff", width=2)))
                fig_v.add_hline(y=1.0, line_dash="dash", line_color="gray")
                fig_v.add_hline(y=1.05, line_color="red", line_width=1)
                fig_v.update_layout(title="PCC VOLTAGE PROFILE (pu)", height=200, margin=dict(l=10, r=10, t=30, b=10), 
                                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.3)', font=dict(color="#ccc", family="Orbitron"),
                                    yaxis=dict(gridcolor='#333', range=[0.9, 1.1]))
                st.plotly_chart(fig_v, use_container_width=True)


    # STATE ESTIMATION DETAILS PANEL & PLOT
    with st.expander("🧠 STATE ESTIMATION (WLS) ENGINE & SECURITY", expanded=True):
        attack_cols = st.columns([1, 3])
        with attack_cols[0]:
            is_attack = st.toggle("☠️ CYBER ATTACK (FDI)", value=st.session_state.fdi_attack)
            st.session_state.fdi_attack = is_attack
            if is_attack:
                st.error("🚨 FALSE DATA INJECTION ACTIVE")
        
        se1, se2, se3 = st.columns(3)
        with se1:
            st.metric("RAW SCADA MEASUREMENT", f"{measured_v:.4f} pu", delta="Noisy Input", delta_color="inverse")
        with se2:
            st.metric("ESTIMATED STATE (x̂)", f"{estimated_v:.4f} pu", delta="Converged Output")
        with se3:
            st.metric("RESIDUAL (J(x))", f"{se_resid:.5f}", delta=f"Chi-Sq: {se_chi:.2f}", delta_color="off")
        
        fig_se_plot = go.Figure()
        x_axis = list(range(len(st.session_state.history_se_meas)))
        fig_se_plot.add_trace(go.Scatter(x=x_axis, y=st.session_state.history_se_meas, mode='markers+lines', name='SCADA (Raw/Bad)', line=dict(color='#ff0055', width=1, dash='dot'), marker=dict(size=4)))
        fig_se_plot.add_trace(go.Scatter(x=x_axis, y=st.session_state.history_se_est, mode='lines', name='Estimated (WLS)', line=dict(color='#00ff00', width=3)))
        fig_se_plot.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20), title=dict(text="REAL-TIME STATE ESTIMATOR CONVERGENCE", font=dict(size=12, color="white", family="Orbitron")), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.3)', xaxis=dict(showgrid=True, gridcolor='#333'), yaxis=dict(showgrid=True, gridcolor='#333'), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_se_plot, use_container_width=True)
        
        fig_j = make_cyber_plot(list(range(len(st.session_state.history_se_j))), st.session_state.history_se_j, "SE RESIDUAL COST J(x) - (BAD DATA DETECTOR)", "#ffae00", height=150)
        st.plotly_chart(fig_j, use_container_width=True)

    current_temp = st.session_state.transformer_thermal
    thermal_color = "#00f3ff"
    if current_temp > 80: thermal_color = "#ffae00"
    if current_temp > 100: thermal_color = "#ff0055"
    
    col_thermal, col_tap = st.columns([1, 1])
    with col_thermal:
        with st.container(border=True):
            st.markdown("#### 🔥 TRANSFORMER THERMAL MODEL")
            prog = min(1.0, current_temp / 120.0)
            st.progress(prog)
            st.caption(f"OIL TEMP (Physics Lag): {current_temp:.1f} °C")
            st.markdown(f"<span style='color:{thermal_color}'><b>THERMAL INERTIA ACTIVE</b></span>", unsafe_allow_html=True)
            
    with col_tap:
        with st.container(border=True):
            st.markdown("#### 🎛️ TRANSFORMER CONTROL")
            auto_tap = st.checkbox("🤖 AUTO-TAP (AVR)", value=st.session_state.auto_tap_mode)
            st.session_state.auto_tap_mode = auto_tap
            if auto_tap:
                st.info(f"AVR ACTIVE: {st.session_state.tap_position:.3f} pu")
            else:
                tap_val = st.slider("TAP POSITION (pu)", 0.90, 1.10, st.session_state.tap_position, step=0.01)
                st.session_state.tap_position = tap_val
            tap_fig = make_cyber_plot(list(range(len(st.session_state.history_tap))), st.session_state.history_tap, "TAP CHANGE HISTORY", "#ffae00", height=100)
            st.plotly_chart(tap_fig, use_container_width=True, key="tap_hist")

    with st.expander("🛡️ SEQUENCE COMPONENTS (PHYSICS)", expanded=True):
        seq_c1, seq_c2, seq_c3, seq_c4 = st.columns(4)
        imbalance_alert = "✅ BALANCED"
        if i2 > 0.1: imbalance_alert = "⚠️ UNBALANCED"
        if i0 > 0.1: imbalance_alert = "⚠️ GND FAULT"
        with seq_c1: st.metric("POS SEQ (I1)", f"{i1:.2f} pu")
        with seq_c2: st.metric("NEG SEQ (I2)", f"{i2:.2f} pu")
        with seq_c3: st.metric("ZERO SEQ (I0)", f"{i0:.2f} pu")
        with seq_c4: st.metric("CONDITION", imbalance_alert)
        seq_df = pd.DataFrame({'Component': ['Pos (I1)', 'Neg (I2)', 'Zero (I0)'], 'Magnitude': [i1, i2, i0]})
        fig_seq = go.Figure(go.Bar(x=seq_df['Component'], y=seq_df['Magnitude'], marker_color=['#00f3ff', '#ffae00', '#ff0055']))
        fig_seq.update_layout(height=150, margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), yaxis=dict(showgrid=True, gridcolor='#333'))
        st.plotly_chart(fig_seq, use_container_width=True, key="seq_chart")

    with st.expander("📉 HARMONICS (IEEE 519)", expanded=False):
        pq_col1, pq_col2 = st.columns([1, 3])
        with pq_col1:
            st.session_state.thd_mode = st.toggle("INJECT NOISE", value=st.session_state.thd_mode)
            thd_active = st.session_state.thd_mode or is_local_fault
            disable_filter = not thd_active
            st.session_state.filter_mode = st.toggle("ACTIVE FILTER", value=st.session_state.filter_mode, disabled=disable_filter)
            if disable_filter and st.session_state.filter_mode: st.session_state.filter_mode = False 
            t_wave, v_wave, thd_val = generate_waveform(thd_active, st.session_state.filter_mode)
            status_delta = "CLEAN" if (thd_active and st.session_state.filter_mode) else ("DIRTY" if thd_val > 5.0 else "NOMINAL")
            st.metric("THD", f"{thd_val:.2f} %", delta=status_delta, delta_color="inverse")
        with pq_col2:
            fig_wave = go.Figure()
            if st.session_state.filter_mode and thd_active:
                 _, v_bad, _ = generate_waveform(True, False)
                 fig_wave.add_trace(go.Scatter(x=t_wave*1000, y=v_bad, name="Raw", line=dict(color="#ff0055", width=1, dash='dot')))
            fig_wave.add_trace(go.Scatter(x=t_wave*1000, y=v_wave, name="Filtered", line=dict(color="#00f3ff", width=3)))
            fig_wave.update_layout(title="OSCILLOSCOPE", xaxis_title="ms", yaxis_title="V", height=250, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.3)", font=dict(color="#ccc", family="Orbitron"), xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"))
            st.plotly_chart(fig_wave, use_container_width=True, key="oscilloscope")

    with st.expander("⚡ APFC CONTROL (Capacitor Switching)", expanded=True):
        c_apfc_1, c_apfc_2, c_apfc_3 = st.columns([1, 2, 1])
        with c_apfc_1: st.metric("P.F.", f"{pf_final:.3f}")
        with c_apfc_2:
            auto = st.checkbox("AI AGENT AUTO", value=st.session_state.apfc_auto_mode)
            st.session_state.apfc_auto_mode = auto
            if not auto: 
                steps = st.select_slider("CAPACITOR STEPS (25 kVAR/Step)", options=[0, 25, 50, 75, 100, 125, 150, 175, 200], value=int(st.session_state.capacitor_bank_kvAr))
                st.session_state.capacitor_bank_kvAr = float(steps)
            else:
                 st.info(f"AI Controlling: {st.session_state.capacitor_bank_kvAr:.0f} kVAR Active")
            active_banks = int(st.session_state.capacitor_bank_kvAr // 25)
            bank_visual = "🔋" * active_banks + "⚫" * (8 - active_banks)
            st.write(f"Active Banks: {bank_visual}")
        with c_apfc_3: st.metric("NET VAR", f"{(display_q - st.session_state.capacitor_bank_kvAr):.1f}")
        cap_fig = make_cyber_plot(list(range(len(st.session_state.history_cap))), st.session_state.history_cap, "kVAR INJECTED", "#00ff00", height=150)
        st.plotly_chart(cap_fig, use_container_width=True, key="cap_hist")

    c_chart, c_phasor = st.columns([2, 1])
    with c_chart:
        try:
             hist_data = df_fa_p[view_bus].iloc[max(0, idx-80):idx].tolist()
        except KeyError:
             hist_data = [45] * 80
        st.plotly_chart(make_cyber_plot(list(range(len(hist_data))), hist_data, f"LOAD: {view_bus}", "#00f3ff", delta_val=delta_feeder), use_container_width=True, key="feeder_load_trend")
    with c_phasor:
        st.plotly_chart(draw_phasor(Ia, Id, Ib, Ibd, Ic, Icd), use_container_width=True, key="phasor_diagram")

# --- REAL-TIME TOPOLOGY RENDERER ---
def get_node_sim_data(bus_name, current_idx):
    try:
        p_kw = df_fa_p[bus_name].iloc[current_idx]
        q_kvar = df_fa_p[bus_name].iloc[current_idx] * 0.4
    except KeyError:
        base_load = 50.0
        if "30" in bus_name: base_load = 80.0
        p_kw = base_load + (10 * np.sin(current_idx * 0.1)) + np.random.uniform(-2, 2)
        q_kvar = p_kw * 0.3

    # PV Injection
    pv_out, _, _ = calculate_pv_physics(bus_name, current_idx, st.session_state.room_temp)

    # Use Physics Engine for Voltage (Reverse Power Flow Aware)
    v_pu = calculate_voltage_profile(bus_name, p_kw, q_kvar, st.session_state.tap_position, p_gen_kw=pv_out)
    
    # Net Power for Current Calc
    p_net = p_kw - pv_out
    i_amps = (np.sqrt(p_net**2 + q_kvar**2) / (0.208 * 1.732)) if bus_name not in TRANSFORMER_NODES else 0.0
    pf = p_net / np.sqrt(p_net**2 + q_kvar**2) if p_net > 0 else 1.0
    
    return v_pu, i_amps, p_kw, q_kvar, pf, pv_out

@st.fragment(run_every=speed if st.session_state.run_simulation else None)
def render_topology():
    if st.session_state.run_simulation: 
        st.session_state.idx = (st.session_state.idx + 1) % len(df_raw)
    
    st.markdown("### 🌐 DIGITAL TWIN TOPOLOGY (SLD)")
    col_map, col_details = st.columns([3, 1])
    selected_node = st.selectbox("🎯 SELECT ASSET (Or click on map nodes below)", ["All"] + bus_list, index=0)
    
    edge_traces = []
    x_norm, y_norm = [], []
    x_heavy, y_heavy = [], []
    x_crit, y_crit = [], []
    
    for i, (b1, b2) in enumerate(EDGE_LIST_RAW):
        if b1 in bus_dict and b2 in bus_dict:
            x0, y0 = bus_dict[b1]
            x1, y1 = bus_dict[b2]
            load_factor = ((i + st.session_state.idx) % 50) / 50.0 
            if load_factor > 0.9:
                x_crit.extend([x0, x1, None]); y_crit.extend([y0, y1, None])
            elif load_factor > 0.7:
                x_heavy.extend([x0, x1, None]); y_heavy.extend([y0, y1, None])
            else:
                x_norm.extend([x0, x1, None]); y_norm.extend([y0, y1, None])

    edge_traces.append(go.Scatter(x=x_norm, y=y_norm, line=dict(width=1, color='#00f3ff'), hoverinfo='none', mode='lines', name='Normal Load'))
    edge_traces.append(go.Scatter(x=x_heavy, y=y_heavy, line=dict(width=2, color='#ffae00'), hoverinfo='none', mode='lines', name='Heavy Load'))
    edge_traces.append(go.Scatter(x=x_crit, y=y_crit, line=dict(width=2, color='#ff0055', dash='dot'), hoverinfo='none', mode='lines', name='Critical'))

    node_x, node_y, node_color, node_size, node_sym, node_text = [], [], [], [], [], []
    for bus, (bx, by) in bus_dict.items():
        node_x.append(bx); node_y.append(by)
        
        # --- NODE STYLING LOGIC ---
        if bus in SOLAR_BUS_CONFIG:
            n_type, col, size, sym = "SOLAR BUS", "#ffff00", 14, "circle-x" # Solar gets Yellow
        elif bus in TRANSFORMER_NODES: 
            n_type, col, size, sym = "TRANSFORMER", "#ffae00", 12, "star"
        elif "source" in bus or bus == "bus1": 
            n_type, col, size, sym = "SUBSTATION", "#ffffff", 18, "diamond"
        else: 
            n_type, col, size, sym = "LOAD BUS", "#00f3ff", 6, "circle"
            
        if selected_node != "All":
            if bus == selected_node: col, size = "#ff00ff", 22
            else: col = "rgba(0, 243, 255, 0.3)"
        
        node_color.append(col); node_size.append(size); node_sym.append(sym)
        node_text.append(f"<b>{bus.upper()}</b><br>{n_type}")

    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', text=node_text, hoverinfo='text', marker=dict(symbol=node_sym, color=node_color, size=node_size, line=dict(width=1, color='#000')), name="Assets")

    with col_map:
        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(showlegend=True, legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0.5)', font=dict(color='white')), hovermode='closest', margin=dict(b=0,l=0,r=0,t=0), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=650, dragmode='pan')
        st.plotly_chart(fig, use_container_width=True)

    with col_details:
        if selected_node != "All":
            v, i, p, q, pf, pv_out = get_node_sim_data(selected_node, st.session_state.idx)
            st.markdown(f"""
            <div style="background: rgba(20,20,30,0.9); border: 1px solid #00f3ff; padding: 15px; border-radius: 5px;">
                <h4 style="margin:0; color: #ff00ff;">INSPECTOR</h4>
                <h2 style="margin:0; color: white;">{selected_node.upper()}</h2>
                <hr style="border-color: #333;">
            </div>
            """, unsafe_allow_html=True)
            c_d1, c_d2 = st.columns(2)
            c_d1.metric("VOLTAGE", f"{v:.3f} pu", delta="-1.2%" if v < 0.98 else "OK")
            c_d2.metric("CURRENT", f"{i:.1f} A")
            
            if pv_out > 0:
                 st.metric("SOLAR GENERATION", f"{pv_out:.1f} kW", delta="Active", delta_color="normal")
            
            st.metric("POWER FACTOR", f"{pf:.2f}")
            fig_mini_phasor = go.Figure()
            fig_mini_phasor.add_trace(go.Scatterpolar(r=[0, v], theta=[0, 0], mode='lines+markers', line=dict(color='#00f3ff', width=3), name='V'))
            fig_mini_phasor.add_trace(go.Scatterpolar(r=[0, i/100], theta=[0, -30], mode='lines+markers', line=dict(color='#ffae00', width=3), name='I'))
            fig_mini_phasor.update_layout(polar=dict(radialaxis=dict(visible=False), angularaxis=dict(visible=False), bgcolor="rgba(0,0,0,0)"), paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=10, r=10, t=20, b=20), height=150, showlegend=False, title=dict(text="LOCAL PHASOR", font=dict(size=10, color="#ccc")))
            st.plotly_chart(fig_mini_phasor, use_container_width=True)
            load_pct = min(100, (p/100)*100)
            st.write(f"**LOAD CAPACITY: {load_pct:.1f}%**")
            st.progress(int(load_pct)/100)
        else:
            st.info("👈 Select a node on the map to view real-time telemetry.")

@st.fragment(run_every=speed if st.session_state.run_simulation else None)
def render_ai_dashboard():
    if st.session_state.run_simulation: 
        st.session_state.idx = (st.session_state.idx + 1) % len(df_raw)
        
    if not PROPHET_AVAILABLE or not LSTM_AVAILABLE:
        st.error(f"⚠️ MISSING LIBS: Prophet={PROPHET_AVAILABLE}, LSTM={LSTM_AVAILABLE}")
    else:
        sim_idx = st.session_state.idx
        with st.spinner("AI ENGINE: Loading Cached Models or Training..."):
            m_prophet, full_fcast, m_lstm, lstm_preds, metrics = load_or_train_models(df_raw["Total_Active_Power"].values)
        
        start_hist = max(0, sim_idx - 72)
        end_hist = sim_idx
        start_pred = sim_idx
        end_pred = min(len(full_fcast), sim_idx + 24)
        
        hist_dates = full_fcast['ds'].iloc[start_hist:end_hist]
        hist_actual = df_raw["Total_Active_Power"].iloc[start_hist:end_hist].values
        
        pred_dates = full_fcast['ds'].iloc[start_pred:end_pred]
        prophet_slice = full_fcast['yhat'].iloc[start_pred:end_pred]
        lstm_slice = lstm_preds[start_pred:end_pred] if lstm_preds is not None else []
        
        hist_prophet = full_fcast['yhat'].iloc[start_hist:end_hist].values
        hist_lstm = lstm_preds[start_hist:end_hist]
        
        if len(hist_actual) > 0:
            p_rmse = np.sqrt(mean_squared_error(hist_actual, hist_prophet))
            p_mae = mean_absolute_error(hist_actual, hist_prophet)
            l_rmse = np.sqrt(mean_squared_error(hist_actual, hist_lstm))
            l_mae = mean_absolute_error(hist_actual, hist_lstm)
        else:
            p_rmse, p_mae, l_rmse, l_mae = 0.0, 0.0, 0.0, 0.0

        st.markdown("### 🏆 MODEL PERFORMANCE COMPARISON (LIVE ROLLING WINDOW)")
        cm1, cm2, cm3, cm4 = st.columns(4)
        
        with cm1: st.metric("PROPHET RMSE", f"{p_rmse:.2f}")
        with cm2: st.metric("PROPHET MAE", f"{p_mae:.2f}")
        with cm3: 
            delta_rmse = p_rmse - l_rmse
            st.metric("LSTM RMSE (Deep Learning)", f"{l_rmse:.2f}", delta=f"{-delta_rmse:.2f} (Better)", delta_color="inverse")
        with cm4: 
            delta_mae = p_mae - l_mae
            st.metric("LSTM MAE (Deep Learning)", f"{l_mae:.2f}", delta=f"{-delta_mae:.2f} (Better)", delta_color="inverse")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_dates, y=hist_actual, name="ACTUAL (History)", line=dict(color='#00f3ff', width=2)))
        fig.add_trace(go.Scatter(x=pred_dates, y=prophet_slice, name="PROPHET (Baseline)", line=dict(color='#ffff00', width=2, dash='dot')))
        fig.add_trace(go.Scatter(x=pred_dates, y=lstm_slice, name="LSTM (Deep Learning)", line=dict(color='#ff00ff', width=4)))

        fig.update_layout(
            title=f"REAL-TIME FORECAST COMPARISON (Sim Hour: {sim_idx})", 
            xaxis_title="TIME", yaxis_title="Active Power (kW)",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.3)',
            font=dict(color="#ccc", family="Orbitron"), height=500, 
            xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333")
        )
        st.plotly_chart(fig, use_container_width=True, key="hybrid_forecast")
        st.info("ℹ️ NOTE: The LSTM model is trained once and cached. If you delete 'lstm_model.h5', it will retrain automatically.")

# ----------------------------------------------------------
# 8. MAIN ROUTER
# ----------------------------------------------------------
if nav == "Live Telemetry":
    st.header("🏠 GRID OVERVIEW")
    render_home()

elif nav == "Grid Topology":
    render_topology()

elif nav == "Feeder Analytics":
    st.header("🔌 FEEDER ANALYTICS")
    v_bus = st.selectbox("SELECT BUS", bus_list)
    render_feeder(v_bus)

elif nav == "AI Forecasting":
    st.header("🔮 HYBRID AI PREDICTION (PROPHET vs LSTM)")
    render_ai_dashboard()
