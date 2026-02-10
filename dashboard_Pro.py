import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objs as go
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
    page_title="NEON GRID CONTROL v28.0 (Industrial Twin)", 
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
# CSV FILES CONFIG (Ensure these exist or code will use dummy data logic)
CSV_PATH = "Total_P&Q.csv"
FEEDER_A_P = "FeederA_P.csv"
FEEDER_A_Q = "FeederA_Q.csv"

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
TRANSFORMER_TAU = 20.0  # Thermal time constant (simulation steps)
SOURCE_IMPEDANCE = 0.01 + 0.1j  # pu (Source stiffness)
LINE_IMPEDANCE_PER_UNIT_DIST = 0.005 + 0.002j # pu impedance per unit distance

# ECONOMICS & ENVIRONMENT
TARIFF_RATE = 25.0 # PKR/kWh (Approx)
CARBON_INTENSITY = 0.45 # kg CO2/kWh

# SWING EQUATION CONSTANTS
INERTIA_H = 5.0 # Inertia Constant (MW-s/MVA)
DAMPING_D = 1.0 # Damping Coefficient
SYSTEM_BASE_MVA = 10.0 # Base MVA for per unit calc

# FAULT IMPEDANCE LIBRARY (Z_fault in pu)
FAULT_LIBRARY = {
    "L-G (Line-to-Ground)":   {"Zf": 0.0, "type": "LG", "color": "#ff9100", "icon": "⚡"},
    "L-L (Line-to-Line)":     {"Zf": 0.01, "type": "LL", "color": "#ff5500", "icon": "🔥"},
    "L-L-G (2-Line-Ground)":  {"Zf": 0.0, "type": "LLG", "color": "#ff0000", "icon": "💥"},
    "L-L-L (3-Phase Bolted)": {"Zf": 0.0, "type": "LLL", "color": "#ff0055", "icon": "☠️"}
}

# --- SOLAR PV CONFIGURATION ---
SOLAR_BUS_CONFIG = {
    "bus1005": 100.0,
    "bus1010": 150.0,
    "bus2008": 80.0,
    "bus2015": 120.0,
    "bus2025": 200.0, 
    "bus3006": 50.0,
    "bus3020": 100.0,
    "bus3035": 150.0,
    "bus3050": 90.0,
    "bus3100": 75.0
}

# --- BESS (BATTERY) CONFIGURATION ---
BESS_CAPACITY_KWH = 500.0
BESS_MAX_POWER = 100.0 # kW Charge/Discharge rate

# ----------------------------------------------------------
# 3. TOPOLOGY & DATA PARSING
# ----------------------------------------------------------
def parse_bus_coords(dss_content):
    """Parses Buscoords.dss text content into a dict {bus_name: [x, y]}"""
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
if "rotor_angle" not in st.session_state: st.session_state.rotor_angle = 0.0 # Delta (radians)
if "mech_power" not in st.session_state: st.session_state.mech_power = 5000.0 # Pm (kW)

if "transformer_thermal" not in st.session_state: st.session_state.transformer_thermal = 40.0 # deg C

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
# NEW: Auto Tap
if "auto_tap_mode" not in st.session_state: st.session_state.auto_tap_mode = False

# NEW: Cyber Attack
if "fdi_attack" not in st.session_state: st.session_state.fdi_attack = False

# NEW: Cloud Shading & BESS
if "cloud_shading" not in st.session_state: st.session_state.cloud_shading = False
if "bess_soc" not in st.session_state: st.sessio