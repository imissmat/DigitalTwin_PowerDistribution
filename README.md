# Digital Twin For Smart Power Distribution

Final Year Project (BS Electrical Engineering)

### MATLAB Power Flow Results File Ready to download link:
https://drive.google.com/file/d/1753HejcxxvsqdLUm8445e8i3gsiKm7Cj/view?usp=sharing

## How to RUN?
1. Download
2. Extract / Unzip
3. In terminal, type 'pip install -r requirements.txt'.
4. Then in terminal, type 'streamlit run dashboard.py' (older) and 'dashboar_Pro' (newer) to start the dashboard.
5. Then it will open the browser and start the simulations and dashboard.


[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly)](https://plotly.com/)

## Overview

**AZU Digital Twin** is a high-fidelity, real-time Cyber-Physical System (CPS) simulation for modern smart grids. Unlike traditional static dashboards, this framework integrates a rigorous **Physics Engine** (Swing Equation, Thermal Inertia) with a **Data-Driven SCADA Layer** (State Estimation, AI Forecasting) to create a "living" digital replica of a distribution network.

This tool is designed for operational analysis, operator training, and research into grid stability, cyber-resilience (FDI attacks), and renewable energy integration.

---

## Key Features

### 1. Core Physics Engine
* **Transient Stability:** Implements the **2nd-Order Differential Swing Equation** to simulate realistic generator frequency inertia and rotor angle oscillations.
* **Thermal Inertia:** Uses differential heating equations (IEEE C57.91) to model transformer temperature rise/fall based on loading history.
* **Impedance-Based Voltage:** Calculates accurate nodal voltage drops ($V = I \times Z$) based on feeder distance and topology.

### 2. Advanced State Estimation (SE)
* **Weighted Least Squares (WLS):** Features a full Iterative Gauss-Newton solver to reconstruct the "True State" from noisy SCADA measurements.
* **Bad Data Detection:** Monitors the **Residual Cost Function $J(x)$** to detect anomalies and sensor failures.
* **Cyber-Resilience:** Visualizes the impact of **False Data Injection (FDI)** attacks and the estimator's ability to filter them.

### 3. Hybrid AI Load Forecasting
* **Dual-Model Architecture:** Combines **Meta Prophet** (for seasonality/trend) and **LSTM Deep Learning** (for non-linear short-term dynamics).
* **Live Metrics:** Calculates RMSE and MAE in real-time on a rolling window to benchmark model performance.

### 4. Renewable & Operational Logic
* **Smart Solar Dispatch:** Simulates a 24-hour cycle with "Master/Slave" handover logic between Grid and Solar generation.
* **Auto-Recloser (Device 79):** Simulates protection logic states (Trip $\rightarrow$ Wait $\rightarrow$ Reclose $\rightarrow$ Lockout) for fault management.
* **Automatic Voltage Regulation (AVR):** Automated tap-changer logic to stabilize voltage during high solar penetration.

---

## System Architecture

The framework operates on a closed-loop feedback system:

1.  **Data Ingestion:** Loads historical P/Q profiles from CSV.
2.  **Physics Core:** Calculates ideal states ($V, f, \delta, T$) based on load and topology.
3.  **SCADA Layer:** Adds Gaussian noise to simulate real-world sensor telemetry.
4.  **Control Plane:** Runs State Estimation, Protection Logic, and AI Inference.
5.  **HMI (Dashboard):** Visualizes the synthesized data for the operator.

---

## Installation

### Prerequisites
* Python 3.8 or higher
* PIP (Python Package Manager)

### 1. Clone the Repository
```bash
git clone https://github.com/imissmat/DigitalTwin_PowerDistribution/
cd DigitalTwin_PowerDistribution
